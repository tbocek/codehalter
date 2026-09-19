package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/BurntSushi/toml"

	"github.com/tbocek/codehalter/acp"
	"github.com/tbocek/codehalter/llm"
)

const sessionDir = ".codehalter"

type Message struct {
	Role     string      `toml:"role"`
	Content  string      `toml:"content"`
	Images   []ImageData `toml:"images,omitempty"`
	ToolUses []ToolUse   `toml:"tool_uses,omitempty"`
	// StartedAt is the wall-clock time the message was created. For user
	// turns this is when the prompt arrived; for assistant turns this is
	// when the first llmStream call of that turn started. Always populated
	// for newly-created messages; older session files decode with the zero
	// value (omitempty keeps them clean).
	StartedAt time.Time `toml:"started_at,omitempty"`
	// DurationMs is meaningful only for assistant turns: cumulative wall-clock
	// time spent in llmStream calls for that turn (the agentic loop may run
	// multiple llmStream → tools → llmStream cycles, all attributed here).
	// Excludes tool execution time — those are timed individually on each
	// ToolUse. Zero on user turns.
	DurationMs int64 `toml:"duration_ms,omitempty"`
	// Phase tags which pipeline stage produced this message: "plan",
	// "execute", "verify" or "document". Empty on user turns
	// and on legacy entries from before this field existed.
	Phase string `toml:"phase,omitempty"`
	// PromptTokens is the server-reported prompt_tokens of the call that produced
	// this assistant message — the cumulative context size at that point. The
	// 400-recovery keep-window (keepWindowStart) sizes by these real tokens. 0 on
	// user turns, on backends that don't report usage, and on legacy entries.
	PromptTokens int `toml:"prompt_tokens,omitempty"`
}

type ImageData struct {
	// ID is the content-addressed handle ("img_<sha256[:8] hex>") assigned at
	// extraction. The bytes themselves live in <cwd>/.codehalter/images/<id>.<ext>;
	// buildLLMContext re-reads them every turn (so the wire shape stays
	// byte-identical for the prefix cache) and view_image fetches them after
	// compaction has dropped the owning message.
	ID       string `toml:"id,omitempty"`
	MimeType string `toml:"mime_type"`
}

type ToolUse struct {
	// ID is a per-process handle ("tu_<n>") generated when the tool loop
	// records the call. Empty for ToolUses loaded from older session files.
	ID string `toml:"id,omitempty"`
	// CallID is the tool_call id the MODEL emitted (OpenAI linkage between the
	// assistant's tool call and its result). It's what the live wire used, so we
	// replay it verbatim from history — using the internal useID (ID) instead
	// would change the bytes and bust the prefix cache at every phase boundary.
	// Empty for older sessions / models that don't send ids → falls back to ID.
	CallID string `toml:"call_id,omitempty"`
	Name   string `toml:"name"`
	Input  string `toml:"input"`
	Output string `toml:"output"`
	// Failed is set by the tool handler when it observed a hard failure
	// (e.g. run_task saw a non-zero exit). Authoritative — verify uses it
	// to override an LLM "success=true" verdict when codehalter itself
	// knows the call failed. omitempty keeps older session files clean.
	Failed bool `toml:"failed,omitempty"`
	// StartedAt and DurationMs are populated by runToolLoop when the call is
	// dispatched. omitempty keeps older session files decoding cleanly. For
	// deduped cache hits DurationMs is near zero and StartedAt is when the
	// hit happened, not when the original call ran.
	StartedAt  time.Time `toml:"started_at,omitempty"`
	DurationMs int64     `toml:"duration_ms,omitempty"`
	// ImageID is the content-addressed id of an image this call PRODUCED
	// (screenshot). Replay rebuilds the multimodal parts from those STORED
	// bytes instead of re-running the tool: unlike view_image, re-rendering is
	// not pure — the page may have changed since — and different bytes in the
	// middle of the prompt reprocess every message behind them. Empty for
	// every other tool and for older sessions.
	ImageID string `toml:"image_id,omitempty"`
}

// summariseTask is one queued background-summariser job: the messages of one
// completed turn to feed the structured-turn prompt. The connection is picked
// at enqueue time so the runner doesn't have to reach back into the agent.
type summariseTask struct {
	Turn []Message
	Conn *llm.Conn
	// Prompt is the SUMMARISE.md body for the paste-mode render (unused in
	// prefix-extension mode, where it is already baked into Msgs).
	Prompt string
	// Msgs, when non-nil, switches the note generation to prefix-extension
	// mode: the full wire context frozen at turn end plus the summarise
	// instruction (see appendSummariseMsgs). Frozen at ENQUEUE time so a
	// queued task still summarises exactly its own turn even when the next
	// turn has already started by the time the worker runs it.
	Msgs []llm.Message
}

type Session struct {
	ID        string    `toml:"id"`
	Cwd       string    `toml:"cwd"`
	CreatedAt time.Time `toml:"created_at"`
	Summary   string    `toml:"summary,omitempty"`
	// FoldedSummary is a shorter rewrite of Summary, produced in the background
	// after a compaction (scheduleSummaryFold) and consumed as the BASE of the
	// next one. Empty means "no rewrite ready", which is not an error: the next
	// compaction then concatenates onto Summary exactly as it always did.
	//
	// It is a separate field rather than an in-place shrink of Summary because
	// Summary leads every request. Rewriting it between compactions would
	// re-render the entire prompt behind it for a saving nothing reads until the
	// next compaction, which is the one moment the prefix is already gone.
	FoldedSummary string `toml:"folded_summary,omitempty"`
	// Title is the thread name shown in the client, derived from the first user
	// message (see setSessionTitle). Persisted so a reloaded thread keeps the
	// name it was given instead of reverting to its id.
	Title string `toml:"title,omitempty"`
	// SystemPrompt holds the rendered skills + project context that leads
	// every LLM call. Set on the first user turn (see Prompt) and refreshed
	// after each foldHistory rotation — so it survives the summariser
	// (which otherwise compresses skills away) and reflects current
	// .codehalter/SKILL-*.md content. Emitted by buildLLMContext as the
	// leading user message before any Summary.
	SystemPrompt string    `toml:"system_prompt,omitempty"`
	Messages     []Message `toml:"messages"`
	// Shadow holds one structured per-turn note (Goal / Constraints / Progress /
	// Decisions / Next Steps / Critical Context, per SUMMARISE.md) for every
	// COMPLETED turn since the last compaction — produced by the background
	// summariser at each turn boundary. foldHistory folds the whole buffer
	// into Summary when it rotates, so a note exists for every turn it archives.
	// Persisted (toml) so the notes survive a restart; the live context never
	// shows them — they exist only to feed the next compaction.
	Shadow   []string `toml:"shadow,omitempty"`
	filePath string
	// mcpOffer holds the editor's own MCP server list, as it arrived on
	// session/new. Not persisted: it's the client's configuration, re-sent on
	// every connect, and offerMCPImport consumes it once at bootstrap.
	mcpOffer []acp.MCPServer
	// phaseActive/phaseCurrent track the plan UI state. Not persisted.
	// phaseActive=true means a phase entry is showing as in_progress and
	// must be marked completed before Prompt returns; phaseCurrent is the
	// 0-based index into phaseNames it refers to. Guarded by phaseMu, NOT
	// the main session mu — llmStream calls setStatus mid-stream during long
	// calls while session writers hold sess.mu, so reusing sess.mu for phase
	// reads would deadlock.
	phaseMu      sync.Mutex
	phaseActive  bool
	phaseCurrent int
	// planTableShown records that the planner's subtasks already streamed into
	// the transcript as a live table (see planTableSink), so renderPlan prints
	// its heading alone instead of repeating the list. Consumed on read: a later
	// mid-run revision never went through submit_plan, has nothing on screen, and
	// must still render in full. Under phaseMu for the same reason as the fields
	// above, it is written from the SSE read loop.
	planTableShown bool
	// mu serialises the Save() encoder write against concurrent mutators —
	// AddUser/AddAssistant/etc. acquire it before touching persisted fields
	// so a Save() landing in parallel doesn't observe a torn slice. Prompt
	// runs synchronously per session so contention is rare; the lock mainly
	// exists for the background summariser path (which reads Messages while
	// the foreground turn may be appending) and for the encoder invariant
	// inside saveLocked. The phaseMu field above
	// intentionally has its own lock — it doesn't touch persisted state and
	// must not block on mu.
	mu sync.Mutex
	// turnStartIdx is the index into Messages where the current top-level turn
	// begins (set by markTurnStart at runTurn entry, after the human/card prompt
	// is appended). Mid-turn compaction keeps Messages[turnStartIdx:] verbatim
	// and folds only the completed turns before it; synthetic user messages a
	// turn injects (subtask/doc prompts, mid-session skills) all land after
	// turnStartIdx and stay in the in-flight turn. In-memory only: a turn never
	// spans a restart, and the next prompt re-marks it. Guarded by mu.
	turnStartIdx int
	// summariseQueue is a FIFO of completed-turn snapshots waiting for the
	// background summariser. backgroundSummarise enqueues exactly one per turn
	// boundary; a single worker goroutine drains the queue sequentially (the LLM
	// connection's slot semaphore would serialise concurrent runners anyway).
	// summariseUndone counts enqueued-minus-finished; waitSummarise blocks while
	// it exceeds 0 — compaction folds the whole Shadow buffer with no anchor held
	// back, so every note must have landed first. summariseCond is
	// lazy-initialised under summariseMu on first use so tests that construct
	// Session{} directly don't need to know about it.
	summariseMu      sync.Mutex
	summariseQueue   []summariseTask
	summariseRunning bool
	summariseUndone  int
	summariseCond    *sync.Cond
	// ctl is the turn gate: one turn at a time, and how a newer one replaces
	// it (turn.go).
	ctl turnControl
	// turn is the current turn's scratch state (turnState) and lineage the
	// prefix-cache comparison point that outlives it (cacheLineage). Both are
	// guarded by turnMu, a leaf lock: nothing is called while holding it.
	turnMu  sync.Mutex
	turn    turnState
	lineage cacheLineage
	// rt is what the session keeps in memory across turns (sessionRuntime).
	rt sessionRuntime
	// promptSkills is the set of SKILL-*.md filenames folded into the current
	// SystemPrompt. A skill seeded on disk AFTER the prompt was built is injected
	// as a user message (NOT folded into the prompt — that would bust the KV
	// prefix cache) until the next compaction re-renders the prompt. Runtime-only.
	promptSkills []string
	// llmHash is hex sha256 of the concatenated global + project
	// settings.toml contents at the time of the last successful LLM probe.
	// ensureLLM short-circuits the probe when the current hash matches AND
	// we still have a reachable connection — the file the user could have
	// edited hasn't actually changed, no need to re-handshake every prompt.
	// Reset to "" by failed probes so the next prompt re-probes from scratch.
	// In-memory only; a restart pays one extra probe on the first turn.
	llmHash string `toml:"-"`
	// knownStacks is the set of language stacks detectStacks reports for
	// this session's cwd, with the meta-tooling entries (bash, devcontainer)
	// filtered out — those aren't stacks, they're scaffolding every project
	// uses. Populated by checkEnv on every Prompt turn so a stack appearing
	// or disappearing mid-session takes effect on the next turn.
	knownStacks []string `toml:"-"`
	// knownRunners is the set of runner-config kinds detectRunnerConfigs
	// reports for cwd (just/make/npm/cargo/go) — driven purely by config-
	// file presence so we can flag "user has a justfile but `just` not on
	// PATH" as a fixable problem, distinct from "no runner at all".
	knownRunners []string `toml:"-"`
	// capabilitiesShown gates the full capabilities banner to once per session.
	// The first prepare (bootstrap) always emits it, so a session opens with a
	// visible statement of what codehalter found; afterwards routine changes
	// (a tool installed, an MCP server starting, a re-probe) surface as
	// one-line notices / fix cards instead of re-dumping the whole setup
	// screen mid-conversation. Not persisted — a restart re-shows it once.
	capabilitiesShown bool `toml:"-"`
	// formatCardShown gates the formatter-config card to once per session. Every
	// other fix card re-offers each turn until its condition goes away, which is
	// right for "a tool is missing" but wrong here: a project can legitimately
	// want no formatter config, and asking again after every single turn would
	// be pure nagging. Declining costs one card per session; `format_config =
	// false` in settings.toml silences it for good. Not persisted.
	formatCardShown bool `toml:"-"`
}

// turnState is what one turn accumulates and the next must not see. runTurn
// gives every turn a fresh one (startTurn), so none of it needs reset code of
// its own; the zero value is ready and the maps are made on first write.
// Background goroutines (summariser, git commit) add their tokens to whichever
// turn is current. In-memory only.
type turnState struct {
	// seen maps each read_file window, list_files directory and search_text
	// query to the fnv hash of what it returned, so a literal repeat with an
	// unchanged result gets a note instead of silently re-running (repeatedRead).
	// fsWrite drops a path's entries, so a post-edit re-read starts fresh.
	seen map[string]uint64
	// readCursor is, per path, the next 1-based line continue_read serves: set
	// when read_file (or continue_read) returns a partial chunk, cleared at EOF
	// and on a write. Lets continue_read page forward with no line math.
	readCursor map[string]int
	// editFailed holds the paths where edit_file returned "not found". A failed
	// edit means the model's remembered content is stale or inexact, so the next
	// read_file on that path bypasses the readContentInContext guard: the model
	// needs a fresh look to get the exact old_text for a retry. Cleared when the
	// path is re-read.
	editFailed map[string]bool

	// start and humanWaitMs give the turn's active time for the "✅ Done" line:
	// wall clock since start minus the time blocked on user-input cards
	// (doPermissionRequest). stats sums every llmStream call's usage; turnStats
	// fills in its activeMs when it reads it.
	start       time.Time
	humanWaitMs int64
	stats       turnReport
}

// cacheLineage is the previous tool-loop call, which noteCacheLineage compares
// the next call's cache split against.
//
// It deliberately does NOT reset per turn: the comparison point is a property
// of the conversation, not of the turn. Zeroing it at every turn start exempted
// the FIRST call of every turn from the check, which is the one place
// message-list mutations actually land (compaction, a summariser fold, a tool
// result that replays differently than it was sent). An 11.6h session logged
// zero CACHE lines with a 30466-token rewind sitting in it, at exactly such a
// boundary. Not persisted: a restart re-prefills anyway, so there is no cache
// to reason about across process boundaries.
type cacheLineage struct {
	prompt int // its prompt_tokens
	// render is its llm.RenderKey: the template-affecting params it was sent with.
	// It answers the question the token counts raise but cannot settle: a
	// rewind means the prompt was re-rendered, and this says whether WE asked
	// for that (a role switch across differing chat_template_kwargs) or the
	// server did it on its own.
	render string
	// at is when it landed. A rewind cannot say why on token counts alone, and
	// the gap separates the two causes better than anything else in the record:
	// on the 11.6h session that motivated this detector, every stable-rendering
	// rewind sat behind an idle gap (2h06 and 13min), while the calls seconds
	// apart never lost a prefix. Servers reclaim idle slots, and no line of
	// settings.toml prevents that.
	at time.Time
}

// sessionRuntime is what a session keeps across turns but never persists: this
// process's view of the pages it fetched, the files it wrote, the background
// jobs that finished and the /spec loop it runs. A restart legitimately forgets
// all of it. One leaf mutex for the lot.
type sessionRuntime struct {
	mu sync.Mutex
	// webBodies caches the full page text from web_read, keyed by URL, so a
	// range re-call (offset/limit) slices from memory instead of re-fetching.
	webBodies map[string]string
	// webResults caches the rendered output keyed by URL+mode, so a literal
	// repeat skips both the fetch and the re-summarise and the model receives
	// the exact same string (friendly to the prefix cache). Range requests
	// bypass it and go through webBodies.
	webResults map[string]string
	// wroteHash is the hash of the bytes codehalter last wrote to each path, and
	// drifted marks the paths a later read found different. Together they detect
	// a file being rewritten by something outside this session between our write
	// and the model's next look at it: an editor's format-on-save is the usual
	// cause, and the symptom is an edit_file whose old_text was copied from a
	// read that is no longer true. Not per turn: that drift routinely spans turns.
	wroteHash map[string]string
	drifted   map[string]bool
	// bgNotes queues the results of finished run_background jobs until a quiet
	// point (see deliverBgNotesWhenIdle / flushBgNotes). The job died with the
	// process that would have reported it.
	bgNotes []bgNote
	// specFenceDir is the spec directory a running /spec loop has made
	// read-only for the file tools (spec_loop.go); "" when no loop runs.
	specFenceDir string
}

// startTurn gives the session a fresh turnState: nothing one turn saw (dedup,
// cursors, failed edits, stats) leaks into the next. The cache lineage is kept
// on purpose, see cacheLineage.
func (s *Session) startTurn(start time.Time) {
	s.turnMu.Lock()
	s.turn = turnState{start: start}
	s.turnMu.Unlock()
}

// addTurnTokens folds one call's usage into the turn: it sums completion and the
// evaluated (sent-but-not-cached) prompt tokens, and tracks the most recent
// call's full prompt size for the no-cache fallback line. evaluated is -1 when
// the backend reported no cache split (then haveServerCache stays false). prompt
// is the full prompt size, used only for the context-size fallback — never summed.
func (s *Session) addTurnTokens(prompt, completion, evaluated int) {
	s.turnMu.Lock()
	st := &s.turn.stats
	st.completion += completion
	if prompt > 0 {
		st.lastPrompt = prompt
	}
	if evaluated >= 0 {
		st.evaluatedPrompt += evaluated
		st.haveServerCache = true
	}
	s.turnMu.Unlock()
}

// cacheRewindSlack is how far below the previous call's prompt this call's
// cached count may sit before it counts as a rewind. Two things eat into it
// legitimately: llama.cpp always drops the last cache chunk (measured at 4
// tokens), and the end-of-turn summariser, when it lands on the foreground
// connection, extends the prefix with an instruction tail that the next call
// then trims back off. Both are in the low hundreds. Every rewind we have
// actually diagnosed was thousands (3129, 4366, 6871, 9998 on one turn), so a
// threshold here is generous without hiding anything worth reporting.
const cacheRewindSlack = 1024

// idleEvictionSuspect is the gap after which a rewind under an unchanged
// rendering is more likely the server reclaiming an idle slot than anything
// codehalter did. There is no protocol signal for an eviction, so this is a
// judgement call from the record: on the 11.6h session that motivated the
// detector, every same-rendering rewind sat behind a gap of minutes or hours
// (2h06 and 13min), while the hundreds of calls seconds apart never lost a
// prefix. One minute is comfortably above the largest gap a tool loop puts
// between two calls on its own and well below any of the observed evictions.
const idleEvictionSuspect = time.Minute

// wastedCompletionFloor is how much discarded decode a turn has to accumulate
// before the Done line names it. A stream-rule abort throws away a few dozen
// tokens as a matter of course and saying so every turn would be noise; the
// events worth seeing are cap-length stalls and retries, which are thousands.
// 256 tokens is ~7s of decode on the hardware this was measured on.
const wastedCompletionFloor = 256

// cacheRewind is one detected rewind. tokens == 0 means there is nothing to
// report; the other fields exist to answer "and whose fault is it", which the
// token counts on their own cannot: the same prompt/cached pair is produced by
// a rendering change, a rewritten message, and a server-side eviction alike.
type cacheRewind struct {
	tokens        int           // prompt the previous call had already sent, re-read now
	prevRender    string        // the previous call's llm.RenderKey
	renderChanged bool          // ... and this call asked for a different one
	idle          time.Duration // wall gap since the previous call (0 if unknown)
}

// noteCacheLineage folds one tool-loop call's cache split into the turn's
// rewind detector and describes what it found. render is this call's llm.RenderKey
// and now is its wall time; both are recorded for the next call to compare
// against, whether or not this one rewound.
//
// When the rendering changed, the rewind is explained: the two calls asked the
// server for two different renderings of the same conversation, and a rendering
// the server has never held has no shared prefix to hit. When it did NOT
// change, that is just as informative: it rules the settings out and leaves the
// idle gap to separate the remaining two causes, a server-side eviction from
// something rewriting the middle of the prompt (a tool result that replayed
// differently than it was sent, or a template that repositions content).
//
// Only the tool loop feeds this (llm.Conn.CacheLineage), because only the
// tool loop guarantees the premise: each call's message list is the previous
// call's plus an append, so the server should serve the whole previous prompt
// from cache and evaluate just the new tail. The premise holds ACROSS turn
// boundaries too (the next turn's first call is the same list plus one user
// message), which is why the comparison point outlives the turn: the one
// place where it genuinely does not hold drops it by hand (resetCacheLineage,
// called by compaction).
//
// cached < 0 means the backend reported no cache split; then there is nothing
// to compare and the lineage restarts at this call.
func (s *Session) noteCacheLineage(prompt, cached int, render string, now time.Time) cacheRewind {
	s.turnMu.Lock()
	defer s.turnMu.Unlock()
	prev, prevRender, prevAt := s.lineage.prompt, s.lineage.render, s.lineage.at
	s.lineage = cacheLineage{prompt: prompt, render: render, at: now}
	out := cacheRewind{prevRender: prevRender}
	if !prevAt.IsZero() && now.After(prevAt) {
		out.idle = now.Sub(prevAt)
	}
	if prev <= 0 || cached < 0 {
		return out
	}
	// A shrinking prompt is not a rewind: an extension can only grow, so if this
	// call is SMALLER than the previous one the message list was rewritten and
	// there was never a prefix to reuse. Reporting prev-cached there invents
	// enormous faults out of ordinary context resets. Measured against one 11.6h
	// session, this guard is the difference between 16 reported rewinds (10 of
	// them a compaction dropping 115135 tokens to 5970) and the 6 that were real.
	if prompt+cacheRewindSlack < prev {
		return out
	}
	rewound := prev - cached
	if rewound <= cacheRewindSlack {
		return out
	}
	s.turn.stats.cacheRewinds++
	s.turn.stats.cacheRewound += rewound
	out.tokens = rewound
	if prevRender != render {
		s.turn.stats.cacheRewindsRender++
		out.renderChanged = true
	}
	return out
}

// addWastedCompletion records completion tokens that were generated and then
// discarded. See turnReport.wastedCompletion.
func (s *Session) addWastedCompletion(n int) {
	s.turnMu.Lock()
	s.turn.stats.wastedCompletion += n
	s.turnMu.Unlock()
}

// resetCacheLineage drops the comparison point so the next call can't be read
// as a rewind. Compaction calls it: rewriting the front of the context throws
// the prefix away by design, and reporting that as a fault would be crying
// wolf at the one moment the user was already told what happened.
func (s *Session) resetCacheLineage() {
	s.turnMu.Lock()
	s.lineage = cacheLineage{}
	s.turnMu.Unlock()
}

// addTurnTiming sums one call's eval/gen times.
func (s *Session) addTurnTiming(promptMs, genMs int64) {
	s.turnMu.Lock()
	s.turn.stats.promptMs += promptMs
	s.turn.stats.genMs += genMs
	s.turnMu.Unlock()
}

// addHumanWait records time spent blocked on a user-input card so it can be
// excluded from the turn's active time.
func (s *Session) addHumanWait(d time.Duration) {
	s.turnMu.Lock()
	s.turn.humanWaitMs += d.Milliseconds()
	s.turnMu.Unlock()
}

// turnReport is the end-of-turn accounting for the "✅ Done" line, summed over
// every llmStream call of the turn.
type turnReport struct {
	activeMs   int64 // wall clock minus time waiting on the user; set by turnStats
	completion int   // Σ completion_tokens
	// evaluatedPrompt is Σ of each call's prompt_tokens − cached_tokens: the
	// real prompt work. The gross prompt_tokens are deliberately NOT summed (that
	// re-counts the cached prefix every call). When no backend reports a cache
	// split, haveServerCache stays false and the line falls back to lastPrompt.
	evaluatedPrompt int
	lastPrompt      int   // final context size (last call's prompt_tokens)
	haveServerCache bool  // a backend reported the cache split
	promptMs        int64 // eval time (server prompt_ms, else TTFT)
	genMs           int64 // generation time
	// cacheRewinds counts the tool-loop calls that had to re-read prompt the
	// previous call already sent (noteCacheLineage); cacheRewound sums those
	// tokens; cacheRewindsRender is how many of them we caused ourselves by
	// changing the template params between the two calls.
	cacheRewinds       int
	cacheRewound       int
	cacheRewindsRender int
	// wastedCompletion is decode the harness threw away: a <think> stall, a
	// stream-rule abort, a cap retry. It is already inside completion, so
	// without a name of its own it reads as productive output. It is usually the
	// largest single cost in a bad turn: one measured stall burned 8192 tokens,
	// which at that server's 37.7 tok/s is 3.6 min, against 57 s for the prefix
	// loss the same event caused.
	wastedCompletion int
}

// turnStats returns the current turn's report. activeMs is 0 before the first
// startTurn.
func (s *Session) turnStats() turnReport {
	s.turnMu.Lock()
	defer s.turnMu.Unlock()
	if s.turn.start.IsZero() {
		return turnReport{}
	}
	r := s.turn.stats
	r.activeMs = max(time.Since(s.turn.start).Milliseconds()-s.turn.humanWaitMs, 0)
	return r
}

// recallWebBody returns a cached page body if the URL was fetched earlier in
// this session. Lets range-style web_read calls slice from cache without
// re-issuing the HTTP request.
func (s *Session) recallWebBody(url string) (string, bool) {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	b, ok := s.rt.webBodies[url]
	return b, ok
}

// rememberWebBody stores the full raw page text from a web_read / web_read_raw
// fetch. Overwrites on re-fetch so a refresh updates the cache; this is what
// the model wants — if it asked for a fresh fetch it should see fresh content
// on subsequent range views.
func (s *Session) rememberWebBody(url, body string) {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	if s.rt.webBodies == nil {
		s.rt.webBodies = make(map[string]string)
	}
	s.rt.webBodies[url] = body
}

// webResultKey distinguishes summarized vs. raw output for the same URL so
// web_read and web_read_raw don't collide in the result cache.
func webResultKey(url string, summarize bool) string {
	if summarize {
		return url + "\x00summary"
	}
	return url + "\x00raw"
}

// recallWebResult returns a previously-rendered web_read / web_read_raw output
// for the same URL+mode. Lets the second call on a duplicate URL skip fetch
// and re-summarize entirely.
func (s *Session) recallWebResult(url string, summarize bool) (string, bool) {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	r, ok := s.rt.webResults[webResultKey(url, summarize)]
	return r, ok
}

// rememberWebResult stores the final rendered output for a URL+mode so a
// duplicate call returns the byte-identical string without redoing any work.
func (s *Session) rememberWebResult(url string, summarize bool, out string) {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	if s.rt.webResults == nil {
		s.rt.webResults = make(map[string]string)
	}
	s.rt.webResults[webResultKey(url, summarize)] = out
}

func loadSession(cwd string, id string) (*Session, error) {
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// TOML refuses invalid UTF-8 outright, so one bad byte made the whole
	// session unloadable: a string cut through a multibyte character before
	// every cut snapped to rune boundaries (tailUTF8), or a command's raw output.
	// Each bad byte becomes U+FFFD, which is exactly what the model already saw:
	// Go's JSON encoder sends every invalid byte as \ufffd, so the repaired
	// session renders to the same wire bytes and keeps its cached prefix.
	if !utf8.Valid(data) {
		slog.Warn("session file has invalid UTF-8, replacing it with U+FFFD", "path", path)
		var fixed bytes.Buffer
		for len(data) > 0 {
			r, n := utf8.DecodeRune(data)
			if r == utf8.RuneError && n == 1 {
				fixed.WriteString("\uFFFD")
			} else {
				fixed.Write(data[:n])
			}
			data = data[n:]
		}
		data = fixed.Bytes()
	}
	var s Session
	if _, err := toml.Decode(string(data), &s); err != nil {
		return nil, err
	}
	s.ID = id
	s.Cwd = cwd
	s.filePath = path
	return &s, nil
}

// newSessionWithID resurrects a session under a Zed-supplied id (used by
// LoadSession when the .toml is missing). The file appears on first Save() —
// typically from prompt.go after the first user message. Zed opens an agent
// connection per editor tab; if a tab is opened and never prompted, this
// avoids leaving an empty stub session on disk that would clutter
// listSessions.
func newSessionWithID(cwd string, id string) *Session {
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)
	return &Session{
		ID:        id,
		Cwd:       cwd,
		CreatedAt: time.Now(),
		filePath:  path,
	}
}

func newSession(cwd string) (*Session, error) {
	if err := os.MkdirAll(filepath.Join(cwd, sessionDir), 0755); err != nil {
		return nil, fmt.Errorf("creating session dir: %w", err)
	}
	now := time.Now()
	// Second granularity collides when one session is opened in the same second
	// as the last, which is exactly what the CLI's /new does. Reusing that id
	// would overwrite the previous session's file, so take the next free name.
	base := now.Format("20060102_150405")
	id := base
	var path string
	for n := 2; ; n++ {
		path = filepath.Join(cwd, sessionDir, fmt.Sprintf("session_%s.toml", id))
		if _, err := os.Stat(path); err != nil {
			break // free, or unreadable: the first save reports the real problem
		}
		id = fmt.Sprintf("%s_%d", base, n)
	}
	return &Session{
		ID:        id,
		Cwd:       cwd,
		CreatedAt: now,
		filePath:  path,
	}, nil
}

func (s *Session) AddUser(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "user", Content: text, StartedAt: time.Now()})
}

func (s *Session) AddUserWithImages(text string, images []ImageData) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "user", Content: text, Images: images, StartedAt: time.Now()})
}

func (s *Session) AddAssistant(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "assistant", Content: text, StartedAt: time.Now()})
}

func (s *Session) AddAssistantWithTools(text string, tools []ToolUse) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "assistant", Content: text, ToolUses: tools, StartedAt: time.Now()})
}

// AppendToolUse adds a tool use to the last assistant message, creating one if needed.
func (s *Session) AppendToolUse(tu ToolUse) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.Messages) == 0 || s.Messages[len(s.Messages)-1].Role != "assistant" {
		s.Messages = append(s.Messages, Message{Role: "assistant"})
	}
	last := &s.Messages[len(s.Messages)-1]
	last.ToolUses = append(last.ToolUses, tu)
}

// markEditFailed records that edit_file failed with "not found" for path,
// allowing the next read_file on that path to bypass readContentInContext.
func (s *Session) markEditFailed(path string) {
	s.turnMu.Lock()
	if s.turn.editFailed == nil {
		s.turn.editFailed = map[string]bool{}
	}
	s.turn.editFailed[path] = true
	s.turnMu.Unlock()
}

// clearEditFailed clears the edit-failed flag for path and returns whether
// it was set. Called by serveRead so the bypass fires exactly once per failure.
func (s *Session) clearEditFailed(path string) bool {
	s.turnMu.Lock()
	defer s.turnMu.Unlock()
	if s.turn.editFailed[path] {
		delete(s.turn.editFailed, path)
		return true
	}
	return false
}

// externalChangeNote is the one-line warning handed to the model when a file it
// is about to work on was rewritten behind us. It names the likely cause because
// the model cannot see the editor: without it, one measured session spent three
// and a half minutes writing python to work out why its edits stopped matching.
const externalChangeNote = "\n\n[NOTE: this file changed on disk after codehalter wrote it — something outside this session rewrote it, and an editor's format-on-save is the usual cause. Any old_text you remember from before that write may no longer match; copy it from THIS read.]"

// recordWrite remembers the exact bytes codehalter just wrote to path, which is
// what a later read is compared against. Any pending drift note for the path is
// dropped: we have just overwritten whatever the other writer did, so warning
// about it would send the model looking for a difference that is gone.
func (s *Session) recordWrite(path, content string) {
	sum := sha256.Sum256([]byte(content))
	s.rt.mu.Lock()
	if s.rt.wroteHash == nil {
		s.rt.wroteHash = map[string]string{}
	}
	s.rt.wroteHash[path] = string(sum[:])
	delete(s.rt.drifted, path)
	s.rt.mu.Unlock()
}

// checkExternalChange compares a fresh full read against the bytes we last wrote
// to that path and, on a mismatch, arms a one-time note for takeDriftNote. The
// stored hash is advanced to the content just read, so each distinct drift is
// reported once rather than on every read until the next write.
//
// Paths codehalter never wrote are not tracked: a file changing before we have
// touched it is not drift, it is just the file.
func (s *Session) checkExternalChange(path, content string) {
	sum := sha256.Sum256([]byte(content))
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	prev, ok := s.rt.wroteHash[path]
	if !ok || prev == string(sum[:]) {
		return
	}
	s.rt.wroteHash[path] = string(sum[:])
	if s.rt.drifted == nil {
		s.rt.drifted = map[string]bool{}
	}
	s.rt.drifted[path] = true
}

// takeDriftNote returns the pending external-change note for path, or "" when
// there is none, and clears it. Tools append it to their own result so the
// warning arrives attached to the content it is about.
func (s *Session) takeDriftNote(path string) string {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	if !s.rt.drifted[path] {
		return ""
	}
	delete(s.rt.drifted, path)
	return externalChangeNote
}

// bgNote is one finished background job: a line for the user and the full
// note (exit code, log path, log tail) for the model.
type bgNote struct {
	line, full string
}

func (s *Session) addBgNote(n bgNote) {
	s.rt.mu.Lock()
	s.rt.bgNotes = append(s.rt.bgNotes, n)
	s.rt.mu.Unlock()
}

func (s *Session) hasBgNotes() bool {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	return len(s.rt.bgNotes) > 0
}

func (s *Session) takeBgNotes() []bgNote {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	notes := s.rt.bgNotes
	s.rt.bgNotes = nil
	return notes
}

// repeatedResult records that the tool call named by key returned a result
// hashing to sum, and reports whether the same call already returned exactly
// that earlier this turn. read_file, list_files and search_text use it to flag
// a literal repeat with readUnchangedMarker instead of silently re-running.
func (s *Session) repeatedResult(key string, sum uint64) bool {
	s.turnMu.Lock()
	defer s.turnMu.Unlock()
	prev, ok := s.turn.seen[key]
	if s.turn.seen == nil {
		s.turn.seen = map[string]uint64{}
	}
	s.turn.seen[key] = sum
	return ok && prev == sum
}

// readContentInContext reports whether the exact bytes `content` are still
// present in the live message window as a prior read_file/continue_read result
// the model can scroll back to. Compaction trims s.Messages, so an archived read
// returns false — the model genuinely needs those bytes re-served. Callers
// restrict this to content that fits whole in context (≤ liveExemptCap); a
// clipped >32 KB read isn't fully present and must not be treated as available.
// Only read tools count: a run_command that happened to echo the same bytes
// isn't the file "in context" for navigation purposes.
func (s *Session) readContentInContext(content string) bool {
	if content == "" {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.Messages {
		for _, tu := range s.Messages[i].ToolUses {
			if tu.Name == "read_file" || tu.Name == "continue_read" {
				if strings.Contains(tu.Output, content) {
					return true
				}
			}
		}
	}
	return false
}

// enqueueSummarise appends a task to the summariser queue and starts the
// single worker goroutine if one isn't already draining it. runner is called
// sequentially for each task in FIFO order; the worker exits when the queue
// empties. summariseUndone tracks queued + in-flight tasks for the joiner
// (waitSummarise). Caller holds no session locks; this method manages its
// own.
func (s *Session) enqueueSummarise(t summariseTask, runner func(summariseTask)) {
	s.summariseMu.Lock()
	s.summariseQueue = append(s.summariseQueue, t)
	s.summariseUndone++
	if s.summariseRunning {
		s.summariseMu.Unlock()
		return
	}
	s.summariseRunning = true
	s.summariseMu.Unlock()

	go func() {
		for {
			s.summariseMu.Lock()
			if len(s.summariseQueue) == 0 {
				s.summariseRunning = false
				s.summariseMu.Unlock()
				return
			}
			next := s.summariseQueue[0]
			s.summariseQueue = s.summariseQueue[1:]
			s.summariseMu.Unlock()

			runner(next)

			s.summariseMu.Lock()
			s.summariseUndone--
			if s.summariseCond != nil {
				s.summariseCond.Broadcast()
			}
			s.summariseMu.Unlock()
		}
	}()
}

// waitSummarise blocks until no summariser task remains outstanding (queued or
// in-flight). foldHistory uses it before folding: every completed turn's
// note must be in the Shadow buffer, since compaction folds the whole buffer
// with no anchor held back. Only ever called past the trigger check, so a
// below-budget turn never stalls on it.
func (s *Session) waitSummarise() {
	s.summariseMu.Lock()
	defer s.summariseMu.Unlock()
	if s.summariseCond == nil {
		s.summariseCond = sync.NewCond(&s.summariseMu)
	}
	for s.summariseUndone > 0 {
		s.summariseCond.Wait()
	}
}

// appendShadow adds one completed-turn note to the Shadow buffer. Guarded by mu
// (same lock as Messages/Save) so a note landing from the background worker
// can't tear a concurrent encode.
func (s *Session) appendShadow(chunk string) {
	chunk = strings.TrimSpace(chunk)
	if chunk == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Shadow = append(s.Shadow, chunk)
}

// drainShadow returns every accumulated turn note joined with blank-line
// separators and clears the buffer. No anchor is held back: compaction rotates
// out exactly the turns these notes cover, so all of them belong in Summary.
// Returns "" when the buffer is empty (no completed turn has a note yet).
func (s *Session) drainShadow() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.Shadow) == 0 {
		return ""
	}
	out := strings.Join(s.Shadow, "\n\n")
	s.Shadow = nil
	return out
}

// markTurnStart records where the current top-level turn begins: the index of
// the just-appended human/card prompt. Called at runTurn entry. Mid-turn
// compaction reads turnStartIdx to decide what to keep verbatim.
func (s *Session) markTurnStart() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turnStartIdx = len(s.Messages) - 1
	if s.turnStartIdx < 0 {
		s.turnStartIdx = 0
	}
}

// lastAssistantIndex returns the index of the most recent assistant message in
// the in-flight large turn (at or after turnStartIdx), i.e. the start of the
// unfinished small turn. The 400 recovery folds everything before it and keeps
// it verbatim. Falls back to turnStartIdx when the turn has no assistant
// message yet (its first call 400'd), so foldHistory then keeps the whole
// in-flight turn rather than slicing into a prompt-only window.
func (s *Session) lastAssistantIndex() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	start := s.turnStartIdx
	if start < 0 || start > len(s.Messages) {
		start = 0
	}
	for i := len(s.Messages) - 1; i >= start; i-- {
		if s.Messages[i].Role == "assistant" {
			return i
		}
	}
	return start
}

// recordLastPromptTokens stamps the server-reported prompt_tokens of the call
// that just produced the trailing assistant message onto it (the cumulative
// context size at that call), so keepWindowStart can size the 400-recovery keep
// window by REAL tokens. No-op when the backend reports no usage.
func (s *Session) recordLastPromptTokens() {
	s.turnMu.Lock()
	pt := s.turn.stats.lastPrompt
	s.turnMu.Unlock()
	if pt <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if n := len(s.Messages); n > 0 && s.Messages[n-1].Role == "assistant" {
		s.Messages[n-1].PromptTokens = pt
	}
}

// keepWindowStart returns the index the 400 recovery should keep verbatim FROM:
// the unfinished small turn (the last assistant message, always kept) plus the
// most recent completed small turns whose combined cost stays under
// maxCompletedTokens. Sizing uses the server's real per-call prompt_tokens
// (Message.PromptTokens = cumulative context at each call), so the cost of
// keeping from K is PromptTokens[unfinished] - PromptTokens[K]. Everything older
// — earlier small turns and all prior large turns — folds into Summary, so an
// oversized in-flight turn is NOT kept whole (that was the 194 KB bug). When the
// server reports no token usage (PromptTokens == 0), it conservatively keeps only
// the unfinished small turn rather than guess.
func (s *Session) keepWindowStart(maxCompletedTokens int) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := len(s.Messages)
	if n == 0 {
		return 0 // nothing to keep or fold
	}
	start := s.turnStartIdx
	if start < 0 || start >= n {
		start = 0 // turnStartIdx past the end (or unset) → scan from the top
	}
	// unfinished small turn = last assistant message (always kept)
	last := start
	for i := n - 1; i >= start; i-- {
		if s.Messages[i].Role == "assistant" {
			last = i
			break
		}
	}
	ref := s.Messages[last].PromptTokens
	if ref <= 0 {
		return last // no server token data → keep only the unfinished small turn
	}
	// Walk back through completed small turns, keeping the most recent ones while
	// (ref - PromptTokens[K]) stays under budget. A PromptTokens that RISES going
	// back marks a prior fold boundary (the context was reset there) — stop, don't
	// keep pre-fold messages verbatim.
	prev := ref
	keep := last
	for i := last - 1; i >= start; i-- {
		if s.Messages[i].Role != "assistant" {
			continue
		}
		pt := s.Messages[i].PromptTokens
		if pt <= 0 || pt > prev || ref-pt > maxCompletedTokens {
			break
		}
		keep = i
		prev = pt
	}
	return keep
}

// UpsertLastAssistant sets the content of the trailing assistant message,
// or appends a new one if the last message is not already an assistant turn.
func (s *Session) UpsertLastAssistant(content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.Messages) > 0 && s.Messages[len(s.Messages)-1].Role == "assistant" {
		s.Messages[len(s.Messages)-1].Content = content
		return
	}
	s.Messages = append(s.Messages, Message{Role: "assistant", Content: content, StartedAt: time.Now()})
}

// appendAssistantNote concatenates a short note (e.g. "Understood: A",
// "User declined execution") onto the trailing assistant message instead of
// creating a fresh assistant turn. Two consecutive assistant messages would
// break strict role alternation; UpsertLastAssistant overwrites in place,
// preserving the planner's JSON content plus any prior tool uses.
func appendAssistantNote(sess *Session, note string) {
	if sess == nil || note == "" {
		return
	}
	sess.mu.Lock()
	var existing string
	if len(sess.Messages) > 0 && sess.Messages[len(sess.Messages)-1].Role == "assistant" {
		existing = sess.Messages[len(sess.Messages)-1].Content
	}
	sess.mu.Unlock()
	if existing != "" {
		sess.UpsertLastAssistant(existing + "\n\n" + note)
	} else {
		sess.UpsertLastAssistant(note)
	}
}

// MarkLastAssistantTiming stamps timing and phase onto the trailing assistant
// message, creating one if no assistant turn is currently the latest entry.
// Used by runToolLoop at the end of an assistant turn so the .toml records
// when generation started, how much wall-clock the LLM calls took, and which
// phase produced the turn. started.IsZero() preserves any existing stamp
// (idempotent — repeated calls with a zero start time don't clobber).
func (s *Session) MarkLastAssistantTiming(started time.Time, durationMs int64, phase string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.Messages) == 0 || s.Messages[len(s.Messages)-1].Role != "assistant" {
		s.Messages = append(s.Messages, Message{Role: "assistant"})
	}
	m := &s.Messages[len(s.Messages)-1]
	if !started.IsZero() {
		m.StartedAt = started
	}
	m.DurationMs = durationMs
	m.Phase = phase
}

func (s *Session) Save() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.saveLocked()
}

// saveOrLog persists the session, logging on failure instead of returning the
// error. Most save sites are best-effort (after each AddUser/AddAssistant in
// the tool loops) where a failure isn't worth unwinding the turn — but it must
// NOT vanish: a full disk or read-only .codehalter should be loud. Replaces the
// former `_ = sess.Save()` discards so the failure is never silent.
func (s *Session) saveOrLog() {
	if err := s.Save(); err != nil {
		slog.Error("session save failed", "id", s.ID, "path", s.filePath, "err", err)
	}
}

// rotate archives the session as it currently is to a new "session_archive_*"
// file, then resets s in place to carry only `keep` raw messages and `summary`
// as the rolled-up prior context. The live session keeps its own ID and
// filePath; only its on-disk contents change. Returns the archive's id.
// rotate itself takes no lock; its sole caller, foldHistory, runs mid-turn on a
// context-overflow 400 where no concurrent writer exists. The follow-up
// live-session Save() (by the caller) does lock.
func (s *Session) rotate(keep []Message, summary string) (string, error) {
	archiveID := fmt.Sprintf("archive_%s_%d", s.ID, time.Now().UnixNano())
	archivePath := filepath.Join(s.Cwd, sessionDir, fmt.Sprintf("session_%s.toml", archiveID))
	archive := &Session{
		ID:           archiveID,
		Cwd:          s.Cwd,
		CreatedAt:    s.CreatedAt,
		Summary:      s.Summary,
		SystemPrompt: s.SystemPrompt,
		Messages:     s.Messages,
		filePath:     archivePath,
	}
	// Fresh struct, no concurrent access — the encoder invariant on
	// saveLocked is vacuously satisfied.
	if err := archive.saveLocked(); err != nil {
		return "", err
	}
	s.Summary = summary
	s.Messages = keep
	return archiveID, nil
}

// sessionFilePath returns where this session's toml/log of basename `name`
// should be written: the project's .codehalter/ directory.
func (s *Session) sessionFilePath(name string) string {
	return filepath.Join(s.Cwd, sessionDir, name)
}

// saveLocked writes the session to disk. Caller must hold s.mu (or own the
// session exclusively, e.g. a freshly-constructed archive in rotate()).
func (s *Session) saveLocked() error {
	f, err := os.Create(s.sessionFilePath(filepath.Base(s.filePath)))
	if err != nil {
		return err
	}
	if err := toml.NewEncoder(f).Encode(s); err != nil {
		f.Close() // best-effort; the encode error is the real failure
		return err
	}
	// Return the close error: a failed flush on close means a truncated /
	// corrupt session TOML, which the encode step alone won't surface.
	return f.Close()
}

func listSessions(cwd string) ([]acp.SessionInfo, error) {
	dir := filepath.Join(cwd, sessionDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var sessions []acp.SessionInfo
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "session_") || !strings.HasSuffix(e.Name(), ".toml") {
			continue
		}
		// Skip post-rotation archives (and subagent sessions an older
		// codehalter left behind): on disk for inspection, not for the picker.
		if strings.HasPrefix(e.Name(), "session_sub_") ||
			strings.HasPrefix(e.Name(), "session_archive_") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		id := strings.TrimPrefix(e.Name(), "session_")
		id = strings.TrimSuffix(id, ".toml")

		sessions = append(sessions, acp.SessionInfo{
			SessionId: id,
			Cwd:       cwd,
			UpdatedAt: info.ModTime().Format(time.RFC3339),
		})
	}

	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].UpdatedAt > sessions[j].UpdatedAt
	})

	return sessions, nil
}
