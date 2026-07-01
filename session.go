package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/BurntSushi/toml"
)

// improveAskCap is the maximum number of improvements /improve may present per
// run; the per-change ask_user loop is held to this in code (tool_ask.go).
const improveAskCap = 3

// improveAskBlocked counts one /improve Apply/Skip prompt and reports whether it
// exceeds improveAskCap. A no-op returning false outside an /improve turn. Once
// the cap is passed it stays blocked, forcing the per-change loop to stop.
func (s *Session) improveAskBlocked() bool {
	if !s.improving.Load() {
		return false
	}
	return s.improveAsks.Add(1) > improveAskCap
}

// beginTurn registers the in-flight turn's cancel.
func (s *Session) beginTurn(c context.CancelFunc) {
	s.turnCancelMu.Lock()
	s.turnCancel = c
	s.turnCancelMu.Unlock()
}

// cancelTurn cancels the in-flight turn's ctx (Cancel button, or a new prompt
// superseding it). The caller does NOT wait for the turn to finish.
func (s *Session) cancelTurn() {
	s.turnCancelMu.Lock()
	c := s.turnCancel
	s.turnCancelMu.Unlock()
	if c != nil {
		c()
	}
}

// markSuperseding flags that a freshly-arrived prompt is about to cancel and
// replace the in-flight turn, so that turn's cancel handler stays silent (the
// new turn speaks for itself) instead of reporting an editor abort. Pairs with
// adoptTurn, which clears the flag once the new turn has taken over.
func (s *Session) markSuperseding() {
	s.turnCancelMu.Lock()
	s.superseding = true
	s.turnCancelMu.Unlock()
}

// adoptTurn clears the supersede flag once the new prompt has acquired turnMu
// and become the active turn (the superseded turn has fully unwound by then).
func (s *Session) adoptTurn() {
	s.turnCancelMu.Lock()
	s.superseding = false
	s.turnCancelMu.Unlock()
}

// superseded reports whether a newly-arrived prompt flagged the in-flight turn
// for replacement (markSuperseding). A superseded turn's cancel handler stays
// silent; a plain editor abort / client-side timeout surfaces its reason.
func (s *Session) superseded() bool {
	s.turnCancelMu.Lock()
	defer s.turnCancelMu.Unlock()
	return s.superseding
}

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
	// "execute", "verify", "document", or "subagent". Empty on user turns
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
	// ID is a per-session stable handle ("tu_<n>") generated when the tool
	// loop records the call. Surfaced in head/tail truncation hints so the
	// model can call `view_output id=tu_N` to retrieve any portion of the
	// full result without re-running the original tool. Empty for ToolUses
	// loaded from older session files; view_output treats those as not found.
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
}

// summariseTask is one queued background-summariser job: the messages of one
// completed turn to feed the structured-turn prompt. The connection is picked
// at enqueue time so the runner doesn't have to reach back into the agent.
type summariseTask struct {
	Turn []Message
	Conn *LLMConnection
	// Prompt is the SUMMARISE.md body for the paste-mode render (unused in
	// prefix-extension mode, where it is already baked into Msgs).
	Prompt string
	// Msgs, when non-nil, switches the note generation to prefix-extension
	// mode: the full wire context frozen at turn end plus the summarise
	// instruction (see appendSummariseMsgs). Frozen at ENQUEUE time so a
	// queued task still summarises exactly its own turn even when the next
	// turn has already started by the time the worker runs it.
	Msgs []llmMessage
}

type Session struct {
	ID        string    `toml:"id"`
	Cwd       string    `toml:"cwd"`
	CreatedAt time.Time `toml:"created_at"`
	Depth     int       `toml:"depth,omitempty"`
	ParentID  string    `toml:"parent_id,omitempty"`
	Summary   string    `toml:"summary,omitempty"`
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
	// launchedSubagents caches results of completed launch_subagent tasks,
	// keyed by a hash of (instructions, context). When the model re-asks for
	// an identical subagent (within or across launch_subagent tool calls in
	// the same session), we return the prior result instead of running it
	// again — small models forget what they already launched. Not persisted
	// across process restarts: re-running after a restart re-launches, which
	// is the safer default than feeding back a possibly-stale cached result.
	launchedSubagentsMu sync.Mutex
	launchedSubagents   map[string]string
	// webBodies caches the full raw page text from web_read / web_read_raw,
	// keyed by URL. When the LLM-visible result is truncated, the model can
	// re-call with offset/limit to view a specific range — we slice from the
	// cache instead of issuing another HTTP fetch, so paging through a long
	// document is free after the first read. In-memory only; lost on restart.
	webBodiesMu sync.Mutex
	webBodies   map[string]string
	// webResults caches the final rendered output (summary or raw-truncated)
	// keyed by URL+mode. A literal-repeat web_read on the same URL skips both
	// the HTTP fetch AND the re-summarize, and the model receives the exact
	// same string — friendly to the LLM prefix cache. Range requests bypass
	// this and go through webBodies/sliceWebBody.
	webResultsMu sync.Mutex
	webResults   map[string]string
	// mu serialises the Save() encoder write against concurrent mutators —
	// AddUser/AddAssistant/etc. acquire it before touching persisted fields
	// so a Save() landing in parallel doesn't observe a torn slice. Prompt
	// runs synchronously per session so contention is rare; the lock mainly
	// exists for the background summariser path (which reads Messages while
	// the foreground turn may be appending) and for the encoder invariant
	// inside saveLocked. The phaseMu and launchedSubagentsMu fields above
	// intentionally have their own locks — they don't touch persisted state
	// and must not block on mu.
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
	// readDedup remembers read_file outcomes for the current Prompt() turn
	// so a literal-repeat read (same path+line+limit, file unchanged) is
	// rejected instead of re-running. Reset at the top of Prompt(); busted
	// for a specific path whenever edit_file / write_file writes through
	// fsWrite. In-memory only.
	readDedupMu sync.Mutex
	readDedup   map[string]readDedupEntry
	// searchDedup remembers search_text outcomes for the current Prompt() turn
	// so a literal-repeat search (same query+path+flags, same results) is flagged
	// with a note instead of silently re-running. Keyed by the full args, value
	// is the fnv hash of the result. Same lifecycle as readDedup: reset at the
	// top of Prompt(). In-memory only.
	searchDedupMu sync.Mutex
	searchDedup   map[string]uint64
	// readCursor remembers, per file path, the next 1-based line continue_read
	// should serve — set when read_file (or continue_read) returns a partial
	// chunk, cleared when a chunk reaches EOF. Lets continue_read page forward
	// with no line math. Same lifecycle as readDedup: reset each Prompt() turn,
	// busted for a path on write. In-memory only.
	readCursorMu sync.Mutex
	readCursor   map[string]int
	// editFailedPaths is the set of paths where edit_file returned "not found"
	// this turn. A failed edit signals that the model's remembered content is
	// stale or inexact, so the next read_file on that path must bypass the
	// readContentInContext guard — the model genuinely needs a fresh look to
	// get the exact old_text for a retry. Cleared when the path is re-read.
	// Same lifecycle as readDedup: reset at the top of Prompt(). In-memory only.
	editFailedPathsMu sync.Mutex
	editFailedPaths   map[string]bool
	// pendingPlan holds a plan whose "Execute / Abort" card the user
	// dismissed by typing (e.g. asking a question) instead of choosing. Prompt
	// re-shows it after the typed message is handled, so a question doesn't throw
	// the plan away. resumePlan is that plan handed back to orchestrate when the
	// user picks Execute on the re-show, so it runs without re-planning. Both are
	// foreground-turn-only (no background goroutine touches them), so unguarded.
	pendingPlan *planResult
	resumePlan  *planResult
	// One turn per session. turnMu is held across the whole turn; a new prompt
	// cancelTurn()s the in-flight one then Lock()s here, so turns never overlap
	// (overlap raced compaction → two divergent context snapshots). turnCancel is
	// the in-flight turn's ctx cancel, fired by the Cancel button or a superseding
	// prompt; guarded by turnCancelMu. superseding (same mutex) is set by a
	// newly-arrived prompt right before it cancels the in-flight turn, so that
	// turn's cancel handler can tell a supersede (stay silent) from a plain
	// editor abort / client-side timeout (surface the reason).
	turnMu       sync.Mutex
	turnCancelMu sync.Mutex
	turnCancel   context.CancelFunc
	superseding  bool
	// fixAutoExec is set while a user-accepted fix card is being dispatched: the
	// user already approved on the card, so confirmPlan skips its "Execute?" gate
	// for that turn. Foreground-turn-only, so unguarded like the plans above.
	fixAutoExec bool
	// improving marks the current turn as an /improve run. It does double duty,
	// since both effects begin and end with the same /improve turn: (1) the
	// per-change ask_user Apply/Skip prompts are capped to improveAskCap in code so
	// a chatty model can't loop through dozens (improveAsks counts them), and (2)
	// this session's toml + log writes are routed to scratchDir (/tmp) instead of
	// .codehalter/, for the run's throwaway, self-referential logs. Set by
	// beginImproveScratch, cleared (deferred) by endImproveScratch; preImproveMsgs /
	// preImproveSummary hold the real conversation snapshot to restore at turn end.
	// Atomic: sessionFilePath/logSession read it from background goroutines (e.g.
	// the background summariser's llmStream logging) concurrently with begin/end.
	improving atomic.Bool
	// improveNoLicense caches, for the current /improve run, that the project has
	// no open-source license — so feedback-API submission is impossible. Set in
	// beginImproveScratch; read to drop the "Submit?" ask deterministically in
	// code (the template's prerequisite footnote alone doesn't stop a weak model
	// from asking, then submit_improvement hard-fails on the same check).
	improveNoLicense atomic.Bool
	// improveDelivered marks that submit_improvement's apply loop has already run
	// this /improve turn, so a duplicate call (same batch, or after a replan) no-ops
	// instead of re-applying, and the respond-funnel stops re-arming. Reset in
	// beginImproveScratch.
	improveDelivered  atomic.Bool
	improveAsks       atomic.Int64
	preImproveMsgs    []Message
	preImproveSummary string
	// promptSkills is the set of SKILL-*.md filenames folded into the current
	// SystemPrompt. A skill seeded on disk AFTER the prompt was built is injected
	// as a user message (NOT folded into the prompt — that would bust the KV
	// prefix cache) until the next compaction re-renders the prompt. Runtime-only.
	promptSkills []string
	// DisclosedSkills is the set of deferred SKILL-*.md filenames already
	// injected into this session under skills="auto" (see discloseSkills). The
	// system-prompt renderer keeps a deferred skill out of the prefix until it
	// appears here. Persisted so a resumed session's rebuilt prompt still
	// carries the skills its history already relies on.
	DisclosedSkills []string `toml:"disclosed_skills,omitempty"`
	// PinnedLLMIdx pins a subagent session to one [[llm]] entry. All LLM
	// calls from this session route to settings.LLM[PinnedLLMIdx] regardless
	// of role — cache-coherence trumps per-role sampler matching, the conn
	// stays warm across the subagent's whole run instead of bouncing on every
	// plan/execute switch. -1 means no pin (main session, tests).
	// Breadth-first assigned at creation by launch_subagent: the first
	// subagent in a batch pins to LLM[0], the rest fan out across LLM[1+]
	// up to each conn's parallel cap.
	PinnedLLMIdx int `toml:"pinned_llm_idx,omitempty"`
	// DisplayLabel is the short human-readable name the runner uses when it
	// surfaces this session's activity to its parent ("subagent 1",
	// "subagent 2", …). Not persisted — purely a UI breadcrumb so the
	// parent's chat can show what each subagent is doing tool-call by
	// tool-call. Set by launch_subagent at creation; empty on main sessions
	// (we don't forward main-session updates anywhere).
	DisplayLabel string `toml:"-"`
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
	// envSnapshot is the canonical string from checkEnv covering everything
	// the consolidated banner can display (container, firefox, run_command,
	// stacks, runner configs, per-tool probe binaries). checkEnv compares
	// against it to decide whether to flag envChanged so prepare re-emits
	// the banner. Not persisted — restart re-emits the banner once on the
	// first turn.
	envSnapshot string `toml:"-"`
	// capabilitiesShown gates the full capabilities banner to once per session.
	// The first prepare (bootstrap) emits it to establish state; afterwards
	// routine changes (a tool installed, an MCP server starting, a re-probe)
	// surface as one-line notices / fix cards instead of re-dumping the whole
	// setup screen mid-conversation. Not persisted — a restart re-shows it once.
	capabilitiesShown bool `toml:"-"`

	// Per-turn stats for the "✅ Done" line: reset at Prompt start, summed
	// during the turn, read at the end. turnStart is wall-clock; turnHumanWaitMs
	// is time blocked on user-input cards (doPermissionRequest) so it can be
	// subtracted out; turn*Tokens sum every llmStream call's reported usage.
	// Guarded by turnStatsMu — background goroutines (summariser, git-commit)
	// add tokens concurrently with the foreground turn.
	turnStatsMu          sync.Mutex
	turnStart            time.Time
	turnHumanWaitMs      int64
	turnCompletionTokens int
	// turnEvaluatedPrompt is the sent-but-not-cached prompt total (Σ of each call's
	// prompt_tokens − cached_tokens) — the real prompt work, shown on the Done line.
	// We deliberately do NOT sum the gross prompt_tokens (which re-counts the cached
	// prefix every call). When no backend reports a cache split, haveServerCache is
	// false and the line falls back to turnLastPrompt (final context size).
	turnEvaluatedPrompt int
	turnLastPrompt      int // most recent call's prompt size
	haveServerCache     bool
	turnPromptMs        int64 // eval time (server prompt_ms, else TTFT)
	turnGenMs           int64 // generation time
}

// resetTurnStats starts a fresh per-turn measurement window at start.
func (s *Session) resetTurnStats(start time.Time) {
	s.turnStatsMu.Lock()
	s.turnStart = start
	s.turnHumanWaitMs = 0
	s.turnCompletionTokens = 0
	s.turnEvaluatedPrompt = 0
	s.turnLastPrompt = 0
	s.haveServerCache = false
	s.turnPromptMs = 0
	s.turnGenMs = 0
	s.turnStatsMu.Unlock()
}

// addTurnTokens folds one call's usage into the turn: it sums completion and the
// evaluated (sent-but-not-cached) prompt tokens, and tracks the most recent
// call's full prompt size for the no-cache fallback line. evaluated is -1 when
// the backend reported no cache split (then haveServerCache stays false). prompt
// is the full prompt size, used only for the context-size fallback — never summed.
func (s *Session) addTurnTokens(prompt, completion, evaluated int) {
	s.turnStatsMu.Lock()
	s.turnCompletionTokens += completion
	if prompt > 0 {
		s.turnLastPrompt = prompt
	}
	if evaluated >= 0 {
		s.turnEvaluatedPrompt += evaluated
		s.haveServerCache = true
	}
	s.turnStatsMu.Unlock()
}

// addTurnTiming sums one call's eval/gen times.
func (s *Session) addTurnTiming(promptMs, genMs int64) {
	s.turnStatsMu.Lock()
	s.turnPromptMs += promptMs
	s.turnGenMs += genMs
	s.turnStatsMu.Unlock()
}

// addHumanWait records time spent blocked on a user-input card so it can be
// excluded from the turn's active time.
func (s *Session) addHumanWait(d time.Duration) {
	s.turnStatsMu.Lock()
	s.turnHumanWaitMs += d.Milliseconds()
	s.turnStatsMu.Unlock()
}

// turnStats returns the active wall-clock for the current turn (elapsed minus
// time spent waiting on the user) and the summed token usage. activeMs is 0
// before the first resetTurnStats.
// turnReport is the end-of-turn accounting for the "✅ Done" line.
type turnReport struct {
	activeMs        int64
	completion      int  // Σ completion_tokens
	evaluatedPrompt int  // Σ sent-but-not-cached prompt tokens (the real prompt work)
	lastPrompt      int  // final context size (last call's prompt_tokens)
	haveServerCache bool // a backend reported the cache split
	promptMs        int64
	genMs           int64
}

func (s *Session) turnStats() turnReport {
	s.turnStatsMu.Lock()
	defer s.turnStatsMu.Unlock()
	if s.turnStart.IsZero() {
		return turnReport{}
	}
	activeMs := time.Since(s.turnStart).Milliseconds() - s.turnHumanWaitMs
	if activeMs < 0 {
		activeMs = 0
	}
	return turnReport{
		activeMs:        activeMs,
		completion:      s.turnCompletionTokens,
		evaluatedPrompt: s.turnEvaluatedPrompt,
		lastPrompt:      s.turnLastPrompt,
		haveServerCache: s.haveServerCache,
		promptMs:        s.turnPromptMs,
		genMs:           s.turnGenMs,
	}
}

// recallSubagent returns a prior result for the given task hash if one exists.
func (s *Session) recallSubagent(hash string) (string, bool) {
	s.launchedSubagentsMu.Lock()
	defer s.launchedSubagentsMu.Unlock()
	if s.launchedSubagents == nil {
		return "", false
	}
	r, ok := s.launchedSubagents[hash]
	return r, ok
}

// rememberSubagent caches the result of a successful subagent run.
func (s *Session) rememberSubagent(hash, result string) {
	s.launchedSubagentsMu.Lock()
	defer s.launchedSubagentsMu.Unlock()
	if s.launchedSubagents == nil {
		s.launchedSubagents = make(map[string]string)
	}
	s.launchedSubagents[hash] = result
}

// recallWebBody returns a cached page body if the URL was fetched earlier in
// this session. Lets range-style web_read calls slice from cache without
// re-issuing the HTTP request.
func (s *Session) recallWebBody(url string) (string, bool) {
	s.webBodiesMu.Lock()
	defer s.webBodiesMu.Unlock()
	if s.webBodies == nil {
		return "", false
	}
	b, ok := s.webBodies[url]
	return b, ok
}

// rememberWebBody stores the full raw page text from a web_read / web_read_raw
// fetch. Overwrites on re-fetch so a refresh updates the cache; this is what
// the model wants — if it asked for a fresh fetch it should see fresh content
// on subsequent range views.
func (s *Session) rememberWebBody(url, body string) {
	s.webBodiesMu.Lock()
	defer s.webBodiesMu.Unlock()
	if s.webBodies == nil {
		s.webBodies = make(map[string]string)
	}
	s.webBodies[url] = body
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
	s.webResultsMu.Lock()
	defer s.webResultsMu.Unlock()
	if s.webResults == nil {
		return "", false
	}
	r, ok := s.webResults[webResultKey(url, summarize)]
	return r, ok
}

// rememberWebResult stores the final rendered output for a URL+mode so a
// duplicate call returns the byte-identical string without redoing any work.
func (s *Session) rememberWebResult(url string, summarize bool, out string) {
	s.webResultsMu.Lock()
	defer s.webResultsMu.Unlock()
	if s.webResults == nil {
		s.webResults = make(map[string]string)
	}
	s.webResults[webResultKey(url, summarize)] = out
}

func loadSession(cwd string, id string) (*Session, error) {
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)

	var s Session
	if _, err := toml.DecodeFile(path, &s); err != nil {
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
	id := now.Format("20060102_150405")
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)
	return &Session{
		ID:        id,
		Cwd:       cwd,
		CreatedAt: now,
		filePath:  path,
	}, nil
}

func newSubagentSession(cwd string, parentID string, index, depth, pinnedLLMIdx int) *Session {
	// Belt-and-suspenders: the parent session already created this dir. If it
	// fails here the first saveLocked will surface it, but don't let the mkdir
	// itself vanish.
	if err := os.MkdirAll(filepath.Join(cwd, sessionDir), 0755); err != nil {
		slog.Warn("newSubagentSession: mkdir failed", "cwd", cwd, "err", err)
	}
	// Nanosecond suffix so sequential launch_subagent calls from the same
	// parent don't collide on id (each call re-starts index at 0).
	now := time.Now()
	id := fmt.Sprintf("sub_%s_%d_%d", parentID, now.UnixNano(), index)
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)
	s := &Session{
		ID:           id,
		Cwd:          cwd,
		Depth:        depth,
		ParentID:     parentID,
		CreatedAt:    now,
		filePath:     path,
		PinnedLLMIdx: pinnedLLMIdx,
	}
	s.saveOrLog()
	return s
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
	s.editFailedPathsMu.Lock()
	if s.editFailedPaths == nil {
		s.editFailedPaths = map[string]bool{}
	}
	s.editFailedPaths[path] = true
	s.editFailedPathsMu.Unlock()
}

// clearEditFailed clears the edit-failed flag for path and returns whether
// it was set. Called by serveRead so the bypass fires exactly once per failure.
func (s *Session) clearEditFailed(path string) bool {
	s.editFailedPathsMu.Lock()
	defer s.editFailedPathsMu.Unlock()
	if s.editFailedPaths[path] {
		delete(s.editFailedPaths, path)
		return true
	}
	return false
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

// FindToolUseOutput scans every message's tool uses for one matching id and
// returns its Output. Used by view_output to re-serve the full output of any
// prior tool call without re-running it. Returns "" when no match (older
// session files that predate ToolUse.ID, or a hallucinated id).
func (s *Session) FindToolUseOutput(id string) string {
	if id == "" {
		return ""
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.Messages {
		for _, tu := range s.Messages[i].ToolUses {
			if tu.ID == id {
				return tu.Output
			}
		}
	}
	return ""
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
// compaction reads it via turnStartIndex to decide what to keep verbatim.
func (s *Session) markTurnStart() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turnStartIdx = len(s.Messages) - 1
	if s.turnStartIdx < 0 {
		s.turnStartIdx = 0
	}
}

// turnStartIndex returns the in-flight turn's start index, clamped to the
// current Messages length (a rotation may have shrunk Messages out from under
// a stale value).
func (s *Session) turnStartIndex() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.turnStartIdx > len(s.Messages) {
		return len(s.Messages)
	}
	return s.turnStartIdx
}

// lastAssistantIndex returns the index of the most recent assistant message in
// the in-flight large turn (at or after turnStartIdx), i.e. the start of the
// unfinished small turn. The 400 recovery folds everything before it and keeps
// it verbatim. Falls back to turnStartIndex when the turn has no assistant
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
	s.turnStatsMu.Lock()
	pt := s.turnLastPrompt
	s.turnStatsMu.Unlock()
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

// resetTurnStart is called right after rotate() trims the message prefix: the
// kept window now begins at index 0, so the in-flight turn does too. rotate
// runs with no concurrent writer (see its doc), so this needs no extra lock.
func (s *Session) resetTurnStart() {
	s.turnStartIdx = 0
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
		Depth:        s.Depth,
		ParentID:     s.ParentID,
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

// scratchDir is where an /improve run's throwaway session files go (its bulky,
// self-referential logs must not bloat or pollute the .codehalter/ logs it
// analyses). A var so tests can point it at a tempdir.
var scratchDir = "/tmp"

// beginImproveScratch marks the turn as an /improve run (arming the ask cap and
// the scratchDir log redirect), snapshots the real conversation IN MEMORY, and
// resets the session to a fresh context (so /improve has full budget to read
// every log). Nothing touches disk here, so the real .codehalter/session_*.toml
// stays exactly as the last turn left it. endImproveScratch restores the
// snapshot at turn end (every exit path), so the user's conversation resumes
// untouched and the /improve bulk is discarded with the /tmp files.
func (s *Session) beginImproveScratch() {
	// Compute the license verdict before taking the lock (it does file I/O) so
	// the submit step can be gated deterministically for this run.
	_, licErr := checkLicense(s.Cwd)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.preImproveMsgs = s.Messages
	s.preImproveSummary = s.Summary
	s.Messages = nil
	s.Summary = ""
	s.improveAsks.Store(0)
	s.improveDelivered.Store(false)
	s.improveNoLicense.Store(licErr != nil)
	s.improving.Store(true)
}

func (s *Session) endImproveScratch() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = s.preImproveMsgs
	s.Summary = s.preImproveSummary
	s.preImproveMsgs = nil
	s.preImproveSummary = ""
	s.improveAsks.Store(0)
	s.improving.Store(false)
}

// sessionFilePath returns where this session's toml/log of basename `name`
// should be written: scratchDir during an /improve run, else .codehalter/.
func (s *Session) sessionFilePath(name string) string {
	if s.improving.Load() {
		return filepath.Join(scratchDir, name)
	}
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

func listSessions(cwd string) ([]SessionInfo, error) {
	dir := filepath.Join(cwd, sessionDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var sessions []SessionInfo
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "session_") || !strings.HasSuffix(e.Name(), ".toml") {
			continue
		}
		// Skip subagent and post-rotation archive sessions — both live on
		// disk for inspection but should not clutter the picker.
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

		sessions = append(sessions, SessionInfo{
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

// SessionInfo is returned by session/list.
type SessionInfo struct {
	SessionId string `json:"sessionId"`
	Cwd       string `json:"cwd"`
	UpdatedAt string `json:"updatedAt,omitempty"`
}
