package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/BurntSushi/toml"
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
	// "execute", "verify", "document", or "subagent". Empty on user turns
	// and on legacy entries from before this field existed.
	Phase string `toml:"phase,omitempty"`
}

type ImageData struct {
	// ID is a per-session stable handle ("img_<n>") generated when the image
	// is captured from the user prompt. Surfaced in the text placeholder that
	// replaces inline data: URLs for older messages so the model can call
	// view_image id=img_N to re-fetch the bytes. Empty for ImageData loaded
	// from older session files.
	ID       string `toml:"id,omitempty"`
	MimeType string `toml:"mime_type"`
	Data     string `toml:"data"` // base64-encoded
}

type ToolUse struct {
	// ID is a per-session stable handle ("tu_<n>") generated when the tool
	// loop records the call. Surfaced in head/tail truncation hints so the
	// model can call `view_output id=tu_N` to retrieve any portion of the
	// full result without re-running the original tool. Empty for ToolUses
	// loaded from older session files; view_output treats those as not found.
	ID     string `toml:"id,omitempty"`
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

// bgJob coalesces fan-out of a fire-and-forget background goroutine and lets
// other goroutines join the in-flight call. TryStart claims the slot AND
// registers the pending goroutine in one atomic step — the caller must defer
// Done on every path after a successful TryStart (including early returns
// from within the launching function, before the goroutine itself fires).
type bgJob struct {
	running atomic.Bool
	pending sync.WaitGroup
}

// TryStart attempts to acquire the slot. Returns true when the caller is now
// the active runner (and MUST call Done exactly once); false when another
// runner is in-flight and this caller should skip.
func (j *bgJob) TryStart() bool {
	if !j.running.CompareAndSwap(false, true) {
		return false
	}
	j.pending.Add(1)
	return true
}

// Done releases the slot and decrements the pending WaitGroup. Pair with a
// successful TryStart.
func (j *bgJob) Done() {
	j.pending.Done()
	j.running.Store(false)
}

// Wait blocks until any in-flight goroutine claimed via TryStart has called
// Done. Used by joiners that need fresh state.
func (j *bgJob) Wait() { j.pending.Wait() }

// summariseTask is one queued background-summariser job: the user/assistant
// snapshot to feed the structured-turn prompt. The connection is picked at
// enqueue time so the runner doesn't have to reach back into the agent.
type summariseTask struct {
	User      Message
	Assistant Message
	Conn      *LLMConnection
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
	// after each compressHistory rotation — so it survives the summariser
	// (which otherwise compresses skills away) and reflects current
	// .codehalter/SKILL-*.md content. Emitted by buildLLMContext as the
	// leading user message before any Summary.
	SystemPrompt string    `toml:"system_prompt,omitempty"`
	Messages     []Message `toml:"messages"`
	filePath     string
	// phaseActive/phaseCurrent track the plan UI state. Not persisted.
	// phaseActive=true means a phase entry is showing as in_progress and
	// must be marked completed before Prompt returns; phaseCurrent is the
	// 0-based index into phaseNames it refers to. Guarded by phaseMu, NOT
	// the main session mu — compressHistory holds sess.mu across long LLM
	// calls and llmStream calls setStatus mid-stream, so reusing sess.mu
	// for phase reads would deadlock.
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
	// shadowEntries holds one structured per-turn note per element (Goal /
	// Constraints / Progress / Decisions / Next Steps / Critical Context)
	// produced by the background summariser fired after each llmStream
	// response. compressHistory drains all-but-the-most-recent entry into
	// Session.Summary and leaves the trailing entry behind as the anchor
	// covering the verbatim "last message" the live session keeps. In-memory
	// only — process restart loses it and the next compaction falls through
	// to the synchronous path.
	shadowMu      sync.Mutex
	shadowEntries []string
	// summariseQueue is a FIFO of per-turn snapshots waiting for the
	// background summariser. Every llmStream response in the tool loop
	// enqueues one — none are dropped, so a 50-iteration loop produces 50
	// shadow notes (one per assistant snapshot). A single worker goroutine
	// drains the queue sequentially; the LLM connection's slot semaphore
	// would serialise concurrent runners anyway, so a worker is simpler
	// than fire-and-forget. summariseUndone counts enqueued-minus-finished;
	// waitSummarise blocks while it exceeds 1 (the most-recent task is
	// always allowed to ride the next compaction so end-of-turn doesn't
	// stall on it). summariseCond is lazy-initialised under summariseMu on
	// first use so tests that construct Session{} directly don't need to
	// know about it.
	summariseMu      sync.Mutex
	summariseQueue   []summariseTask
	summariseRunning bool
	summariseUndone  int
	summariseCond    *sync.Cond
	// gitCommitJob coalesces per-LLM-call fan-out: TryStart drops the call
	// when a prior one is in-flight, and Wait() lets callers
	// (cleanupGitCommitIfClean) join any pending goroutine before they
	// read the resulting state. Drop-on-busy is appropriate here because
	// gitCommitLastHash already short-circuits identical snapshots — there
	// is no value in committing intermediate file states.
	gitCommitJob bgJob
	// gitCommitLastHash is the sha256 of (status + diff) from the last
	// successful backgroundGitCommit write. Subsequent calls short-circuit
	// when the snapshot hashes identical — the file on disk is already
	// up to date, so there's no reason to spend an LLM call. In-memory
	// only; a restart pays one redundant regeneration on the next turn.
	gitCommitMu       sync.Mutex
	gitCommitLastHash [32]byte
	// readDedup remembers read_file outcomes for the current Prompt() turn
	// so a literal-repeat read (same path+line+limit, file unchanged) is
	// rejected instead of re-running. Reset at the top of Prompt(); busted
	// for a specific path whenever edit_file / write_file writes through
	// fsWrite. In-memory only.
	readDedupMu sync.Mutex
	readDedup   map[string]readDedupEntry
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
	// lastCompletePromptTokens is the server-reported prompt_tokens count
	// from the most recent llmStream call on this session, captured via
	// stream_options.include_usage. "Complete" because every llmStream call
	// re-sends the whole conversation prefix (system prompt + summary +
	// every prior message + the new turn) — this single reading is the
	// cumulative context size, not a per-message delta. Drives the
	// compressHistory trigger: once it crosses ~80% of mainSlotTokens we
	// rotate, no estimator required. In-memory only — a restart resets to
	// 0, which is correct (the first turn after restart cannot exceed
	// n_ctx, and the next llmStream call will refresh the count). Guarded
	// by lastTokensMu so llmStream's background-thread writes don't race
	// compressHistory's read.
	lastTokensMu             sync.Mutex
	lastCompletePromptTokens int
}

// SetLastCompletePromptTokens records the server-reported prompt_tokens for
// the most recent llmStream call on this session. Called from llmStream's
// hot path.
func (s *Session) SetLastCompletePromptTokens(n int) {
	s.lastTokensMu.Lock()
	s.lastCompletePromptTokens = n
	s.lastTokensMu.Unlock()
}

// LastCompletePromptTokens returns the server-reported prompt_tokens for the
// most recent llmStream call, or 0 when no call has run yet on this session.
// "Complete" because the value already covers the whole conversation prefix
// re-sent on that call — callers should NOT sum across calls.
func (s *Session) LastCompletePromptTokens() int {
	s.lastTokensMu.Lock()
	defer s.lastTokensMu.Unlock()
	return s.lastCompletePromptTokens
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
	os.MkdirAll(filepath.Join(cwd, sessionDir), 0755)
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
	_ = s.Save()
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

// FindImage scans every message's images for one matching id and returns a
// copy. Used by view_image to re-deliver an attachment after buildLLMContext
// has elided its data: URL into a text placeholder. Returns nil when no match.
func (s *Session) FindImage(id string) *ImageData {
	if id == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.Messages {
		for j := range s.Messages[i].Images {
			if s.Messages[i].Images[j].ID == id {
				img := s.Messages[i].Images[j]
				return &img
			}
		}
	}
	return nil
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

// waitSummarise blocks until at most one summariser task remains outstanding
// (queued or in-flight). compressHistory uses it at end-of-turn: the older
// notes must be in the shadow buffer before we drain them into Summary, but
// the very latest task is allowed to ride the next compaction so the user
// isn't stalled waiting for a fresh background LLM call to come back.
func (s *Session) waitSummarise() {
	s.summariseMu.Lock()
	defer s.summariseMu.Unlock()
	if s.summariseCond == nil {
		s.summariseCond = sync.NewCond(&s.summariseMu)
	}
	for s.summariseUndone > 1 {
		s.summariseCond.Wait()
	}
}

// appendShadow adds a structured per-turn note to the shadow buffer. Each
// note is stored as its own slice entry so drainShadow can leave the last one
// behind as an anchor. Caller must NOT hold s.mu (shadowMu is independent).
func (s *Session) appendShadow(chunk string) {
	chunk = strings.TrimSpace(chunk)
	if chunk == "" {
		return
	}
	s.shadowMu.Lock()
	defer s.shadowMu.Unlock()
	s.shadowEntries = append(s.shadowEntries, chunk)
}

// drainShadow returns the accumulated structured notes EXCEPT the most recent
// one, joined with blank-line separators. The trailing entry stays in the
// buffer as an anchor covering the verbatim "last message" that compressHistory
// keeps — without this, the kept message and the freshly-drained note would
// duplicate each other in the next prompt. Returns "" when there's fewer than
// 2 entries, falling the caller through to the synchronous summarize path.
func (s *Session) drainShadow() string {
	s.shadowMu.Lock()
	defer s.shadowMu.Unlock()
	if len(s.shadowEntries) < 2 {
		return ""
	}
	out := strings.Join(s.shadowEntries[:len(s.shadowEntries)-1], "\n\n")
	s.shadowEntries = s.shadowEntries[len(s.shadowEntries)-1:]
	return out
}

// peekShadow returns every accumulated structured note (including the anchor)
// joined with blank-line separators, WITHOUT draining the buffer. Used by the
// document phase to read per-turn summaries while still letting compressHistory
// drain the older ones at the next compaction.
func (s *Session) peekShadow() string {
	s.shadowMu.Lock()
	defer s.shadowMu.Unlock()
	return strings.Join(s.shadowEntries, "\n\n")
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

// MarkLastAssistantTiming stamps timing and phase onto the trailing assistant
// message, creating one if no assistant turn is currently the latest entry.
// Used by runToolLoopOn at the end of an assistant turn so the .toml records
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

// rotate archives the session as it currently is to a new "session_archive_*"
// file, then resets s in place to carry only `keep` raw messages and `summary`
// as the rolled-up prior context. The live session keeps its own ID and
// filePath; only its on-disk contents change. Returns the archive's id.
// Called only from compressHistory in the post-orchestrate epilogue, where
// no concurrent writer exists for the session — so the in-place mutation of
// s.Summary/s.Messages here doesn't take the lock; the follow-up Save() does.
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

// saveLocked writes the session to disk. Caller must hold s.mu (or own the
// session exclusively, e.g. a freshly-constructed archive in rotate()).
func (s *Session) saveLocked() error {
	f, err := os.Create(s.filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	return toml.NewEncoder(f).Encode(s)
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
