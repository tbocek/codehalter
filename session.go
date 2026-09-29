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
)

const sessionDir = ".codehalter"

type Message struct {
	Role      string      `toml:"role"`
	Content   string      `toml:"content"`
	Images    []ImageData `toml:"images,omitempty"`
	ToolUses  []ToolUse   `toml:"tool_uses,omitempty"`
	StartedAt time.Time   `toml:"started_at,omitempty"`
	// PromptTokens is the context size at the call that produced this assistant
	// message (keepWindowStart sizes by it); 0 when unknown.
	PromptTokens int `toml:"prompt_tokens,omitempty"`
}

type ImageData struct {
	// The bytes live in .codehalter/images and are re-read every turn, so the wire
	// stays byte-identical for the prefix cache.
	ID       string `toml:"id,omitempty"`
	MimeType string `toml:"mime_type"`
}

type ToolUse struct {
	ID string `toml:"id,omitempty"`
	// CallID is the model's own tool_call id, replayed verbatim: substituting ID
	// would change the bytes and bust the prefix cache. Empty falls back to ID.
	CallID string `toml:"call_id,omitempty"`
	Name   string `toml:"name"`
	Input  string `toml:"input"`
	Output string `toml:"output"`
	// Failed is authoritative: it overrides an LLM "success=true" verdict.
	Failed bool `toml:"failed,omitempty"`
	// For a dedup hit these describe the hit, not the original call.
	StartedAt  time.Time `toml:"started_at,omitempty"`
	DurationMs int64     `toml:"duration_ms,omitempty"`
	// ImageID is an image this call produced. Replay uses the stored bytes, never
	// a re-run: re-rendering is not pure, and new bytes would bust the cache.
	ImageID string `toml:"image_id,omitempty"`
}

type Session struct {
	ID        string    `toml:"id"`
	Cwd       string    `toml:"cwd"`
	CreatedAt time.Time `toml:"created_at"`
	Summary   string    `toml:"summary,omitempty"`
	// FoldedSummary is the base for the next compaction. Separate from Summary,
	// which leads every request: shrinking that in place would re-render the prompt.
	FoldedSummary string    `toml:"folded_summary,omitempty"`
	Title         string    `toml:"title,omitempty"`
	SystemPrompt  string    `toml:"system_prompt,omitempty"`
	Messages      []Message `toml:"messages"`
	// Shadow holds one note per completed turn since the last compaction; never in the live context.
	Shadow   []string `toml:"shadow,omitempty"`
	mcpOffer []acpMCPServer
	// phaseMu, not mu: llmStream updates the status while writers hold mu.
	phaseMu        sync.Mutex
	phaseActive    bool
	phaseCurrent   int
	planTableShown bool
	// mu guards the persisted fields so a parallel Save never encodes a torn slice.
	mu sync.Mutex
	// turnStartIdx is guarded by mu.
	turnStartIdx int
	// summariseUndone is enqueued minus finished; waitSummarise blocks on it because
	// compaction folds the whole Shadow. summariseCond is created on first use.
	summariseMu      sync.Mutex
	summariseQueue   []func()
	summariseRunning bool
	summariseUndone  int
	summariseCond    *sync.Cond
	ctl              turnControl
	// turnMu is a leaf lock: nothing is called while holding it.
	turnMu  sync.Mutex
	turn    turnState
	lineage cacheLineage
	rt      sessionRuntime
	// promptSkills are the skills in SystemPrompt. A skill applicable later goes in
	// as a user message until compaction re-renders: folding it in would bust the cache.
	promptSkills []string
	// llmHash is the settings files' hash at the last successful probe.
	llmHash           string   `toml:"-"`
	knownStacks       []string `toml:"-"`
	capabilitiesShown bool     `toml:"-"`
}

// turnState is replaced per turn, so it needs no reset code. Background calls
// add their tokens to whichever turn is current.
type turnState struct {
	start       time.Time
	humanWaitMs int64
	stats       turnReport
}

// cacheLineage does not reset per turn: a turn's first call is exactly where
// message-list mutations land.
type cacheLineage struct {
	prompt int // its prompt_tokens
	// render tells a rewind we caused (a different renderKey) from one the server did.
	render string
	// at: servers reclaim idle slots, so the gap since it hints at an eviction.
	at time.Time
}

// sessionRuntime is never persisted. mu is a leaf.
type sessionRuntime struct {
	mu        sync.Mutex
	webBodies map[string]string
	// drifted marks paths rewritten from outside after our write (format-on-save).
	// Not per turn: that drift routinely spans turns.
	wroteHash map[string]string
	drifted   map[string]bool
	pending   []pendingInput
	specStop  bool
	// specFenceDir is read-only for the file tools while a /spec loop runs.
	specFenceDir string
	// specHandoff is a /spec command the planner asked for, run after the planning turn.
	specHandoff string
	// specAudit: a bare /spec redo is auditing, so the planner's list is not asked about.
	specAudit bool
	// stuckCalls: the calls a step repeated until the loop ended it, keyed like the
	// repetition tracker, with their output. The turn's later steps get the output
	// back instead of a run, until they write something (see runToolLoop).
	stuckCalls map[string]string
	prevCall   toolCallBrief
	// Later calls of the same reply were batched already and get no batching note.
	replyStart bool
	// jobRuns are the background jobs that exited: a /spec gate that ran as one counts.
	jobRuns []jobRun
}

type jobRun struct {
	cmd            string
	code           int
	started, ended time.Time
}

func (s *Session) recordJobRun(j jobRun) {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	s.rt.jobRuns = append(s.rt.jobRuns, j)
	if len(s.rt.jobRuns) > 20 {
		s.rt.jobRuns = s.rt.jobRuns[1:]
	}
}

func (s *Session) markReplyStart() {
	s.rt.mu.Lock()
	s.rt.replyStart = true
	s.rt.mu.Unlock()
}

type toolCallBrief struct {
	name, args string
	failed     bool
}

// startTurn keeps the cache lineage on purpose, see cacheLineage.
func (s *Session) startTurn(start time.Time) {
	s.turnMu.Lock()
	s.turn = turnState{start: start}
	s.turnMu.Unlock()
}

// evaluated < 0 means no cache split was reported. prompt is never summed.
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

// Covers llama.cpp dropping the last cache chunk and a trimmed summariser tail.
const cacheRewindSlack = 1024

// No protocol signal marks an idle-slot eviction; a minute is above any tool-loop gap.
const idleEvictionSuspect = time.Minute

// Below this, discarded decode is routine noise; stalls and retries are thousands.
const wastedCompletionFloor = 256

// tokens == 0 means no rewind.
type cacheRewind struct {
	tokens        int // prompt the previous call had already sent, re-read now
	prevRender    string
	renderChanged bool
	idle          time.Duration // wall gap since the previous call (0 if unknown)
}

// Only the tool loop feeds this: only there is each call the previous one plus
// an append. cached < 0 (no split reported) restarts the lineage.
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
	// A shrinking prompt is not a rewind: the message list was rewritten.
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

func (s *Session) addWastedCompletion(n int) {
	s.turnMu.Lock()
	s.turn.stats.wastedCompletion += n
	s.turnMu.Unlock()
}

// Compaction drops the prefix by design, so that must not read as a rewind.
func (s *Session) resetCacheLineage() {
	s.turnMu.Lock()
	s.lineage = cacheLineage{}
	s.turnMu.Unlock()
}

func (s *Session) addTurnTiming(promptMs, genMs int64) {
	s.turnMu.Lock()
	s.turn.stats.promptMs += promptMs
	s.turn.stats.genMs += genMs
	s.turnMu.Unlock()
}

func (s *Session) addHumanWait(d time.Duration) {
	s.turnMu.Lock()
	s.turn.humanWaitMs += d.Milliseconds()
	s.turnMu.Unlock()
}

type turnReport struct {
	activeMs   int64 // wall clock minus user wait; set by turnStats
	completion int
	// Σ (prompt_tokens - cached_tokens)
	evaluatedPrompt int
	lastPrompt      int
	haveServerCache bool
	promptMs        int64 // server prompt_ms, else TTFT
	genMs           int64
	// cacheRewindsRender counts the rewinds we caused by changing template params.
	cacheRewinds       int
	cacheRewound       int
	cacheRewindsRender int
	// wastedCompletion is decode thrown away (cap hit, truncation); it is inside completion.
	wastedCompletion int
}

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

func (s *Session) recallWebBody(url string) (string, bool) {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	b, ok := s.rt.webBodies[url]
	return b, ok
}

func (s *Session) rememberWebBody(url, body string) {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	if s.rt.webBodies == nil {
		s.rt.webBodies = make(map[string]string)
	}
	s.rt.webBodies[url] = body
}

func sessionPath(cwd, id, ext string) string {
	return filepath.Join(cwd, sessionDir, "session_"+id+"."+ext)
}

func loadSession(cwd string, id string) (*Session, error) {
	path := sessionPath(cwd, id, "toml")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	// TOML refuses invalid UTF-8. U+FFFD is what Go's JSON encoder already sent for
	// those bytes, so the wire and the cache stay the same.
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
	return &s, nil
}

// Nothing is written until the first Save, so an unprompted tab leaves no stub on disk.
func newSessionWithID(cwd string, id string) *Session {
	return &Session{
		ID:        id,
		Cwd:       cwd,
		CreatedAt: time.Now(),
	}
}

func newSession(cwd string) (*Session, error) {
	if err := os.MkdirAll(filepath.Join(cwd, sessionDir), 0755); err != nil {
		return nil, fmt.Errorf("creating session dir: %w", err)
	}
	// Ids are second-granular; reusing one would overwrite that session's file.
	base := time.Now().Format("20060102_150405")
	id := base
	for n := 2; ; n++ {
		if _, err := os.Stat(sessionPath(cwd, id, "toml")); err != nil {
			break // free, or unreadable: the first save reports the real problem
		}
		id = fmt.Sprintf("%s_%d", base, n)
	}
	return newSessionWithID(cwd, id), nil
}

func (s *Session) AddUser(text string, images ...ImageData) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "user", Content: text, Images: images, StartedAt: time.Now()})
}

func (s *Session) AddAssistant(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "assistant", Content: text, StartedAt: time.Now()})
}

func (s *Session) AppendToolUse(tu ToolUse) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.Messages) == 0 || s.Messages[len(s.Messages)-1].Role != "assistant" {
		s.Messages = append(s.Messages, Message{Role: "assistant"})
	}
	last := &s.Messages[len(s.Messages)-1]
	last.ToolUses = append(last.ToolUses, tu)
}

// Names the likely cause because the model cannot see the editor.
const externalChangeNote = "\n\n[NOTE: this file changed on disk after codehalter wrote it — something outside this session rewrote it, and an editor's format-on-save is the usual cause. Any old_text you remember from before that write may no longer match; copy it from THIS read.]"

// recordWrite drops any pending drift note: we just overwrote the other writer's change.
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

// The hash advances to the content read, so each drift is reported once. Paths we
// never wrote are not tracked.
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

func (s *Session) takeDriftNote(path string) string {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	if !s.rt.drifted[path] {
		return ""
	}
	delete(s.rt.drifted, path)
	return externalChangeNote
}

// line is for the user; full (exit code, log path, tail) is for the model.
type bgNote struct {
	line, full string
}

// pendingInput is the user's text typed during a turn, or a job note when note is set.
type pendingInput struct {
	text string
	note *bgNote
}

func (s *Session) addSteer(text string) {
	s.rt.mu.Lock()
	s.rt.pending = append(s.rt.pending, pendingInput{text: text})
	s.rt.mu.Unlock()
}

func (s *Session) addBgNote(n bgNote) {
	s.rt.mu.Lock()
	s.rt.pending = append(s.rt.pending, pendingInput{note: &n})
	s.rt.mu.Unlock()
}

func (s *Session) hasPending() bool {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	return len(s.rt.pending) > 0
}

func (s *Session) takePending() []pendingInput {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	items := s.rt.pending
	s.rt.pending = nil
	return items
}

// One worker runs jobs FIFO and exits when the queue empties. Caller holds no session locks.
func (s *Session) enqueueSummarise(job func()) {
	s.summariseMu.Lock()
	s.summariseQueue = append(s.summariseQueue, job)
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

			next()

			s.summariseMu.Lock()
			s.summariseUndone--
			if s.summariseCond != nil {
				s.summariseCond.Broadcast()
			}
			s.summariseMu.Unlock()
		}
	}()
}

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

func (s *Session) appendShadow(chunk string) {
	chunk = strings.TrimSpace(chunk)
	if chunk == "" {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Shadow = append(s.Shadow, chunk)
}

// No anchor held back: compaction rotates out exactly the turns these notes cover.
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

func (s *Session) markTurnStart() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.turnStartIdx = max(len(s.Messages)-1, 0)
}

// Without an assistant message yet it returns turnStartIdx, so foldHistory keeps
// the whole in-flight turn.
func (s *Session) lastAssistantIndex() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.Messages) - 1; i >= s.turnStartIdx; i-- {
		if s.Messages[i].Role == "assistant" {
			return i
		}
	}
	return s.turnStartIdx
}

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

// keepWindowStart keeps the unfinished small turn plus the recent completed ones
// under maxCompletedTokens, sized by real prompt_tokens.
func (s *Session) keepWindowStart(maxCompletedTokens int) int {
	last := s.lastAssistantIndex()
	s.mu.Lock()
	defer s.mu.Unlock()
	if last >= len(s.Messages) {
		return last
	}
	ref := s.Messages[last].PromptTokens
	if ref <= 0 {
		return last
	}
	// PromptTokens rising going back marks a prior fold boundary: stop there.
	prev := ref
	keep := last
	for i := last - 1; i >= s.turnStartIdx; i-- {
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

func (s *Session) Save() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.saveLocked()
}

func (s *Session) saveOrLog() {
	if err := s.Save(); err != nil {
		slog.Error("session save failed", "id", s.ID, "path", sessionPath(s.Cwd, s.ID, "toml"), "err", err)
	}
}

// rotate takes no lock: its sole caller, foldHistory, runs where no concurrent
// writer exists. The caller saves the live session.
func (s *Session) rotate(keep []Message, summary string) (string, error) {
	archiveID := fmt.Sprintf("archive_%s_%d", s.ID, time.Now().UnixNano())
	archive := &Session{
		ID:           archiveID,
		Cwd:          s.Cwd,
		CreatedAt:    s.CreatedAt,
		Summary:      s.Summary,
		SystemPrompt: s.SystemPrompt,
		Messages:     s.Messages,
	}
	if err := archive.saveLocked(); err != nil {
		return "", err
	}
	s.Summary = summary
	s.Messages = keep
	return archiveID, nil
}

// Caller must hold s.mu or own the session exclusively.
func (s *Session) saveLocked() error {
	f, err := os.Create(sessionPath(s.Cwd, s.ID, "toml"))
	if err != nil {
		return err
	}
	if err := toml.NewEncoder(f).Encode(s); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

type SessionInfo struct {
	SessionId string `json:"sessionId"`
	Cwd       string `json:"cwd"`
	UpdatedAt string `json:"updatedAt,omitempty"`
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
		// Archives and old subagent sessions are for inspection, not the picker.
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
