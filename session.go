package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
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
	MimeType string `toml:"mime_type"`
	Data     string `toml:"data"` // base64-encoded
}

type ToolUse struct {
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

type Session struct {
	ID        SessionId      `toml:"id"`
	Cwd       string         `toml:"cwd"`
	Title     string         `toml:"title"`
	CreatedAt time.Time      `toml:"created_at"`
	Depth     int            `toml:"depth,omitempty"`
	ParentID  SessionId      `toml:"parent_id,omitempty"`
	Summary   string         `toml:"summary,omitempty"`
	Messages  []Message      `toml:"messages"`
	filePath  string
	// phaseActive/phaseCurrent track the plan UI state. Not persisted.
	// phaseActive=true means a phase entry is showing as in_progress and
	// must be marked completed before Prompt returns; phaseCurrent is the
	// 0-based index into phaseNames it refers to. Guarded by phaseMu, NOT
	// the main session mu — compressHistory holds sess.mu across long LLM
	// calls and llmStream calls notifyPhaseSuffix mid-stream, so reusing
	// sess.mu for phase reads would deadlock.
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
	// mu serialises mutations of the persisted fields (Title, Messages,
	// Summary, …) and the Save() encoder. Prompt runs synchronously per
	// session, but generateTitle runs as a background goroutine and would
	// otherwise race on Title + the TOML file. The phaseMu and
	// launchedSubagentsMu fields above intentionally have their own locks —
	// they don't touch persisted state and must not block on the long-held
	// mu (compressHistory in particular).
	mu sync.Mutex
	// PinnedSubLLMIdx pins a subagent session to one [[subllm]] entry. All
	// LLM calls from this session route to settings.SubLLM[PinnedSubLLMIdx]
	// regardless of role tag — cache-coherence trumps per-role sampler
	// matching: the slot stays warm across the whole subagent run instead of
	// bouncing between entries on every plan/execute switch. -1 means no
	// pin (main session, tests). Round-robin assigned at creation by
	// launch_subagent so concurrent subagents fan across [[subllm]] entries.
	PinnedSubLLMIdx int `toml:"pinned_subllm_idx,omitempty"`
	// DisplayLabel is the short human-readable name the runner uses when it
	// surfaces this session's activity to its parent ("subagent 1",
	// "subagent 2", …). Not persisted — purely a UI breadcrumb so the
	// parent's chat can show what each subagent is doing tool-call by
	// tool-call. Set by launch_subagent at creation; empty on main sessions
	// (we don't forward main-session updates anywhere).
	DisplayLabel string `toml:"-"`
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

func loadSession(cwd string, id SessionId) (*Session, error) {
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
func newSessionWithID(cwd string, id SessionId) *Session {
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
	id := SessionId(now.Format("20060102_150405"))
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)
	return &Session{
		ID:        id,
		Cwd:       cwd,
		CreatedAt: now,
		filePath:  path,
	}, nil
}

func newSubagentSession(cwd string, parentID SessionId, index, depth, pinnedSubLLMIdx int) *Session {
	os.MkdirAll(filepath.Join(cwd, sessionDir), 0755)
	// Nanosecond suffix so sequential launch_subagent calls from the same
	// parent don't collide on id (each call re-starts index at 0).
	now := time.Now()
	id := SessionId(fmt.Sprintf("sub_%s_%d_%d", parentID, now.UnixNano(), index))
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)
	s := &Session{
		ID:              id,
		Cwd:             cwd,
		Depth:           depth,
		ParentID:        parentID,
		CreatedAt:       now,
		filePath:        path,
		PinnedSubLLMIdx: pinnedSubLLMIdx,
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

// SetTitle updates the session title under the lock. Used by generateTitle
// (background goroutine) and retitle (from compressHistory).
func (s *Session) SetTitle(t string) {
	s.mu.Lock()
	s.Title = t
	s.mu.Unlock()
}

// UpdateLastMessageContent mutates the Content field of message at idx. Used
// by the prompt loop to inject plan context into the user turn.
func (s *Session) UpdateLastMessageContent(idx int, content string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if idx < 0 || idx >= len(s.Messages) {
		return
	}
	s.Messages[idx].Content = content
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
// Caller must hold s.mu.
func (s *Session) rotate(keep []Message, summary string) (SessionId, error) {
	archiveID := SessionId(fmt.Sprintf("archive_%s_%d", s.ID, time.Now().UnixNano()))
	archivePath := filepath.Join(s.Cwd, sessionDir, fmt.Sprintf("session_%s.toml", archiveID))
	archive := &Session{
		ID:        archiveID,
		Cwd:       s.Cwd,
		Title:     s.Title,
		CreatedAt: s.CreatedAt,
		Depth:     s.Depth,
		Summary:   s.Summary,
		Messages:  s.Messages,
		filePath:  archivePath,
	}
	// Fresh struct, no concurrent access — saveLocked's "caller holds mu"
	// invariant is vacuously satisfied.
	if err := archive.saveLocked(); err != nil {
		return "", err
	}
	s.Summary = summary
	s.Messages = keep
	return archiveID, nil
}

// saveLocked writes the session to disk. Caller must hold s.mu.
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

		// Read title from file.
		var header struct {
			Title string `toml:"title"`
		}
		_, _ = toml.DecodeFile(filepath.Join(dir, e.Name()), &header)

		sessions = append(sessions, SessionInfo{
			SessionId: SessionId(id),
			Cwd:       cwd,
			Title:     header.Title,
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
	SessionId SessionId `json:"sessionId"`
	Cwd       string    `json:"cwd"`
	Title     string    `json:"title,omitempty"`
	UpdatedAt string    `json:"updatedAt,omitempty"`
}
