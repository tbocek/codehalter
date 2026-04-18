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
}

type ImageData struct {
	MimeType string `toml:"mime_type"`
	Data     string `toml:"data"` // base64-encoded
}

type ToolUse struct {
	Name   string `toml:"name"`
	Input  string `toml:"input"`
	Output string `toml:"output"`
}

type Session struct {
	ID        SessionId      `toml:"id"`
	Cwd       string         `toml:"cwd"`
	Title     string         `toml:"title"`
	CreatedAt time.Time      `toml:"created_at"`
	Depth     int            `toml:"depth,omitempty"`
	ParentID  SessionId      `toml:"parent_id,omitempty"`
	History   []HistoryLevel `toml:"history"`
	Messages  []Message      `toml:"messages"`
	filePath  string
	// mu serialises all mutations of the fields above and the Save() encoder.
	// Prompt runs synchronously per session, but generateTitle runs as a
	// background goroutine and would otherwise race on Title + the TOML file.
	mu sync.Mutex
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

func newSessionWithID(cwd string, id SessionId) *Session {
	os.MkdirAll(filepath.Join(cwd, sessionDir), 0755)
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)
	s := &Session{
		ID:        id,
		Cwd:       cwd,
		CreatedAt: time.Now(),
		filePath:  path,
	}
	_ = s.Save()
	return s
}

func newSession(cwd string) (*Session, error) {
	if err := os.MkdirAll(filepath.Join(cwd, sessionDir), 0755); err != nil {
		return nil, fmt.Errorf("creating session dir: %w", err)
	}

	now := time.Now()
	id := SessionId(now.Format("20060102_150405"))
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)

	s := &Session{
		ID:        id,
		Cwd:       cwd,
		CreatedAt: now,
		filePath:  path,
	}
	return s, s.Save()
}

func newSubagentSession(cwd string, parentID SessionId, index, depth int) *Session {
	os.MkdirAll(filepath.Join(cwd, sessionDir), 0755)
	// Nanosecond suffix so sequential launch_subagent calls from the same
	// parent don't collide on id (each call re-starts index at 0).
	now := time.Now()
	id := SessionId(fmt.Sprintf("sub_%s_%d_%d", parentID, now.UnixNano(), index))
	filename := fmt.Sprintf("session_%s.toml", id)
	path := filepath.Join(cwd, sessionDir, filename)
	s := &Session{
		ID:        id,
		Cwd:       cwd,
		Depth:     depth,
		ParentID:  parentID,
		CreatedAt: now,
		filePath:  path,
	}
	_ = s.Save()
	return s
}

func (s *Session) AddUser(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "user", Content: text})
}

func (s *Session) AddUserWithImages(text string, images []ImageData) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "user", Content: text, Images: images})
}

func (s *Session) AddAssistant(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "assistant", Content: text})
}

func (s *Session) AddAssistantWithTools(text string, tools []ToolUse) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Messages = append(s.Messages, Message{Role: "assistant", Content: text, ToolUses: tools})
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
	s.Messages = append(s.Messages, Message{Role: "assistant", Content: content})
}

func (s *Session) Save() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.saveLocked()
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
		// Skip subagent sessions.
		if strings.HasPrefix(e.Name(), "session_sub_") {
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
