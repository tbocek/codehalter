package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

const sessionDir = ".codehalter"

type Message struct {
	Role    string `toml:"role"`
	Content string `toml:"content"`
}

type Session struct {
	ID        SessionId      `toml:"id"`
	Cwd       string         `toml:"cwd"`
	Title     string         `toml:"title"`
	CreatedAt time.Time      `toml:"created_at"`
	History   []HistoryLevel `toml:"history"`
	Messages  []Message      `toml:"messages"`
	filePath  string
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

func (s *Session) AddUser(text string) {
	s.Messages = append(s.Messages, Message{Role: "user", Content: text})
}

func (s *Session) AddAssistant(text string) {
	s.Messages = append(s.Messages, Message{Role: "assistant", Content: text})
}

func (s *Session) Save() error {
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
		info, err := e.Info()
		if err != nil {
			continue
		}
		id := strings.TrimPrefix(e.Name(), "session_")
		id = strings.TrimSuffix(id, ".toml")

		// Read title from file.
		var header struct{ Title string `toml:"title"` }
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
