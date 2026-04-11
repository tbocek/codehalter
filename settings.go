package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

type Settings struct {
	LLMConnections []LLMConnection `toml:"llmconnections"`
	path           string
}

type LLMConnection struct {
	Name  string `toml:"name"`
	URL   string `toml:"url"`
	Token string `toml:"token,omitempty"`
	Model string `toml:"model"`
}

// loadSettings looks for settings.toml in this order:
// 1. <cwd>/.codehalter/settings.toml (project-local, takes priority)
// 2. ~/.config/codehalter/settings.toml (global fallback)
// Always creates .codehalter/ in the project if it doesn't exist.
func loadSettings(cwd string) (Settings, error) {
	// Ensure project .codehalter dir exists.
	projectDir := filepath.Join(cwd, sessionDir)
	_ = os.MkdirAll(projectDir, 0755)

	// Try project-local first.
	projectPath := filepath.Join(projectDir, "settings.toml")
	if _, err := os.Stat(projectPath); err == nil {
		return decodeSettings(projectPath)
	}

	// Fall back to global config.
	home, err := os.UserHomeDir()
	if err == nil {
		globalPath := filepath.Join(home, ".config", "codehalter", "settings.toml")
		if _, err := os.Stat(globalPath); err == nil {
			return decodeSettings(globalPath)
		}
	}

	return Settings{}, fmt.Errorf("no settings.toml found (checked %s and ~/.config/codehalter/settings.toml)", projectPath)
}

func decodeSettings(path string) (Settings, error) {
	var s Settings
	if _, err := toml.DecodeFile(path, &s); err != nil {
		return s, fmt.Errorf("loading %s: %w", path, err)
	}
	s.path = path
	return s, nil
}

func (s *Settings) LLM(name string) *LLMConnection {
	for i := range s.LLMConnections {
		if s.LLMConnections[i].Name == name {
			return &s.LLMConnections[i]
		}
	}
	return nil
}

// SummaryLLM returns the summary connection, falling back to thinking.
func (s *Settings) SummaryLLM() *LLMConnection {
	if c := s.LLM("summary"); c != nil {
		return c
	}
	return s.LLM("thinking")
}
