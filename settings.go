package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"
)

type Settings struct {
	LLMConnections []LLMConnection `toml:"llmconnections"`
	path           string
}

type LLMConnection struct {
	Name      string         `toml:"name"`
	URL       string         `toml:"url"`
	APIKey    string         `toml:"api_key,omitempty"`
	Model     string         `toml:"model"`
	ExtraBody map[string]any `toml:"extra_body,omitempty"`
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

// loadGlobalSettings reads only the user-level settings at
// ~/.config/codehalter/settings.toml. Unlike loadSettings it never touches the
// project directory, so it is safe to call before a session (and a cwd) exist.
func loadGlobalSettings() (Settings, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return Settings{}, err
	}
	globalPath := filepath.Join(home, ".config", "codehalter", "settings.toml")
	if _, err := os.Stat(globalPath); err != nil {
		return Settings{}, fmt.Errorf("no global settings.toml at %s", globalPath)
	}
	return decodeSettings(globalPath)
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

// requiredRoles are the LLM connection names that must be configured.
var requiredRoles = []string{"thinking", "execute", "summary"}

// SummaryLLM returns the summary connection. Validate guarantees it exists.
func (s *Settings) SummaryLLM() *LLMConnection {
	return s.LLM("summary")
}

// Validate ensures every required role is present. Returns a human-readable
// error listing the missing roles, or nil if the configuration is complete.
func (s *Settings) Validate() error {
	var missing []string
	for _, name := range requiredRoles {
		if s.LLM(name) == nil {
			missing = append(missing, name)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	src := s.path
	if src == "" {
		src = "settings.toml"
	}
	return fmt.Errorf(
		"missing required LLM connection(s): %s — add an [[llmconnections]] entry with name=\"%s\" (and any others listed) in %s",
		strings.Join(missing, ", "), missing[0], src,
	)
}
