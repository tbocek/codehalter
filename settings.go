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

// LLMConnection describes one llama.cpp/OpenAI-compatible endpoint.
//
// Position in Settings.LLMConnections determines the tier: index 0 is the
// "main" tier (foreground agent), indices 1+ are the "subagent" tier
// (parallel/offloaded work). Each connection declares per-role sampler
// overrides via extra_body_<role>; LLMFor merges the matching one into the
// request body. ExtraBody and Tag are runtime-only — they are populated by
// LLMFor on the returned clone, never read from TOML.
type LLMConnection struct {
	URL               string         `toml:"url"`
	APIKey            string         `toml:"api_key,omitempty"`
	Model             string         `toml:"model"`
	ExtraBodyThinking map[string]any `toml:"extra_body_thinking,omitempty"`
	ExtraBodyExecute  map[string]any `toml:"extra_body_execute,omitempty"`
	ExtraBodySummary  map[string]any `toml:"extra_body_summary,omitempty"`

	ExtraBody map[string]any `toml:"-"`
	Tag       string         `toml:"-"`
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

// LLMFor selects the connection to use for the given role and caller tier.
// tier == "main" prefers index 0, then 1+; tier == "subagent" prefers 1+,
// then 0. Within that ordering, the first connection that declares
// extra_body_<role> wins. If none does, the first connection in tier order
// is returned with no role-specific overrides — so single-endpoint setups
// keep working without per-role keys.
//
// The returned value is a clone with ExtraBody (the merged request overrides)
// and Tag (for the per-session log) populated. nil if no connections exist.
func (s *Settings) LLMFor(role, tier string) *LLMConnection {
	if len(s.LLMConnections) == 0 {
		return nil
	}
	var order []int
	if tier == "subagent" && len(s.LLMConnections) > 1 {
		for i := 1; i < len(s.LLMConnections); i++ {
			order = append(order, i)
		}
		order = append(order, 0)
	} else {
		order = append(order, 0)
		for i := 1; i < len(s.LLMConnections); i++ {
			order = append(order, i)
		}
	}

	extraFor := func(c *LLMConnection) map[string]any {
		switch role {
		case "thinking":
			return c.ExtraBodyThinking
		case "execute":
			return c.ExtraBodyExecute
		case "summary":
			return c.ExtraBodySummary
		}
		return nil
	}

	for _, i := range order {
		c := s.LLMConnections[i]
		if eb := extraFor(&c); eb != nil {
			c.ExtraBody = eb
			c.Tag = role
			return &c
		}
	}
	c := s.LLMConnections[order[0]]
	c.Tag = role
	return &c
}

// Validate ensures the connection list is non-empty. Per-role overrides are
// optional; LLMFor falls back to defaults when none are declared.
func (s *Settings) Validate() error {
	if len(s.LLMConnections) > 0 {
		return nil
	}
	src := s.path
	if src == "" {
		src = "settings.toml"
	}
	return fmt.Errorf("no [[llmconnections]] entries in %s — add at least one with url and model", src)
}
