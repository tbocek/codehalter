package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

// defaultMaxTokens is the max_tokens injected into an LLM request when the
// user's params block doesn't set one. Bounds a runaway completion that loops
// inside a single LLM round-trip — the per-tool-loop iteration cap can't help
// there. 8192 is generous headroom (execute ~2-4k, plan/verify <1k); override
// per-role with `max_tokens` inside params_thinking / params_execute.
const defaultMaxTokens = 8192

type Settings struct {
	// LLM is the ordered list of OpenAI-compatible endpoints codehalter can
	// dispatch to. LLM[0] is the "main" connection: the foreground session
	// always runs on it, its KV cache holds the parent's history, and
	// background work (summariser) avoids it to keep that cache warm. LLM[1+]
	// are extras — used to fan out subagents in parallel. Each entry's
	// Parallel field caps how many concurrent requests it accepts.
	LLM []LLMConnection `toml:"llm"`

	path string
}

// LLMConnection describes one llama.cpp/OpenAI-compatible endpoint.
//
// Sampler params can be split by role: `params_thinking` for plan/title/
// history (higher temperature, exploratory) and `params_execute` for
// execute/verify/document/summarize (lower temperature, follow-instruction).
// `params` is the legacy single-set field — still honoured as the fallback
// when the role-specific variant is empty. Each role-specific set hits the
// SAME prefix cache on the server because sampler params never enter the KV
// cache key — only prompt tokens do.
//
// Parallel is the per-conn concurrent-call cap. Each in-flight llmStream
// acquires one of N tokens from this conn's semaphore; excess calls block
// until a token is released. Default 1 when zero. Holds *per LLM call*, not
// per subagent — between calls (during local tool dispatch) the conn is free
// for another caller, so nested subagents on the same conn just queue.
type LLMConnection struct {
	URL            string         `toml:"url"`
	APIKey         string         `toml:"api_key,omitempty"`
	Model          string         `toml:"model"`
	Tag            string         `toml:"tag,omitempty"`
	Parallel       int            `toml:"parallel,omitempty"`
	Params         map[string]any `toml:"params,omitempty"`
	ParamsThinking map[string]any `toml:"params_thinking,omitempty"`
	ParamsExecute  map[string]any `toml:"params_execute,omitempty"`

	// ExtraBody is the runtime alias for the role-resolved Params used by
	// llmStream when assembling the OpenAI request body. Populated by
	// pickAvailable so callers don't have to know which of Params /
	// ParamsThinking / ParamsExecute applies.
	ExtraBody map[string]any `toml:"-"`
}

// paramsFor returns the sampler params for the given role, falling back to
// the legacy single `params` set when the role-specific one isn't configured.
// An empty map (nil) is fine — llmStream just won't add any extra body keys.
func (c *LLMConnection) paramsFor(role string) map[string]any {
	switch role {
	case "thinking":
		if len(c.ParamsThinking) > 0 {
			return c.ParamsThinking
		}
	case "execute":
		if len(c.ParamsExecute) > 0 {
			return c.ParamsExecute
		}
	}
	return c.Params
}

// parallelCap returns the effective concurrent-call cap for this conn,
// defaulting to 1 when unset or invalid.
func (c *LLMConnection) parallelCap() int {
	if c.Parallel < 1 {
		return 1
	}
	return c.Parallel
}

// loadSettings looks for settings.toml in this order:
// 1. ~/.config/codehalter/settings.toml (global, preferred)
// 2. <cwd>/.codehalter/settings.toml (project-local fallback)
// Always creates .codehalter/ in the project if it doesn't exist. When
// neither file exists, returns an empty Settings (with path "") and a nil
// error so callers can prompt the user to create one without aborting the
// session.
func loadSettings(cwd string) (Settings, error) {
	// Ensure project .codehalter dir exists so scaffoldSettings can write a
	// skeleton into it later.
	projectDir := filepath.Join(cwd, sessionDir)
	_ = os.MkdirAll(projectDir, 0755)

	// Global first — once a user has a global config it serves every project,
	// so we don't need to nag with the project-local prompt anymore.
	if home, err := os.UserHomeDir(); err == nil {
		globalPath := filepath.Join(home, ".config", "codehalter", "settings.toml")
		if _, err := os.Stat(globalPath); err == nil {
			return decodeSettings(globalPath)
		}
	}

	// Project-local fallback (typically the skeleton written by scaffoldSettings
	// on first run; users are encouraged to promote it to the global path).
	projectPath := filepath.Join(projectDir, "settings.toml")
	if _, err := os.Stat(projectPath); err == nil {
		return decodeSettings(projectPath)
	}

	return Settings{}, nil
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

// MainLLM returns the foreground connection (LLM[0]) with role-resolved
// ExtraBody and Tag, or nil when no LLM is configured. Used by startup probes
// and the main session's tool loop.
func (s *Settings) MainLLM(role string) *LLMConnection {
	if len(s.LLM) == 0 {
		return nil
	}
	c := s.LLM[0]
	c.ExtraBody = c.paramsFor(role)
	c.Tag = role
	return &c
}

// ConnAt returns LLM[idx] with role-resolved ExtraBody, or nil when idx is
// out of range. Used by pickAvailable to resolve pinned subagent sessions.
func (s *Settings) ConnAt(idx int, role string) *LLMConnection {
	if idx < 0 || idx >= len(s.LLM) {
		return nil
	}
	c := s.LLM[idx]
	c.ExtraBody = c.paramsFor(role)
	c.Tag = role
	return &c
}

// PinSlot pairs a conn index with the slot-within-conn occupied by a subagent
// task. For caps [1, 3] the breadth-first interleave yields
// [{0,0}, {1,0}, {1,1}, {1,2}] — the Slot field disambiguates multiple
// subagents pinned to the same conn so display labels can show conn/slot.
type PinSlot struct {
	Conn int
	Slot int
}

// SubagentPinOrder returns the breadth-first slot interleave used to assign
// pinned conn indices to subagent tasks. Caps total = sum of every LLM
// entry's parallelCap. For caps [1, 3] the returned slice is
// [{0,0}, {1,0}, {1,1}, {1,2}] — task 0 pins to LLM[0] slot 0, tasks 1..3
// to LLM[1] slots 0..2. Tasks beyond the slice length wrap (k % len), so
// the same per-conn cap still applies once they reach the conn's semaphore
// at dispatch time.
func (s *Settings) SubagentPinOrder() []PinSlot {
	if len(s.LLM) == 0 {
		return nil
	}
	maxCap := 0
	for i := range s.LLM {
		if c := s.LLM[i].parallelCap(); c > maxCap {
			maxCap = c
		}
	}
	var out []PinSlot
	for slot := 0; slot < maxCap; slot++ {
		for i := range s.LLM {
			if slot < s.LLM[i].parallelCap() {
				out = append(out, PinSlot{Conn: i, Slot: slot})
			}
		}
	}
	return out
}

// settingsLooksPlaceholder reports whether the loaded settings still hold the
// skeleton's placeholder values. Lets renderLLMStatus print a clear "edit your
// settings.toml" warning instead of a generic "unreachable" or "model not
// loaded" one when the user hasn't filled them in yet.
func settingsLooksPlaceholder(s Settings) bool {
	if len(s.LLM) == 0 {
		return false
	}
	return s.LLM[0].Model == "your-model-id"
}

// allConnections enumerates every distinct LLMConnection across the [[llm]]
// list. Used by probeAllLLMs for the prepare-phase probe and by slash.go for
// the /status summary. Returns clones safe to mutate.
func (s *Settings) allConnections() []LLMConnection {
	out := make([]LLMConnection, len(s.LLM))
	copy(out, s.LLM)
	return out
}
