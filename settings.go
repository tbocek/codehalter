package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

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
	// Ensure project .codehalter dir exists so ensureSettings can write a
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

	// Project-local fallback (typically the skeleton written by ensureSettings
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

// ensureSettings writes a commented skeleton to .codehalter/settings.toml when
// neither the global nor project-local file exists, and reloads a.settings so
// the rest of startup (checkLLM, capability checks) sees the new file. The
// skeleton's URL/model are placeholders — codehalter will still report the
// LLM as unreachable until the user edits them, but the file appears in Zed's
// file tree so they have something concrete to open and edit.
//
// Asks once per project. On "Skip", we don't write a marker — without
// settings the agent can't do anything useful, so re-prompting is the right
// behaviour. On the next session, the user either has a global file (no
// prompt) or still doesn't (prompt again).
func (a *agent) ensureSettings(ctx context.Context, cwd string, sid string) {
	if a.settings.path != "" {
		return
	}

	a.sendUpdate(ctx, sid, MessageChunk(KindAgentMessage, TextBlock(
		"No settings.toml found at ~/.config/codehalter/settings.toml or "+
			".codehalter/settings.toml. Codehalter needs at least one [[llm]] "+
			"entry pointing at your LLM server to function. I can write a "+
			"commented skeleton into this project's .codehalter/ folder; once "+
			"you've edited it, move it to ~/.config/codehalter/ to share it "+
			"across every project on this machine.\n\n")))

	tcId := a.StartToolCall(ctx, sid, "Write skeleton .codehalter/settings.toml?", "think", nil)
	ok, err := a.askYesNoAuto(ctx, sid, tcId, "Create", "Skip")
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if !ok {
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Skipped — codehalter cannot run any LLM until you create the file.")})
		return
	}

	path := filepath.Join(cwd, sessionDir, "settings.toml")
	if err := os.WriteFile(path, []byte(defaultSettingsTOML), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}

	// Reload so checkLLM sees the new file. The skeleton has placeholder
	// values that won't work as-is — checkLLM will surface that with its
	// own placeholder-detection warning.
	if loaded, err := loadSettings(cwd); err == nil {
		a.settings = loaded
		a.buildSlotSems()
	}

	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Wrote " + path)})
	a.sendUpdate(ctx, sid, MessageChunk(KindAgentMessage, TextBlock(
		"⚠ Wrote "+path+" with placeholder values — codehalter will not be able to call the LLM until you edit it.\n\n"+
			"1. Open the file (in Zed: Ctrl/Cmd+P → type `settings.toml`).\n"+
			"2. Replace `url` and `model` with values that match your LLM server (run `curl <url>/v1/models` to see the model ids your server reports).\n"+
			"3. Restart this Zed session so the new settings are loaded.\n"+
			"4. Optional: move the edited file to ~/.config/codehalter/settings.toml to share it across every project.\n\n")))
}

// settingsLooksPlaceholder reports whether the loaded settings still hold the
// skeleton's placeholder values. Lets checkLLM print a clear "edit your
// settings.toml" warning instead of a generic "unreachable" or "model not
// loaded" one when the user hasn't filled them in yet.
func settingsLooksPlaceholder(s Settings) bool {
	if len(s.LLM) == 0 {
		return false
	}
	return s.LLM[0].Model == "your-model-id"
}

// allConnections enumerates every distinct LLMConnection across the [[llm]]
// list. Used by checkLLM for the startup probe and by slash.go for the
// /status summary. Returns clones safe to mutate.
func (s *Settings) allConnections() []LLMConnection {
	out := make([]LLMConnection, len(s.LLM))
	copy(out, s.LLM)
	return out
}
