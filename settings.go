package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

type Settings struct {
	// LLM is the singular main-tier connection — the small fast model that
	// owns the main session's KV cache. It serves every role at tier="main"
	// (plan, document, history compaction). Required for codehalter to run.
	LLM *LLMConnection `toml:"llm"`
	// SubLLM are the subagent-tier connections. Each entry declares a Tag
	// identifying the role it serves ("thinking" or "execute"); LLMCandidates
	// matches by tag. Untagged entries fall through as wildcards. Empty list
	// is valid — the main LLM then serves both tiers.
	SubLLM []LLMConnection `toml:"subllm"`

	path string
}

// LLMConnection describes one llama.cpp/OpenAI-compatible endpoint. The same
// struct is used for both the main [llm] and subagent [[subllm]] entries.
//
// Sampler params can be split by role: `params_thinking` for plan/title/
// history (higher temperature, exploratory) and `params_execute` for
// execute/verify/document/summarize (lower temperature, follow-instruction).
// `params` is the legacy single-set field — still honoured as the fallback
// when the role-specific variant is empty. Each role-specific set hits the
// SAME prefix cache on the server because sampler params never enter the KV
// cache key — only prompt tokens do.
//
// Tag is now purely informational (logged in llmStream); routing no longer
// filters by tag, every entry serves every role with the right param set.
type LLMConnection struct {
	URL            string         `toml:"url"`
	APIKey         string         `toml:"api_key,omitempty"`
	Model          string         `toml:"model"`
	Tag            string         `toml:"tag,omitempty"`
	Params         map[string]any `toml:"params,omitempty"`
	ParamsThinking map[string]any `toml:"params_thinking,omitempty"`
	ParamsExecute  map[string]any `toml:"params_execute,omitempty"`

	// ExtraBody is the runtime alias for the role-resolved Params used by
	// llmStream when assembling the OpenAI request body. Populated by
	// LLMCandidates / pickAvailable so callers don't have to know which
	// of Params / ParamsThinking / ParamsExecute applies.
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

// LLMFor returns the first connection in role/tier preference order. Kept for
// startup probes where slot state is irrelevant; runtime call sites should
// use agent.pickAvailable so a busy server can be skipped.
func (s *Settings) LLMFor(role, tier string) *LLMConnection {
	cs := s.LLMCandidates(role, tier)
	if len(cs) == 0 {
		return nil
	}
	return &cs[0]
}

// LLMCandidates returns every connection in preference order for the given
// role and caller tier. tier == "main" returns the [llm] entry (one slot).
// tier == "subagent" returns every [[subllm]] entry (round-robined by the
// caller), with the main [llm] as a last-resort fallback if no subllm is
// configured.
//
// Each candidate's ExtraBody is populated from the role-specific param set
// (params_thinking / params_execute / legacy params) and Tag is overridden
// to the requesting role for logging.
func (s *Settings) LLMCandidates(role, tier string) []LLMConnection {
	mainCopy := func() *LLMConnection {
		if s.LLM == nil {
			return nil
		}
		c := *s.LLM
		c.ExtraBody = c.paramsFor(role)
		c.Tag = role
		return &c
	}

	if tier == "main" {
		c := mainCopy()
		if c == nil {
			return nil
		}
		return []LLMConnection{*c}
	}

	// tier == "subagent": every entry serves every role with its own
	// per-role params. Round-robin happens in pickAvailable.
	var out []LLMConnection
	for _, c := range s.SubLLM {
		cc := c
		cc.ExtraBody = cc.paramsFor(role)
		cc.Tag = role
		out = append(out, cc)
	}
	if len(out) == 0 {
		if c := mainCopy(); c != nil {
			out = append(out, *c)
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
func (a *agent) ensureSettings(ctx context.Context, cwd string, sid SessionId) {
	if a.settings.path != "" {
		return
	}

	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
		"No settings.toml found at ~/.config/codehalter/settings.toml or "+
			".codehalter/settings.toml. Codehalter needs an [llm] entry "+
			"pointing at your LLM server to function. I can write a commented "+
			"skeleton into this project's .codehalter/ folder; once you've "+
			"edited it, move it to ~/.config/codehalter/ to share it across "+
			"every project on this machine.\n\n")))

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
	}

	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Wrote " + path)})
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
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
	if s.LLM == nil {
		return false
	}
	return s.LLM.Model == "your-model-id"
}

// allConnections enumerates every distinct LLMConnection across [llm] and
// [[subllm]]. Used by checkLLM for the startup probe and by slash.go for the
// /status summary. Returns clones safe to mutate.
func (s *Settings) allConnections() []LLMConnection {
	var out []LLMConnection
	if s.LLM != nil {
		out = append(out, *s.LLM)
	}
	out = append(out, s.SubLLM...)
	return out
}
