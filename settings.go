package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
)

// defaultMaxTokens bounds a completion that loops within one round trip, which the
// tool-loop iteration cap cannot.
const defaultMaxTokens = 8192

const purposeSummary = "summary"

type Settings struct {
	// LLM[0] runs the foreground session and its KV cache holds the conversation, so
	// background work avoids it.
	LLM []LLMConnection `toml:"llm"`

	// Prewarm caches the prompt prefix with a 1-token call at session open. nil means on.
	Prewarm *bool `toml:"prewarm,omitempty"`

	// KeepWarm refreshes an idle prefix before a local server reclaims its slot. Empty means
	// keepWarmEvery; "off" suits hosted endpoints, which cache on their own and bill per request.
	KeepWarm string `toml:"keep_warm,omitempty"`

	// UpdateCheck nil means on; CODEHALTER_UPDATE=skip disables it for one run.
	UpdateCheck *bool `toml:"update_check,omitempty"`

	path string
}

// Per-role sampler params share one prefix cache because samplers never enter the KV cache key.
type LLMConnection struct {
	// Server is the base URL (host root plus any proxy prefix), without /v1/chat/completions.
	Server string `toml:"server"`
	APIKey string `toml:"api_key,omitempty"`
	Model  string `toml:"model"`
	Tag    string `toml:"tag,omitempty"`
	// Purpose "summary" routes the summariser here. Named, not inferred, so with two
	// extras it does not land on whichever happens to be idle.
	Purpose string `toml:"purpose,omitempty"`

	// Parallel is held per LLM call, not during tool dispatch; probeAllLLMs fills it from
	// llama.cpp total_slots when unset.
	Parallel       *int           `toml:"parallel,omitempty"`
	Params         map[string]any `toml:"params,omitempty"`
	ParamsThinking map[string]any `toml:"params_thinking,omitempty"`
	ParamsExecute  map[string]any `toml:"params_execute,omitempty"`

	// ContextSize overrides probing; required for backends without llama.cpp-style discovery.
	ContextSize *int `toml:"context_size,omitempty"`
	// ImageSupport nil means probe via /props or /v1/models launch args.
	ImageSupport *bool `toml:"image_support,omitempty"`

	// ExtraBody is the role-resolved params, set at runtime by ConnAt.
	ExtraBody map[string]any `toml:"-"`

	// Slot is a display index only (llm[0] foreground, llm[1] background); llama.cpp
	// picks the real KV slot.
	Slot int `toml:"-"`

	// noThinkPrefill disables reasoning by appending a closed think block: changing
	// chat_template_kwargs would re-render the conversation and re-prefill from zero.
	noThinkPrefill bool

	// noPrefill marks a server (Halogen) that 400s on continuation prefill with a forced
	// tool_choice. Set by llmStream on the first such 400 for the process lifetime; thinking
	// off then uses the server's own mechanism, at the cost of a separate cache per role.
	noPrefill bool

	// noTurnStats keeps a warm call's prefill out of the stats of a turn started meanwhile.
	noTurnStats bool

	// cacheLineage opts into the rewind check; only calls sharing a history (the tool loop) qualify.
	cacheLineage bool
}

// samplerParams never reach the chat template, so they don't change the rendered prompt.
// Every other params key is assumed to.
var samplerParams = map[string]bool{
	"frequency_penalty": true, "max_tokens": true, "min_p": true,
	"n": true, "presence_penalty": true, "repeat_penalty": true,
	"seed": true, "stop": true, "temperature": true, "top_k": true, "top_p": true,
	// tool_choice only adds a grammar; the rendered prompt is unchanged.
	"tool_choice": true,
}

// renderKey fingerprints the template-affecting params so a detected cache rewind can
// name its cause. "" means none are configured.
func renderKey(extra map[string]any) string {
	keep := map[string]any{}
	for k, v := range extra {
		if !samplerParams[k] {
			keep[k] = v
		}
	}
	if len(keep) == 0 {
		return ""
	}
	// encoding/json sorts map keys, so the key is deterministic.
	b, err := json.Marshal(keep)
	if err != nil {
		return ""
	}
	return string(b)
}

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
		return c.Params
	}
	return c.Params
}

func (c *LLMConnection) endpoint(path string) string {
	return strings.TrimRight(c.Server, "/") + path
}

func (c *LLMConnection) parallelCap() int {
	if c.Parallel != nil && *c.Parallel >= 1 {
		return *c.Parallel
	}
	return 1
}

// Selection is whole-file, never a merge: the first existing candidate wins.
type settingsSource struct {
	Path   string
	Scope  string // "project" | "global"
	Exists bool
	Active bool
}

func settingsSources(cwd string) []settingsSource {
	out := []settingsSource{{Path: filepath.Join(cwd, sessionDir, "settings.toml"), Scope: "project"}}
	if path, err := globalConfigPath("settings.toml"); err == nil {
		out = append(out, settingsSource{Path: path, Scope: "global"})
	}
	claimed := false
	for i := range out {
		if _, err := os.Stat(out[i].Path); err != nil {
			continue
		}
		out[i].Exists = true
		out[i].Active = !claimed
		claimed = true
	}
	return out
}

// renderSettingsSources reads disk, not the loaded Settings, so it reflects files changed since load.
func renderSettingsSources(cwd string) string {
	var b strings.Builder
	b.WriteString("**Settings**\n\n")
	found := false
	for _, src := range settingsSources(cwd) {
		switch {
		case src.Active:
			found = true
			fmt.Fprintf(&b, "✅ in use: `%s` (%s)\n\n", src.Path, src.Scope)
		case src.Exists:
			fmt.Fprintf(&b, "❕ shadowed: `%s` (%s): the file above wins, this one is never read.\n\n", src.Path, src.Scope)
		default:
			fmt.Fprintf(&b, "· absent: `%s` (%s)\n\n", src.Path, src.Scope)
		}
	}
	if !found {
		b.WriteString("🟡 No settings.toml at either path: codehalter cannot run a turn until one exists.\n\n")
	}
	return b.String()
}

// loadSettings returns an empty Settings and nil error when no file exists.
func loadSettings(cwd string) (Settings, error) {
	for _, s := range settingsSources(cwd) {
		if s.Active {
			return decodeSettings(s.Path)
		}
	}
	return Settings{}, nil
}

// globalConfigPath is ~/.config/codehalter/<name>.
func globalConfigPath(name string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".config", "codehalter", name), nil
}

func loadGlobalSettings() (Settings, error) {
	globalPath, err := globalConfigPath("settings.toml")
	if err != nil {
		return Settings{}, err
	}
	if _, err := os.Stat(globalPath); err != nil {
		return Settings{}, fmt.Errorf("no global settings.toml at %s", globalPath)
	}
	return decodeSettings(globalPath)
}

// GlobalConfig holds host facts written by install.sh to ~/.config/codehalter/global.toml.
type GlobalConfig struct {
	HasGitconfigInHome bool `toml:"has_gitconfig_in_home"`
}

func loadGlobalConfig() GlobalConfig {
	var g GlobalConfig
	path, err := globalConfigPath("global.toml")
	if err != nil {
		slog.Warn("loadGlobalConfig: no home directory; optional host mounts disabled", "err", err)
		return g
	}
	md, err := toml.DecodeFile(path, &g)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Warn("loadGlobalConfig: unreadable global config (ignored, optional host mounts disabled)", "file", path, "err", err)
		}
		return GlobalConfig{}
	}
	for _, key := range md.Undecoded() {
		slog.Warn("unknown global config key (ignored, check for a typo)", "key", key.String(), "file", path)
	}
	return g
}

func decodeSettings(path string) (Settings, error) {
	var s Settings
	md, err := toml.DecodeFile(path, &s)
	if err != nil {
		return s, fmt.Errorf("loading %s: %w", path, err)
	}
	// BurntSushi silently drops unknown keys, so a typo like `url` would otherwise
	// surface much later as an opaque probe error.
	for _, key := range md.Undecoded() {
		slog.Warn("unknown settings key (ignored, check for a typo)", "key", key.String(), "file", path)
	}
	for i := range s.LLM {
		if p := s.LLM[i].Purpose; p != "" && !strings.EqualFold(p, purposeSummary) {
			slog.Warn("unknown llm purpose (ignored, the only valid value is \"summary\")", "purpose", p, "llm", i, "file", path)
		}
	}
	s.path = path
	return s, nil
}

func (a *agent) prewarmEnabled() bool {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	return a.settings.Prewarm == nil || *a.settings.Prewarm
}

// keepWarmEvery must stay under a local server's idle-slot reclaim time.
const keepWarmEvery = 3 * time.Minute

// keepWarmFor stops refreshing this long after the last real call, so an abandoned
// session frees the GPU slot.
const keepWarmFor = 30 * time.Minute

func (a *agent) keepWarmInterval() time.Duration {
	a.cfgMu.RLock()
	raw := strings.TrimSpace(a.settings.KeepWarm)
	a.cfgMu.RUnlock()
	switch strings.ToLower(raw) {
	case "":
		return keepWarmEvery
	case "off", "false", "no", "0":
		return 0
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d <= 0 {
		slog.Warn("settings: keep_warm is not a duration, using the default", "value", raw, "default", keepWarmEvery)
		return keepWarmEvery
	}
	return d
}

func (s *Settings) ConnAt(idx int, role string) *LLMConnection {
	if idx < 0 || idx >= len(s.LLM) {
		return nil
	}
	c := s.LLM[idx]
	c.ExtraBody = c.paramsFor(role)
	c.Tag = role
	c.Slot = idx
	return &c
}
