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

// defaultMaxTokens is the max_tokens sent when the params set none. It bounds
// a completion that loops inside one round trip, which the tool-loop iteration
// cap cannot. Override per role with `max_tokens`.
const defaultMaxTokens = 8192

// purposeSummary is the [[llm]] `purpose` value that hosts the per-turn
// summariser (see LLMConnection.Purpose and connForBackgroundLLM).
const purposeSummary = "summary"

type Settings struct {
	// LLM is the ordered list of endpoints. LLM[0] is the main connection: the
	// foreground session runs on it and its KV cache holds the conversation, so
	// background work avoids it. LLM[1+] are extras; one may host the summariser
	// (purpose = "summary").
	LLM []LLMConnection `toml:"llm"`

	// Prewarm fires one 1-token call at session open, so the server caches the
	// prompt prefix before the first message. Only useful with prefix caching
	// (llama.cpp); elsewhere it wastes one tiny request. nil means on.
	Prewarm *bool `toml:"prewarm,omitempty"`

	// KeepWarm is how long a prefix may sit unused before codehalter refreshes
	// it with Prewarm's 1-token call, since a local server reclaims an idle slot
	// and the next turn then re-reads the whole prompt (measured: 174k tokens
	// after a 2m59s gap). Empty means keepWarmEvery; "off" disables it, which is
	// what a hosted endpoint wants: it caches on its own and bills per request.
	KeepWarm string `toml:"keep_warm,omitempty"`

	// FormatConfig controls the setup card that offers to pin a formatter config
	// for a project that has none (see formatterConfigNeeds). nil means on. Set
	// it to false for a project that deliberately keeps no formatter config, so
	// the card stops being offered at the start of every session.
	FormatConfig *bool `toml:"format_config,omitempty"`

	// UpdateCheck controls the once-a-day release check against GitHub (see
	// version.go). nil means on; false is for a machine that must not reach
	// github.com on its own. CODEHALTER_UPDATE=skip does the same for one run.
	UpdateCheck *bool `toml:"update_check,omitempty"`

	path string
}

// LLMConnection describes one llama.cpp/OpenAI-compatible endpoint.
//
// Sampler params split by role: `params_thinking` for planning, `params_execute`
// for execute/document/summarise, with `params` as the fallback. Both roles hit
// the same prefix cache, because samplers never enter the KV cache key.
//
// Parallel caps concurrent calls, held per LLM call, so the conn is free during
// tool dispatch. probeAllLLMs fills it from llama.cpp's total_slots when left
// at 0; set it for backends that report no slots. 0 undetected means 1.
type LLMConnection struct {
	// Server is the base URL of the OpenAI-compatible server — host root plus
	// any reverse-proxy path prefix, e.g. "http://localhost:8080" or
	// "https://gw.example/myllm". codehalter appends the API paths itself
	// (see endpoint): /v1/chat/completions for completions, /v1/models and the
	// root-level /props for probing. Do NOT put /v1/chat/completions here.
	Server string `toml:"server"`
	APIKey string `toml:"api_key,omitempty"`
	Model  string `toml:"model"`
	Tag    string `toml:"tag,omitempty"`
	// Purpose names the background work this entry hosts: "summary" sends the
	// per-turn summariser here instead of LLM[0]. Named rather than inferred as
	// "the first free extra", which with two extras would route the summariser
	// to whichever happened to be idle. On LLM[0] it means the same as unset.
	Purpose string `toml:"purpose,omitempty"`

	Parallel       *int           `toml:"parallel,omitempty"`
	Params         map[string]any `toml:"params,omitempty"`
	ParamsThinking map[string]any `toml:"params_thinking,omitempty"`
	ParamsExecute  map[string]any `toml:"params_execute,omitempty"`

	// ContextSize is the model's max prompt+output tokens. Optional — when
	// set, codehalter trusts this and skips metadata-endpoint probing for
	// ctx size. Required for backends that don't expose llama.cpp-style
	// discovery (OpenAI, Ollama, vLLM, OpenWebUI, LiteLLM, …).
	ContextSize *int `toml:"context_size,omitempty"`
	// ImageSupport declares whether the model accepts image inputs.
	// Optional — *bool so unset (probe), true (force on), and false (force
	// off) are distinct. nil falls through to discovery via /props or
	// /v1/models launch args; everywhere else the user must set it
	// explicitly to enable inline image_url blocks.
	ImageSupport *bool `toml:"image_support,omitempty"`

	// ExtraBody is the runtime alias for the role-resolved Params used by
	// llmStream when assembling the OpenAI request body. Populated by
	// connFor so callers don't have to know which of Params /
	// ParamsThinking / ParamsExecute applies.
	ExtraBody map[string]any `toml:"-"`

	// Slot is the display index shown in the meter and the log header as
	// llm[<Slot>]: the foreground turn is llm[0], background work llm[1], even
	// when both are one physical connection with parallel >= 2 (llama.cpp picks
	// the real KV slot). Runtime-only.
	Slot int `toml:"-"`

	// noThinkPrefill suppresses reasoning by APPENDING a closed think block
	// instead of changing chat_template_kwargs (set by withThinkingDisabled). A
	// kwargs change re-renders the whole conversation and re-prefills from zero;
	// an append leaves every earlier token matching. Measured on a 13,972-token
	// prompt: the kwargs change came back cached=0, the append cached=13,968.
	noThinkPrefill bool

	// noTurnStats keeps this call out of the turn's "✅ Done" stats. Set by
	// prewarm and keepWarm: a turn that starts while a warm call is still
	// streaming resets the counters first, so the warm's prefill would otherwise
	// inflate that turn's "uncached" number. Runtime-only.
	noTurnStats bool

	// streamRulesArmed opts this call into the stream-rule check (rules.go).
	// Opt-in because an abort is only useful where something re-asks, which is
	// the tool loop's retry ladder alone: on the summariser it would silently
	// downgrade the note to the raw fallback. Set by forToolLoop.
	streamRulesArmed bool

	// cacheLineage folds this call into the rewind check (noteCacheLineage).
	// Opt-in, because comparing against the PREVIOUS call only means something
	// when the two share a history: the tool loop's calls do, the summariser's
	// one-shot prompt does not. Set by forToolLoop. Runtime-only.
	cacheLineage bool
}

// samplerParams are the request fields that only steer generation. They never
// reach the server's chat template, so two calls that differ only in these
// render the same tokens and share a KV prefix. Everything else in a params
// table is assumed to change the rendering.
var samplerParams = map[string]bool{
	"frequency_penalty": true, "max_tokens": true, "min_p": true,
	"n": true, "presence_penalty": true, "repeat_penalty": true,
	"seed": true, "stop": true, "temperature": true, "top_k": true, "top_p": true,
	// tool_choice constrains generation with a grammar and renders the same
	// prompt (measured: prompt=250679 cached=250178 with "none"). Listed here so
	// a phase switch differing only in it is not REPORTED as a rendering change.
	"tool_choice": true,
}

// renderKey fingerprints the params that reach the server's chat template:
// everything the role configured except the samplers. Two calls with the same
// key render to the same tokens, so the second extends the first's KV prefix;
// two keys are two token sequences, and a one-slot server holds only one. It
// lets a detected rewind NAME its cause instead of listing suspects. "" means
// nothing template-affecting was configured.
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
	// encoding/json sorts map keys, so the same params always yield the same
	// key regardless of TOML ordering or map iteration order.
	b, err := json.Marshal(keep)
	if err != nil {
		return ""
	}
	return string(b)
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
		return c.Params
	}
	return c.Params
}

// paramsFor never adds chat_template_kwargs of its own, though turning
// reasoning off for execute is worth ~70% of its decode time. That field is a
// template argument, so differing per role gives the two roles two renderings,
// and on a single slot every plan→execute switch then re-prefills from zero:
// measured, that costs more than the decode it saves. codehalter takes the win
// with an appended closed <think></think> instead (withThinkingDisabled), which
// is a suffix and not a re-render. The full measurement, and when the kwargs
// route IS affordable (two or more slots), is in res/settings.toml.

// endpoint joins the configured server base with an API path. The user sets
// only Server (the host root); codehalter owns the path layout. Trailing
// slashes on Server are tolerated.
func (c *LLMConnection) endpoint(path string) string {
	return strings.TrimRight(c.Server, "/") + path
}

// parallelCap returns the effective concurrent-call cap for this conn,
// defaulting to 1 when unset or invalid.
func (c *LLMConnection) parallelCap() int {
	if c.Parallel != nil && *c.Parallel >= 1 {
		return *c.Parallel
	}
	return 1
}

// settingsSource is one candidate settings.toml and its fate. Selection is
// whole-file, never a merge: the first candidate that exists is Active and
// every later one that exists is shadowed by it.
type settingsSource struct {
	Path   string
	Scope  string // "project" | "global"
	Exists bool
	Active bool // the one loadSettings reads
}

// settingsSources lists the candidates in precedence order: the project's
// .codehalter/settings.toml, then ~/.config/codehalter/settings.toml for every
// project without one. loadSettings reads the active one and /settings prints
// the list, so the order is defined once.
func settingsSources(cwd string) []settingsSource {
	out := []settingsSource{{Path: filepath.Join(cwd, sessionDir, "settings.toml"), Scope: "project"}}
	if home, err := os.UserHomeDir(); err == nil {
		out = append(out, settingsSource{Path: filepath.Join(home, ".config", "codehalter", "settings.toml"), Scope: "global"})
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

// renderSettingsSources reports which settings.toml is in force and which
// candidates it shadows. Read off disk each call rather than from the loaded
// Settings: the question /settings answers is "which file am I reading", and a
// path cached in memory cannot answer it if the file changed underneath.
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
			fmt.Fprintf(&b, "❕ shadowed: `%s` (%s) — the file above wins, this one is never read.\n\n", src.Path, src.Scope)
		default:
			fmt.Fprintf(&b, "· absent: `%s` (%s)\n\n", src.Path, src.Scope)
		}
	}
	if !found {
		b.WriteString("🟡 No settings.toml at either path — codehalter cannot run a turn until one exists.\n\n")
	}
	return b.String()
}

// loadSettings decodes the highest-precedence settings.toml that exists (see
// settingsSources). Always creates .codehalter/ in the project if it doesn't
// exist. When neither file exists, returns an empty Settings (with path "") and
// a nil error so callers can prompt the user to create one without aborting the
// session.
func loadSettings(cwd string) (Settings, error) {
	// Ensure project .codehalter dir exists so scaffoldSettings can write a
	// skeleton into it later.
	projectDir := filepath.Join(cwd, sessionDir)
	if err := os.MkdirAll(projectDir, 0755); err != nil {
		// Non-fatal here: a global settings.toml (checked next) makes the
		// project dir irrelevant, and the project-local write path surfaces
		// its own error. Log so a genuine permission problem isn't silent.
		slog.Warn("loadSettings: could not create project .codehalter dir", "dir", projectDir, "err", err)
	}
	for _, s := range settingsSources(cwd) {
		if s.Active {
			return decodeSettings(s.Path)
		}
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

// GlobalConfig holds host-level facts captured at install time (install.sh writes
// ~/.config/codehalter/global.toml). Read when scaffolding a devcontainer to
// decide which optional bind mounts are safe — a bind whose source is missing
// fails the container start, so e.g. .gitconfig is only mounted when the host has
// one.
type GlobalConfig struct {
	HasGitconfigInHome bool `toml:"has_gitconfig_in_home"`
}

// loadGlobalConfig reads ~/.config/codehalter/global.toml, best-effort: the
// zero value (no optional mounts) is always returned, never an error. A missing
// file is normal and silent; one that exists and fails to parse is logged,
// since reading it as "no gitconfig on the host" sends you debugging a bind
// mount instead of a typo.
func loadGlobalConfig() GlobalConfig {
	var g GlobalConfig
	home, err := os.UserHomeDir()
	if err != nil {
		slog.Warn("loadGlobalConfig: no home directory; optional host mounts disabled", "err", err)
		return g
	}
	path := filepath.Join(home, ".config", "codehalter", "global.toml")
	md, err := toml.DecodeFile(path, &g)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Warn("loadGlobalConfig: unreadable global config (ignored — optional host mounts disabled)", "file", path, "err", err)
		}
		return GlobalConfig{}
	}
	for _, key := range md.Undecoded() {
		slog.Warn("unknown global config key (ignored — check for a typo)", "key", key.String(), "file", path)
	}
	return g
}

func decodeSettings(path string) (Settings, error) {
	var s Settings
	md, err := toml.DecodeFile(path, &s)
	if err != nil {
		return s, fmt.Errorf("loading %s: %w", path, err)
	}
	// BurntSushi silently drops keys that don't map to a struct field, so a
	// typo like `url = ...` (the field is `server`) leaves Server empty and the
	// LLM probe later fails with an opaque `unsupported protocol scheme ""`.
	// Surface every unmatched key here so the misconfiguration is named at load
	// time instead of misdiagnosed three layers down.
	for _, key := range md.Undecoded() {
		// skill_variant selected a per-model pruned skill set; those were folded
		// into the single .codehalter/SKILL-*.md set. Named here so an existing
		// config does not get told it has a typo.
		if key.String() == "skills" {
			slog.Warn("skills is no longer supported (ignored — every SKILL-*.md is always in the system prompt now; delete the line)", "file", path)
			continue
		}
		if k := key.String(); strings.HasSuffix(k, "skill_variant") {
			slog.Warn("skill_variant is no longer supported (ignored — there is one skill set now, .codehalter/SKILL-*.md; delete the line)", "key", k, "file", path)
			continue
		}
		slog.Warn("unknown settings key (ignored — check for a typo)", "key", key.String(), "file", path)
	}
	// A typo'd purpose is silent otherwise: the entry just never receives the
	// summariser and the work stays on LLM[0], which looks like the flag not
	// working rather than the flag not being read.
	for i := range s.LLM {
		if p := s.LLM[i].Purpose; p != "" && !strings.EqualFold(p, purposeSummary) {
			slog.Warn("unknown llm purpose (ignored — the only valid value is \"summary\")", "purpose", p, "llm", i, "file", path)
		}
	}
	s.path = path
	return s, nil
}

// prewarmEnabled reads the prewarm flag under cfgMu.
// Settings hot-reload each turn (prepare), so an edit takes effect on the
// next turn without a restart.
func (a *agent) prewarmEnabled() bool {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	return a.settings.Prewarm == nil || *a.settings.Prewarm
}

// keepWarmEvery is the default gap between keep-alive calls: short enough to
// sit under an idle-slot reclaim, long enough that an untouched session is not
// generating traffic every few seconds.
const keepWarmEvery = 3 * time.Minute

// keepWarmFor bounds how long codehalter keeps refreshing after the last real
// call. Past this the session is not idle, it is over: a machine left running
// overnight should not hold a GPU slot until morning.
const keepWarmFor = 30 * time.Minute

// keepWarmInterval resolves the configured gap, or 0 when keep-alive is off.
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

// MainLLM returns the foreground connection (LLM[0]) with role-resolved
// ExtraBody and Tag, or nil when no LLM is configured. Used by startup probes
// and the main session's tool loop.
func (s *Settings) MainLLM(role string) *LLMConnection {
	return s.ConnAt(0, role)
}

// ConnAt returns LLM[idx] with role-resolved ExtraBody, or nil when idx is
// out of range.
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

// allConnections enumerates every distinct LLMConnection across the [[llm]]
// list. Used by probeAllLLMs for the prepare-phase probe and by slash.go for
// the /status summary. Returns clones safe to mutate.
func (s *Settings) allConnections() []LLMConnection {
	out := make([]LLMConnection, len(s.LLM))
	copy(out, s.LLM)
	return out
}
