package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"
)

// defaultMaxTokens is the max_tokens injected into an LLM request when the
// user's params block doesn't set one. Bounds a runaway completion that loops
// inside a single LLM round-trip — the per-tool-loop iteration cap can't help
// there. 8192 is generous headroom (execute ~2-4k, plan/verify <1k); override
// per-role with `max_tokens` inside params_thinking / params_execute.
const defaultMaxTokens = 8192

// purposeSummary is the [[llm]] `purpose` value that hosts the per-turn
// summariser (see LLMConnection.Purpose and connForBackgroundLLM).
const purposeSummary = "summary"

type Settings struct {
	// LLM is the ordered list of OpenAI-compatible endpoints codehalter can
	// dispatch to. LLM[0] is the "main" connection: the foreground session
	// always runs on it, its KV cache holds the parent's history, and
	// background work (summariser) avoids it to keep that cache warm. LLM[1+]
	// are extras: one may host the summariser (purpose = "summary"). Each entry's
	// Parallel field caps how many concurrent requests it accepts.
	LLM []LLMConnection `toml:"llm"`

	// Prewarm fires one background 1-token LLM call at session open so the
	// server tokenizes and caches the prompt prefix (system prompt + tools)
	// before the user's first message; turn one then only pays for its own
	// delta. Only useful on backends with prefix caching (llama.cpp); elsewhere
	// it wastes one tiny request. nil means on.
	Prewarm *bool `toml:"prewarm,omitempty"`

	// FormatConfig controls the setup card that offers to pin a formatter config
	// for a project that has none (see formatterConfigNeeds). nil means on. Set
	// it to false for a project that deliberately keeps no formatter config, so
	// the card stops being offered at the start of every session.
	FormatConfig *bool `toml:"format_config,omitempty"`

	// UpdateCheck controls the once-a-day "a newer release exists" check
	// against the GitHub releases API (see version.go). nil means on. Set it to
	// false for a machine that should never reach github.com on its own, or for
	// an installation someone else keeps up to date; CODEHALTER_UPDATE=skip
	// does the same for a single run.
	UpdateCheck *bool `toml:"update_check,omitempty"`

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
// until a token is released. Held *per LLM call*: between calls (during local
// tool dispatch) the conn is free for another caller. Optional for llama.cpp:
// probeAllLLMs auto-fills it from /props total_slots (-np) when left at 0. Set
// it explicitly only for backends that don't report slots (vLLM, OpenAI, …) or
// to cap concurrency below the server's capacity; 0 with no detection means 1.
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
	// Purpose designates which non-foreground work routes to this entry.
	// "summary" sends the per-turn summariser here instead of LLM[0].
	// Empty means no designated background work.
	//
	// Named explicitly rather than inferred as "the first free entry after
	// LLM[0]": with two extras the inferred rule sends the summariser to
	// whichever happens to be idle, a small fast model on one turn, a slow
	// reasoning model the next. Naming the entry makes the routing stable and
	// lets the summariser live on a machine picked for it. Marking LLM[0] is allowed and simply means "summarise on the main
	// conn", which is also what no marking at all yields.
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
	// connForSession so callers don't have to know which of Params /
	// ParamsThinking / ParamsExecute applies.
	ExtraBody map[string]any `toml:"-"`

	// Slot is the flat display index shown in the live meter and the session-
	// log header as llm[<Slot>]. The foreground turn runs as llm[0]; background
	// work (summariser / git-commit) runs as llm[1] — the same physical
	// connection when there's a single [[llm]] entry with parallel >= 2, a
	// distinct slot so you can see which is in use (llama.cpp assigns the real
	// KV slot). Stamped by MainLLM / ConnAt / connForBackgroundLLM; runtime-only.
	Slot int `toml:"-"`

	// noThinkPrefill suppresses reasoning by APPENDING a closed think block to
	// the messages instead of changing chat_template_kwargs. Set (on a copy) by
	// withThinkingDisabled. A kwargs change re-runs the template over the whole
	// conversation, so the server sees a token sequence it has never held and
	// re-prefills from zero; an appended message is an extension, so every token
	// before it still matches. Measured against ai.jos.li on a 13,972-token
	// prompt: enable_thinking=false came back cached=0, the prefill came back
	// cached=13,968 of 13,978 and suppressed reasoning just as completely.
	// Runtime-only.
	noThinkPrefill bool

	// noTurnStats excludes this call from the per-turn "✅ Done" usage stats.
	// Set (on a copy) by prewarm: its call logs under the real sid for
	// diagnosability, but a turn that starts while the warm is still streaming
	// resets the counters BEFORE the warm's usage lands, so without this flag
	// the warm's ~10k prefill inflates that turn's "uncached" number.
	// Runtime-only.
	noTurnStats bool

	// streamRulesArmed opts this call into the stream-rule check (rules.go): a
	// pattern match aborts the generation mid-token and returns a
	// streamRuleError. Opt-IN rather than on-by-default because a rule abort is
	// only useful where something catches it and re-asks — that is the tool
	// loop's retry ladder and nowhere else. The background summariser, in
	// particular, passes the foreground's full tools array (for prefix-cache
	// reasons, see summariseCall) but has no ladder: a rule firing there would
	// silently downgrade the turn's note to the raw fallback. Set on a copy by
	// runToolLoop (forToolLoop). Runtime-only.
	streamRulesArmed bool

	// cacheLineage folds this call into the session's prefix-cache rewind check
	// (Session.noteCacheLineage). Opt-in for the same reason: the check compares
	// this call's cached count against the PREVIOUS call's prompt size, which is
	// only meaningful when the two share a message history. The tool loop's calls
	// do (each is the last plus an append); the background summariser's do not.
	// It runs a one-shot prompt on (usually) another server, and counting it would
	// report a rewind on every turn. Set on a copy by forToolLoop. Runtime-only.
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
}

// renderKey fingerprints the params that reach the server's chat template:
// everything the role configured except the samplers. Two calls with the same
// key render the same messages to the same tokens, so the second extends the
// first's KV prefix. Two different keys are two different token sequences, and
// a server with one slot can only hold one of them.
//
// Built from the role's params, not from the assembled request body: model,
// messages, tools and stream are codehalter's own and identical by
// construction. "" means "nothing that touches the template was configured".
//
// It exists so a detected rewind can NAME its cause. Without it the log can
// only list the four things that could have done it and let the user guess,
// which is what turned one real diagnosis into an offline analysis of a 397 MB
// session log.
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

// Why paramsFor hands back the role's params untouched, and in particular never
// adds chat_template_kwargs of its own:
//
// Turning reasoning off for execute is worth a lot on a thinking model.
// Measured over one 11.6h session against Qwen3.8-27B: 308 execute calls,
// 169030 completion tokens, 71.2 minutes of pure decode, of which reasoning was
// roughly 70%. chat_template_kwargs.enable_thinking=false delivers it: 71
// responses, 0 with reasoning, 20.7s -> 8.8s per call.
//
// Setting it here anyway would be a bad trade on a single-slot server, which is
// the default. That field is an argument to the server's Jinja chat template,
// so giving execute a different value from thinking gives the two roles
// different renderings, and the two alternate: 35 role switches over 436 calls
// in that same session, one every ~12 calls. With identical renderings those 35
// switches re-evaluated 121748 tokens between them (median 1526 per switch,
// 97.4% cached), which is 4.2 minutes at the server's measured 483 tok/s
// prefill. The same 35 calls carried 2414262 prompt tokens, so re-prefilling
// each from scratch is 83.3 minutes. That is more than the ~50 minutes of
// decode the change was buying.
//
// Sum, not 35x the median: the switch prompts are right-skewed (median 55308,
// mean 68978, max 136803) and the deep ones dominate the total.
//
// And re-prefill is what was actually observed. The one time the two renderings
// diverged in that session (the stuck-thinking retry, 22:32Z) the server came
// back with cached=0 on a 71997-token prompt, then 27585 re-evaluated switching
// back: a complete cache loss in both directions, on this server, at this
// context depth. n=1, but it is the only direct measurement and it points the
// conservative way.
//
// codehalter takes the decode win a third way, which costs nothing at all: the
// execute-role phases append an already-closed <think></think> for the model to
// continue (withThinkingDisabled). That is a suffix, not a re-render, so both
// roles keep asking for the same rendering and the 83 minutes never come due.
// Probed against the same server on a 13978-token prompt: the kwargs change
// came back cached=0, the append cached=13968, reasoning suppressed either way.
//
// The kwargs field stays a per-connection decision the user makes, not a
// default codehalter imposes, because whether it is affordable depends on the
// slot count of the server in front of it. res/settings.toml documents when to
// take it. On a server holding two or more slots each rendering keeps its own
// KV cache and the switch is cheap; on one slot it is the 83 minutes above.

// endpoint joins the configured server base with an API path, e.g.
// endpoint("/v1/models") → "http://host:8080/v1/models". The user configures
// only Server (the host root); codehalter owns the path layout — the
// OpenAI-compatible /v1/chat/completions and /v1/models, plus llama.cpp's
// root-level /props. Trailing slashes on Server are tolerated.
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

// settingsSources lists the candidates in precedence order:
//  1. <cwd>/.codehalter/settings.toml (project-local, preferred — per-project
//     overrides win even when a machine-wide config exists)
//  2. ~/.config/codehalter/settings.toml (global fallback, serving every project
//     without a local file)
//
// loadSettings reads the Active one; /settings prints the whole list so the
// user can see which file is in force and which it shadows. Both go through
// here, so the order is defined once.
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

// loadGlobalConfig reads ~/.config/codehalter/global.toml, best-effort. A missing
// or unreadable file yields the zero value (all false), the safe default —
// codehalter just won't add the corresponding optional mount.
//
// Best-effort means the zero value is always returned, never an error, but NOT
// that the failure goes unmentioned: absence is the normal case (install.sh may
// not have run) and stays silent, while a file that exists and fails to parse is
// a real misconfiguration. Silently treating it as "no gitconfig on the host"
// sends you debugging a devcontainer bind mount instead of a typo, the same
// misdiagnosis decodeSettings warns about below.
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
