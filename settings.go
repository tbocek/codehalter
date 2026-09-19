package main

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"

	"github.com/tbocek/codehalter/llm"
)

// purposeSummary is the [[llm]] `purpose` value that hosts the per-turn
// summariser (see llm.Conn.Purpose and connForBackgroundLLM).
const purposeSummary = "summary"

type Settings struct {
	// LLM is the ordered list of OpenAI-compatible endpoints codehalter can
	// dispatch to. LLM[0] is the "main" connection: the foreground session
	// always runs on it, its KV cache holds the parent's history, and
	// background work (summariser) avoids it to keep that cache warm. LLM[1+]
	// are extras: one may host the summariser (purpose = "summary"). Each entry's
	// Parallel field caps how many concurrent requests it accepts.
	LLM []llm.Conn `toml:"llm"`

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

// Why llm.Conn.ParamsFor hands back the role's params untouched, and in particular never
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
// continue (llm.Conn.WithThinkingDisabled). That is a suffix, not a re-render, so both
// roles keep asking for the same rendering and the 83 minutes never come due.
// Probed against the same server on a 13978-token prompt: the kwargs change
// came back cached=0, the append cached=13968, reasoning suppressed either way.
//
// The kwargs field stays a per-connection decision the user makes, not a
// default codehalter imposes, because whether it is affordable depends on the
// slot count of the server in front of it. res/settings.toml documents when to
// take it. On a server holding two or more slots each rendering keeps its own
// KV cache and the switch is cheap; on one slot it is the 83 minutes above.

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
func (s *Settings) MainLLM(role string) *llm.Conn {
	return s.ConnAt(0, role)
}

// ConnAt returns LLM[idx] with role-resolved ExtraBody, or nil when idx is
// out of range.
func (s *Settings) ConnAt(idx int, role string) *llm.Conn {
	if idx < 0 || idx >= len(s.LLM) {
		return nil
	}
	c := s.LLM[idx]
	c.ExtraBody = c.ParamsFor(role)
	c.Tag = role
	c.Slot = idx
	return &c
}

// allConnections enumerates every distinct llm.Conn across the [[llm]]
// list. Used by probeAllLLMs for the prepare-phase probe and by slash.go for
// the /status summary. Returns clones safe to mutate.
func (s *Settings) allConnections() []llm.Conn {
	out := make([]llm.Conn, len(s.LLM))
	copy(out, s.LLM)
	return out
}
