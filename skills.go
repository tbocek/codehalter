package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"
)

//go:embed res/SKILL-go.md
var skillGo string

//go:embed res/SKILL-ts.md
var skillTS string

//go:embed res/SKILL-js.md
var skillJS string

//go:embed res/SKILL-java.md
var skillJava string

//go:embed res/SKILL-bash.md
var skillBash string

//go:embed res/SKILL-c.md
var skillC string

//go:embed res/SKILL-base.md
var skillBase string

//go:embed res/SKILL-makefile.md
var skillMakefile string

//go:embed res/SKILL-justfile.md
var skillJustfile string

//go:embed res/SKILL-alpine.md
var skillAlpine string

//go:embed res/SKILL-arch.md
var skillArch string

//go:embed res/SKILL-debian.md
var skillDebian string

//go:embed res/SKILL-fedora.md
var skillFedora string

//go:embed res/SKILL-ubuntu.md
var skillUbuntu string

// defaultSkills maps a per-stack key (language stack from detectStacks)
// to the embedded skill body. The container / per-OS / per-runner skills
// live in their own seed paths inside ensureSkills, not here, because
// they're not driven by detectStacks.
var defaultSkills = map[string]string{
	"go":   skillGo,
	"ts":   skillTS,
	"js":   skillJS,
	"java": skillJava,
	"bash": skillBash,
	"c":    skillC,
}

// osSkills maps an /etc/os-release ID (as returned by readOSInfo) to the
// embedded skill body. Only IDs we have a SKILL-<id>.md for are present;
// readOSInfo filters everything else to "" before lookup.
var osSkills = map[string]string{
	"alpine": skillAlpine,
	"arch":   skillArch,
	"debian": skillDebian,
	"fedora": skillFedora,
	"ubuntu": skillUbuntu,
}

// ensureSkills seeds .codehalter/SKILL-*.md from the embeds based on what is
// PRESENT in the project tree (justfile / Makefile / language stacks) and in
// the container (/etc/os-release ID), NOT on what tooling is installed on PATH.
// This is load-bearing: the LLM needs SKILL-justfile.md loaded BEFORE it
// installs `just` for the user, otherwise the fix-dispatch turn has no idea what
// justfile syntax looks like.
//
// Seed-once: a skill is written only when missing; once it exists — codehalter's
// default or a user edit, doesn't matter — it's left alone. To pull in an
// updated embed, delete the file and it re-seeds. Per-OS handling additionally
// prunes every SKILL-<other-os>.md, because codehalter supports exactly one OS
// per session and a stale skill from a prior run on a different host would
// otherwise keep getting concatenated into every system prompt.
func ensureSkills(cwd string, stacks []string, osi osInfo) error {
	dir := filepath.Join(cwd, ".codehalter")
	seed := func(name, body string) error {
		if body == "" {
			return nil
		}
		path := filepath.Join(dir, name)
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			return nil // already seeded (or stat failed otherwise) — leave it
		}
		// Atomic publish (temp + rename): a concurrent reader (loadSkills) never sees
		// a half-written seed, and two same-cwd sessions seeding at once just overwrite
		// with identical bytes instead of racing a partial file. The ".seed-*" temp
		// name can't match the "SKILL-*.md" load glob.
		f, err := os.CreateTemp(dir, ".seed-*")
		if err != nil {
			return fmt.Errorf("seeding %s: %w", path, err)
		}
		tmp := f.Name()
		if _, err := f.Write([]byte(body)); err != nil {
			f.Close()
			os.Remove(tmp)
			return fmt.Errorf("seeding %s: %w", path, err)
		}
		if err := f.Close(); err != nil {
			os.Remove(tmp)
			return fmt.Errorf("seeding %s: %w", path, err)
		}
		if err := os.Rename(tmp, path); err != nil {
			os.Remove(tmp)
			return fmt.Errorf("seeding %s: %w", path, err)
		}
		return nil
	}
	exists := func(names ...string) bool {
		for _, n := range names {
			if _, err := os.Stat(filepath.Join(cwd, n)); err == nil {
				return true
			}
		}
		return false
	}

	// Always-on container skill.
	if err := seed("SKILL-base.md", skillBase); err != nil {
		return err
	}
	// Per-stack skills (driven by files in the tree).
	for _, stack := range stacks {
		if body, ok := defaultSkills[stack]; ok {
			if err := seed("SKILL-"+stack+".md", body); err != nil {
				return err
			}
		}
	}
	// Per-runner skills.
	if exists("justfile", "Justfile", ".justfile") {
		if err := seed("SKILL-justfile.md", skillJustfile); err != nil {
			return err
		}
	}
	if exists("Makefile", "makefile", "GNUmakefile") {
		if err := seed("SKILL-makefile.md", skillMakefile); err != nil {
			return err
		}
	}
	// Per-OS skill: prune the other-OS copies, then seed the active one
	// (rendered with this container's /etc/os-release values).
	if osi.ID != "" {
		for other := range osSkills {
			if other == osi.ID {
				continue
			}
			path := filepath.Join(dir, "SKILL-"+other+".md")
			if _, err := os.Stat(path); err != nil {
				continue
			}
			if err := os.Remove(path); err != nil {
				return fmt.Errorf("pruning stale %s: %w", path, err)
			}
		}
		// The per-OS skills carry no special templating: their os-release
		// values come through the same load-time {{cmd:...}} expansion
		// (sourcing /etc/os-release) as every other skill.
		if body, ok := osSkills[osi.ID]; ok {
			if err := seed("SKILL-"+osi.ID+".md", body); err != nil {
				return err
			}
		}
	}
	return nil
}

// cmdPlaceholder matches a {{cmd:...}} placeholder in a skill body (single
// line — `.` doesn't cross newlines). Expanded at LOAD time via `sh -c`,
// inside the devcontainer, so a skill bakes live facts (today's date, tool
// versions) into its text instead of spending model turns probing. Load-time
// (not seed-time) so per-session facts stay fresh — the date must move — while
// staying byte-stable WITHIN a session: the expanded result is built once at
// session init (SystemPrompt) and stored, so the cached prefix never shifts
// mid-session. The placeholder stays in the seeded file, so it also works in
// user-added and user-edited skills. Trust note: skills live in the project's
// .codehalter/, so their commands are project-controlled — same trust level as
// mcp.toml's child processes, and both run inside the devcontainer sandbox.
var cmdPlaceholder = regexp.MustCompile(`\{\{cmd:(.+?)\}\}`)

// expandCmdPlaceholders runs each {{cmd:...}} through the shell and splices
// in its trimmed stdout. A failing command (missing binary — e.g. a session
// on a host without the container's toolchain) leaves the placeholder
// verbatim in the rendered prompt and logs a warning, instead of silently
// substituting an empty string.
func expandCmdPlaceholders(body string) string {
	return cmdPlaceholder.ReplaceAllStringFunc(body, func(m string) string {
		cmd := m[len("{{cmd:") : len(m)-2]
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, "sh", "-c", cmd).Output()
		if err != nil {
			slog.Warn("skill {{cmd:}} failed — leaving the placeholder in place", "cmd", cmd, "err", err)
			return m
		}
		return strings.TrimSpace(string(out))
	})
}

// readSkillBody returns the body of one .codehalter/SKILL-*.md, or "" if absent.
// Used to inject a mid-session-seeded skill as a user message (see checkEnv).
func readSkillBody(cwd, name string) string {
	data, err := os.ReadFile(filepath.Join(cwd, ".codehalter", name))
	if err != nil {
		return ""
	}
	return expandCmdPlaceholders(string(data))
}

// skillFiles returns the SKILL-*.md filenames in .codehalter/, sorted — a
// deterministic order keeps loadSkills's cache prefix and listSkills's banner
// stable. Non-skill files are filtered out.
func skillFiles(cwd string) []string {
	entries, err := os.ReadDir(filepath.Join(cwd, ".codehalter"))
	if err != nil {
		return nil
	}
	var names []string
	for _, e := range entries {
		n := e.Name()
		if !e.IsDir() && strings.HasPrefix(n, "SKILL-") && strings.HasSuffix(n, ".md") {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	return names
}

// loadSkills concatenates every SKILL-*.md present in .codehalter/. Detection
// (detectStacks) decides which to seed initially, but loading honors whatever
// the user actually has on disk — drop a SKILL-rust.md in there manually and
// it gets picked up; delete one and it stops loading. checkEnv rebuilds the
// system prompt every turn and assigns only on a byte diff, so a skill added or
// removed mid-session takes effect on the next turn while an unchanged set
// keeps the cache prefix stable. skip (nil = keep all) excludes individual
// files — skills="auto" passes deferredSkillSkip to withhold untouched
// language skills from the prefix.
func loadSkills(cwd string, skip func(name string) bool) string {
	dir := filepath.Join(cwd, ".codehalter")
	var b strings.Builder
	for _, n := range skillFiles(cwd) {
		if skip != nil && skip(n) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			continue
		}
		content := expandCmdPlaceholders(string(data))
		if content != "" {
			b.WriteString(content)
			if !strings.HasSuffix(content, "\n") {
				b.WriteString("\n")
			}
			b.WriteString("\n")
		}
	}
	return b.String()
}

// listSkills returns the bare names (no "SKILL-" prefix, no ".md" suffix) of
// every SKILL-*.md in .codehalter/, sorted. notifyCapabilities shows these so
// the user sees which skill bodies got concatenated into the system prompt this
// turn — implicitly revealing what was pruned (anything missing from the list).
func listSkills(cwd string) []string {
	files := skillFiles(cwd)
	names := make([]string, len(files))
	for i, n := range files {
		names[i] = strings.TrimSuffix(strings.TrimPrefix(n, "SKILL-"), ".md")
	}
	return names
}

// ---------------------------------------------------------------------------
// skills="auto": deferred skills, disclosed on first touch
// ---------------------------------------------------------------------------

// deferredSkill describes one SKILL file that skills="auto" withholds from the
// system prompt until the session actually touches its stack. A skill is
// triggered by tool NAME (the per-runner tools imply their stack), by a token
// matcher run over the whitespace-split string arguments of every tool call —
// file paths in edit/read calls, filenames inside run_command strings — or by
// matchRaw over each WHOLE string argument (line structure intact), which is
// how shell scripts without a .sh extension are caught: the shebang inside a
// write_file/edit_file content argument classifies the file the same way
// `file`(1) would, without exec'ing anything. Skills not listed here — the
// container base, the OS skills, and anything the user drops in manually —
// always stay inline: there is no reliable touch signal to defer them on.
type deferredSkill struct {
	name     string
	tools    []string
	match    func(tok string) bool
	matchRaw func(s string) bool
}

// shebangShell matches a #! interpreter line naming a shell (sh, bash, dash,
// zsh — not fish, whose "sh" is mid-word). Appears in write_file/edit_file
// content the moment the model authors a script, extension or not.
var shebangShell = regexp.MustCompile(`#![^\n]*\b(?:ba|da|z)?sh\b`)

// Slice, not map: disclosure order inside one batch stays deterministic.
var deferredSkills = []deferredSkill{
	{name: "SKILL-bash.md", match: suffixAny(".sh", ".bash"), matchRaw: shebangShell.MatchString},
	{name: "SKILL-c.md", match: suffixAny(".c", ".h", ".cpp", ".cc", ".cxx", ".hpp")},
	{name: "SKILL-go.md", tools: []string{"go"}, match: suffixAny(".go")},
	{name: "SKILL-java.md", tools: []string{"gradle"}, match: suffixAny(".java")},
	{name: "SKILL-js.md", tools: []string{"npm"}, match: suffixAny(".js", ".jsx", ".mjs", ".cjs")},
	{name: "SKILL-justfile.md", tools: []string{"just"}, match: baseAny("justfile", "Justfile", ".justfile")},
	{name: "SKILL-makefile.md", tools: []string{"make"}, match: func(tok string) bool {
		return strings.HasSuffix(tok, ".mk") || baseAny("Makefile", "makefile", "GNUmakefile")(tok)
	}},
	{name: "SKILL-ts.md", match: suffixAny(".ts", ".tsx")},
}

func suffixAny(exts ...string) func(string) bool {
	return func(tok string) bool {
		for _, e := range exts {
			if strings.HasSuffix(tok, e) {
				return true
			}
		}
		return false
	}
}

func baseAny(names ...string) func(string) bool {
	return func(tok string) bool {
		return slices.Contains(names, path.Base(tok))
	}
}

func isDeferredSkill(name string) bool {
	return slices.ContainsFunc(deferredSkills, func(d deferredSkill) bool { return d.name == name })
}

// collectArgStrings returns every string value in a tool call's JSON
// arguments, intact (line structure preserved, for matchRaw). Non-JSON
// arguments degrade to the raw string as one value.
func collectArgStrings(rawArgs string) []string {
	var v any
	if json.Unmarshal([]byte(rawArgs), &v) != nil {
		return []string{rawArgs}
	}
	var out []string
	var walk func(any)
	walk = func(x any) {
		switch t := x.(type) {
		case string:
			out = append(out, t)
		case []any:
			for _, e := range t {
				walk(e)
			}
		case map[string]any:
			for _, e := range t {
				walk(e)
			}
		}
	}
	walk(v)
	return out
}

// argTokens splits already-extracted argument strings into whitespace tokens
// trimmed of shell punctuation, so both {"path":"cmd/main.go"} and
// {"command":"cat cmd/main.go && ls"} yield "cmd/main.go". Fed from
// collectArgStrings so each call's JSON is parsed exactly once.
func argTokens(strs []string) []string {
	var out []string
	for _, s := range strs {
		for _, tok := range strings.Fields(s) {
			if tok = strings.Trim(tok, "\"'`;&|()<>,="); tok != "" {
				out = append(out, tok)
			}
		}
	}
	return out
}

// disclosedSnapshot returns a copy of sess.DisclosedSkills under the session
// lock, safe to read after release.
func disclosedSnapshot(sess *Session) []string {
	sess.mu.Lock()
	defer sess.mu.Unlock()
	return append([]string(nil), sess.DisclosedSkills...)
}

// deferredSkillSkip returns the loadSkills filter for this session: nil in
// inline mode (everything loads), else a closure excluding deferred skills the
// session hasn't disclosed yet. Disclosed ones load again, so the
// compaction-time re-render (history.go) folds them into the cached prefix.
func (a *agent) deferredSkillSkip(sess *Session) func(name string) bool {
	if !a.skillsAuto() {
		return nil
	}
	disclosed := disclosedSnapshot(sess)
	return func(name string) bool {
		return isDeferredSkill(name) && !slices.Contains(disclosed, name)
	}
}

// discloseSkills is the skills="auto" trigger: called after each executed tool
// batch, it returns the wrapped body of every deferred skill this batch
// touches for the first time. Each disclosure is recorded on the session —
// DisclosedSkills (persisted; the prompt renderer stops excluding the skill at
// the next compaction re-render) and promptSkills (so prepare's mid-session
// seed loop never injects it a second time) — and appended to history via
// AddUser so a rebuild or resume replays it. The caller puts the returned
// messages on the live wire; both paths are append-only, so the cached prefix
// is untouched.
func (a *agent) discloseSkills(sid string, calls []toolCall) []string {
	if len(calls) == 0 || !a.skillsAuto() {
		return nil
	}
	sess := a.getSession(sid)
	if sess == nil {
		return nil
	}
	disclosed := disclosedSnapshot(sess)

	var names []string
	var tokens []string
	var raws []string
	for _, tc := range calls {
		names = append(names, tc.Function.Name)
		rs := collectArgStrings(tc.Function.Arguments)
		raws = append(raws, rs...)
		tokens = append(tokens, argTokens(rs)...)
	}

	var out []string
	for _, d := range deferredSkills {
		if slices.Contains(disclosed, d.name) {
			continue
		}
		hit := slices.ContainsFunc(names, func(n string) bool { return slices.Contains(d.tools, n) })
		if !hit && d.match != nil {
			hit = slices.ContainsFunc(tokens, d.match)
		}
		if !hit && d.matchRaw != nil {
			hit = slices.ContainsFunc(raws, d.matchRaw)
		}
		if !hit {
			continue
		}
		body := readSkillBody(sess.Cwd, d.name)
		if body == "" {
			continue // not seeded in this project (yet) — check again next batch
		}
		msg := "[Skill loaded on first use — " + d.name + ". Follow it for all matching work. It enters the system prompt at the next history compaction; until then it lives here.]\n\n" + body
		sess.AddUser(msg)
		sess.mu.Lock()
		sess.DisclosedSkills = append(sess.DisclosedSkills, d.name)
		sess.mu.Unlock()
		if !slices.Contains(sess.promptSkills, d.name) {
			sess.promptSkills = append(sess.promptSkills, d.name)
		}
		a.logSession(sid, "SKILL", "disclosed %s (skills=auto)", d.name)
		out = append(out, msg)
	}
	if len(out) > 0 {
		sess.saveOrLog()
	}
	return out
}
