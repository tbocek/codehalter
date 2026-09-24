package main

import (
	"context"
	_ "embed" // for the //go:embed skill bodies below
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"
)

//go:embed res/SKILL-layout.md
var skillLayout string

//go:embed res/SKILL-base.md
var skillBase string

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

// shippedSkills is every skill codehalter carries, by filename. A project
// loads the applicable subset (skillSet) straight out of the binary, so a skill
// edited here reaches every project on the next session instead of only the
// ones set up after the change. A file of the same name in .codehalter
// overrides the shipped text; nothing is ever written there.
var shippedSkills = func() map[string]string {
	m := map[string]string{
		"SKILL-base.md":     skillBase,
		"SKILL-layout.md":   skillLayout,
		"SKILL-justfile.md": skillJustfile,
	}
	for id, body := range osSkills {
		m["SKILL-"+id+".md"] = body
	}
	return m
}()

// skillSet returns the skills this project loads, sorted so the system prompt's
// cached prefix is byte-stable across turns.
//
// What applies is decided by what is PRESENT in the tree and in the container,
// never by what is installed on PATH: the model needs SKILL-justfile.md before
// it installs `just` for the user, not after. There are no per-language skills,
// because a capable model knows its languages; the skills cover what it cannot
// know, which is this container, its distro, the task runner and how to check a
// rendered page. Anything else the user drops in .codehalter joins the set.
func skillSet(cwd string, stacks []string) []string {
	names := []string{"SKILL-base.md"}
	if slices.Contains(stacks, "css") {
		names = append(names, "SKILL-layout.md")
	}
	for _, n := range []string{"justfile", "Justfile", ".justfile"} {
		if _, err := os.Stat(filepath.Join(cwd, n)); err == nil {
			names = append(names, "SKILL-justfile.md")
			break
		}
	}
	// The distro skill carries no special templating: its os-release values come
	// through the same load-time {{cmd:...}} expansion as every other skill.
	if osi := readOSInfo(); osSkills[osi.ID] != "" {
		names = append(names, "SKILL-"+osi.ID+".md")
	}
	for _, n := range diskSkillFiles(cwd) {
		if shippedSkills[n] == "" && !slices.Contains(names, n) {
			names = append(names, n)
		}
	}
	sort.Strings(names)
	return names
}

// cmdPlaceholder matches a {{cmd:...}} placeholder in a skill body (single
// line — `.` doesn't cross newlines). Expanded at LOAD time via `sh -c`,
// inside the devcontainer, so a skill bakes live facts (today's date, tool
// versions) into its text instead of spending model turns probing. Load-time
// so per-session facts stay fresh (the date must move) while
// staying byte-stable WITHIN a session: the expanded result is built once at
// session init (SystemPrompt) and stored, so the cached prefix never shifts
// mid-session. It runs on the shipped text and on a user's own skill alike.
// Trust note: skills live in the project's
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

// overriddenBuiltins names the files in .codehalter that shadow something the
// binary ships: a phase prompt, a skill or a template macro. Sorted. A file of
// a new name (the user's own skill or macro) is an addition, not an override,
// and is not listed; the banner's skills line already shows those.
func overriddenBuiltins(cwd string) []string {
	entries, err := os.ReadDir(filepath.Join(cwd, ".codehalter"))
	if err != nil {
		return nil
	}
	shippedTemplates := map[string]bool{}
	if res, err := templateFS.ReadDir("res"); err == nil {
		for _, e := range res {
			shippedTemplates[e.Name()] = true
		}
	}
	var over []string
	for _, e := range entries {
		n := e.Name()
		if e.IsDir() {
			continue
		}
		if _, ok := shippedPrompts[n]; ok || shippedSkills[n] != "" || shippedTemplates[n] {
			over = append(over, n)
		}
	}
	sort.Strings(over)
	return over
}

// diskSkillFiles returns the SKILL-*.md filenames actually present in
// .codehalter/, sorted. These are the user's: either an override of a shipped
// skill, or one of their own.
func diskSkillFiles(cwd string) []string {
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

// skillBody returns one skill's text, {{cmd:}} expanded: the project's own copy
// in .codehalter when it has one, else the copy in the binary. "" when the name
// is neither, which is how a deleted user skill stops loading.
func skillBody(cwd, name string) string {
	body, ok := shippedSkills[name]
	if data, err := os.ReadFile(filepath.Join(cwd, ".codehalter", name)); err == nil {
		body, ok = string(data), true
	}
	if !ok {
		return ""
	}
	return expandCmdPlaceholders(body)
}

// loadSkills concatenates the given skills into the block that opens the system
// prompt. checkEnv rebuilds the prompt every turn and assigns only on a byte
// diff, so a skill that becomes applicable mid-session (a justfile appears)
// takes effect on the next turn, while an unchanged set keeps the prefix stable.
func loadSkills(cwd string, names []string) string {
	var b strings.Builder
	for _, n := range names {
		content := skillBody(cwd, n)
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
