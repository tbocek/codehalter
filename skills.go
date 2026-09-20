package main

import (
	"context"
	_ "embed" // for the //go:embed skill bodies below
	"fmt"
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

// retiredSkills names skills codehalter used to seed and has since dropped.
var retiredSkills = []string{"SKILL-makefile.md"}

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

// seedFile writes a default into .codehalter/ once: a file that exists, whether
// codehalter's copy or the user's edit, is left alone, and deleting it re-seeds.
// The publish is atomic (temp + rename), so a concurrent reader never sees a
// half-written file and two sessions seeding the same project at once simply
// overwrite each other with identical bytes.
func seedFile(dir, name, body string) error {
	if body == "" {
		return nil
	}
	path := filepath.Join(dir, name)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		return nil // already seeded (or stat failed otherwise) — leave it
	}
	// First seed of a fresh project creates .codehalter/ itself.
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("seeding %s: %w", path, err)
	}
	// Atomic publish (temp + rename): a concurrent reader (loadSkills) never sees
	// a half-written seed, and two same-cwd sessions seeding at once just overwrite
	// with identical bytes instead of racing a partial file. The ".seed-*" temp
	// name can't match the "SKILL-*.md" load glob.
	f, err := os.CreateTemp(filepath.Dir(path), ".seed-*")
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
	exists := func(names ...string) bool {
		for _, n := range names {
			if _, err := os.Stat(filepath.Join(cwd, n)); err == nil {
				return true
			}
		}
		return false
	}

	// Always-on container skill.
	if err := seedFile(dir, "SKILL-base.md", skillBase); err != nil {
		return err
	}
	// There are no per-language skills: a capable model knows its languages, and
	// what it cannot know (this container, the distro, the task runner, how to
	// check a rendered page) is what the skills below cover. The layout skill is
	// the one driven by the tree: it teaches the screenshot/measure loop, which
	// only matters where there are stylesheets or HTML templates.
	if slices.Contains(stacks, "css") {
		if err := seedFile(dir, "SKILL-layout.md", skillLayout); err != nil {
			return err
		}
	}
	// The only per-runner skill. There is no Makefile one: everything it said
	// about `make` (.PHONY, tab indentation, `:=` vs `=`, a subshell per recipe
	// line) is knowledge any model already has, whereas `just` is young enough
	// that its recipes-are-never-incremental rule has to be spelled out.
	if exists("justfile", "Justfile", ".justfile") {
		if err := seedFile(dir, "SKILL-justfile.md", skillJustfile); err != nil {
			return err
		}
	}
	// Seeding only ever adds, so a project set up before a skill was dropped
	// keeps its copy and loadSkills keeps concatenating it into every system
	// prompt. Sweeping the filenames is what actually retires them.
	for _, name := range retiredSkills {
		if err := os.Remove(filepath.Join(dir, name)); err != nil && !os.IsNotExist(err) {
			slog.Warn("removing a retired skill", "name", name, "err", err)
		}
	}
	// Migration: earlier versions seeded per-model pruned skill sets into
	// .codehalter/skills/<variant>/. Those were folded back into the single
	// SKILL-*.md set, so the tree is dead weight nothing reads — drop it, same
	// as the stale per-OS copies below. Codehalter wrote it, codehalter clears
	// it; a failure here is not worth aborting the session over.
	variants := filepath.Join(dir, "skills")
	if fi, err := os.Stat(variants); err == nil && fi.IsDir() {
		if err := os.RemoveAll(variants); err != nil {
			slog.Warn("could not remove the obsolete per-model skill directory (harmless — nothing loads it)", "dir", variants, "err", err)
		} else {
			slog.Info("removed obsolete per-model skill directory", "dir", variants)
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
			if err := seedFile(dir, "SKILL-"+osi.ID+".md", body); err != nil {
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

// skillPath resolves the file backing one skill NAME. One skill set, one
// location: .codehalter/<name>.
func skillPath(cwd, name string) string {
	return filepath.Join(cwd, ".codehalter", name)
}

// readSkillBody returns the body of one skill, or "" if absent. Used to inject a mid-session-seeded skill as a user message (see
// checkEnv) and by skills="auto" disclosure.
func readSkillBody(cwd, name string) string {
	data, err := os.ReadFile(skillPath(cwd, name))
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
// keeps the cache prefix stable.
func loadSkills(cwd string) string {
	var b strings.Builder
	for _, n := range skillFiles(cwd) {
		data, err := os.ReadFile(skillPath(cwd, n))
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
