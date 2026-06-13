package main

import (
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
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
		if _, err := os.Stat(path); os.IsNotExist(err) {
			if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
				return fmt.Errorf("seeding %s: %w", path, err)
			}
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
		if body, ok := osSkills[osi.ID]; ok {
			if err := seed("SKILL-"+osi.ID+".md", renderOSSkill(body, osi.Fields)); err != nil {
				return err
			}
		}
	}
	return nil
}

// osSkillPlaceholder matches a {{KEY}} placeholder in a per-OS skill body.
var osSkillPlaceholder = regexp.MustCompile(`{{(\w+)}}`)

// renderOSSkill substitutes {{KEY}} placeholders in the per-OS skill body with
// the matching /etc/os-release field. A key absent from fields (a non-standard
// image missing VERSION_ID, say) resolves to "" rather than leaking a literal
// {{X}} to the LLM. Applied ONLY to per-OS skills — the justfile skill's
// {{var}}/{{args}} examples are seeded raw and never pass through here.
func renderOSSkill(body string, fields map[string]string) string {
	return osSkillPlaceholder.ReplaceAllStringFunc(body, func(m string) string {
		return fields[m[2:len(m)-2]] // strip the {{ }}, look up (empty if missing)
	})
}

// readSkillBody returns the body of one .codehalter/SKILL-*.md, or "" if absent.
// Used to inject a mid-session-seeded skill as a user message (see checkEnv).
func readSkillBody(cwd, name string) string {
	data, err := os.ReadFile(filepath.Join(cwd, ".codehalter", name))
	if err != nil {
		return ""
	}
	return string(data)
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
	dir := filepath.Join(cwd, ".codehalter")
	var b strings.Builder
	for _, n := range skillFiles(cwd) {
		data, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			continue
		}
		content := string(data)
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
