package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestSkillSetNeedsNothingOnDisk is the point of reading skills out of the
// binary: a project with an empty .codehalter still gets the shipped text, so a
// skill edited in res/ reaches every project on its next session. Copying the
// skills out used to pin a project to the release that created it.
func TestSkillSetNeedsNothingOnDisk(t *testing.T) {
	cwd := t.TempDir()
	names := skillSet(cwd, nil)
	if !slices.Contains(names, "SKILL-base.md") {
		t.Fatalf("skillSet on a bare dir = %v, want the base skill in it", names)
	}
	body := loadSkills(cwd, names)
	if !strings.Contains(body, "# Container skill") && !strings.HasPrefix(strings.TrimSpace(body), "#") {
		t.Errorf("the base skill rendered as %q", truncate(body, 120))
	}
	if entries, err := os.ReadDir(cwd); err != nil || len(entries) != 0 {
		t.Errorf("loading skills wrote to the project: %v (err %v)", entries, err)
	}
}

// TestSkillSetApplicability: what applies is read off the tree and the
// container, not off what happens to be on disk.
func TestSkillSetApplicability(t *testing.T) {
	cwd := t.TempDir()
	if slices.Contains(skillSet(cwd, nil), "SKILL-justfile.md") {
		t.Error("the justfile skill applied to a project with no justfile")
	}
	if err := os.WriteFile(filepath.Join(cwd, "justfile"), []byte("test:\n\ttrue\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(skillSet(cwd, nil), "SKILL-justfile.md") {
		t.Error("a justfile in the tree did not pull in its skill")
	}
	if slices.Contains(skillSet(cwd, nil), "SKILL-layout.md") {
		t.Error("the layout skill applied to a project with no stylesheets")
	}
	if !slices.Contains(skillSet(cwd, []string{"css"}), "SKILL-layout.md") {
		t.Error("the css stack did not pull in the layout skill")
	}
}

// TestSkillOverrideAndOwnSkills pins the only two reasons a SKILL file exists
// in .codehalter: it replaces shipped text of the same name, or it is the
// user's own skill and joins the set. Deleting either goes back to the default.
func TestSkillOverrideAndOwnSkills(t *testing.T) {
	cwd := t.TempDir()
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	shipped := skillBody(cwd, "SKILL-base.md")
	if shipped == "" {
		t.Fatal("no shipped base skill")
	}

	override := filepath.Join(dir, "SKILL-base.md")
	if err := os.WriteFile(override, []byte("my own base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := skillBody(cwd, "SKILL-base.md"); got != "my own base\n" {
		t.Errorf("the override did not win: %q", truncate(got, 80))
	}
	if n := len(skillSet(cwd, nil)); n != len(skillSet(t.TempDir(), nil)) {
		t.Error("an override changed the size of the set; it replaces, it does not add")
	}
	if err := os.Remove(override); err != nil {
		t.Fatal(err)
	}
	if got := skillBody(cwd, "SKILL-base.md"); got != shipped {
		t.Error("removing the override did not go back to the shipped skill")
	}

	if err := os.WriteFile(filepath.Join(dir, "SKILL-house-rules.md"), []byte("# ours\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	names := skillSet(cwd, nil)
	if !slices.Contains(names, "SKILL-house-rules.md") {
		t.Fatalf("a project's own skill was not picked up: %v", names)
	}
	if !strings.Contains(loadSkills(cwd, names), "# ours") {
		t.Error("a project's own skill was not rendered")
	}
	if !slices.IsSorted(names) {
		t.Errorf("skillSet must be sorted so the cached prefix is byte-stable: %v", names)
	}
}

// TestOverriddenBuiltins: only files that carry a shipped name count, across
// all three kinds; a skill of the user's own and the config files do not.
func TestOverriddenBuiltins(t *testing.T) {
	cwd := t.TempDir()
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if got := overriddenBuiltins(cwd); len(got) != 0 {
		t.Fatalf("empty dir reports overrides: %v", got)
	}
	for _, n := range []string{"EXECUTE.md", "SKILL-base.md", "TEMPLATE-commit.md", "SKILL-house-rules.md", "settings.toml", "mcp.toml", "checks.done"} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got := overriddenBuiltins(cwd)
	want := []string{"EXECUTE.md", "SKILL-base.md", "TEMPLATE-commit.md"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("overrides = %v, want %v", got, want)
	}
}

// TestExpandCmdPlaceholders pins the {{cmd:...}} templating: stdout is spliced
// in trimmed, several placeholders on one line all expand, a failing command
// leaves its placeholder verbatim (visible in the prompt instead of baking a
// silent empty string), and the justfile skill's literal {{var}} examples and
// the os-release keys don't match.
func TestExpandCmdPlaceholders(t *testing.T) {
	got := expandCmdPlaceholders("v={{cmd:echo  1.2.3 }} on {{cmd:echo alpine}}!")
	if got != "v=1.2.3 on alpine!" {
		t.Errorf("expand = %q", got)
	}
	in := "v={{cmd:definitely-not-a-binary-xyz --version}}"
	if got := expandCmdPlaceholders(in); got != in {
		t.Errorf("failed cmd should leave the placeholder, got %q", got)
	}
	in = "Base: {{PRETTY_NAME}} and {{var}} stay"
	if got := expandCmdPlaceholders(in); got != in {
		t.Errorf("non-cmd placeholders must pass through, got %q", got)
	}
}

// TestSkillCmdExpandsAtLoad: the command runs when the skill is read, not when
// it is stored, so a skill's live facts (the date, a tool version) are current
// every session. An override keeps that property.
func TestSkillCmdExpandsAtLoad(t *testing.T) {
	cwd := t.TempDir()
	orig := osSkills["alpine"]
	osSkills["alpine"] = "# Alpine skill\nBase: {{cmd:echo Alpine Test}}, {{cmd:echo apk-tools 9.9}}.\n"
	shippedSkills["SKILL-alpine.md"] = osSkills["alpine"]
	defer func() {
		osSkills["alpine"] = orig
		shippedSkills["SKILL-alpine.md"] = orig
	}()

	want := "Base: Alpine Test, apk-tools 9.9."
	if body := skillBody(cwd, "SKILL-alpine.md"); !strings.Contains(body, want) {
		t.Errorf("shipped skill did not expand:\n%s", body)
	}
	if all := loadSkills(cwd, []string{"SKILL-alpine.md"}); !strings.Contains(all, want) {
		t.Errorf("loadSkills did not expand:\n%s", all)
	}
}

// TestLoadSkillsDeterministic verifies the concatenation is byte-stable across
// calls: a moving skill order would invalidate the prompt cache every session.
func TestLoadSkillsDeterministic(t *testing.T) {
	cwd := t.TempDir()
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Written in non-alphabetical order; readdir order is filesystem-dependent.
	for name, body := range map[string]string{
		"SKILL-ts.md": "# TS\n", "SKILL-go.md": "# Go\n",
		"SKILL-bash.md": "# Bash\n", "SKILL-java.md": "# Java\n",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	names := skillSet(cwd, nil)
	first := loadSkills(cwd, names)
	for i := range 5 {
		if got := loadSkills(cwd, names); got != first {
			t.Fatalf("run %d differs from run 0:\n  %q\n  %q", i, truncate(first, 120), truncate(got, 120))
		}
	}
	idx := func(needle string) int { return strings.Index(first, needle) }
	if !(idx("# Bash") < idx("# Go") && idx("# Go") < idx("# Java") && idx("# Java") < idx("# TS")) {
		t.Errorf("not alphabetical:\n%s", first)
	}
}
