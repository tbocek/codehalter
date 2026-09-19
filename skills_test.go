package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// readSkill is a tiny helper: the on-disk body of a skill, or "" if absent.
func readSkill(t *testing.T, dir, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return ""
	}
	return string(b)
}

// TestEnsureSkillsSeedsOnceAndLeavesEdits: a skill is written when missing, and
// a later pass leaves an existing copy — user edit or not — untouched.
func TestEnsureSkillsSeedsOnceAndLeavesEdits(t *testing.T) {
	cwd := t.TempDir()
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// First pass seeds the always-on base skill from the embed.
	if err := ensureSkills(cwd, nil, osInfo{}); err != nil {
		t.Fatalf("ensureSkills (seed): %v", err)
	}
	if readSkill(t, dir, "SKILL-base.md") == "" {
		t.Fatal("SKILL-base.md should have been seeded")
	}
	// User edits it; a second pass must NOT overwrite (seed-once).
	if err := os.WriteFile(filepath.Join(dir, "SKILL-base.md"), []byte("my edits"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ensureSkills(cwd, nil, osInfo{}); err != nil {
		t.Fatalf("ensureSkills (re-run): %v", err)
	}
	if got := readSkill(t, dir, "SKILL-base.md"); got != "my edits" {
		t.Errorf("SKILL-base.md = %q, want 'my edits' (seed-once must not overwrite existing)", got)
	}
}

func TestEnsureSkillsPrunesOtherOS(t *testing.T) {
	cwd := t.TempDir()
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// A stale skill from a different OS must be pruned for the active OS.
	if err := os.WriteFile(filepath.Join(dir, "SKILL-debian.md"), []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ensureSkills(cwd, nil, osInfo{ID: "arch", Fields: map[string]string{}}); err != nil {
		t.Fatalf("ensureSkills err: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "SKILL-debian.md")); !os.IsNotExist(err) {
		t.Errorf("SKILL-debian.md should have been pruned")
	}
	if readSkill(t, dir, "SKILL-arch.md") == "" {
		t.Errorf("SKILL-arch.md should have been seeded for the active OS")
	}
}

// TestEnsureSkillsPrunesVariantDir: projects seeded by an older codehalter
// still carry .codehalter/skills/<variant>/ from the per-model skill split.
// Nothing loads it now (skillFiles globs the top level only), so ensureSkills
// clears it rather than leaving dead prompt copies in the tree.
func TestEnsureSkillsPrunesVariantDir(t *testing.T) {
	cwd := t.TempDir()
	dir := filepath.Join(cwd, ".codehalter")
	vdir := filepath.Join(dir, "skills", "gemma-4-31b")
	if err := os.MkdirAll(vdir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(vdir, "SKILL-base.md"), []byte("pruned variant"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := ensureSkills(cwd, nil, osInfo{}); err != nil {
		t.Fatalf("ensureSkills err: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "skills")); !os.IsNotExist(err) {
		t.Errorf(".codehalter/skills should have been removed, stat err = %v", err)
	}
	if readSkill(t, dir, "SKILL-base.md") == "" {
		t.Errorf("the generic skill set must still seed after the prune")
	}
}

// TestExpandCmdPlaceholders pins the seed-time {{cmd:...}} templating: stdout
// is spliced in trimmed, several placeholders on one line all expand, a
// failing command leaves its placeholder verbatim (visible in the seeded file
// instead of baking a silent empty string), and the justfile skill's literal
// {{var}} examples don't match.
func TestExpandCmdPlaceholders(t *testing.T) {
	got := expandCmdPlaceholders("v={{cmd:echo  1.2.3 }} on {{cmd:echo alpine}}!")
	if got != "v=1.2.3 on alpine!" {
		t.Errorf("expand = %q", got)
	}
	// Failing command → placeholder stays.
	in := "v={{cmd:definitely-not-a-binary-xyz --version}}"
	if got := expandCmdPlaceholders(in); got != in {
		t.Errorf("failed cmd should leave the placeholder, got %q", got)
	}
	// Non-cmd placeholders (os-release keys, justfile examples) pass through.
	in = "Base: {{PRETTY_NAME}} and {{var}} stay"
	if got := expandCmdPlaceholders(in); got != in {
		t.Errorf("non-cmd placeholders must pass through, got %q", got)
	}
}

// TestSkillCmdExpandsAtLoad pins the load-time templating contract: the
// seeded file keeps its {{cmd:...}} placeholder verbatim (user-ownable,
// re-expands fresh each session), while every model-facing read — loadSkills
// for the system prompt, readSkillBody for mid-session disclosure — renders
// the command output.
func TestSkillCmdExpandsAtLoad(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".codehalter"), 0o755); err != nil {
		t.Fatal(err)
	}
	orig := osSkills["alpine"]
	osSkills["alpine"] = "# Alpine skill\nBase: {{cmd:echo Alpine Test}}, {{cmd:echo apk-tools 9.9}}.\n"
	defer func() { osSkills["alpine"] = orig }()

	if err := ensureSkills(dir, nil, osInfo{ID: "alpine"}); err != nil {
		t.Fatalf("ensureSkills: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, ".codehalter", "SKILL-alpine.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "{{cmd:echo Alpine Test}}") {
		t.Errorf("seeded file should keep the placeholder verbatim:\n%s", raw)
	}
	if body := readSkillBody(dir, "SKILL-alpine.md"); !strings.Contains(body, "Base: Alpine Test, apk-tools 9.9.") {
		t.Errorf("readSkillBody should expand:\n%s", body)
	}
	if all := loadSkills(dir); !strings.Contains(all, "Base: Alpine Test, apk-tools 9.9.") {
		t.Errorf("loadSkills should expand:\n%s", all)
	}
}

// TestLoadSkillsDeterministic verifies loadSkills sorts entries so the
// concatenated system-prompt prefix is byte-stable across calls — a moving
// SKILL order would invalidate the cache on every session start.
func TestLoadSkillsDeterministic(t *testing.T) {
	dir := t.TempDir()
	cfgDir := filepath.Join(dir, ".codehalter")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// Write in non-alphabetical order; readdir order is filesystem-dependent.
	files := map[string]string{
		"SKILL-ts.md":   "# TS\n",
		"SKILL-go.md":   "# Go\n",
		"SKILL-bash.md": "# Bash\n",
		"SKILL-java.md": "# Java\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(cfgDir, name), []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	first := loadSkills(dir)
	for i := 0; i < 5; i++ {
		got := loadSkills(dir)
		if got != first {
			t.Errorf("loadSkills run %d differs from run 0:\n  run 0: %q\n  run %d: %q", i, first, i, got)
		}
	}
	// Bash should come first alphabetically; TS should be last. Looking at
	// the order via index ensures we catch a swap, not just presence.
	idx := func(needle string) int { return strings.Index(first, needle) }
	if idx("# Bash") != 0 {
		t.Errorf("expected loadSkills to start with Bash; got %q", truncate(first, 80))
	}
	if !(idx("# Bash") < idx("# Go") && idx("# Go") < idx("# Java") && idx("# Java") < idx("# TS")) {
		t.Errorf("loadSkills not alphabetical:\n%s", first)
	}
}
