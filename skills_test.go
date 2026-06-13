package main

import (
	"os"
	"path/filepath"
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
	// First pass seeds the always-on container skill from the embed.
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
