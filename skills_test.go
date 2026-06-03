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

func TestSyncSkillSeedsWhenMissing(t *testing.T) {
	dir := t.TempDir()
	state := map[string]string{}
	changed, err := syncSkill(dir, "SKILL-x.md", "v1", false, state)
	if err != nil || !changed {
		t.Fatalf("syncSkill(missing) = (%v, %v), want (true, nil)", changed, err)
	}
	if got := readSkill(t, dir, "SKILL-x.md"); got != "v1" {
		t.Errorf("body = %q, want v1", got)
	}
	if state["SKILL-x.md"] != skillHash("v1") {
		t.Errorf("provenance not recorded on seed")
	}
}

func TestSyncSkillRefreshesUnedited(t *testing.T) {
	dir := t.TempDir()
	state := map[string]string{}
	// Seed v1, then ship v2 with the file untouched → must refresh.
	_, _ = syncSkill(dir, "SKILL-x.md", "v1", false, state)
	changed, err := syncSkill(dir, "SKILL-x.md", "v2", false, state)
	if err != nil || !changed {
		t.Fatalf("syncSkill(unedited, new embed) = (%v, %v), want (true, nil)", changed, err)
	}
	if got := readSkill(t, dir, "SKILL-x.md"); got != "v2" {
		t.Errorf("body = %q, want v2 (un-edited copy should refresh)", got)
	}
}

func TestSyncSkillPreservesUserEdit(t *testing.T) {
	dir := t.TempDir()
	state := map[string]string{}
	_, _ = syncSkill(dir, "SKILL-x.md", "v1", false, state)
	// User hand-edits the file.
	if err := os.WriteFile(filepath.Join(dir, "SKILL-x.md"), []byte("my edits"), 0o644); err != nil {
		t.Fatal(err)
	}
	changed, err := syncSkill(dir, "SKILL-x.md", "v3", false, state)
	if err != nil || changed {
		t.Fatalf("syncSkill(edited) = (%v, %v), want (false, nil)", changed, err)
	}
	if got := readSkill(t, dir, "SKILL-x.md"); got != "my edits" {
		t.Errorf("body = %q, want 'my edits' (user edit must survive)", got)
	}
}

func TestSyncSkillAuthoritativeMigration(t *testing.T) {
	dir := t.TempDir()
	// Pre-existing copy, no provenance (the migration case): an authoritative
	// (per-OS) skill must adopt the current embed — this is what keeps a
	// codehalter skill edit propagating to existing projects.
	if err := os.WriteFile(filepath.Join(dir, "SKILL-arch.md"), []byte("old arch"), 0o644); err != nil {
		t.Fatal(err)
	}
	state := map[string]string{}
	changed, err := syncSkill(dir, "SKILL-arch.md", "new arch", true, state)
	if err != nil || !changed {
		t.Fatalf("syncSkill(authoritative migration) = (%v, %v), want (true, nil)", changed, err)
	}
	if got := readSkill(t, dir, "SKILL-arch.md"); got != "new arch" {
		t.Errorf("body = %q, want 'new arch'", got)
	}
}

func TestSyncSkillNonAuthoritativeMigrationPreserves(t *testing.T) {
	dir := t.TempDir()
	// Pre-existing copy, no provenance, non-authoritative (stack/runner): an
	// untracked difference might be a genuine user edit, so leave it.
	if err := os.WriteFile(filepath.Join(dir, "SKILL-go.md"), []byte("old go"), 0o644); err != nil {
		t.Fatal(err)
	}
	state := map[string]string{}
	changed, err := syncSkill(dir, "SKILL-go.md", "new go", false, state)
	if err != nil || changed {
		t.Fatalf("syncSkill(non-authoritative migration) = (%v, %v), want (false, nil)", changed, err)
	}
	if got := readSkill(t, dir, "SKILL-go.md"); got != "old go" {
		t.Errorf("body = %q, want 'old go' (untracked copy preserved)", got)
	}
}

func TestSyncSkillRecordsAlreadyCurrent(t *testing.T) {
	dir := t.TempDir()
	// Untracked copy that happens to match the embed: no rewrite, but provenance
	// is recorded so a later embed bump can refresh it.
	if err := os.WriteFile(filepath.Join(dir, "SKILL-go.md"), []byte("same"), 0o644); err != nil {
		t.Fatal(err)
	}
	state := map[string]string{}
	changed, err := syncSkill(dir, "SKILL-go.md", "same", false, state)
	if err != nil || changed {
		t.Fatalf("syncSkill(already current) = (%v, %v), want (false, nil)", changed, err)
	}
	if state["SKILL-go.md"] != skillHash("same") {
		t.Errorf("provenance not recorded for already-current copy")
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
