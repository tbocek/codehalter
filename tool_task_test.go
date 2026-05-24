package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// TestClassifyTask covers the task-name heuristic. The classifier must be
// narrow enough that "checkout" doesn't get labelled "lint" just because it
// contains "check".
func TestClassifyTask(t *testing.T) {
	cases := map[string]string{
		"build":      "build",
		"compile":    "build",
		"bundle":     "build",
		"test":       "test",
		"unit-test":  "test",
		"spec":       "test",
		"lint":       "lint",
		"vet":        "lint",
		"check":      "lint",
		"clippy":     "lint",
		"fmt":        "format",
		"format":     "format",
		"prettier":   "format",
		"clean":      "", // not in any category yet
		"start":      "",
		"checkout":   "",     // segment match: "checkout" is one segment, not "check"
		"checkstyle": "",     // same — "checkstyle" is one segment
		"test-build": "test", // first segment wins
		"test:unit":  "test", // colon separator
		"build.all":  "build",
	}
	for task, want := range cases {
		if got := classifyTask(task); got != want {
			t.Errorf("classifyTask(%q): got %q, want %q", task, got, want)
		}
	}
}

// TestClassifyRunners verifies each task gets slotted into its category and
// the distinct runner list is preserved.
func TestClassifyRunners(t *testing.T) {
	runners := []taskRunner{
		{Name: "go", Tasks: []string{"build", "test", "vet", "fmt"}},
		{Name: "make", Tasks: []string{"build", "clean", "release"}},
	}
	caps := classifyRunners(runners)

	if !slices.Equal(caps.runners, []string{"go", "make"}) {
		t.Errorf("runners: got %v, want [go make]", caps.runners)
	}
	if !slices.Equal(caps.build, []string{"go:build", "make:build"}) {
		t.Errorf("build: got %v", caps.build)
	}
	if !slices.Equal(caps.test, []string{"go:test"}) {
		t.Errorf("test: got %v", caps.test)
	}
	if !slices.Equal(caps.lint, []string{"go:vet"}) {
		t.Errorf("lint: got %v", caps.lint)
	}
	if !slices.Equal(caps.format, []string{"go:fmt"}) {
		t.Errorf("format: got %v", caps.format)
	}
}

// TestDiscoverGoAndCargo verifies the zero-parse runners fire when their
// manifest exists and produce the standard subcommand list. Go discovery
// is a fallback that defers to a justfile/Makefile when one is present.
func TestDiscoverGoAndCargo(t *testing.T) {
	goDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(goDir, "go.mod"), []byte("module x\n"), 0644); err != nil {
		t.Fatalf("go.mod: %v", err)
	}
	r := discoverGo(goDir)
	if r == nil || r.Name != "go" || !slices.Contains(r.Tasks, "test") || !slices.Contains(r.Tasks, "vet") {
		t.Errorf("discoverGo: got %+v", r)
	}
	if args := r.Args("test"); !slices.Equal(args, []string{"test", "./..."}) {
		t.Errorf("go Args(test): got %v, want [test ./...]", args)
	}

	// A Makefile alongside go.mod should suppress the Go fallback.
	if err := os.WriteFile(filepath.Join(goDir, "Makefile"), []byte("test:\n\tgo test ./...\n"), 0644); err != nil {
		t.Fatalf("Makefile: %v", err)
	}
	if r := discoverGo(goDir); r != nil {
		t.Errorf("discoverGo with Makefile present: got %+v, want nil", r)
	}

	cargoDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(cargoDir, "Cargo.toml"), []byte("[package]\nname=\"x\"\n"), 0644); err != nil {
		t.Fatalf("Cargo.toml: %v", err)
	}
	r = discoverCargo(cargoDir)
	if r == nil || r.Name != "cargo" || !slices.Contains(r.Tasks, "clippy") {
		t.Errorf("discoverCargo: got %+v", r)
	}

	// Absent manifests return nil cleanly.
	if r := discoverGo(t.TempDir()); r != nil {
		t.Errorf("discoverGo on empty dir: got %+v, want nil", r)
	}
	if r := discoverCargo(t.TempDir()); r != nil {
		t.Errorf("discoverCargo on empty dir: got %+v, want nil", r)
	}
}

// TestEmptyProjectFlag covers the deferred-bootstrap path: a fresh empty
// dir sets a.emptyProject so the first user turn can inject a hint asking
// what language/runner to use. Populated dirs must not be flagged.
func TestEmptyProjectFlag(t *testing.T) {
	dir := t.TempDir()
	if !isEmptyProject(dir) {
		t.Fatal("expected fresh tempdir to be empty")
	}

	s, err := newSession(dir) // creates .codehalter/ but no source files
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	if !isEmptyProject(dir) {
		t.Error("expected dir with only .codehalter/ to still count as empty")
	}

	a := &agent{sessions: map[string]*Session{s.ID: s}}
	a.discoverRunners(dir)

	if !a.emptyProject {
		t.Error("expected emptyProject=true for empty dir")
	}
	if len(a.capabilities.runners) != 0 {
		t.Errorf("expected no runners discovered, got %v", a.capabilities.runners)
	}
	if _, err := os.Stat(filepath.Join(dir, "Makefile")); err == nil {
		t.Error("bootstrap must be deferred — no Makefile should be written")
	}

	// Non-empty project: isEmptyProject=false and flag stays off.
	populated := t.TempDir()
	if err := os.WriteFile(filepath.Join(populated, "main.go"), []byte("package main\n"), 0644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if isEmptyProject(populated) {
		t.Error("expected dir with main.go to not be empty")
	}
	a2 := &agent{sessions: map[string]*Session{}}
	a2.discoverRunners(populated)
	if a2.emptyProject {
		t.Error("expected emptyProject=false when source files are present")
	}
}
