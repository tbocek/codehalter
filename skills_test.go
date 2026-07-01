package main

import (
	"strings"
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

// TestArgTokens: string values inside JSON args split on whitespace and shed
// shell punctuation, so paths surface from both structured fields and
// run_command strings; non-JSON input degrades to a raw split. Exercises the
// collectArgStrings → argTokens pair the way discloseSkills composes them.
func TestArgTokens(t *testing.T) {
	cases := []struct {
		args string
		want string // one token that must be present
	}{
		{`{"path":"cmd/main.go"}`, "cmd/main.go"},
		{`{"command":"cat cmd/main.go && ls"}`, "cmd/main.go"},
		{`{"command":"grep -r foo 'src/app.ts';"}`, "src/app.ts"},
		{`{"nested":{"files":["a/b.c","d.h"]}}`, "a/b.c"},
		{`not json at all Makefile here`, "Makefile"},
	}
	for _, c := range cases {
		toks := argTokens(collectArgStrings(c.args))
		found := false
		for _, tok := range toks {
			if tok == c.want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("argTokens(collectArgStrings(%q)) = %v, want it to contain %q", c.args, toks, c.want)
		}
	}
}

// TestDeferredSkillTriggers walks the trigger table: each stack's touch signal
// (path token or runner tool name) must map to exactly the expected skill.
func TestDeferredSkillTriggers(t *testing.T) {
	byName := map[string]deferredSkill{}
	for _, d := range deferredSkills {
		byName[d.name] = d
	}
	tokenCases := []struct {
		tok  string
		want string
	}{
		{"cmd/main.go", "SKILL-go.md"},
		{"src/app.ts", "SKILL-ts.md"},
		{"web/index.jsx", "SKILL-js.md"},
		{"lib/foo.cpp", "SKILL-c.md"},
		{"include/foo.h", "SKILL-c.md"},
		{"src/Main.java", "SKILL-java.md"},
		{"justfile", "SKILL-justfile.md"},
		{"sub/dir/Makefile", "SKILL-makefile.md"},
		{"rules.mk", "SKILL-makefile.md"},
		{"install.sh", "SKILL-bash.md"},
		{"scripts/run.bash", "SKILL-bash.md"},
	}
	for _, c := range tokenCases {
		var hits []string
		for _, d := range deferredSkills {
			if d.match != nil && d.match(c.tok) {
				hits = append(hits, d.name)
			}
		}
		if len(hits) != 1 || hits[0] != c.want {
			t.Errorf("token %q matched %v, want exactly [%s]", c.tok, hits, c.want)
		}
	}
	// Runner tool names imply their stack without any path token.
	for tool, want := range map[string]string{
		"go": "SKILL-go.md", "just": "SKILL-justfile.md",
		"make": "SKILL-makefile.md", "npm": "SKILL-js.md", "gradle": "SKILL-java.md",
	} {
		d, ok := byName[want]
		if !ok {
			t.Fatalf("skill %s missing from deferredSkills", want)
		}
		found := false
		for _, tn := range d.tools {
			if tn == tool {
				found = true
			}
		}
		if !found {
			t.Errorf("tool %q should trigger %s (tools: %v)", tool, want, d.tools)
		}
	}
	// Never-deferred skills must stay off the table: they have no touch signal.
	for _, name := range []string{"SKILL-base.md", "SKILL-arch.md", "SKILL-rust.md"} {
		if isDeferredSkill(name) {
			t.Errorf("%s must not be deferred", name)
		}
	}
}

// TestShebangShell pins the extensionless-script signal: a shell shebang in a
// whole argument string (a write_file content, say) triggers the bash skill,
// the same classification `file`(1) would make, while other interpreters and
// mid-word "sh" don't.
func TestShebangShell(t *testing.T) {
	hits := []string{
		"#!/usr/bin/env bash\nset -euo pipefail\n",
		"#!/bin/sh\necho hi\n",
		"#!/bin/dash\n",
		"#!/usr/bin/zsh\n",
		"leading text\n#!/bin/bash\n", // shebang not at byte 0 still counts
	}
	misses := []string{
		"#!/usr/bin/env python3\nprint()\n",
		"#!/usr/bin/fish\n", // "sh" mid-word must not match
		"echo no shebang here",
		"",
	}
	for _, s := range hits {
		if !shebangShell.MatchString(s) {
			t.Errorf("shebangShell should match %q", s)
		}
	}
	for _, s := range misses {
		if shebangShell.MatchString(s) {
			t.Errorf("shebangShell should NOT match %q", s)
		}
	}
	// End-to-end through the raw extractor: an extensionless script write.
	raws := collectArgStrings(`{"path":"install","content":"#!/usr/bin/env bash\nset -euo pipefail\n"}`)
	found := false
	for _, r := range raws {
		if shebangShell.MatchString(r) {
			found = true
		}
	}
	if !found {
		t.Error("shebang inside write_file content should be caught via collectArgStrings")
	}
}

// TestLoadSkillsSkip: the skip filter drops exactly the named files while the
// rest keep loading — the mechanism skills="auto" uses to withhold untouched
// language skills from the system prompt.
func TestLoadSkillsSkip(t *testing.T) {
	cwd := t.TempDir()
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"SKILL-base.md": "base body",
		"SKILL-go.md":   "go body",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	all := loadSkills(cwd, nil)
	if !strings.Contains(all, "base body") || !strings.Contains(all, "go body") {
		t.Fatalf("nil skip should load everything, got %q", all)
	}
	filtered := loadSkills(cwd, func(name string) bool { return name == "SKILL-go.md" })
	if strings.Contains(filtered, "go body") {
		t.Errorf("SKILL-go.md should have been skipped, got %q", filtered)
	}
	if !strings.Contains(filtered, "base body") {
		t.Errorf("SKILL-base.md should still load, got %q", filtered)
	}
}
