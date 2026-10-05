package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestRenderMacro(t *testing.T) {
	if got, msg := renderMacro("grill", "do {{}} now", "the thing"); got != "do the thing now" || msg != "" {
		t.Errorf("substitute: got %q msg %q", got, msg)
	}
	if got, msg := renderMacro("grill", "do {{}} now", "   "); got != "" || msg == "" {
		t.Errorf("missing-arg: got %q msg %q (want empty render + a message)", got, msg)
	}
	if got, _ := renderMacro("grill", "fixed body", "extra"); got != "fixed body\n\nextra" {
		t.Errorf("append: got %q", got)
	}
	if got, _ := renderMacro("grill", "fixed body", ""); got != "fixed body" {
		t.Errorf("as-is: got %q", got)
	}
}

// A template with {{}} stops without args and takes them in place otherwise.
func TestExpandMacro(t *testing.T) {
	a, sess := newTestAgent(t)
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, sessionDir), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, sessionDir, "TEMPLATE-ask.md"), []byte("do {{}} now"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in       string
		handled  bool
		rendered string
		stops    bool
	}{
		{in: "hello world"},
		{in: "/nope-not-a-template here"},
		{in: ""},
		{in: "no slash here"},
		{in: "/ask", handled: true, stops: true},
		{in: "/ask the thing", handled: true, rendered: "do the thing now"},
	} {
		rendered, stopMsg, handled := a.expandMacro(context.Background(), sess.ID, dir, c.in)
		if handled != c.handled || rendered != c.rendered || (stopMsg != "") != c.stops {
			t.Errorf("expandMacro(%q) = %q, %q, %v; want %q, stop=%v, %v", c.in, rendered, stopMsg, handled, c.rendered, c.stops, c.handled)
		}
	}
}

// Only session files go; everything else in .codehalter stays.
func TestHandleClean(t *testing.T) {
	for _, c := range []struct {
		name    string
		files   []string
		wantMsg string
	}{
		{"session files", []string{"session_20260614.log", "session_20260614.toml", "session_20260615.log", "PLAN.md"}, "Cleaned 3"},
		{"none to clean", []string{"PLAN.md"}, "No session"},
	} {
		dir := t.TempDir()
		ch := filepath.Join(dir, sessionDir)
		for _, f := range c.files {
			writeFiles(t, ch, f)
		}
		if msg := handleClean(dir); !strings.Contains(msg, c.wantMsg) {
			t.Errorf("%s: handleClean = %q, want it to say %q", c.name, msg, c.wantMsg)
		}
		entries, err := os.ReadDir(ch)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 1 || entries[0].Name() != "PLAN.md" {
			t.Errorf("%s: left %v, want only PLAN.md", c.name, entries)
		}
	}
}

// A same-name file in .codehalter replaces a shipped macro; a new name adds a command.
func TestTemplatesComeFromTheBinary(t *testing.T) {
	dir := t.TempDir()
	ch := filepath.Join(dir, ".codehalter")
	if err := os.MkdirAll(ch, 0o755); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(templateNames(dir), "commit") {
		t.Fatalf("shipped macros missing from the menu: %v", templateNames(dir))
	}
	if entries, _ := os.ReadDir(ch); len(entries) != 0 {
		t.Errorf("listing the macros wrote to the project: %v", entries)
	}
	shipped, _ := loadTemplate(dir, "commit")

	if err := os.WriteFile(filepath.Join(ch, "TEMPLATE-commit.md"), []byte("mine"), 0o644); err != nil {
		t.Fatal(err)
	}
	if body, _ := loadTemplate(dir, "commit"); body != "mine" {
		t.Errorf("the override did not win: %q", body)
	}
	if err := os.Remove(filepath.Join(ch, "TEMPLATE-commit.md")); err != nil {
		t.Fatal(err)
	}
	if body, _ := loadTemplate(dir, "commit"); body != shipped {
		t.Error("removing the override did not go back to the shipped macro")
	}

	if err := os.WriteFile(filepath.Join(ch, "TEMPLATE-standup.md"), []byte("ours"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(templateNames(dir), "standup") {
		t.Errorf("a project's own macro is not in the menu: %v", templateNames(dir))
	}
}

// With no [[llm]] configured the report must still carry the models half: the "add one" warning.
func TestExpandMacroSettingsIsCodeLevel(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // no global settings.toml to find
	a, sess := newTestAgent(t)
	rendered, stopMsg, handled := a.expandMacro(context.Background(), sess.ID, sess.Cwd, "/settings")
	if !handled || rendered != "" || stopMsg == "" {
		t.Fatalf("/settings: handled=%v rendered=%q stopMsg=%q (want handled, nothing to run, a report)", handled, rendered, stopMsg)
	}
	if !strings.Contains(stopMsg, "no [[llm]] in settings.toml") {
		t.Errorf("/settings report does not cover the models: %q", stopMsg)
	}
}

// The menu shows a template's first line, a heading without its marks.
func TestAvailableCommandsDescribeThemselves(t *testing.T) {
	if got := templateSummary("commit", "Commit my changes (do NOT push).\nMore text.\n"); got != "Commit my changes (do NOT push)." {
		t.Errorf("summary should be the first line, got %q", got)
	}
	if got := templateSummary("commit", "# Commit helper\n\nbody\n"); got != "Commit helper" {
		t.Errorf("a heading should lose its marks and serve as the description, got %q", got)
	}
	if got := templateSummary("empty", "#\n\n"); got != "Run the empty prompt template" {
		t.Errorf("an unusable body should fall back, got %q", got)
	}
	if got := templateSummary("args", "{{}}\nDo the thing.\n"); got != "Do the thing." {
		t.Errorf("the placeholder is not a description, got %q", got)
	}
}
