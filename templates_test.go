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
	// {{}} + args → substituted.
	if got, msg := renderMacro("grill", "do {{}} now", "the thing"); got != "do the thing now" || msg != "" {
		t.Errorf("substitute: got %q msg %q", got, msg)
	}
	// {{}} + no args → user-facing stop message, nothing rendered.
	if got, msg := renderMacro("grill", "do {{}} now", "   "); got != "" || msg == "" {
		t.Errorf("missing-arg: got %q msg %q (want empty render + a message)", got, msg)
	}
	// no {{}} + args → appended.
	if got, _ := renderMacro("grill", "fixed body", "extra"); got != "fixed body\n\nextra" {
		t.Errorf("append: got %q", got)
	}
	// no {{}} + no args → as-is.
	if got, _ := renderMacro("grill", "fixed body", ""); got != "fixed body" {
		t.Errorf("as-is: got %q", got)
	}
}

func TestExpandMacroNonCommand(t *testing.T) {
	dir := t.TempDir() // no on-disk templates → embed-only lookup
	a, sess := newTestAgent(t)
	for _, s := range []string{"hello world", "/nope-not-a-template here", "", "no slash here"} {
		if _, _, handled := a.expandMacro(context.Background(), sess.ID, dir, s); handled {
			t.Errorf("expandMacro(%q) handled=true, want false", s)
		}
	}
}

func TestTemplateNamesIncludesGrillMe(t *testing.T) {
	dir := t.TempDir()
	if !contains(templateNames(dir), "grill-me") {
		t.Errorf("templateNames() = %v, want it to include grill-me (res/TEMPLATE-grill-me.md)", templateNames(dir))
	}
}

func TestHandleClean(t *testing.T) {
	dir := t.TempDir()
	ch := filepath.Join(dir, ".codehalter")
	os.MkdirAll(ch, 0o755)
	// Create some session files.
	for _, f := range []string{"session_20260614.log", "session_20260614.toml", "session_20260615.log"} {
		os.WriteFile(filepath.Join(ch, f), []byte("test"), 0o644)
	}
	// Create a non-session file that should not be deleted.
	os.WriteFile(filepath.Join(ch, "PLAN.md"), []byte("keep"), 0o644)

	msg, handled := handleClean(dir)
	if !handled {
		t.Fatal("handleClean returned handled=false")
	}
	if !strings.Contains(msg, "Cleaned 3") {
		t.Errorf("handleClean: got %q, want message mentioning 3 files", msg)
	}
	// Verify session files are gone.
	entries, _ := os.ReadDir(ch)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "session_") {
			t.Errorf("session file still present: %s", e.Name())
		}
	}
	// PLAN.md should still exist.
	if _, err := os.Stat(filepath.Join(ch, "PLAN.md")); os.IsNotExist(err) {
		t.Error("PLAN.md was incorrectly deleted")
	}
}

func TestHandleCleanNoFiles(t *testing.T) {
	dir := t.TempDir()
	ch := filepath.Join(dir, ".codehalter")
	os.MkdirAll(ch, 0o755)
	msg, handled := handleClean(dir)
	if !handled {
		t.Fatal("handleClean returned handled=false")
	}
	if !strings.Contains(msg, "No session") {
		t.Errorf("handleClean: got %q, want message about no session files", msg)
	}
}

// expandMacro on the real /grill-me template: it carries a {{}}, so no args
// must stop with a message and args must land in the rendered prompt. Embed
// fallback (empty temp cwd) so the shipped default is what's exercised.
func TestExpandMacroGrillMe(t *testing.T) {
	dir := t.TempDir()
	a, sess := newTestAgent(t)
	if _, stopMsg, handled := a.expandMacro(context.Background(), sess.ID, dir, "/grill-me"); !handled || stopMsg == "" {
		t.Errorf("/grill-me with no args: handled=%v stopMsg=%q (want handled + a stop message)", handled, stopMsg)
	}
	rendered, stopMsg, handled := a.expandMacro(context.Background(), sess.ID, dir, "/grill-me the auth design")
	if !handled || stopMsg != "" || !strings.Contains(rendered, "the auth design") {
		t.Errorf("/grill-me <args>: handled=%v stopMsg=%q rendered=%q", handled, stopMsg, rendered)
	}
}

// expandMacro on the real /commit template: it carries no placeholder, which
// means a bare /commit RUNS (renders, no stop message) rather than being
// rejected for a missing arg. Embed fallback (empty temp cwd) exercises the
// shipped default.
func TestExpandMacroCommitRunsBare(t *testing.T) {
	dir := t.TempDir()
	a, sess := newTestAgent(t)
	rendered, stopMsg, handled := a.expandMacro(context.Background(), sess.ID, dir, "/commit")
	if !handled || stopMsg != "" || rendered == "" {
		t.Fatalf("bare /commit should run: handled=%v stopMsg=%q renderedEmpty=%v", handled, stopMsg, rendered == "")
	}
}

// TestTemplatesComeFromTheBinary: the shipped macros are in the slash menu with
// nothing on disk, a file of the same name replaces one, and a file of a new
// name adds a command. Retiring a macro is deleting its res/ file, because
// nothing was ever copied into the project to outlive it.
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

// TestExpandMacroSettingsIsCodeLevel: /settings must be handled without any
// TEMPLATE-settings.md on disk (it is code, not a template) and must come back
// with the LLM half of the report. With no [[llm]] configured that is the
// "add one" warning, which is exactly the state a user runs /settings in.
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

// TestAvailableCommandsDescribeThemselves pins what the editor's slash menu
// shows: a template's own first line (a heading counts, without its marks)
// rather than a generic label.
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
