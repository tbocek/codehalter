package main

import (
	"os"
	"path/filepath"
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
	for _, s := range []string{"hello world", "/nope-not-a-template here", "", "no slash here"} {
		if _, _, handled := expandMacro(dir, s); handled {
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
	if _, stopMsg, handled := expandMacro(dir, "/grill-me"); !handled || stopMsg == "" {
		t.Errorf("/grill-me with no args: handled=%v stopMsg=%q (want handled + a stop message)", handled, stopMsg)
	}
	rendered, stopMsg, handled := expandMacro(dir, "/grill-me the auth design")
	if !handled || stopMsg != "" || !strings.Contains(rendered, "the auth design") {
		t.Errorf("/grill-me <args>: handled=%v stopMsg=%q rendered=%q", handled, stopMsg, rendered)
	}
}
