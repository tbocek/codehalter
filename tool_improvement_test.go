package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func submitImprovementArgs(t *testing.T, m map[string]string) string {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}
	return string(b)
}

// TestApplyImprovement pins the code-side edit applied once the user clicks Apply:
// replace/remove swap the `original` text, add inserts after an anchor or appends,
// and bad inputs (missing original, path escape, unknown type) error without
// touching the file.
func TestApplyImprovement(t *testing.T) {
	dir := t.TempDir()
	ch := filepath.Join(dir, ".codehalter")
	if err := os.MkdirAll(ch, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(body string) { os.WriteFile(filepath.Join(ch, "PLAN.md"), []byte(body), 0o644) }
	read := func() string { b, _ := os.ReadFile(filepath.Join(ch, "PLAN.md")); return string(b) }

	write("alpha OLD omega")
	if err := applyImprovement(dir, improvementEntry{File: "PLAN.md", Type: "replace", Original: "OLD", New: "NEW"}); err != nil {
		t.Fatalf("replace: %v", err)
	}
	if read() != "alpha NEW omega" {
		t.Errorf("replace: got %q", read())
	}

	write("head\nanchor\ntail")
	if err := applyImprovement(dir, improvementEntry{File: "PLAN.md", Type: "add", Original: "anchor", New: "MORE"}); err != nil {
		t.Fatalf("add-anchor: %v", err)
	}
	if !strings.Contains(read(), "anchor\n\nMORE") {
		t.Errorf("add-anchor: got %q", read())
	}

	write("body")
	if err := applyImprovement(dir, improvementEntry{File: "PLAN.md", Type: "add", New: "APPENDED"}); err != nil {
		t.Fatalf("add-append: %v", err)
	}
	if !strings.HasSuffix(read(), "APPENDED\n") {
		t.Errorf("add-append: got %q", read())
	}

	write("keep DROP keep")
	if err := applyImprovement(dir, improvementEntry{File: "PLAN.md", Type: "remove", Original: " DROP"}); err != nil {
		t.Fatalf("remove: %v", err)
	}
	if read() != "keep keep" {
		t.Errorf("remove: got %q", read())
	}

	// Errors, none of which should write.
	write("nothing here")
	if err := applyImprovement(dir, improvementEntry{File: "PLAN.md", Type: "replace", Original: "MISSING", New: "X"}); err == nil {
		t.Error("missing original should error")
	}
	if err := applyImprovement(dir, improvementEntry{File: "../escape.md", Type: "add", New: "x"}); err == nil {
		t.Error("path escape should error")
	}
	if err := applyImprovement(dir, improvementEntry{File: "PLAN.md", Type: "frobnicate", New: "x"}); err == nil {
		t.Error("unknown type should error")
	}
	if read() != "nothing here" {
		t.Errorf("error cases must not write: got %q", read())
	}
}

// TestImprovementExecuteApplyAndSubmit drives the whole code-side flow in
// autopilot (Apply + Submit auto-answered yes): the structured change is applied
// to the .codehalter/ file AND the applied entry is POSTed with the license
// header, no auth token, and the model stamped on.
func TestImprovementExecuteApplyAndSubmit(t *testing.T) {
	var gotBody, gotLicense, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotLicense, gotAuth = r.Header.Get("X-License"), r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Model: "test-model"}}}
	a.mu.Lock()
	a.mode = "Autopilot" // auto-answer the Apply + Submit cards (no conn needed)
	a.mu.Unlock()

	ch := filepath.Join(s.Cwd, ".codehalter")
	os.MkdirAll(ch, 0o755)
	os.WriteFile(filepath.Join(ch, "PLAN.md"), []byte("alpha OLD omega"), 0o644)
	os.WriteFile(filepath.Join(s.Cwd, "LICENSE"), []byte("MIT License"), 0o644)

	var tc toolCall
	tc.Function.Name = submitImprovementToolName
	tc.Function.Arguments = submitImprovementArgs(t, map[string]string{
		"endpoint":     srv.URL,
		"improvements": `[{"title":"t","file":"PLAN.md","type":"replace","original":"OLD","new":"NEW","reasoning":"r"}]`,
	})
	out, failed := a.executeTool(context.Background(), s.ID, tc)
	if failed {
		t.Fatalf("failed=true: %s", out)
	}
	if b, _ := os.ReadFile(filepath.Join(ch, "PLAN.md")); string(b) != "alpha NEW omega" {
		t.Errorf("file not edited: %q", b)
	}
	if gotLicense != "MIT" {
		t.Errorf("X-License = %q, want MIT", gotLicense)
	}
	if gotAuth != "" {
		t.Errorf("endpoint is keyless — no Authorization expected, got %q", gotAuth)
	}
	var p improvementPayload
	if json.Unmarshal([]byte(gotBody), &p) != nil || len(p.Improvements) != 1 || p.Improvements[0].Model != "test-model" {
		t.Errorf("payload wrong (want 1 entry stamped test-model): %s", gotBody)
	}
	if !strings.Contains(out, "Applied 1") || !strings.Contains(out, "submitted") {
		t.Errorf("summary wrong: %s", out)
	}
}

// TestImprovementExecuteOnceOnly pins that the apply loop runs at most once per
// run: a second submit_improvement call (a duplicate in one batch, or a replan
// re-entry) no-ops instead of re-applying its side effects.
func TestImprovementExecuteOnceOnly(t *testing.T) {
	a, s := newTestAgent(t)
	a.mu.Lock()
	a.mode = "Autopilot"
	a.mu.Unlock()
	ch := filepath.Join(s.Cwd, ".codehalter")
	os.MkdirAll(ch, 0o755)
	os.WriteFile(filepath.Join(ch, "PLAN.md"), []byte("alpha OLD omega"), 0o644)
	// No LICENSE: applies locally, no POST.

	mk := func() toolCall {
		var tc toolCall
		tc.Function.Name = submitImprovementToolName
		tc.Function.Arguments = submitImprovementArgs(t, map[string]string{
			"improvements": `[{"title":"t","file":"PLAN.md","type":"replace","original":"OLD","new":"NEW","reasoning":"r"}]`,
		})
		return tc
	}

	out1, _ := a.executeTool(context.Background(), s.ID, mk())
	if !strings.Contains(out1, "Applied 1") {
		t.Fatalf("first call should apply: %s", out1)
	}
	if b, _ := os.ReadFile(filepath.Join(ch, "PLAN.md")); string(b) != "alpha NEW omega" {
		t.Fatalf("not applied: %q", b)
	}
	out2, _ := a.executeTool(context.Background(), s.ID, mk())
	if !strings.Contains(out2, "already handled") {
		t.Errorf("second call must no-op (already handled), got: %s", out2)
	}
}

// TestImprovementExecuteErrors covers the early validation branches (no ask, no
// conn): missing/empty/invalid improvements.
func TestImprovementExecuteErrors(t *testing.T) {
	a, s := newTestAgent(t)
	call := func(m map[string]string) string {
		var tc toolCall
		tc.Function.Name = submitImprovementToolName
		tc.Function.Arguments = submitImprovementArgs(t, m)
		out, _ := a.executeTool(context.Background(), s.ID, tc)
		return out
	}
	if out := call(map[string]string{"improvements": ""}); !strings.Contains(out, "improvements is required") {
		t.Errorf("empty string: %s", out)
	}
	if out := call(map[string]string{"improvements": "[]"}); !strings.Contains(out, "nothing to apply") {
		t.Errorf("empty array: %s", out)
	}
	if out := call(map[string]string{"improvements": "not json"}); !strings.Contains(out, "invalid improvements JSON") {
		t.Errorf("bad json: %s", out)
	}
}

// TestImprovementExecuteNoLicense: without a LICENSE the accepted edit is still
// applied locally, but the submission is refused (no POST), so a missing license
// can't block the user from improving their own prompts.
func TestImprovementExecuteNoLicense(t *testing.T) {
	a, s := newTestAgent(t)
	a.mu.Lock()
	a.mode = "Autopilot"
	a.mu.Unlock()
	ch := filepath.Join(s.Cwd, ".codehalter")
	os.MkdirAll(ch, 0o755)
	os.WriteFile(filepath.Join(ch, "PLAN.md"), []byte("alpha OLD omega"), 0o644)
	// No LICENSE file written.

	var tc toolCall
	tc.Function.Name = submitImprovementToolName
	tc.Function.Arguments = submitImprovementArgs(t, map[string]string{
		"endpoint":     "http://127.0.0.1:0", // must never be hit
		"improvements": `[{"title":"t","file":"PLAN.md","type":"replace","original":"OLD","new":"NEW","reasoning":"r"}]`,
	})
	out, failed := a.executeTool(context.Background(), s.ID, tc)
	if failed {
		t.Fatalf("failed=true: %s", out)
	}
	if b, _ := os.ReadFile(filepath.Join(ch, "PLAN.md")); string(b) != "alpha NEW omega" {
		t.Errorf("file should still be edited locally: %q", b)
	}
	if !strings.Contains(out, "no open-source license") {
		t.Errorf("expected no-license note, got: %s", out)
	}
}
