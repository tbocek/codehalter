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

// TestSubmitImprovementPostsPayload verifies the happy path: the tool wraps the
// improvements array in {"improvements":[...]}, POSTs it with the license header
// and NO auth token (the endpoint is keyless), and reports success on a 2xx.
func TestSubmitImprovementPostsPayload(t *testing.T) {
	var gotMethod, gotAuth, gotCT, gotBody, gotLicense string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod, gotAuth, gotCT = r.Method, r.Header.Get("Authorization"), r.Header.Get("Content-Type")
		gotLicense = r.Header.Get("X-License")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Model: "test-model"}}} // stamped onto each entry
	// Write a LICENSE file so checkLicense passes.
	if err := os.WriteFile(filepath.Join(s.Cwd, "LICENSE"), []byte("MIT License"), 0644); err != nil {
		t.Fatalf("write LICENSE: %v", err)
	}
	improvements := `[{"title":"t","file":"PLAN.md","type":"remove","original":"x","new":"","reasoning":"r"}]`
	var tc toolCall
	tc.Function.Name = "submit_improvement"
	tc.Function.Arguments = submitImprovementArgs(t, map[string]string{
		"endpoint": srv.URL, "improvements": improvements,
	})

	out, failed := a.executeTool(context.Background(), s.ID, tc)
	if failed {
		t.Fatalf("happy path returned failed=true: %s", out)
	}
	if !strings.Contains(out, "Submitted 1 improvement") {
		t.Errorf("success message wrong: %s", out)
	}
	if gotMethod != http.MethodPost {
		t.Errorf("method = %s, want POST", gotMethod)
	}
	if gotAuth != "" {
		t.Errorf("endpoint is keyless — no Authorization header expected, got %q", gotAuth)
	}
	if gotCT != "application/json" {
		t.Errorf("content-type = %q", gotCT)
	}
	if gotLicense != "MIT" {
		t.Errorf("X-License = %q, want %q", gotLicense, "MIT")
	}
	var p improvementPayload
	if err := json.Unmarshal([]byte(gotBody), &p); err != nil || len(p.Improvements) != 1 || p.Improvements[0].File != "PLAN.md" {
		t.Errorf("body not the wrapped payload: %s", gotBody)
	}
	if p.Improvements[0].Model != "test-model" {
		t.Errorf("model not stamped onto the entry: got %q, want %q", p.Improvements[0].Model, "test-model")
	}
}

// TestSubmitImprovementErrors covers the validation and HTTP-error branches.
func TestSubmitImprovementErrors(t *testing.T) {
	a, s := newTestAgent(t)
	// Write a LICENSE file so checkLicense passes (errors below are not about license).
	if err := os.WriteFile(filepath.Join(s.Cwd, "LICENSE"), []byte("MIT License"), 0644); err != nil {
		t.Fatalf("write LICENSE: %v", err)
	}
	call := func(m map[string]string) (string, bool) {
		var tc toolCall
		tc.Function.Name = "submit_improvement"
		tc.Function.Arguments = submitImprovementArgs(t, m)
		return a.executeTool(context.Background(), s.ID, tc)
	}

	if out, _ := call(map[string]string{"api_key": "k", "improvements": "[]"}); !strings.Contains(out, "empty") {
		t.Errorf("empty improvements array should error: %s", out)
	}
	if out, _ := call(map[string]string{"api_key": "k", "improvements": "not json"}); !strings.Contains(out, "invalid improvements JSON") {
		t.Errorf("bad JSON should error: %s", out)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		io.WriteString(w, "boom")
	}))
	defer srv.Close()
	out, _ := call(map[string]string{
		"endpoint": srv.URL, "api_key": "k",
		"improvements": `[{"title":"t","file":"f","type":"remove"}]`,
	})
	if !strings.Contains(out, "HTTP 500") || !strings.Contains(out, "boom") {
		t.Errorf("HTTP error not surfaced: %s", out)
	}
}

// TestSubmitImprovementNoLicense rejects submissions from projects without an
// open-source license.
func TestSubmitImprovementNoLicense(t *testing.T) {
	a, s := newTestAgent(t)
	// Make sure there's no LICENSE file.
	var tc toolCall
	tc.Function.Name = "submit_improvement"
	tc.Function.Arguments = submitImprovementArgs(t, map[string]string{
		"endpoint": "http://example.com", "api_key": "k",
		"improvements": `[{"title":"t","file":"f","type":"remove"}]`,
	})
	out, _ := a.executeTool(context.Background(), s.ID, tc)
	if !strings.Contains(out, "no open-source license") {
		t.Errorf("expected license error, got: %s", out)
	}
}
