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
// improvements array in {"improvements":[...]}, POSTs it with a Bearer header,
// and reports success on a 2xx.
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
	// Write a LICENSE file so checkLicense passes.
	if err := os.WriteFile(filepath.Join(s.Cwd, "LICENSE"), []byte("MIT License"), 0644); err != nil {
		t.Fatalf("write LICENSE: %v", err)
	}
	improvements := `[{"title":"t","file":"PLAN.md","type":"remove","original":"x","new":"","reasoning":"r"}]`
	var tc toolCall
	tc.Function.Name = "submit_improvement"
	tc.Function.Arguments = submitImprovementArgs(t, map[string]string{
		"endpoint": srv.URL, "api_key": "SECRET", "improvements": improvements,
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
	if gotAuth != "Bearer SECRET" {
		t.Errorf("auth = %q, want %q", gotAuth, "Bearer SECRET")
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
}

// TestSubmitImprovementAnonymous pins that api_key is optional: with no key the
// submission still POSTs (the gate is the user's Yes + the license), and the
// Authorization header is omitted entirely rather than sent as a bare "Bearer ".
func TestSubmitImprovementAnonymous(t *testing.T) {
	var gotAuth string
	var hadAuth bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		_, hadAuth = r.Header["Authorization"]
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	a, s := newTestAgent(t)
	if err := os.WriteFile(filepath.Join(s.Cwd, "LICENSE"), []byte("MIT License"), 0644); err != nil {
		t.Fatalf("write LICENSE: %v", err)
	}
	var tc toolCall
	tc.Function.Name = "submit_improvement"
	tc.Function.Arguments = submitImprovementArgs(t, map[string]string{
		"endpoint":     srv.URL, // no api_key
		"improvements": `[{"title":"t","file":"PLAN.md","type":"remove","original":"x","new":"","reasoning":"r"}]`,
	})
	out, failed := a.executeTool(context.Background(), s.ID, tc)
	if failed || !strings.Contains(out, "Submitted 1 improvement") {
		t.Fatalf("anonymous submit should succeed, got: %s", out)
	}
	if hadAuth || gotAuth != "" {
		t.Errorf("no api_key should omit the Authorization header, got %q (present=%v)", gotAuth, hadAuth)
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
