package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestConfirmPlanFixAutoExecSkipsGate pins that a user-accepted fix dispatch
// (fixAutoExec set) skips confirmPlan's "Execute?" gate. Every fix card —
// missing-tools/npm, lsmcp, mcp errors — routes through proposeFix, which sets
// the flag, so this one check covers them all. With a nil conn, reaching the
// gate would call conn.AskChoice and panic; returning nil proves it was skipped.
func TestConfirmPlanFixAutoExecSkipsGate(t *testing.T) {
	a, s := newTestAgent(t)
	plan := &planResult{Subtasks: []subtask{{Description: "install npm + prettier"}}}

	s.fixAutoExec = true
	if err := a.confirmPlan(context.Background(), s.ID, plan, false); err != nil {
		t.Errorf("fixAutoExec set: confirmPlan should skip the gate and return nil, got %v", err)
	}
}

// TestParseLineRange covers the line-range encodings Zed may put in a
// resource_link fragment.
func TestParseLineRange(t *testing.T) {
	cases := []struct {
		frag             string
		wantStart, wantE int
	}{
		{"L810-845", 810, 845},
		{"810:845", 810, 845},
		{"810-845", 810, 845},
		{"L810", 810, 810},
		{"", 0, 0},
		{"nodigits", 0, 0},
	}
	for _, c := range cases {
		if s, e := parseLineRange(c.frag); s != c.wantStart || e != c.wantE {
			t.Errorf("parseLineRange(%q) = %d,%d, want %d,%d", c.frag, s, e, c.wantStart, c.wantE)
		}
	}
}

// TestReadLinkedResource verifies a resource_link is inlined with its line
// range, full-file when no range, and refused outside the workspace.
func TestReadLinkedResource(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.go")
	if err := os.WriteFile(path, []byte("a\nb\nc\nd\ne\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Line range 2-4 → "b\nc\nd".
	if snip, label, ok := readLinkedResource(dir, "file://"+path+"#L2-4"); !ok || snip != "b\nc\nd" || label != "f.go:2-4" {
		t.Errorf("range read = %q,%q,%v", snip, label, ok)
	}
	// No range → whole file, label is basename.
	if snip, label, ok := readLinkedResource(dir, "file://"+path); !ok || snip != "a\nb\nc\nd\ne\n" || label != "f.go" {
		t.Errorf("full read = %q,%q,%v", snip, label, ok)
	}
	// Outside cwd → refused.
	if _, _, ok := readLinkedResource(dir, "file:///etc/passwd"); ok {
		t.Errorf("expected refusal for path outside workspace")
	}
	// Missing file → refused.
	if _, _, ok := readLinkedResource(dir, "file://"+filepath.Join(dir, "nope.go")); ok {
		t.Errorf("expected refusal for missing file")
	}
}

// TestResourcePath pins the URI → read_file path mapping used when an embedded
// "resource" / "resource_link" block is folded into the prompt: file:// URIs
// collapse to their percent-decoded path with the fragment stripped, non-file
// URIs pass through verbatim.
func TestResourcePath(t *testing.T) {
	cases := map[string]string{
		"file:///workspaces/codehalter/llm.go":          "/workspaces/codehalter/llm.go",
		"file:///workspaces/codehalter/llm.go#L801-836": "/workspaces/codehalter/llm.go",
		"file:///a%20b/c.go":                            "/a b/c.go",
		"/plain/path.go":                                "/plain/path.go",
		"https://example.com/x":                         "https://example.com/x",
		"":                                              "",
	}
	for uri, want := range cases {
		if got := resourcePath(uri); got != want {
			t.Errorf("resourcePath(%q) = %q, want %q", uri, got, want)
		}
	}
}

// TestResourceLabel checks the human-readable header: the path plus the URI
// fragment (an editor line range) in parentheses when present.
func TestResourceLabel(t *testing.T) {
	cases := map[string]string{
		"file:///workspaces/codehalter/llm.go":          "/workspaces/codehalter/llm.go",
		"file:///workspaces/codehalter/llm.go#L801-836": "/workspaces/codehalter/llm.go (L801-836)",
		"": "attachment",
	}
	for uri, want := range cases {
		if got := resourceLabel(uri); got != want {
			t.Errorf("resourceLabel(%q) = %q, want %q", uri, got, want)
		}
	}
}
