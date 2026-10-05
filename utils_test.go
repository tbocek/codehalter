package main

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestTrimJSON(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{name: "plain", in: `{"ok":true}`, want: `{"ok":true}`},
		{name: "leading whitespace", in: "  \n{\"ok\":true}\n  ", want: `{"ok":true}`},
		{name: "json fence", in: "```json\n{\"ok\":true}\n```", want: `{"ok":true}`},
		{name: "bare fence", in: "```\n{\"ok\":true}\n```", want: `{"ok":true}`},
		{name: "prose prefix", in: "Sure, here's the JSON:\n{\"ok\":true}", want: `{"ok":true}`},
		{name: "prose suffix", in: "{\"ok\":true}\nLet me know if you need more.", want: `{"ok":true}`},
		{name: "prose both sides", in: "Here you go: {\"ok\":true}, that's it!", want: `{"ok":true}`},
		{name: "nested", in: "noise {\"a\":{\"b\":1}} noise", want: `{"a":{"b":1}}`},
		{name: "brace in string", in: `{"s":"} not the end"}`, want: `{"s":"} not the end"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := trimJSON(tc.in); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// TestClipUTF8 also covers tailUTF8.
func TestClipUTF8(t *testing.T) {
	s := "hé llo" // é is 2 bytes (0xC3 0xA9): bytes are h, 0xC3, 0xA9, ' ', l, l, o
	if got := clipUTF8(s, 2); got != "h" {
		t.Errorf("clipUTF8(%q, 2) = %q, want %q (must not split é)", s, got, "h")
	}
	if got := clipUTF8(s, 3); got != "hé" {
		t.Errorf("clipUTF8(%q, 3) = %q, want %q", s, got, "hé")
	}
	if got := clipUTF8(s, 100); got != s {
		t.Errorf("clipUTF8 past the end should return the whole string, got %q", got)
	}
	for n := 0; n <= len(s); n++ {
		if !utf8.ValidString(clipUTF8(s, n)) {
			t.Errorf("clipUTF8(%q, %d) is not valid UTF-8: %q", s, n, clipUTF8(s, n))
		}
		if !utf8.ValidString(tailUTF8(s, n)) {
			t.Errorf("tailUTF8(%q, %d) is not valid UTF-8: %q", s, n, tailUTF8(s, n))
		}
	}
}

// Replaced whole, with its mode, and no temp file left beside it either way.
func TestWriteFileAtomic(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "session_x.toml")
	if err := os.WriteFile(path, []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := writeFileAtomic(path, []byte("new"), 0o644); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(path); string(b) != "new" {
		t.Errorf("content = %q", b)
	}
	if st, _ := os.Stat(path); st.Mode().Perm() != 0o644 {
		t.Errorf("mode = %v", st.Mode().Perm())
	}
	if err := writeFileAtomic(filepath.Join(dir, "gone", "x.toml"), []byte("x"), 0o644); err == nil {
		t.Error("a write into a missing directory succeeded")
	}
	if entries, _ := os.ReadDir(dir); len(entries) != 1 {
		t.Errorf("left behind: %v", entries)
	}
}

// A long trace becomes one line and the message above it stays; short traces and
// ordinary output, `grep -n` lines included, are left alone.
func TestCollapseStackTraces(t *testing.T) {
	var rust strings.Builder
	rust.WriteString("---- sec_07 stdout ----\nthread 'sec_07' panicked at tests/narrate_surface_widgets.rs:451:5:\nassertion `left == right` failed: with a voice and words the sample is being synthesized\nstack backtrace:\n")
	for i := range 28 {
		fmt.Fprintf(&rust, "  %2d:     0x563482bd51aa - std::sync::once::Once::call_once::h%d\n                               at /rustc/31fca3/library/std/src/sync/once.rs:166:41\n", i, i)
	}
	rust.WriteString("failures:\n    sec_07\n\ntest result: FAILED. 3 passed; 1 failed\n")
	python := "Traceback (most recent call last):\n" + strings.Repeat("  File \"/app/x.py\", line 12, in f\n    return g(x)\n", 7) + "AssertionError: the total is 4, want 3\n"
	golang := "panic: index out of range [5] with length 3\n\ngoroutine 1 [running]:\n" + strings.Repeat("main.walk(...)\n\t/app/main.go:21 +0x1d\n", 7) + "exit status 2\n"
	js := "Error: expected 200, got 500\n" + strings.Repeat("    at Server.handle (/app/server.js:40:11)\n", 8) + "1 failing\n"
	for name, tc := range map[string]struct{ in, keep, gone, tail string }{
		"rust":       {rust.String(), "with a voice and words the sample is being synthesized", "0x563482bd51aa", "test result: FAILED"},
		"python":     {python, "Traceback", `File "/app/x.py"`, "AssertionError: the total is 4, want 3"},
		"go":         {golang, "panic: index out of range", "/app/main.go:21", "exit status 2"},
		"javascript": {js, "Error: expected 200, got 500", "Server.handle", "1 failing"},
	} {
		got := collapseStackTraces(tc.in)
		if !strings.Contains(got, tc.keep) || !strings.Contains(got, tc.tail) || strings.Contains(got, tc.gone) || !strings.Contains(got, "stack-trace lines left out") {
			t.Errorf("%s:\n%s", name, got)
		}
	}
	plain := "12: fn walk() {\n13:     for x in xs {\n14:     }\n  3: a short trace frame\n      at src/a.rs:3\nok\n"
	if got := collapseStackTraces(plain); got != plain {
		t.Errorf("ordinary output changed:\n%s", got)
	}
}

// Lowercase, punctuation-stripped, order-independent, so rewordings match.
func TestIssueBagTokenisation(t *testing.T) {
	a := issueBag([]string{"Missing import!", "Syntax error."})
	b := issueBag([]string{"syntax  ERROR", "missing\timport"})
	if !slices.Equal(sortedKeys(a), sortedKeys(b)) {
		t.Errorf("expected equivalent bags, got %v vs %v", sortedKeys(a), sortedKeys(b))
	}

	// No empty tokens from adjacent separators.
	bag := issueBag([]string{"foo--bar...baz"})
	want := []string{"bar", "baz", "foo"}
	if !slices.Equal(sortedKeys(bag), want) {
		t.Errorf("got %v, want %v", sortedKeys(bag), want)
	}
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

// A reworded near-duplicate scores above the threshold, unrelated failures below.
func TestJaccardSimilarity(t *testing.T) {
	// Two empty bags are treated as identical (degenerate but well-defined).
	if got := jaccard(map[string]bool{}, map[string]bool{}); got != 1 {
		t.Errorf("empty/empty: got %v, want 1", got)
	}

	// |∩|=2, |∪|=3: 0.67, above the threshold.
	a := issueBag([]string{"missing import"})
	b := issueBag([]string{"import is missing"})
	if s := jaccard(a, b); s < failureSimilarityThreshold {
		t.Errorf("reworded duplicate: got %v, want >= %v", s, failureSimilarityThreshold)
	}

	c := issueBag([]string{"missing import in foo.go"})
	d := issueBag([]string{"unused variable x"})
	if s := jaccard(c, d); s >= failureSimilarityThreshold {
		t.Errorf("disjoint issues: got %v, want < %v", s, failureSimilarityThreshold)
	}

	if jaccard(a, b) != jaccard(b, a) {
		t.Errorf("expected jaccard to be symmetric")
	}
}
