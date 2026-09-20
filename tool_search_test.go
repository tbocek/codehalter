package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// TestFormatMatchBlock pins the search-result rendering: file:line header, the
// match line marked ">", context clamped at the file bounds, and a bare
// file:line fallback when there were no lines to render context from.
func TestFormatMatchBlock(t *testing.T) {
	lines := []string{"a", "b", "c", "d", "e"}

	got := formatMatchBlock("f.go", lines, 3, 2)
	if want := "f.go:3\n  1| a\n  2| b\n> 3| c\n  4| d\n  5| e\n"; got != want {
		t.Errorf("middle match:\ngot  %q\nwant %q", got, want)
	}

	got = formatMatchBlock("f.go", lines, 1, 2)
	if want := "f.go:1\n> 1| a\n  2| b\n  3| c\n"; got != want {
		t.Errorf("top match (context clamped):\ngot  %q\nwant %q", got, want)
	}

	if got := formatMatchBlock("f.go", nil, 7, 2); got != "f.go:7" {
		t.Errorf("no-context fallback: got %q, want %q", got, "f.go:7")
	}
}

// TestSearchBuckets pins the cap-hit hint: searchBucket collapses a path to its
// first two segments (root files → "(root)"), and topBuckets renders the
// busiest buckets highest-first with name-tie-breaking so the model is pointed
// at where matches cluster (e.g. a gitignored bench/results log tree).
func TestSearchBuckets(t *testing.T) {
	cases := map[string]string{
		"bench/results/2026-05-16/x.log": "bench/results",
		"improve/store.go":               "improve",
		"main.go":                        "(root)",
	}
	for in, want := range cases {
		if got := searchBucket(in); got != want {
			t.Errorf("searchBucket(%q) = %q, want %q", in, got, want)
		}
	}

	// Highest first, capped at n entries (cmd/build dropped).
	counts := map[string]int{"bench/results": 63, "improve": 12, "cmd/build": 5}
	if got := topBuckets(counts, 2); got != "bench/results (63) · improve (12)" {
		t.Errorf("topBuckets top-2 = %q", got)
	}
	// Equal counts break by name ("alpha" before "zeta").
	if got := topBuckets(map[string]int{"zeta": 5, "alpha": 5}, 5); got != "alpha (5) · zeta (5)" {
		t.Errorf("topBuckets tie = %q", got)
	}
	if got := topBuckets(map[string]int{}, 5); got != "" {
		t.Errorf("topBuckets(empty) = %q, want empty", got)
	}
}

// TestSearchInFileMatchers verifies the matcher callback cleanly swaps
// literal substring and regex behavior. Literal mode treats '.' as a dot;
// regex mode treats it as any-char.

// TestSearchSkipsBinary pins the fix for the "plan not valid JSON" stall: a
// search must NOT scan a binary file (NUL bytes) — returning its bytes as match
// blocks poisons the LLM context. Covers both the line scanner (searchInFile)
// and the multiline path (searchInFileMultiline), and confirms a text file with
// the same token still matches.
func TestSearchSkipsBinary(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "blob.zip")
	txt := filepath.Join(dir, "code.txt")
	// "name" appears in both, but the zip has a NUL byte → must be skipped.
	if err := os.WriteFile(bin, []byte("PK\x03\x04\x00name\x00\x00duckduckgo"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(txt, []byte("the name is here\n"), 0644); err != nil {
		t.Fatal(err)
	}

	lit := func(s string) bool { return strings.Contains(s, "name") }
	if m := searchInFile(bin, lit, 10); m != nil {
		t.Errorf("searchInFile scanned a binary file: got matches %v", m)
	}
	if m := searchInFile(txt, lit, 10); len(m) != 1 {
		t.Errorf("searchInFile missed the text match: got %v, want 1", m)
	}
	re := regexp.MustCompile("name")
	if m := searchInFileMultiline(bin, re, 10); m != nil {
		t.Errorf("searchInFileMultiline scanned a binary file: got %v", m)
	}
	if m := searchInFileMultiline(txt, re, 10); len(m) != 1 {
		t.Errorf("searchInFileMultiline missed the text match: got %v, want 1", m)
	}
}

func TestSearchInFileMatchers(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f.txt")
	body := "alpha\nbeta.go\nbeta_go\ngamma\n"
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}

	// Literal: "beta.go" matches only line 2 (the actual dot).
	lit := searchInFile(path, func(s string) bool { return strings.Contains(s, "beta.go") }, 10)
	if !slices.Equal(lit, []int{2}) {
		t.Errorf("literal: got %v, want [2]", lit)
	}

	// Regex: "beta.go" matches both lines 2 and 3 (. = any char).
	re := regexp.MustCompile("beta.go")
	rx := searchInFile(path, re.MatchString, 10)
	if !slices.Equal(rx, []int{2, 3}) {
		t.Errorf("regex: got %v, want [2 3]", rx)
	}

	// Case-insensitive via inline flag.
	reI := regexp.MustCompile("(?i)ALPHA")
	ci := searchInFile(path, reI.MatchString, 10)
	if !slices.Equal(ci, []int{1}) {
		t.Errorf("case-insensitive: got %v, want [1]", ci)
	}

	// Limit is honoured.
	anyLine := regexp.MustCompile(".+")
	limited := searchInFile(path, anyLine.MatchString, 2)
	if len(limited) != 2 {
		t.Errorf("limit=2: got %d matches, want 2", len(limited))
	}
}

// TestSearchTextDedupOnRepeat pins the search_text dedup: re-running an identical
// search this turn still returns the hits but leads with the unchanged marker the
// tool loop scans for, while a search with a different scope is NOT flagged.
func TestSearchTextDedupOnRepeat(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()
	if err := os.WriteFile(filepath.Join(s.Cwd, "f.go"), []byte("package main\n// needle here\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	var tc toolCall
	tc.Function.Name = "search_text"
	tc.Function.Arguments = `{"query":"needle"}`

	if first, _ := a.executeTool(ctx, s.ID, tc); strings.Contains(first, readUnchangedMarker) {
		t.Fatalf("first search must not be flagged:\n%s", first)
	}
	if second, _ := a.executeTool(ctx, s.ID, tc); !strings.Contains(second, readUnchangedMarker) {
		t.Errorf("a repeated identical search should carry the unchanged marker:\n%s", second)
	}

	// Same query, different path → different dedup key → NOT a repeat.
	var scoped toolCall
	scoped.Function.Name = "search_text"
	scoped.Function.Arguments = `{"query":"needle","path":"."}`
	if out, _ := a.executeTool(ctx, s.ID, scoped); strings.Contains(out, readUnchangedMarker) {
		t.Errorf("a search with a different path must not be flagged as a repeat:\n%s", out)
	}
}

// TestSearchContextAndFilePath pins the two things the partial-read note sends
// the model here for: `context` widens the lines shown around a hit (capped),
// and `path` may name a single file, so "just the part of this big file I
// need" is one call.
func TestSearchContextAndFilePath(t *testing.T) {
	a, s := newTestAgent(t)
	var body strings.Builder
	for i := 1; i <= 40; i++ {
		fmt.Fprintf(&body, "line %d\n", i)
	}
	if err := os.WriteFile(filepath.Join(s.Cwd, "big.txt"), []byte(body.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(s.Cwd, "other.txt"), []byte("line 20\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	search := func(args string) string {
		out, _ := searchTextTool.Execute(context.Background(), a, s.ID, args)
		return out
	}

	narrow := search(`{"query":"line 20","path":"big.txt"}`)
	if strings.Contains(narrow, "other.txt") {
		t.Errorf("path=big.txt must search that file only:\n%s", narrow)
	}
	if !strings.Contains(narrow, "line 18") || strings.Contains(narrow, "line 15") {
		t.Errorf("default context should be %d lines a side:\n%s", searchContextLines, narrow)
	}

	wide := search(`{"query":"line 20","path":"big.txt","context":5}`)
	if !strings.Contains(wide, "line 15") || !strings.Contains(wide, "line 25") || strings.Contains(wide, "line 14\n") {
		t.Errorf("context=5 should show lines 15-25:\n%s", wide)
	}

	capped := search(`{"query":"line 20","path":"big.txt","context":500}`)
	if strings.Contains(capped, "line 9\n") || !strings.Contains(capped, "line 10") {
		t.Errorf("context is capped at %d:\n%s", maxSearchContext, capped)
	}
}
