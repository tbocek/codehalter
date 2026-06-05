package main

import (
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
