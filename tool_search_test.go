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
