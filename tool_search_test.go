package main

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

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
