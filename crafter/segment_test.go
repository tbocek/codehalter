package main

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestLocateSpan(t *testing.T) {
	file := strings.Split(strings.TrimPrefix(`
# Go skill
## Errors
- Errors = values. Check: if err != nil.
- NO panic for control flow.

## Idioms
- := declare, = reassign.
User=non-root dev, sudo NOPASSWD. Write ops need sudo. Read probes no sudo.
`, "\n"), "\n")

	cases := []struct {
		name, source       string
		wantStart, wantEnd int
		wantFrag           bool
	}{
		{"single bullet", "- NO panic for control flow.", 4, 4, false},
		{"first bullet", "- Errors = values. Check: if err != nil.", 3, 3, false},
		{"no such line", "- Idioms line? no", 0, 0, false},
		{"multi contiguous", "## Idioms\n- := declare, = reassign.", 6, 7, false},
		{"absent", "this is not present", 0, 0, false},
		// Sub-line fragments: one sentence of a multi-sentence line.
		{"fragment middle sentence", "Write ops need sudo.", 8, 8, true},
		{"fragment first sentence", "User=non-root dev, sudo NOPASSWD.", 8, 8, true},
		{"fragment without bullet marker", "NO panic for control flow.", 4, 4, true},
		// "sudo" appears in several lines → ambiguous, must not match.
		{"ambiguous fragment", "sudo", 0, 0, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gs, ge, frag := locateSpan(file, c.source)
			if gs != c.wantStart || ge != c.wantEnd || frag != c.wantFrag {
				t.Fatalf("locateSpan(%q) = (%d,%d,%v), want (%d,%d,%v)", c.source, gs, ge, frag, c.wantStart, c.wantEnd, c.wantFrag)
			}
		})
	}
}

func TestLocateSpanSkipsBlankLines(t *testing.T) {
	file := []string{"- first", "", "- second"}
	// A source quoting both bullets with the blank between should still match.
	gs, ge, frag := locateSpan(file, "- first\n- second")
	if gs != 1 || ge != 3 || frag {
		t.Fatalf("got (%d,%d,%v), want (1,3,false)", gs, ge, frag)
	}
}

func TestClaimCacheInvalidation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "go.json")
	claims := []Claim{{ID: "go#abc", Skill: "go", Text: "t", Source: "- s", StartLine: 1, EndLine: 1}}
	if err := writeClaimCache(path, "hashA", claims); err != nil {
		t.Fatal(err)
	}
	// Same content hash → cache hit.
	if got, ok := readClaimCache(path, "hashA"); !ok || len(got) != 1 || got[0].ID != "go#abc" {
		t.Fatalf("expected hit, got ok=%v claims=%v", ok, got)
	}
	// Changed content hash → miss, forcing re-segmentation.
	if _, ok := readClaimCache(path, "hashB"); ok {
		t.Fatal("stale cache must miss when content hash differs")
	}
}

func TestHashOfStable(t *testing.T) {
	if hashOf([]byte("x")) != hashOf([]byte("x")) {
		t.Fatal("hash not deterministic")
	}
	if hashOf([]byte("x")) == hashOf([]byte("y")) {
		t.Fatal("distinct inputs collided")
	}
}

func TestNonBlankTrimmed(t *testing.T) {
	got := nonBlankTrimmed([]string{"  a  ", "", "   ", "b"})
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("nonBlankTrimmed = %v", got)
	}
}
