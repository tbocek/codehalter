package main

import (
	"testing"
	"unicode/utf8"
)

// TestClipUTF8 pins that truncation snaps back to a rune boundary so a multibyte
// character at the cut is never split into a replacement char.
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
	// Valid UTF-8 at EVERY byte cut.
	for n := 0; n <= len(s); n++ {
		if !utf8.ValidString(clipUTF8(s, n)) {
			t.Errorf("clipUTF8(%q, %d) is not valid UTF-8: %q", s, n, clipUTF8(s, n))
		}
	}
}

// TestSliceWebBodyRunes pins that a model-supplied offset/limit range read never
// yields invalid UTF-8, regardless of where the cuts land.
func TestSliceWebBodyRunes(t *testing.T) {
	body := "aé€bc" // 1 + 2 + 3 + 1 + 1 = 8 bytes across mixed-width runes
	for off := 0; off <= len(body)+1; off++ {
		for lim := 0; lim <= len(body)+1; lim++ {
			if got := sliceWebBody(body, off, lim); !utf8.ValidString(got) {
				t.Fatalf("sliceWebBody(%q, %d, %d) = %q is not valid UTF-8", body, off, lim, got)
			}
		}
	}
}
