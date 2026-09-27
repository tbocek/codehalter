package main

import (
	"testing"
	"unicode/utf8"
)

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
