package main

import (
	"os"
	"path/filepath"
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
