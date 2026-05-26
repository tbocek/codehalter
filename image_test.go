package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestImageHashIDContentAddressed covers the content-addressing invariant:
// the same bytes produce the same id, different bytes produce different ids.
// Re-paste of an identical screenshot must collide so we don't bloat the store.
func TestImageHashIDContentAddressed(t *testing.T) {
	id1 := imageHashID([]byte("hello"))
	id2 := imageHashID([]byte("hello"))
	id3 := imageHashID([]byte("world"))
	if id1 != id2 {
		t.Errorf("same bytes produced different ids: %q vs %q", id1, id2)
	}
	if id1 == id3 {
		t.Errorf("different bytes produced same id: %q", id1)
	}
	const want = "img_2cf24dba5fb0a30e"
	if id1 != want {
		t.Errorf("imageHashID(\"hello\") = %q, want %q", id1, want)
	}
}

// TestImageFileRoundTrip exercises the write/read pair across each supported
// mime: the file lands at the right path with the right extension and the
// readback recovers both bytes and mime from the extension alone (caller need
// not remember the original mime).
func TestImageFileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		mime, ext string
	}{
		{"image/png", "png"},
		{"image/jpeg", "jpg"},
		{"image/gif", "gif"},
		{"image/webp", "webp"},
		{"image/unknown", "bin"},
	}
	for _, c := range cases {
		t.Run(c.ext, func(t *testing.T) {
			payload := []byte("payload-" + c.ext)
			id := imageHashID(payload)
			if err := writeImageFile(dir, id, c.mime, payload); err != nil {
				t.Fatalf("writeImageFile: %v", err)
			}
			wantPath := filepath.Join(dir, ".codehalter", "images", id+"."+c.ext)
			if _, err := os.Stat(wantPath); err != nil {
				t.Errorf("expected file at %s: %v", wantPath, err)
			}
			data, mime, err := readImageFile(dir, id)
			if err != nil {
				t.Fatalf("readImageFile: %v", err)
			}
			if string(data) != string(payload) {
				t.Errorf("bytes round-trip: got %q, want %q", data, payload)
			}
			// "bin" extension recovers as application/octet-stream — caller
			// keeps the original mime from ImageData in that case.
			if c.ext != "bin" && mime != c.mime {
				t.Errorf("mime recovery: got %q, want %q", mime, c.mime)
			}
		})
	}
}

// TestImageFileNotFound: a hallucinated id surfaces a clean error, doesn't
// panic, doesn't return zero-length bytes silently.
func TestImageFileNotFound(t *testing.T) {
	dir := t.TempDir()
	_, _, err := readImageFile(dir, "img_doesnotexist")
	if err == nil {
		t.Fatal("expected error for missing image, got nil")
	}
}

// TestImageDedupOnRepaste: writing the same bytes twice produces exactly one
// file on disk. Re-pasting a screenshot in a long session must not bloat the
// store.
func TestImageDedupOnRepaste(t *testing.T) {
	dir := t.TempDir()
	payload := []byte("repeated bytes")
	id := imageHashID(payload)
	if err := writeImageFile(dir, id, "image/png", payload); err != nil {
		t.Fatalf("first write: %v", err)
	}
	if err := writeImageFile(dir, id, "image/png", payload); err != nil {
		t.Fatalf("second write: %v", err)
	}
	entries, err := os.ReadDir(filepath.Join(dir, ".codehalter", "images"))
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Errorf("expected 1 file (content-addressed dedup), got %d", len(entries))
	}
}
