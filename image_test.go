package main

import (
	"os"
	"path/filepath"
	"testing"
)

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
			id := "img_test_" + c.ext
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
