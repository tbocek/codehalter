package main

import (
	"os"
	"path/filepath"
	"testing"
)

// Pins that readImageFile recovers the mime from the extension alone.
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
			id, err := storeImage(dir, c.mime, payload)
			if err != nil {
				t.Fatalf("storeImage: %v", err)
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
			// "bin" reads back as application/octet-stream.
			if c.ext != "bin" && mime != c.mime {
				t.Errorf("mime recovery: got %q, want %q", mime, c.mime)
			}
		})
	}
}
