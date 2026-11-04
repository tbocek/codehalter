package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
)

// Order matters: a MIME type is written under its first extension; "jpeg" is only read.
var imageTypes = []struct{ ext, mime string }{
	{"png", "image/png"},
	{"jpg", "image/jpeg"},
	{"jpeg", "image/jpeg"},
	{"gif", "image/gif"},
	{"webp", "image/webp"},
}

// storeImage returns the content-addressed id even on error (for the log); cwd "" only computes it.
func storeImage(cwd, mime string, data []byte) (string, error) {
	sum := sha256.Sum256(data)
	id := "img_" + hex.EncodeToString(sum[:8])
	if cwd == "" {
		return id, nil
	}
	ext := "bin"
	for _, t := range imageTypes {
		if t.mime == mime {
			ext = t.ext
			break
		}
	}
	dir := filepath.Join(cwd, ".codehalter", "images")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return id, err
	}
	// Unique temp + rename: readers never see a half-written file, parallel same-id
	// writes don't clobber, and "tmp-*" can't match readImageFile's "<id>.*" glob.
	f, err := os.CreateTemp(dir, "tmp-*")
	if err != nil {
		return id, err
	}
	tmp := f.Name()
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(tmp)
		return id, err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return id, err
	}
	if err := os.Rename(tmp, filepath.Join(dir, id+"."+ext)); err != nil {
		os.Remove(tmp)
		return id, err
	}
	return id, nil
}

func readImageFile(cwd, id string) ([]byte, string, error) {
	matches, err := filepath.Glob(filepath.Join(cwd, ".codehalter", "images", id+".*"))
	if err != nil {
		return nil, "", err
	}
	if len(matches) == 0 {
		return nil, "", fmt.Errorf("image not found: %s", id)
	}
	data, err := os.ReadFile(matches[0])
	if err != nil {
		return nil, "", err
	}
	mime := "application/octet-stream"
	for _, t := range imageTypes {
		if "."+t.ext == filepath.Ext(matches[0]) {
			mime = t.mime
			break
		}
	}
	return data, mime, nil
}
