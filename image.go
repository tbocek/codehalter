package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func writeImageFile(cwd, id, mime string, data []byte) error {
	ext := "bin"
	switch mime {
	case "image/png":
		ext = "png"
	case "image/jpeg":
		ext = "jpg"
	case "image/gif":
		ext = "gif"
	case "image/webp":
		ext = "webp"
	}
	dir := filepath.Join(cwd, ".codehalter", "images")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	// Write to a UNIQUE temp, then rename for an atomic publish a concurrent reader
	// (view_image / LoadSession replay) can't catch half-written. The temp is
	// "tmp-*", NOT "<id>.<ext>.tmp": unique so two parallel same-id writes don't
	// clobber each other, and dotless so it can't match readImageFile's "<id>.*"
	// glob (a leftover from a crashed write would otherwise pollute it).
	f, err := os.CreateTemp(dir, "tmp-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, filepath.Join(dir, id+"."+ext)); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// readImageFile finds the file by id (mime unknown to the caller — e.g.
// view_image), reads it, and recovers the mime from the file extension.
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
	switch strings.TrimPrefix(filepath.Ext(matches[0]), ".") {
	case "png":
		mime = "image/png"
	case "jpg", "jpeg":
		mime = "image/jpeg"
	case "gif":
		mime = "image/gif"
	case "webp":
		mime = "image/webp"
	}
	return data, mime, nil
}
