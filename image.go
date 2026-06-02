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
	path := filepath.Join(cwd, ".codehalter", "images", id+"."+ext)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
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
