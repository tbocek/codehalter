package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// imageHashID returns the per-image handle "img_<sha256[:8] hex>". Content
// addressed: the same bytes always produce the same id, so a re-paste of the
// same screenshot collides with the existing file in the image store and
// nothing extra is written.
func imageHashID(data []byte) string {
	h := sha256.Sum256(data)
	return "img_" + hex.EncodeToString(h[:8])
}

func extForMime(mime string) string {
	switch mime {
	case "image/png":
		return "png"
	case "image/jpeg":
		return "jpg"
	case "image/gif":
		return "gif"
	case "image/webp":
		return "webp"
	default:
		return "bin"
	}
}

func mimeForExt(ext string) string {
	switch strings.TrimPrefix(ext, ".") {
	case "png":
		return "image/png"
	case "jpg", "jpeg":
		return "image/jpeg"
	case "gif":
		return "image/gif"
	case "webp":
		return "image/webp"
	default:
		return "application/octet-stream"
	}
}

func imageFilePath(cwd, id, mime string) string {
	return filepath.Join(cwd, ".codehalter", "images", id+"."+extForMime(mime))
}

func writeImageFile(cwd, id, mime string, data []byte) error {
	path := imageFilePath(cwd, id, mime)
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
	return data, mimeForExt(filepath.Ext(matches[0])), nil
}
