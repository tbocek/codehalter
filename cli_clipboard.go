package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// Ctrl+V in the CLI attaches an image from the clipboard: a terminal paste only ever carries text, so the
// picture is read off the clipboard itself, the way a browser chat or Zed takes a pasted screenshot. On
// macOS by osascript, which ships with the system; elsewhere from the Wayland clipboard by wl-paste
// (wl-clipboard).

// clipboardImage is the system's clipboard reader: osascript on macOS, wl-paste elsewhere.
func clipboardImage() ([]byte, string, error) {
	if runtime.GOOS == "darwin" {
		return macClipboardImage()
	}
	return waylandClipboardImage()
}

// macClipboardScript writes the clipboard's picture, as PNG, to the file named by its argument and says
// "ok", or says "none" when the clipboard holds no picture. PNGf is the type screenshots and copied images
// offer (a TIFF on the clipboard is converted to it); reading it as text would be the hex of the bytes.
var macClipboardScript = []string{
	"on run argv",
	"try",
	"set png to the clipboard as «class PNGf»",
	"on error",
	`return "none"`,
	"end try",
	"set f to open for access (POSIX file (item 1 of argv)) with write permission",
	"set eof f to 0",
	"write png to f",
	"close access f",
	`return "ok"`,
	"end run",
}

// macClipboardImage reads the clipboard's picture on macOS through osascript, into a temporary file.
func macClipboardImage() ([]byte, string, error) {
	if _, err := exec.LookPath("osascript"); err != nil {
		return nil, "", errors.New("osascript is not on PATH")
	}
	dir, err := os.MkdirTemp("", "codehalter-clip-")
	if err != nil {
		return nil, "", err
	}
	defer os.RemoveAll(dir)
	file := filepath.Join(dir, "clipboard.png")
	var args []string
	for _, line := range macClipboardScript {
		args = append(args, "-e", line)
	}
	ctx, cancel := context.WithTimeout(context.Background(), clipboardWait)
	defer cancel()
	said, err := exec.CommandContext(ctx, "osascript", append(args, file)...).Output()
	if err != nil {
		return nil, "", fmt.Errorf("osascript: %w", err)
	}
	if strings.TrimSpace(string(said)) != "ok" {
		return nil, "", errNoClipboardImage
	}
	data, err := os.ReadFile(file)
	if err != nil || len(data) == 0 {
		return nil, "", errNoClipboardImage
	}
	return data, "image/png", nil
}

// errNoClipboardImage is a clipboard holding something, but no picture.
var errNoClipboardImage = errors.New("no image on the clipboard")

// clipboardWait bounds each clipboard tool call: a clipboard that never answers must not hang the editor.
const clipboardWait = 3 * time.Second

// clipboardImageTypes are the image types offered, best first: PNG is lossless and what screenshot
// tools put there; the rest are taken when that is all there is.
var clipboardImageTypes = []string{"image/png", "image/jpeg", "image/webp", "image/gif"}

// waylandClipboardImage reads the clipboard's image: the types it offers, then the bytes of the best one.
func waylandClipboardImage() ([]byte, string, error) {
	if os.Getenv("WAYLAND_DISPLAY") == "" {
		return nil, "", errors.New("no Wayland session (WAYLAND_DISPLAY is not set)")
	}
	if _, err := exec.LookPath("wl-paste"); err != nil {
		return nil, "", errors.New("wl-paste is not installed (the wl-clipboard package)")
	}
	types, err := wlPaste("--list-types")
	if err != nil {
		// wl-paste fails this way on an empty clipboard: nothing to list is no image.
		return nil, "", errNoClipboardImage
	}
	mime := pickClipboardImageType(string(types))
	if mime == "" {
		return nil, "", errNoClipboardImage
	}
	data, err := wlPaste("--no-newline", "--type", mime)
	if err != nil {
		return nil, "", fmt.Errorf("wl-paste --type %s: %w", mime, err)
	}
	if len(data) == 0 {
		return nil, "", errNoClipboardImage
	}
	return data, mime, nil
}

func wlPaste(args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), clipboardWait)
	defer cancel()
	return exec.CommandContext(ctx, "wl-paste", args...).Output()
}

// pickClipboardImageType is the best image type in wl-paste's --list-types answer (one type per
// line), or "" when it offers no image.
func pickClipboardImageType(listed string) string {
	offered := strings.Fields(listed)
	for _, want := range clipboardImageTypes {
		for _, t := range offered {
			if t == want {
				return want
			}
		}
	}
	return ""
}
