package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The best image the clipboard offers is taken: PNG before the lossy ones, any of them before none.
func TestPickClipboardImageType(t *testing.T) {
	for _, c := range []struct{ listed, want string }{
		{"text/plain\nimage/jpeg\nimage/png\n", "image/png"},
		{"image/webp\ntext/uri-list", "image/webp"},
		{"text/plain;charset=utf-8\nUTF8_STRING", ""},
		{"", ""},
	} {
		if got := pickClipboardImageType(c.listed); got != c.want {
			t.Errorf("%q: got %q, want %q", c.listed, got, c.want)
		}
	}
}

// fakeWlPaste puts a wl-paste on PATH that offers `types` and answers `--type <t>` with `data`; an empty
// `types` is an empty clipboard, which wl-paste answers with a failure.
func fakeWlPaste(t *testing.T, types string, data []byte) {
	t.Helper()
	requireSh(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "clip.bin"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "types"), []byte(types), 0o600); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\n" +
		"case \"$1\" in\n" +
		"--list-types) [ -s '" + dir + "/types' ] || { echo 'Nothing is copied' >&2; exit 1; }; cat '" + dir + "/types';;\n" +
		"*) cat '" + dir + "/clip.bin';;\n" +
		"esac\n"
	if err := os.WriteFile(filepath.Join(dir, "wl-paste"), []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("WAYLAND_DISPLAY", "wayland-test")
}

// Ctrl+V's reader on Wayland: the picture's bytes and type through wl-paste, and a reason when there is none.
func TestWaylandClipboardImage(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\nfake")
	fakeWlPaste(t, "text/plain\nimage/png\n", png)
	data, mime, err := waylandClipboardImage()
	if err != nil || mime != "image/png" || !bytes.Equal(data, png) {
		t.Fatalf("got %q %q %v, want the PNG", data, mime, err)
	}

	fakeWlPaste(t, "text/plain\n", []byte("just words"))
	if _, _, err := waylandClipboardImage(); !errors.Is(err, errNoClipboardImage) {
		t.Errorf("text only: %v, want errNoClipboardImage", err)
	}

	fakeWlPaste(t, "", nil)
	if _, _, err := waylandClipboardImage(); !errors.Is(err, errNoClipboardImage) {
		t.Errorf("empty clipboard: %v, want errNoClipboardImage", err)
	}

	t.Setenv("PATH", t.TempDir())
	if _, _, err := waylandClipboardImage(); err == nil || !strings.Contains(err.Error(), "wl-clipboard") {
		t.Errorf("no wl-paste: %v, want the package to install named", err)
	}

	t.Setenv("WAYLAND_DISPLAY", "")
	if _, _, err := waylandClipboardImage(); err == nil || !strings.Contains(err.Error(), "WAYLAND_DISPLAY") {
		t.Errorf("no Wayland session: %v", err)
	}
}

// fakeOsascript puts an osascript on PATH that does what the real one does with macClipboardScript: writes
// `png` to the file named by its last argument and says "ok", or says "none" when png is nil.
func fakeOsascript(t *testing.T, png []byte) {
	t.Helper()
	requireSh(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "clip.png"), png, 0o600); err != nil {
		t.Fatal(err)
	}
	answer := "for last; do :; done\ncat '" + dir + "/clip.png' > \"$last\"\necho ok\n"
	if png == nil {
		answer = "echo none\n"
	}
	if err := os.WriteFile(filepath.Join(dir, "osascript"), []byte("#!/bin/sh\n"+answer), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

// Ctrl+V's reader on macOS: osascript writes the clipboard's PNG to a file and the bytes come back; a
// clipboard with no picture is said as such.
func TestMacClipboardImage(t *testing.T) {
	png := []byte("\x89PNG\r\n\x1a\nmac")
	fakeOsascript(t, png)
	data, mime, err := macClipboardImage()
	if err != nil || mime != "image/png" || !bytes.Equal(data, png) {
		t.Fatalf("got %q %q %v, want the PNG", data, mime, err)
	}

	fakeOsascript(t, nil)
	if _, _, err := macClipboardImage(); !errors.Is(err, errNoClipboardImage) {
		t.Errorf("no picture: %v, want errNoClipboardImage", err)
	}

	t.Setenv("PATH", t.TempDir())
	if _, _, err := macClipboardImage(); err == nil || !strings.Contains(err.Error(), "osascript") {
		t.Errorf("no osascript: %v", err)
	}
}
