package main

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestFormatterCmdByExtension(t *testing.T) {
	if cmd := formatterCmd("/x/file.unknownext", ""); cmd != nil {
		t.Errorf("unknown ext: got %v, want nil", cmd)
	}
	// .go → gofmt when it's on PATH (it ships with the Go toolchain the tests
	// run under).
	if _, err := exec.LookPath("gofmt"); err == nil {
		cmd := formatterCmd("/x/main.go", "")
		if len(cmd) == 0 || filepath.Base(cmd[0]) != "gofmt" {
			t.Errorf(".go: got %v, want gofmt", cmd)
		}
	}
}

// TestFormatGuardedOnlyFormatsAlreadyCleanFiles is the defensive-formatting
// contract: format the edit ONLY when the pre-edit source was already canonical;
// leave an edit to an unformatted file untouched; format a new (empty) file.
func TestFormatGuardedOnlyFormatsAlreadyCleanFiles(t *testing.T) {
	if _, err := exec.LookPath("gofmt"); err != nil {
		t.Skip("gofmt not available")
	}
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "main.go")
	messyNew := "package main\n\nfunc main() {\n        x := 1\n        _ = x\n}\n" // 8-space indent

	// Clean old → new content gets gofmt'd (spaces become a tab).
	cleanOld := "package main\n\nfunc main() {}\n"
	if got := a.formatGuarded(s.ID, path, cleanOld, messyNew); !strings.Contains(got, "\tx := 1") {
		t.Errorf("clean old: new should be gofmt'd (tabs), got:\n%q", got)
	}

	// New/empty file is clean by definition → formatted.
	if got := a.formatGuarded(s.ID, path, "", messyNew); !strings.Contains(got, "\tx := 1") {
		t.Errorf("new file: should be formatted, got:\n%q", got)
	}

	// Dirty old (gofmt would change it) → leave the edit untouched.
	dirtyOld := "package main\nfunc main(){}\n"
	if got := a.formatGuarded(s.ID, path, dirtyOld, messyNew); got != messyNew {
		t.Errorf("dirty old: new must be left unformatted, got:\n%q", got)
	}

	// No formatter for the extension → passthrough.
	if got := a.formatGuarded(s.ID, filepath.Join(s.Cwd, "f.unknownext"), "", "x"); got != "x" {
		t.Errorf("no formatter: passthrough, got %q", got)
	}
}
