package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestFormatterCmdByExtension(t *testing.T) {
	if cmd := formatterCmd("/x/file.unknownext", ""); cmd != nil {
		t.Errorf("unknown ext: got %v, want nil", cmd)
	}
	if _, err := exec.LookPath("gofmt"); err == nil {
		cmd := formatterCmd("/x/main.go", "")
		if len(cmd) == 0 || filepath.Base(cmd[0]) != "gofmt" {
			t.Errorf(".go: got %v, want gofmt", cmd)
		}
	}
}

func TestFormatGuardedOnlyFormatsAlreadyCleanFiles(t *testing.T) {
	if _, err := exec.LookPath("gofmt"); err != nil {
		t.Skip("gofmt not available")
	}
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "main.go")
	messyNew := "package main\n\nfunc main() {\n        x := 1\n        _ = x\n}\n"

	cleanOld := "package main\n\nfunc main() {}\n"
	if got := a.formatGuarded(s.ID, path, cleanOld, messyNew); !strings.Contains(got, "\tx := 1") {
		t.Errorf("clean old: new should be gofmt'd (tabs), got:\n%q", got)
	}

	if got := a.formatGuarded(s.ID, path, "", messyNew); !strings.Contains(got, "\tx := 1") {
		t.Errorf("new file: should be formatted, got:\n%q", got)
	}

	dirtyOld := "package main\nfunc main(){}\n"
	if got := a.formatGuarded(s.ID, path, dirtyOld, messyNew); got != messyNew {
		t.Errorf("dirty old: new must be left unformatted, got:\n%q", got)
	}

	if got := a.formatGuarded(s.ID, filepath.Join(s.Cwd, "f.unknownext"), "", "x"); got != "x" {
		t.Errorf("no formatter: passthrough, got %q", got)
	}
}

// An edit that leaves the file unparsable is refused instead of written: the last run
// put a scrambled test file on disk and only `go test` noticed.
func TestParseRefusalBlocksUnparsableContent(t *testing.T) {
	if _, err := exec.LookPath("gofmt"); err != nil {
		t.Skip("gofmt not available")
	}
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "main.go")
	cleanOld := "package main\n\nfunc main() {}\n"
	broken := "package main\n\nfunc main() {\n\tx := 1\n}\n}\n"
	// The gate only judges a file that exists: what it is about to break must be on disk.
	if err := os.WriteFile(path, []byte(cleanOld), 0o644); err != nil {
		t.Fatal(err)
	}

	if msg := a.parseRefusal(s.ID, path, cleanOld, broken); !strings.Contains(msg, "does not parse") || !strings.Contains(msg, "NOTHING was written") {
		t.Errorf("broken new over an existing clean file: want a refusal, got %q", msg)
	} else if !strings.Contains(msg, "expected declaration") {
		t.Errorf("the refusal should quote gofmt's complaint, got %q", msg)
	}

	// gofmt parses the file it is given, so an already-broken source must not be blamed.
	if msg := a.parseRefusal(s.ID, filepath.Join(s.Cwd, "broken.go"), "package main\nfunc main() {\n\tx := 1\n}\n}\n", broken); msg != "" {
		t.Errorf("already-broken file must not be refused, got %q", msg)
	}

	// Valid content, however badly formatted, passes: formatting is not refusing.
	if msg := a.parseRefusal(s.ID, path, cleanOld, "package main\n\nfunc main() {\n        x := 1\n        _ = x\n}\n"); msg != "" {
		t.Errorf("valid but messy new: want no refusal, got %q", msg)
	}

	// A file with neither a parser nor a formatter proves nothing about syntax.
	if msg := a.parseRefusal(s.ID, filepath.Join(s.Cwd, "notes.unknownext"), "# t\n", "```\n}\n"); msg != "" {
		t.Errorf("a file with no parser must not be refused, got %q", msg)
	}
}

// node --check picks the module kind from the file name and the nearest package.json,
// so only .mjs/.cjs are safe to probe: a valid ES module in a .js would be refused.
func TestNodeProbeOnlyForUnambiguousModules(t *testing.T) {
	if _, err := exec.LookPath("node"); err != nil {
		t.Skip("node not available")
	}
	a, s := newTestAgent(t)
	esm := "import x from \"y\";\nexport const z = x;\n"

	if v := probeParses(filepath.Join(s.Cwd, "a.mjs"), esm); v.have && !v.ok {
		t.Errorf("valid ESM .mjs rejected: %s", v.detail)
	}
	if syntaxProbe(filepath.Join(s.Cwd, "a.js")) != nil {
		t.Error(".js must not be probed: node needs the project's package.json to judge it")
	}
	// The refusal paths need a file that exists: a new one is never blamed.
	for _, name := range []string{"a.js", "a.mjs"} {
		if err := os.WriteFile(filepath.Join(s.Cwd, name), []byte("const a = 1;\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if msg := a.parseRefusal(s.ID, filepath.Join(s.Cwd, "a.js"), "const a = 1;\n", esm); msg != "" {
		t.Errorf("valid ESM in a .js must not be refused, got %q", msg)
	}
	if msg := a.parseRefusal(s.ID, filepath.Join(s.Cwd, "a.mjs"), "const a = 1;\n", esm); msg != "" {
		t.Errorf("valid ESM in an .mjs must not be refused, got %q", msg)
	}
	if msg := a.parseRefusal(s.ID, filepath.Join(s.Cwd, "a.mjs"), "const a = 1;\n", "const a = ;\n"); !strings.Contains(msg, "does not parse") {
		t.Errorf("broken .mjs must be refused, got %q", msg)
	}
}

// The language's own parser answers even where no formatter is configured.
