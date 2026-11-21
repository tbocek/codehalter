package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The gate through the real tool path: what a model actually calls, and what lands on disk.
// Each case skips when its parser is not installed, so the file says nothing false either way.
func TestParseGateThroughTheTools(t *testing.T) {
	a, s := newTestAgent(t)
	read := func(p string) string {
		t.Helper()
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}

	t.Run("an edit that breaks Go is refused and the file keeps its bytes", func(t *testing.T) {
		if _, err := exec.LookPath("gofmt"); err != nil {
			t.Skip("gofmt not available")
		}
		path := filepath.Join(s.Cwd, "gate.go")
		src := "package main\n\nfunc main() {\n\tx := 1\n\t_ = x\n}\n"
		if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
		// Scrambled the way a bad multi-edit scrambles a file: a stray closing brace.
		tu := callTool(t, a, s.ID, "edit_file", fmt.Sprintf(`{"path":%q,"old_text":"\tx := 1","new_text":"\tx := 1\n}"}`, path))
		if !tu.Failed || !strings.Contains(tu.Output, "does not parse") {
			t.Fatalf("want a refusal, got failed=%v %q", tu.Failed, tu.Output)
		}
		if !strings.Contains(tu.Output, "expected declaration") {
			t.Errorf("the refusal should quote gofmt: %q", tu.Output)
		}
		if read(path) != src {
			t.Errorf("a refused edit changed the file:\n%s", read(path))
		}
		// The same call fixed up goes through, so the gate is not simply refusing everything.
		tu = callTool(t, a, s.ID, "edit_file", fmt.Sprintf(`{"path":%q,"old_text":"\tx := 1","new_text":"\tx := 2"}`, path))
		if tu.Failed || !strings.Contains(read(path), "x := 2") {
			t.Fatalf("a valid edit after a refusal: failed=%v %q", tu.Failed, tu.Output)
		}
	})

	t.Run("one bad edit in a list writes nothing", func(t *testing.T) {
		if _, err := exec.LookPath("gofmt"); err != nil {
			t.Skip("gofmt not available")
		}
		path := filepath.Join(s.Cwd, "list.go")
		src := "package main\n\nfunc main() {}\n"
		if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
		args, _ := json.Marshal(map[string]any{"path": path, "edits": []map[string]string{
			{"old_text": "func main() {}", "new_text": "func main() { println(\"hi\") }"},
			{"old_text": "package main", "new_text": "package main\n}"},
		}})
		tu := callTool(t, a, s.ID, "edit_file", string(args))
		if !tu.Failed || !strings.Contains(tu.Output, "does not parse") {
			t.Fatalf("the second edit breaks the file; want the whole list refused: %v %q", tu.Failed, tu.Output)
		}
		if read(path) != src {
			t.Errorf("the first edit leaked to disk:\n%s", read(path))
		}
	})

	t.Run("write_file is gated too", func(t *testing.T) {
		if _, err := exec.LookPath("python3"); err != nil {
			t.Skip("python3 not available")
		}
		path := filepath.Join(s.Cwd, "gate.py")
		if err := os.WriteFile(path, []byte("x = 1\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		tu := callTool(t, a, s.ID, "write_file", fmt.Sprintf(`{"path":%q,"content":"x = (1 +\n"}`, path))
		if !tu.Failed || !strings.Contains(tu.Output, "was never closed") {
			t.Fatalf("want a python refusal quoting SyntaxError: %v %q", tu.Failed, tu.Output)
		}
		if read(path) != "x = 1\n" {
			t.Errorf("a refused write changed the file: %q", read(path))
		}
		tu = callTool(t, a, s.ID, "write_file", fmt.Sprintf(`{"path":%q,"content":"x=(1+2)\n"}`, path))
		if tu.Failed || read(path) != "x=(1+2)\n" {
			t.Fatalf("valid messy python must be written: %v %q", tu.Failed, tu.Output)
		}
	})

	t.Run("a brand new unparsable file is allowed: nothing was broken", func(t *testing.T) {
		if _, err := exec.LookPath("rustfmt"); err != nil {
			t.Skip("rustfmt not available")
		}
		path := filepath.Join(s.Cwd, "brand.rs")
		tu := callTool(t, a, s.ID, "write_file", fmt.Sprintf(`{"path":%q,"content":"fn main() { let x = 1;\n"}`, path))
		if tu.Failed {
			t.Errorf("a new file cannot have been broken by this write, so it must be written: %q", tu.Output)
		}
	})

	t.Run("shell syntax is caught", func(t *testing.T) {
		path := filepath.Join(s.Cwd, "gate.sh")
		if err := os.WriteFile(path, []byte("echo hi\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		tu := callTool(t, a, s.ID, "write_file", fmt.Sprintf(`{"path":%q,"content":"if true; then\n"}`, path))
		if !tu.Failed || !strings.Contains(tu.Output, "does not parse") {
			t.Fatalf("an unterminated if must be refused: failed=%v %q", tu.Failed, tu.Output)
		}
		if read(path) != "echo hi\n" {
			t.Errorf("a refused write changed the file: %q", read(path))
		}
	})

	t.Run("a file with no parser is never gated", func(t *testing.T) {
		path := filepath.Join(s.Cwd, "notes.unknownext")
		tu := callTool(t, a, s.ID, "write_file", fmt.Sprintf(`{"path":%q,"content":"}\n]\nnot anything\n"}`, path))
		if tu.Failed {
			t.Errorf("no parser means no opinion: %q", tu.Output)
		}
	})

	_ = context.Background
}
