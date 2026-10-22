package main

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// search finds text in the project's own files, in grep's format with line numbers and context,
// through the same call the model makes: what git ignores and binary files stay out, a glob works
// by name and by path under `path`, a cut-short result says so, and a bad pattern or a path
// outside the project is refused.
func TestSearchFindsInProjectFiles(t *testing.T) {
	a, sess := newTestAgent(t)
	var long strings.Builder
	for i := range 60 {
		fmt.Fprintf(&long, "let hit_%d = 1;\n", i)
	}
	writeTree(t, sess.Cwd, map[string]string{
		".gitignore":         "target/\n",
		"rust/src/cut.rs":    "use std::path::Path;\n\npub fn load(path: &Path) {\n    open(path);\n}\n\nfn other() {}\n\npub fn LOAD_twice() {}\n",
		"rust/tests/cut.rs":  "#[test]\nfn loads() { load(p) }\n",
		"rust/notes.md":      "load the cut\n",
		"rust/target/out.rs": "pub fn load() {}\n",
		"rust/data.bin":      "load\x00\x01",
		"rust/src/many.rs":   long.String(),
	})
	gitInit(t, sess.Cwd)
	search := func(args string) (string, bool) {
		t.Helper()
		var tc toolCall
		tc.ID, tc.Function.Name, tc.Function.Arguments = "s", "search", args
		tu, _ := a.runToolCall(context.Background(), sess.ID, tc)
		return tu.Output, tu.Failed
	}

	out, failed := search(`{"pattern": "fn load", "context": 1}`)
	want := "rust/src/cut.rs-2-\nrust/src/cut.rs:3:pub fn load(path: &Path) {\nrust/src/cut.rs-4-    open(path);\n"
	if failed || !strings.HasPrefix(out, want) || !strings.HasSuffix(out, "rust/tests/cut.rs:2:fn loads() { load(p) }\n\n2 matches in 2 files.") {
		t.Errorf("literal search with context:\n%s", out)
	}
	if strings.Contains(out, "target/out.rs") || strings.Contains(out, "data.bin") {
		t.Errorf("an ignored or binary file was searched:\n%s", out)
	}

	out, _ = search(`{"pattern": "load", "glob": "*.rs", "context": 0}`)
	if strings.Contains(out, "notes.md") || !strings.Contains(out, "rust/tests/cut.rs:2:fn loads() { load(p) }") {
		t.Errorf("a glob by name:\n%s", out)
	}
	out, _ = search(`{"pattern": "load", "path": "rust", "glob": "tests/**/*.rs"}`)
	if !strings.Contains(out, "rust/tests/cut.rs:2:") || strings.Contains(out, "rust/src/") {
		t.Errorf("a glob under path:\n%s", out)
	}

	out, _ = search(`{"pattern": "^pub fn (load|LOAD_\\w+)", "regex": true, "ignore_case": true, "context": 0}`)
	if !strings.Contains(out, "rust/src/cut.rs:3:") || !strings.Contains(out, "rust/src/cut.rs:9:") || !strings.Contains(out, "--\nrust/src/cut.rs:9:") {
		t.Errorf("regex, ignoring case, groups apart:\n%s", out)
	}

	if out, _ = search(`{"pattern": "hit_", "path": "rust/src/many.rs", "limit": 5, "context": 0}`); !strings.Contains(out, "Showing the first 5 of 60 matches, in 1 files") || strings.Count(out, "rust/src/many.rs:") != 5 {
		t.Errorf("a result cut short:\n%s", out)
	}
	if out, failed = search(`{"pattern": "nothing like this"}`); failed || !strings.Contains(out, `No match for "nothing like this" in the project.`) {
		t.Errorf("no match: %q failed=%v", out, failed)
	}
	if out, failed = search(`{"pattern": "(", "regex": true}`); !failed || !strings.Contains(out, "not a regular expression") {
		t.Errorf("a bad regex: %q failed=%v", out, failed)
	}
	if out, failed = search(`{"pattern": "x", "path": "/etc"}`); !failed || !strings.Contains(out, "outside project") {
		t.Errorf("a path outside the project: %q failed=%v", out, failed)
	}
}
