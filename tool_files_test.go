package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func writeLines(t *testing.T, path string, n int) {
	t.Helper()
	var b strings.Builder
	for i := 1; i <= n; i++ {
		fmt.Fprintf(&b, "L%d\n", i)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

var nextReadRe = regexp.MustCompile(`read_file (\{"path": "[^"]*", "start_line": \d+\})`)

// A missed old_text reports failed=true (for the fail cap) and steers to a small
// retry, not a whole-file rewrite.
func TestEditFileMissFailsAndSteers(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()
	path := filepath.Join(s.Cwd, "f.go")
	if err := os.WriteFile(path, []byte("package main\n\nfunc A() {}\n"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	var miss toolCall
	miss.Function.Name = "edit_file"
	miss.Function.Arguments = fmt.Sprintf(`{"path":%q,"old_text":"func ZZZ() {}","new_text":"x"}`, path)
	tu, _ := a.runToolCall(ctx, s.ID, miss)
	out, failed := tu.Output, tu.Failed
	if !failed {
		t.Errorf("missed old_text: failed=false, want true (must feed the fail cap)")
	}
	for _, want := range []string{"not found", "Do NOT read the file again", "whole file"} {
		if !strings.Contains(out, want) {
			t.Errorf("miss message missing %q steering:\n%s", want, out)
		}
	}

	var hit toolCall
	hit.Function.Name = "edit_file"
	hit.Function.Arguments = fmt.Sprintf(`{"path":%q,"old_text":"func A() {}","new_text":"func A() { return }"}`, path)
	if tu, _ := a.runToolCall(ctx, s.ID, hit); tu.Failed {
		t.Errorf("successful edit: failed=true, want false")
	}
}

// Recovers trailing-whitespace and indentation drift, re-indenting new_text, and
// stays unique-or-fail.
func TestTolerantReplace(t *testing.T) {
	file := "func f() {\n\treturn 1 \n}\n"
	old := "func f() {\n\treturn 1\n}"
	out, n := tolerantReplace(file, old, "func f() {\n\treturn 2\n}")
	if n != 1 || !strings.Contains(out, "return 2") {
		t.Fatalf("trailing-ws: n=%d out=%q", n, out)
	}

	file = "x\n\t\tcall(a)\n\t\tcall(b)\ny\n"
	old = "call(a)\ncall(b)"
	out, n = tolerantReplace(file, old, "call(a)\ncall(c)")
	if n != 1 {
		t.Fatalf("indent: n=%d", n)
	}
	if !strings.Contains(out, "\t\tcall(c)") || strings.Contains(out, "\ncall(c)") {
		t.Errorf("indent not reapplied to new_text:\n%q", out)
	}

	file = "a\n  p()\nb\n  p()\nc\n"
	if _, n = tolerantReplace(file, "p()", "q()"); n != 2 {
		t.Errorf("ambiguous: want n=2, got %d", n)
	}

	if out, n = tolerantReplace("alpha\nbeta\n", "gamma", "x"); n != 0 || out != "" {
		t.Errorf("no-match: want n=0 empty, got n=%d out=%q", n, out)
	}
}

// The negative cases matter as much: quoting the wrong region sends the model to
// edit the wrong place.
func TestNearMiss(t *testing.T) {
	file := "package main\n\nfunc load(p string) error {\n\tf, err := os.Open(p)\n\tif err != nil {\n\t\treturn err\n\t}\n\treturn nil\n}\n"

	// One drifted line (the old parameter name): located and quoted verbatim.
	old := "func load(path string) error {\n\tf, err := os.Open(path)\n\tif err != nil {"
	line, snippet, ok := nearMiss(file, old)
	if !ok {
		t.Fatal("drifted snippet: no near miss found")
	}
	if line != 3 {
		t.Errorf("start line = %d, want 3", line)
	}
	if !strings.Contains(snippet, "os.Open(p)") {
		t.Errorf("snippet is not the file's CURRENT text:\n%s", snippet)
	}
	if strings.Contains(snippet, "os.Open(path)") {
		t.Errorf("snippet echoed the model's stale text back at it:\n%s", snippet)
	}

	if _, _, ok := nearMiss(file, "type Server struct {\n\taddr string\n\tport int\n}"); ok {
		t.Error("unrelated snippet produced a near miss")
	}

	// A lone closing brace appears twice and says nothing about the region.
	if _, _, ok := nearMiss("a\n}\nb\n}\nc\n", "}"); ok {
		t.Error("bare boilerplate line produced a near miss")
	}

	if _, _, ok := nearMiss(file, "func load(p string) error {\n\tzzz()\n\tyyy()\n\txxx()"); ok {
		t.Error("sub-threshold overlap produced a near miss")
	}

	for _, old := range []string{"", "\n\n", strings.Repeat("x\n", 100)} {
		if _, _, ok := nearMiss(file, old); ok {
			t.Errorf("degenerate old_text %q produced a near miss", truncate(old, 20))
		}
	}
}

// An old_text not in the file comes back with the file's current bytes, the first line
// where the copy differs, and says NOT to re-read; it claims no drift it cannot know.
func TestEditFileMissQuotesNearbyRegion(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()
	path := filepath.Join(s.Cwd, "g.go")
	body := "package main\n\nfunc load(p string) error {\n\tf, err := os.Open(p)\n\tif err != nil {\n\t\treturn err\n\t}\n\treturn nil\n}\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	var miss toolCall
	miss.Function.Name = "edit_file"
	miss.Function.Arguments = fmt.Sprintf(`{"path":%q,"old_text":"func load(path string) error {\n\tf, err := os.Open(path)\n\tif err != nil {","new_text":"x"}`, path)
	tu, _ := a.runToolCall(ctx, s.ID, miss)
	out, failed := tu.Output, tu.Failed
	if !failed {
		t.Error("drifted edit: failed=false, want true (must feed the fail cap)")
	}
	if !strings.Contains(out, "os.Open(p)") {
		t.Errorf("miss message did not quote the current region:\n%s", out)
	}
	if !strings.Contains(out, "Do NOT read the file again") {
		t.Errorf("miss message still sends the model back to reading:\n%s", out)
	}
	if !strings.Contains(out, `differs from the file first at line 3: you wrote "func load(path string) error {", the file has "func load(p string) error {"`) || strings.Contains(out, "drifted") {
		t.Errorf("miss message does not name the first differing line, or claims drift:\n%s", out)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(after) != body {
		t.Errorf("failed edit modified the file:\n%s", after)
	}
}

// Without a client fs capability the session does its own disk I/O, as ACP
// forbids unclaimed methods. a.conn is nil, so an attempted wire call panics.
func TestFsGatedOnClientCapabilities(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()
	path := filepath.Join(s.Cwd, "f.txt")

	if err := fsWrite(a, ctx, s.ID, path, "hello\n"); err != nil {
		t.Fatalf("fsWrite with no client fs capability: %v", err)
	}
	got, err := fsRead(a, ctx, s.ID, path, nil, nil)
	if err != nil {
		t.Fatalf("fsRead with no client fs capability: %v", err)
	}
	if got != "hello\n" {
		t.Errorf("fsRead = %q, want %q", got, "hello\n")
	}

	// With the capability it takes the wire path: the nil conn panics.
	a.clientCaps.Fs.ReadTextFile = true
	func() {
		defer func() { _ = recover() }()
		if _, err := fsRead(a, ctx, s.ID, path, nil, nil); err == nil {
			t.Error("fsRead with fs.readTextFile advertised took the disk path, want the ACP wire")
		}
	}()
}

// An ambiguous start or a missing end changes nothing and says why.
func TestEditFileByAnchors(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "w.rs")
	src := "fn a() {}\n\nfn target() {\n    one();\n    two();\n} // end target\n\nfn b() {}\n"
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	edit := func(args string) (string, bool) {
		var tc toolCall
		tc.Function.Name = "edit_file"
		tc.Function.Arguments = args
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		return tu.Output, tu.Failed
	}
	out, failed := edit(fmt.Sprintf(`{"path":%q,"start":"fn target()","end":"// end target","new_text":"fn target() {\n    three();\n}"}`, path))
	if failed || !strings.Contains(out, "lines 3-6 replaced") {
		t.Fatalf("anchor edit = %v %s", failed, out)
	}
	got, _ := os.ReadFile(path)
	if want := "fn a() {}\n\nfn target() {\n    three();\n}\n\nfn b() {}\n"; string(got) != want {
		t.Errorf("file =\n%s\nwant\n%s", got, want)
	}
	if out, failed := edit(fmt.Sprintf(`{"path":%q,"start":"fn ","end":"}","new_text":"x"}`, path)); !failed || !strings.Contains(out, "on 3 lines") {
		t.Errorf("ambiguous start = %v %s", failed, out)
	}
	if out, failed := edit(fmt.Sprintf(`{"path":%q,"start":"three();","end":"nowhere","new_text":"x"}`, path)); !failed || !strings.Contains(out, "`end`") {
		t.Errorf("missing end on a line that opens no block = %v %s", failed, out)
	}
	if out, failed := edit(fmt.Sprintf(`{"path":%q,"start":"fn b()","end":"}","new_text":"fn b() { b(); }"}`, path)); failed || !strings.Contains(out, "lines 7-7 replaced, its end found by braces") {
		t.Errorf("a one-line block ended by its brace = %v %s", failed, out)
	}
	if out, failed := edit(fmt.Sprintf(`{"path":%q,"new_text":"x"}`, path)); !failed || !strings.Contains(out, "either") {
		t.Errorf("neither form = %v %s", failed, out)
	}
}

// Edits chain in order and write once; one failing edit writes nothing and is
// named.
func TestEditFileSeveralEdits(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "w.rs")
	src := "let zoom = 1.0;\n\nfn wire_zoom() {\n    old();\n} // wire_zoom\n"
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	edit := func(args string) (string, bool) {
		var tc toolCall
		tc.Function.Name = "edit_file"
		tc.Function.Arguments = args
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		return tu.Output, tu.Failed
	}
	out, failed := edit(fmt.Sprintf(`{"path":%q,"edits":[{"old_text":"let zoom = 1.0;","new_text":"let zoom = ZOOM;"},{"start":"fn wire_zoom(","end":"// wire_zoom","new_text":"fn wire_zoom() {\n    new();\n}"}]}`, path))
	if failed || !strings.Contains(out, "all 2 edits applied in order") {
		t.Fatalf("edits = %v %s", failed, out)
	}
	got, _ := os.ReadFile(path)
	if want := "let zoom = ZOOM;\n\nfn wire_zoom() {\n    new();\n}\n"; string(got) != want {
		t.Errorf("file =\n%s\nwant\n%s", got, want)
	}
	out, failed = edit(fmt.Sprintf(`{"path":%q,"edits":[{"old_text":"let zoom = ZOOM;","new_text":"let zoom = 2.0;"},{"old_text":"not in the file at all","new_text":"x"}]}`, path))
	if !failed || !strings.Contains(out, "edit 2 of 2 failed, so NOTHING was written") {
		t.Errorf("failing list = %v %s", failed, out)
	}
	if after, _ := os.ReadFile(path); string(after) != string(got) {
		t.Errorf("a failed list changed the file:\n%s", after)
	}
	// The list sent as a string holding its JSON is taken as the list; one cut off writes nothing,
	// not even its whole edits: a list is all or none.
	out, failed = edit(fmt.Sprintf(`{"path":%q,"edits":%q}`, path, `[{"old_text":"let zoom = ZOOM;","new_text":"let zoom = 3.0;"}]`))
	if after, _ := os.ReadFile(path); failed || !strings.Contains(string(after), "let zoom = 3.0;") {
		t.Errorf("edits as a string = %v %s\n%s", failed, out, after)
	}
	out, failed = edit(fmt.Sprintf(`{"path":%q,"edits":%q}`, path, `[{"old_text":"let zoom = 3.0;","new_text":"let zoom = 4.0;"}, {"old_text": "let zoom`))
	if after, _ := os.ReadFile(path); !failed || !strings.Contains(out, "a list that is cut off. Nothing was written") || !strings.Contains(string(after), "let zoom = 3.0;") {
		t.Errorf("edits as a cut string = %v %s\n%s", failed, out, after)
	}
}

// Calls batched in one reply, a tool in between, another file, a failed call or
// an existing list get no note.
func TestBatchHints(t *testing.T) {
	_, s := newTestAgent(t)
	call := func(name, args string, failed bool) (string, string) {
		s.markReplyStart() // one call per reply, the unbatched case
		return s.batchHint(name, args, failed)
	}
	s.markReplyStart()
	s.batchHint("edit_file", `{"path":"x.rs","old_text":"p","new_text":"q"}`, false)
	if got, _ := s.batchHint("edit_file", `{"path":"x.rs","old_text":"r","new_text":"s"}`, false); got != "" {
		t.Errorf("a batched second edit got a note: %q", got)
	}
	s.batchHint("run_command", `{"command":"ls"}`, false)
	call("edit_file", `{"path":"y.rs","old_text":"a","new_text":"b"}`, false)
	if got, _ := call("edit_file", `{"path":"y.rs","old_text":"a","new_text":"b"}`, false); got != "" {
		t.Errorf("an identical repeat got a batching note, which hides the repeat: %q", got)
	}

	call("run_command", `{"command":"ls"}`, false)
	if got, _ := call("edit_file", `{"path":"w.rs","old_text":"a","new_text":"b"}`, false); got != "" {
		t.Errorf("first edit got a hint: %q", got)
	}
	if got, _ := call("edit_file", `{"path":"other.rs","old_text":"c","new_text":"d"}`, false); got != "" {
		t.Errorf("edits to two files got a hint: %q", got)
	}
	if got, _ := call("edit_file", `{"path":"other.rs","old_text":"x","new_text":"y"}`, true); got != "" {
		t.Errorf("a failed edit got a hint: %q", got)
	}
	call("edit_file", `{"path":"w.rs","old_text":"a","new_text":"b"}`, false)
	got, told := call("edit_file", `{"path":"w.rs","start":"fn z(","end":"} // z","new_text":"fn z() {}"}`, false)
	if !strings.Contains(got, `{"path": "w.rs", "edits": [{"new_text":"b","old_text":"a"}, {"end":"} // z","new_text":"fn z() {}","start":"fn z("}]}`) || !strings.Contains(told, "several edits to w.rs") {
		t.Errorf("edit hint = %q / %q", got, told)
	}
}

// edit_file strips the `N|` numbers from a snippet copied with them.
func TestEditFileStripsLineNumbers(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "w.rs")
	if err := os.WriteFile(path, []byte("fn a() {}\n\nfn target() {\n    one();\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	run := func(name, args string) (string, bool) {
		var tc toolCall
		tc.Function.Name, tc.Function.Arguments = name, args
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		return tu.Output, tu.Failed
	}
	out, failed := run("edit_file", fmt.Sprintf(`{"path":%q,"old_text":"4|    one();","new_text":"4|    two();"}`, path))
	if failed || !strings.Contains(out, "line-number prefixes") {
		t.Fatalf("numbered old_text = %v %s", failed, out)
	}
	if got, _ := os.ReadFile(path); !strings.Contains(string(got), "    two();") || strings.Contains(string(got), "4|") {
		t.Errorf("file after the numbered edit:\n%s", got)
	}
	if _, ok := stripLineNumbers("let x = 1;\n2|y"); ok {
		t.Error("stripped numbers from a snippet where not every line had one")
	}
}

// Over budget, the project brief may shrink or stay, never grow; elsewhere the
// same name is an ordinary file.
func TestAgentsFileMayNotGrowOverBudget(t *testing.T) {
	a, s := newTestAgent(t)
	brief := filepath.Join(s.Cwd, "AGENT.md")
	body := "# Brief\n" + strings.Repeat("- a line the next agent needs\n", 450) // about 13 KB
	if err := os.WriteFile(brief, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	edit := func(path, oldText, newText string) string {
		var tc toolCall
		tc.Function.Name = "edit_file"
		b, _ := json.Marshal(map[string]string{"path": path, "old_text": oldText, "new_text": newText})
		tc.Function.Arguments = string(b)
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		return tu.Output
	}
	if out := edit(brief, "# Brief\n", "# Brief\n- F4.2 added the call module\n"); !strings.HasPrefix(out, "refused:") {
		t.Errorf("growing an over-budget brief was allowed: %q", out)
	}
	if out := edit(brief, "# Brief\n- a line the next agent needs\n", "# Brief\n"); !strings.HasPrefix(out, "file written") {
		t.Errorf("shrinking it was refused: %q", out)
	}
	nested := filepath.Join(s.Cwd, "docs", "AGENT.md")
	if err := os.MkdirAll(filepath.Dir(nested), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(nested, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if out := edit(nested, "# Brief\n", "# Brief\n- more\n"); !strings.HasPrefix(out, "file written") {
		t.Errorf("a docs/AGENT.md is not the brief, but was refused: %q", out)
	}
	// While /spec runs the brief is read-only, even an edit that shrinks it.
	s.setSpecFence(filepath.Join(s.Cwd, "spec"))
	if out := edit(brief, "# Brief\n", "# The brief\n"); !strings.Contains(out, "read-only while /spec runs") {
		t.Errorf("an AGENT.md edit during /spec went through: %q", out)
	}
}

// write_file never erases a file over a non-string content, creates a new file,
// and tells the model when the file changed on disk since codehalter wrote it.
func TestWriteFile(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "a.txt")
	read := func(p string) string {
		t.Helper()
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	if tu := callTool(t, a, s.ID, "write_file", `{"path":"a.txt","content":"v1\n"}`); tu.Failed || read(path) != "v1\n" {
		t.Fatalf("new file: %+v", tu)
	}
	if tu := callTool(t, a, s.ID, "write_file", `{"path":"a.txt","content":123}`); !strings.Contains(tu.Output, "must be a JSON string") || read(path) != "v1\n" {
		t.Errorf("a number as content: %q, file %q; want refused and the file kept", tu.Output, read(path))
	}
	if err := os.WriteFile(path, []byte("changed outside\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if tu := callTool(t, a, s.ID, "write_file", `{"path":"a.txt","content":"v2\n"}`); !strings.Contains(tu.Output, "changed on disk") || read(path) != "v2\n" {
		t.Errorf("after an outside change: %q, file %q; want written with the drift note", tu.Output, read(path))
	}
	if tu := callTool(t, a, s.ID, "write_file", `{"path":"a.txt","content":"v3\n"}`); strings.Contains(tu.Output, "changed on disk") {
		t.Error("the drift note came again for a change already reported")
	}
}

// A model that drops the leading "/" of an absolute path inside the project still
// reaches the file.
func TestToolPathWithoutLeadingSlash(t *testing.T) {
	a, s := newTestAgent(t)
	if err := os.WriteFile(filepath.Join(s.Cwd, "a.txt"), []byte("found\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rel := strings.TrimPrefix(filepath.Join(s.Cwd, "a.txt"), "/")
	if tu := callTool(t, a, s.ID, "edit_file", `{"path":"`+rel+`","old_text":"found","new_text":"changed"}`); tu.Failed {
		t.Errorf("edit of %q: %q", rel, tu.Output)
	}
	if got, _ := os.ReadFile(filepath.Join(s.Cwd, "a.txt")); string(got) != "changed\n" {
		t.Errorf("a.txt = %q", got)
	}
}

// A function's last line is a bare `}`, which also closes its inner blocks, so the model
// cannot name it: given the bare brace, or a marker that is on no line (the model's own
// "} // placeholder-not-used"), the block opening on the start line is replaced through
// its own closing line, found by braces or, for Python, by indentation.
func TestEditFileBlockEndsWhereItCloses(t *testing.T) {
	a, s := newTestAgent(t)
	edit := func(name, src, args string) (string, string, bool) {
		t.Helper()
		path := filepath.Join(s.Cwd, name)
		if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
		var tc toolCall
		tc.Function.Name = "edit_file"
		tc.Function.Arguments = fmt.Sprintf(`{"path":%q,`, path) + args
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		got, _ := os.ReadFile(path)
		return tu.Output, string(got), tu.Failed
	}
	rs := "fn a() {}\n\npub fn trim(t: f64) -> Option<f64> {\n    if t < 0.0 {\n        return None;\n    }\n    Some(t)\n}\n\nfn b() {}\n"
	want := "fn a() {}\n\npub fn trim(t: f64) -> Option<f64> {\n    Some(t.max(0.0))\n}\n\nfn b() {}\n"
	for _, end := range []string{"} // placeholder-not-used", "}"} {
		out, got, failed := edit("w.rs", rs, `"start":"pub fn trim(t: f64) -> Option<f64> {","end":`+strconv.Quote(end)+`,"new_text":"pub fn trim(t: f64) -> Option<f64> {\n    Some(t.max(0.0))\n}"}`)
		if failed || got != want || !strings.Contains(out, "lines 3-8 replaced, its end found by braces") {
			t.Errorf("end %q: failed=%v %s\nfile:\n%s", end, failed, out, got)
		}
	}
	// Any language: the closing keyword, a body's own braces, a signature over several lines.
	for _, c := range []struct{ name, file, src, start, end, newText, want, how string }{
		{"ruby", "w.rb", "def a\nend\n\ndef trim(t)\n  if t < 0\n    return nil\n  end\n  t\nend\n\ndef b\nend\n",
			"def trim(t)", "end", "def trim(t)\n  t.abs\nend", "def a\nend\n\ndef trim(t)\n  t.abs\nend\n\ndef b\nend\n", "indentation"},
		{"lua", "w.lua", "function trim(t)\n  if t < 0 then\n    return nil\n  end\n  return t\nend\nprint(1)\n",
			"function trim(t)", "end", "function trim(t)\n  return math.abs(t)\nend", "function trim(t)\n  return math.abs(t)\nend\nprint(1)\n", "indentation"},
		{"shell", "w.sh", "if [ -f x ]; then\n  for f in a b; do\n    echo $f\n  done\nfi\necho end\n",
			"if [ -f x ]", "fi", "if [ -f y ]; then\n  echo y\nfi", "if [ -f y ]; then\n  echo y\nfi\necho end\n", "indentation"},
		{"python dict in the body", "d.py", "def f():\n    x = {\n        'a': 1,\n    }\n    return x\n\ny = 2\n",
			"def f():", "# end f", "def f():\n    return {'a': 1}", "def f():\n    return {'a': 1}\n\ny = 2\n", "indentation"},
		{"rust, signature over lines, braces in a char and a comment", "s.rs", "pub fn trim(\n    t: f64,\n) -> f64 {\n    let open = '{';\n    /* { */\n    t\n}\nfn b() {}\n",
			"pub fn trim(", "}", "pub fn trim(t: f64) -> f64 {\n    t\n}", "pub fn trim(t: f64) -> f64 {\n    t\n}\nfn b() {}\n", "braces"},
		{"c, a for loop's semicolons", "l.c", "void f(int n) {\n    for (int i = 0; i < n; i++) {\n        g(i);\n    }\n}\nint x;\n",
			"for (int i = 0; i < n; i++) {", "}", "    for (int i = 0; i < n; i++) g(i);", "void f(int n) {\n    for (int i = 0; i < n; i++) g(i);\n}\nint x;\n", "braces"},
		{"a bare word is a word", "w.rs", "fn f() {\n    stopwatch();\n    stop();\n    after();\n}\n",
			"stopwatch();", "stop", "    go();", "fn f() {\n    go();\n    after();\n}\n", ""},
	} {
		out, got, failed := edit(c.file, c.src, `"start":`+strconv.Quote(c.start)+`,"end":`+strconv.Quote(c.end)+`,"new_text":`+strconv.Quote(c.newText)+`}`)
		if failed || got != c.want || !strings.Contains(out, c.how) {
			t.Errorf("%s: failed=%v %s\nfile:\n%s", c.name, failed, out, got)
		}
	}

	py := "def a():\n    return 1\n\ndef trim(t):\n    if t < 0:\n        return None\n    return t\n\ndef b():\n    pass\n"
	out, got, failed := edit("w.py", py, `"start":"def trim(t):","end":"# end trim","new_text":"def trim(t):\n    return max(t, 0)"}`)
	if failed || got != "def a():\n    return 1\n\ndef trim(t):\n    return max(t, 0)\n\ndef b():\n    pass\n" || !strings.Contains(out, "indentation") {
		t.Errorf("python block: failed=%v %s\nfile:\n%s", failed, out, got)
	}
}

// Where a block ends, across languages and the shapes that trip a naive count: else and
// catch chains, braces in strings, char literals and comments, comments and here-documents
// at column 0, a variable named end, decorators. "error" is a correct refusal: no block
// opens on the start line and the end is on no line after it.
func TestBlockEndEdgeCases(t *testing.T) {
	for _, c := range []struct{ name, src, start, end, want string }{
		{"ruby << append is no heredoc", "def f(list)\n  list << item\n  x = 1<<y\n  list\nend\nputs 1\n", "def f(list)", "end", "<NEW>\nputs 1\n"},
		{"ruby squiggly heredoc", "def sql\n  <<~SQL\nselect 1\n  SQL\nend\nputs 1\n", "def sql", "end", "<NEW>\nputs 1\n"},
		{"shell heredoc at column 0", "if [ -f x ]; then\n  cat <<EOF\nfi in text\n}\nEOF\n  echo x\nfi\necho y\n", "if [ -f x ]", "fi", "<NEW>\necho y\n"},
		{"python docstring at column 0", "def f():\n    s = \"\"\"\nline at col 0\nend = 1\n\"\"\"\n    return s\n\ny = 2\n", "def f():", "nowhere", "<NEW>\n\ny = 2\n"},
		{"python one-line docstring", "def f():\n    \"\"\"Doc.\"\"\"\n    return 1\n\ny = 2\n", "def f():", "nowhere", "<NEW>\n\ny = 2\n"},
		{"js if/else chain, end }", "function f(x) {\n  if (x) {\n    a();\n  } else {\n    b();\n  }\n  c();\n}\n", "if (x) {", "}", "function f(x) {\n<NEW>\n  c();\n}\n"},
		{"java try/catch, invented end", "void f() {\n  try {\n    a();\n  } catch (E e) {\n    b();\n  }\n  c();\n}\n", "try {", "} // end try", "void f() {\n<NEW>\n  c();\n}\n"},
		{"do/while", "void f() {\n  do {\n    a();\n  } while (x);\n  c();\n}\n", "do {", "}", "void f() {\n<NEW>\n  c();\n}\n"},
		{"allman braces", "void f()\n{\n    a();\n}\nvoid g() {}\n", "void f()", "}", "<NEW>\nvoid g() {}\n"},
		{"escaped quote in string", "fn f() {\n    let s = \"a\\\"{\";\n    g();\n}\nfn h() {}\n", "fn f() {", "}", "<NEW>\nfn h() {}\n"},
		{"char literals", "fn f() {\n    let a = '}';\n    let b = '\\'';\n    let c = '\\\\';\n}\nfn h() {}\n", "fn f() {", "}", "<NEW>\nfn h() {}\n"},
		{"lifetimes and a label", "fn f<'a>(x: &'a str) -> &'a str {\n    'outer: loop { break 'outer; }\n    x\n}\nfn h() {}\n", "fn f<'a>", "}", "<NEW>\nfn h() {}\n"},
		{"python comment at column 0 in body", "def f():\n    a = 1\n# note\n    return a\n\ny = 2\n", "def f():", "# end", "<NEW>\n\ny = 2\n"},
		{"python if/else", "def f(x):\n    if x:\n        a()\n    else:\n        b()\n    c()\n", "    if x:", "nowhere", "def f(x):\n<NEW>\n    c()\n"},
		{"python try/except/finally", "try:\n    a()\nexcept E:\n    b()\nfinally:\n    c()\nd()\n", "try:", "nowhere", "<NEW>\nd()\n"},
		{"variable named end after a block", "def f():\n    x = 1\nend = 5\n", "def f():", "nowhere", "<NEW>\nend = 5\n"},
		{"ruby if/elsif/else/end", "def f(x)\n  if x\n    a\n  elsif y\n    b\n  else\n    c\n  end\n  d\nend\n", "  if x", "end", "def f(x)\n<NEW>\n  d\nend\n"},
		{"lua end)", "f(function()\n  a()\nend)\nb()\n", "f(function()", "end)", "<NEW>\nb()\n"},
		{"python comment with brace at base", "def f():\n    a = 1\n# { stray\n    return a\n\ny = 2\n", "def f():", "nowhere", "<NEW>\n\ny = 2\n"},
		{"css one-liners with # colours", "a {\n  color: #fff;\n}\nb { color: #000; }\n", "a {", "}", "<NEW>\nb { color: #000; }\n"},
		{"shell comment with quote", "if [ -f x ]; then\n  # it's here {\n  echo x\nfi\necho y\n", "if [ -f x ]", "fi", "<NEW>\necho y\n"},
		{"multi-line call, end )", "fn f() {\n    foo(\n        a,\n        b,\n    );\n    bar();\n}\n", "    foo(", ")", "fn f() {\n<NEW>\n    bar();\n}\n"},
		{"start inside a block, end }", "fn f() {\n    let x = 1;\n    let y = 2;\n}\nfn g() {}\n", "    let x = 1;", "}", "fn f() {\n<NEW>\nfn g() {}\n"},
		{"go multi-line string with brace", "func f() {\n\ts := `a {\nb`\n\tg()\n}\nfunc h() {}\n", "func f() {", "}", "<NEW>\nfunc h() {}\n"},
		{"crlf", "fn f() {\r\n    a();\r\n}\r\nfn g() {}\r\n", "fn f() {", "}", "<NEW>\nfn g() {}\r\n"},
		{"tabs", "func f() {\n\tif x {\n\t\ta()\n\t}\n}\nfunc g() {}\n", "func f() {", "}", "<NEW>\nfunc g() {}\n"},
		{"yaml key", "a:\n  b: 1\n  c:\n    d: 2\ne: 3\n", "a:", "nowhere", "<NEW>\ne: 3\n"},
		{"python decorator start", "@cache\ndef f():\n    return 1\n\ny = 2\n", "@cache", "nowhere", "<NEW>\n\ny = 2\n"},
		{"elixir do/end", "defmodule M do\n  def f(x) do\n    x\n  end\nend\n", "  def f(x) do", "end", "defmodule M do\n<NEW>\nend\n"},
		{"} then a comment with {", "fn f() {\n    a();\n} // { trailing\nfn g() {}\n", "fn f() {", "}", "<NEW>\nfn g() {}\n"},
		{"else if chain", "if (a) {\n  x();\n} else if (b) {\n  y();\n} else {\n  z();\n}\nw();\n", "if (a) {", "}", "<NEW>\nw();\n"},
		{"shell function, # } in a comment", "f() {\n  # } not the end\n  echo hi\n}\ng() {\n  echo\n}\n", "f() {", "}", "<NEW>\ng() {\n  echo\n}\n"},
		{"shell case/esac", "case $x in\n  a) foo;;\n  b) bar;;\nesac\necho done\n", "case $x in", "esac", "<NEW>\necho done\n"},
		{"ruby do block", "items.each do |x|\n  puts x\nend\nputs 1\n", "items.each do |x|", "end", "<NEW>\nputs 1\n"},
		{"python one-line def", "def f(): return 1\ny = 2\n", "def f(): return 1", "nowhere", "error"},
		{"rust impl inner method", "impl T {\n    fn a(&self) {\n        if x {\n            b();\n        }\n    }\n\n    fn c(&self) {}\n}\n", "    fn a(&self) {", "}", "impl T {\n<NEW>\n\n    fn c(&self) {}\n}\n"},
		{"json-ish closings", "const o = {\n  a: {\n    b: 1,\n  },\n  c: 2,\n};\nf();\n", "const o = {", "};", "<NEW>\nf();\n"},
		{"promise chain", "load()\n  .then(() => {\n    a();\n  })\n  .catch(() => {\n    b();\n  });\nnext();\n", "  .then(() => {", "})", "load()\n<NEW>\n  .catch(() => {\n    b();\n  });\nnext();\n"},
		{"start on the last line", "fn f() {}\nlet x = 1;", "let x = 1;", "}", "error"},
		{"kotlin when", "fun f(x: Int) = when (x) {\n    1 -> \"a\"\n    else -> \"b\"\n}\nfun g() {}\n", "fun f(x: Int) = when (x) {", "}", "<NEW>\nfun g() {}\n"},
		{"swift guard else", "func f() {\n    guard x else {\n        return\n    }\n    a()\n}\nfunc g() {}\n", "func f() {", "}", "<NEW>\nfunc g() {}\n"},
		{"c macro line with brace", "#define X {\nint f() {\n    return 1;\n}\nint g() {}\n", "int f() {", "}", "#define X {\n<NEW>\nint g() {}\n"},
		{"php # comment with brace", "<?php\nfunction f() {\n    # } note\n    return 1;\n}\nfunction g() {}\n", "function f() {", "}", "<?php\n<NEW>\nfunction g() {}\n"},
	} {
		a, s := newTestAgent(t)
		path := filepath.Join(s.Cwd, "f.src")
		if err := os.WriteFile(path, []byte(c.src), 0o644); err != nil {
			t.Fatal(err)
		}
		var tc toolCall
		tc.Function.Name = "edit_file"
		tc.Function.Arguments = fmt.Sprintf(`{"path":%q,"start":%s,"end":%s,"new_text":"<NEW>"}`, path, strconv.Quote(c.start), strconv.Quote(c.end))
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		b, _ := os.ReadFile(path)
		got := string(b)
		if tu.Failed {
			got = "error"
		}
		if got != c.want {
			t.Errorf("%s:\ngot  %q\nwant %q (%s)", c.name, got, c.want, tu.Output)
		}
	}
}
