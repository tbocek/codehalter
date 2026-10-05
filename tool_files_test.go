package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"
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

// Follows each "file continues" note's call to EOF: window clip, partial and
// complete markers, and the next-part call.
func TestServeReadChunks(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "big.txt")
	writeLines(t, path, 350)
	ctx := context.Background()

	out, failed := a.serveRead(ctx, s.ID, path, 1, readChunkLines, "tc1", false)
	if failed {
		t.Fatalf("chunk 1 failed: %s", out)
	}
	if !strings.Contains(out, "L1\n") || !strings.Contains(out, "L150\n") {
		t.Errorf("chunk 1 missing lines 1-150:\n%s", out)
	}
	if strings.Contains(out, "L151") {
		t.Errorf("chunk 1 leaked line 151 — window not clipped:\n%s", out)
	}
	if !strings.Contains(out, "the file continues") {
		t.Errorf("chunk 1 should be marked partial:\n%s", out)
	}
	for i, want := range []struct {
		start      int
		first, end string
	}{{151, "L151\n", "L300\n"}, {301, "L301\n", "L350\n"}} {
		m := nextReadRe.FindStringSubmatch(out)
		if m == nil || m[1] != fmt.Sprintf(`{"path": %q, "start_line": %d}`, path, want.start) {
			t.Fatalf("chunk %d: the note names %v, want start_line %d:\n%s", i+1, m, want.start, out)
		}
		var tc toolCall
		tc.Function.Name = "read_file"
		tc.Function.Arguments = m[1]
		tu, _ := a.runToolCall(ctx, s.ID, tc)
		out = tu.Output
		if !strings.Contains(out, want.first) || !strings.Contains(out, want.end) || strings.Contains(out, fmt.Sprintf("L%d\n", want.start-1)) {
			t.Errorf("chunk %d should start at %q and reach %q:\n%s", i+2, want.first, want.end, out)
		}
	}
	if !strings.Contains(out, "end of file") || nextReadRe.MatchString(out) {
		t.Errorf("final chunk should be marked complete, with no next read:\n%s", out)
	}
}

// Exactly readChunkLines lines is complete; one more is partial and names the
// next line.
func TestServeReadCompleteBoundary(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()

	exact := filepath.Join(s.Cwd, "exact.txt")
	writeLines(t, exact, readChunkLines)
	out, _ := a.serveRead(ctx, s.ID, exact, 1, readChunkLines, "tc", false)
	if !strings.Contains(out, "end of file") || nextReadRe.MatchString(out) {
		t.Errorf("exactly readChunkLines should be complete:\n%s", out)
	}

	over := filepath.Join(s.Cwd, "over.txt")
	writeLines(t, over, readChunkLines+1)
	out, _ = a.serveRead(ctx, s.ID, over, 1, readChunkLines, "tc", false)
	if !strings.Contains(out, "the file continues") || !strings.Contains(out, fmt.Sprintf(`"start_line": %d}`, readChunkLines+1)) {
		t.Errorf("readChunkLines+1 should be partial, continuing at line %d:\n%s", readChunkLines+1, out)
	}
}

// An oversized read stops on a line boundary under liveExemptCap so its note
// survives; numbered lines count too, and a single huge line is cut inside.
func TestServeReadByteCapKeepsNote(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()
	line := strings.Repeat("x", 199) + "\n"
	wide := filepath.Join(s.Cwd, "wide.txt")
	if err := os.WriteFile(wide, []byte(strings.Repeat(line, 1000)), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, numbered := range []bool{false, true} {
		out, _ := a.serveRead(ctx, s.ID, wide, 1, maxReadLines, "tc", numbered)
		if live := liveToolOutput("read_file", "{}", out); live != out {
			t.Errorf("numbered=%v: liveToolOutput clipped the read (%d bytes)", numbered, len(out))
		}
		if !strings.Contains(out, "the file continues") || !nextReadRe.MatchString(out) {
			t.Errorf("numbered=%v: the note naming the next part is missing:\n%s", numbered, out[max(0, len(out)-600):])
		}
	}
	out, _ := a.serveRead(ctx, s.ID, wide, 1, maxReadLines, "tc", false)
	n := readByteBudget / len(line)
	if !strings.Contains(out, fmt.Sprintf("showing lines 1-%d,", n)) || !strings.Contains(out, fmt.Sprintf(`"start_line": %d}`, n+1)) {
		t.Errorf("want whole lines 1-%d served and %d next:\n%s", n, n+1, out[max(0, len(out)-600):])
	}

	long := filepath.Join(s.Cwd, "long.txt")
	if err := os.WriteFile(long, []byte(strings.Repeat("é", liveExemptCap)+"\nnext\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	out, _ = a.serveRead(ctx, s.ID, long, 1, readChunkLines, "tc", false)
	if live := liveToolOutput("read_file", "{}", out); live != out || !utf8.ValidString(out) {
		t.Errorf("a cut mega-line must fit whole and stay valid UTF-8 (%d bytes)", len(out))
	}
	if !strings.Contains(out, "line 1 is longer than") || !strings.Contains(out, `"start_line": 2}`) {
		t.Errorf("the cut line should be named and the read continue at line 2:\n%s", out[max(0, len(out)-600):])
	}
}

// JSON-number line/limit, as the schema declares them, and quoted strings both
// take effect.
func TestReadFileHonoursNumericLineAndLimit(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()
	path := filepath.Join(s.Cwd, "big.txt")
	writeLines(t, path, 350)

	read := func(t *testing.T, rawArgs string) string {
		t.Helper()
		var tc toolCall
		tc.Function.Name = "read_file"
		tc.Function.Arguments = rawArgs
		tu, _ := a.runToolCall(ctx, s.ID, tc)
		out, failed := tu.Output, tu.Failed
		if failed {
			t.Fatalf("read_file %s failed: %s", rawArgs, out)
		}
		return out
	}

	numeric := read(t, fmt.Sprintf(`{"path":%q,"line":42,"limit":5}`, path))
	if strings.Contains(numeric, "L1\n") {
		t.Errorf("numeric line=42 read from the top instead:\n%s", numeric)
	}
	for _, want := range []string{"L42\n", "L46\n"} {
		if !strings.Contains(numeric, want) {
			t.Errorf("numeric line/limit missing %q:\n%s", want, numeric)
		}
	}
	if strings.Contains(numeric, "L47\n") {
		t.Errorf("numeric limit=5 served past line 46:\n%s", numeric)
	}

	quoted := read(t, fmt.Sprintf(`{"path":%q,"line":"200","limit":"5"}`, path))
	if !strings.Contains(quoted, "L200\n") || strings.Contains(quoted, "L205\n") {
		t.Errorf("quoted line/limit not honoured:\n%s", quoted)
	}
}

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
	for _, want := range []string{"not found", "read_file", "whole file"} {
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

// A drifted edit_file comes back with the file's current bytes and says NOT to
// re-read.
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
	if !strings.Contains(out, "Do NOT call read_file") {
		t.Errorf("miss message still sends the model back to read_file:\n%s", out)
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

func TestLocateSymbol(t *testing.T) {
	rust := strings.Join([]string{
		"use gtk::prelude::*;",              // 1
		"",                                  // 2
		"/// Builds the form.",              // 3
		"#[allow(dead_code)]",               // 4
		"pub fn cut_form_column(x: i32) {",  // 5
		`    let s = "}{";`,                 // 6
		"    if x > 0 {",                    // 7
		"        println!(\"{}\", x);",      // 8
		"    }",                             // 9
		"}",                                 // 10
		"fn other() { cut_form_column(1) }", // 11
	}, "\n")
	loc := locateSymbol(rust, "fn cut_form_column")
	if loc.start != 3 || loc.end != 10 || loc.how != "braces" {
		t.Errorf("rust = %+v, want 3-10 by braces", loc)
	}

	goSrc := "package x\n\nfunc (r *Runner) Step(n int) error {\n\treturn nil\n}\n"
	if loc := locateSymbol(goSrc, "Step"); loc.start != 3 || loc.end != 5 {
		t.Errorf("go method = %+v, want 3-5", loc)
	}

	py := "import os\n\n@cache\ndef load(path):\n    with open(path) as f:\n\n        return f.read()\n\nx = load('a')\n"
	if loc := locateSymbol(py, "load"); loc.start != 3 || loc.end != 7 || loc.how != "indentation" {
		t.Errorf("python = %+v, want 3-7 by indentation", loc)
	}

	proto := "trait T {\n    fn draw(&self);\n    fn size(&self) -> u32 { 1 }\n}\n"
	if loc := locateSymbol(proto, "draw"); loc.start != 2 || loc.end != 2 {
		t.Errorf("prototype = %+v, want line 2 alone", loc)
	}

	var broken []string
	broken = append(broken, "fn half_written() {")
	for i := 0; i < 80; i++ {
		broken = append(broken, "    step();")
	}
	if loc := locateSymbol(strings.Join(broken, "\n"), "half_written"); loc.end != symbolFallbackLines || !strings.Contains(loc.how, "fallback") {
		t.Errorf("broken = %+v, want the 50-line fallback", loc)
	}

	if loc := locateSymbol("let a = helper(1);\nlet b = helper(2);\n", "helper"); loc.start != 0 || len(loc.mentions) != 2 {
		t.Errorf("undeclared = %+v, want no definition and two mentions", loc)
	}
}

func TestReadFileBySymbol(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "w.rs")
	src := "fn a() {}\n\nfn target() {\n    one();\n    two();\n}\n\nfn b() {}\n"
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	var tc toolCall
	tc.Function.Name = "read_file"
	tc.Function.Arguments = fmt.Sprintf(`{"path":%q,"symbol":"target"}`, path)
	tu, _ := a.runToolCall(context.Background(), s.ID, tc)
	out, failed := tu.Output, tu.Failed
	if failed || !strings.HasPrefix(out, "[`target`: lines 3-6, block end found by braces]") || !strings.Contains(out, "two();") || strings.Contains(out, "fn b()") {
		t.Errorf("symbol read = failed %v:\n%s", failed, out)
	}
	tc.Function.Arguments = fmt.Sprintf(`{"path":%q,"symbol":"missing"}`, path)
	if tu, _ := a.runToolCall(context.Background(), s.ID, tc); !tu.Failed || !strings.Contains(tu.Output, "grep -rn") {
		t.Errorf("unknown symbol = failed %v: %s", tu.Failed, tu.Output)
	}
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
	if out, failed := edit(fmt.Sprintf(`{"path":%q,"start":"fn b()","end":"nowhere","new_text":"x"}`, path)); !failed || !strings.Contains(out, "`end`") {
		t.Errorf("missing end = %v %s", failed, out)
	}
	if out, failed := edit(fmt.Sprintf(`{"path":%q,"new_text":"x"}`, path)); !failed || !strings.Contains(out, "either") {
		t.Errorf("neither form = %v %s", failed, out)
	}
}

// A failing read does not stop the rest; reads past the cap are named as not
// served.
func TestReadFileSeveralReads(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "w.rs")
	if err := os.WriteFile(path, []byte("fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var tc toolCall
	tc.Function.Name = "read_file"
	tc.Function.Arguments = fmt.Sprintf(`{"reads":[{"path":%q,"symbol":"b"},{"path":%q,"symbol":"nope"},{"path":%q,"line":1,"limit":1}]}`, path, path, path)
	tu, _ := a.runToolCall(context.Background(), s.ID, tc)
	out, failed := tu.Output, tu.Failed
	if failed {
		t.Fatalf("reads failed: %s", out)
	}
	for _, want := range []string{"=== read 1 of 3: " + path + " b ===", "two();", "=== read 2 of 3:", "no definition of `nope`", "=== read 3 of 3: " + path + " from line 1 ===", "fn a() {"} {
		if !strings.Contains(out, want) {
			t.Errorf("reads output lacks %q:\n%s", want, out)
		}
	}
	if strings.Index(out, "read 1 of 3") > strings.Index(out, "read 3 of 3") {
		t.Error("reads out of order")
	}
	var many []string
	for i := 0; i < maxReadsPerCall+2; i++ {
		many = append(many, fmt.Sprintf(`{"path":%q,"line":%d,"limit":1}`, path, i+1))
	}
	tc.Function.Arguments = `{"reads":[` + strings.Join(many, ",") + `]}`
	if tu, _ := a.runToolCall(context.Background(), s.ID, tc); !strings.Contains(tu.Output, "not served: at most") {
		t.Errorf("the cap was not reported:\n%s", tu.Output)
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
	s.batchHint("read_file", `{"path":"x.rs","symbol":"p"}`, false)
	if got, _ := s.batchHint("read_file", `{"path":"y.rs","symbol":"q"}`, false); got != "" {
		t.Errorf("a batched second read got a note: %q", got)
	}
	s.batchHint("run_command", `{"command":"ls"}`, false)

	if got, _ := call("read_file", `{"path":"a.rs","symbol":"f"}`, false); got != "" {
		t.Errorf("first read got a hint: %q", got)
	}
	if got, _ := call("read_file", `{"path":"a.rs","symbol":"f"}`, false); got != "" {
		t.Errorf("an identical re-read got a batching note, which hides the repeat: %q", got)
	}
	got, told := call("read_file", `{"path":"b.rs","line":10,"limit":20}`, false)
	if !strings.Contains(got, `{"reads": [{"path":"a.rs","symbol":"f"}, {"limit":20,"line":10,"path":"b.rs"}]}`) || !strings.HasPrefix(told, "💡 told the model:") {
		t.Errorf("read hint = %q / %q", got, told)
	}
	if got, _ := call("read_file", `{"path":"c.rs","symbol":"g"}`, false); got == "" {
		t.Error("the third read in a row got no hint; every occurrence gets one")
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
	got, told = call("edit_file", `{"path":"w.rs","start":"fn z(","end":"} // z","new_text":"fn z() {}"}`, false)
	if !strings.Contains(got, `{"path": "w.rs", "edits": [{"new_text":"b","old_text":"a"}, {"end":"} // z","new_text":"fn z() {}","start":"fn z("}]}`) || !strings.Contains(told, "several edits to w.rs") {
		t.Errorf("edit hint = %q / %q", got, told)
	}
}

// edit_file strips the `N|` numbers from a snippet copied with them.
func TestReadFileNumbered(t *testing.T) {
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
	if out, _ := run("read_file", fmt.Sprintf(`{"path":%q,"line":3,"limit":2,"numbered":true}`, path)); !strings.Contains(out, "3|fn target() {\n4|    one();\n") {
		t.Errorf("numbered window:\n%s", out)
	}
	if out, _ := run("read_file", fmt.Sprintf(`{"path":%q,"symbol":"target","numbered":true}`, path)); !strings.Contains(out, "5|}") {
		t.Errorf("numbered symbol:\n%s", out)
	}
	if out, _ := run("read_file", fmt.Sprintf(`{"reads":[{"path":%q,"line":1,"limit":1,"numbered":true}]}`, path)); !strings.Contains(out, "1|fn a() {}") {
		t.Errorf("numbered reads item:\n%s", out)
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

// Both ends inclusive, like `sed -n 'a,bp'`; view_range too.
func TestReadFileStartEndLine(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "big.txt")
	writeLines(t, path, 50)
	read := func(args string) string {
		var tc toolCall
		tc.Function.Name, tc.Function.Arguments = "read_file", args
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		out := tu.Output
		return out
	}
	if out := read(fmt.Sprintf(`{"path":%q,"start_line":10,"end_line":12}`, path)); !strings.Contains(out, "L10\nL11\nL12\n") || strings.Contains(out, "L13\n") {
		t.Errorf("start/end:\n%s", out)
	}
	if out := read(fmt.Sprintf(`{"path":%q,"view_range":[20,21]}`, path)); !strings.Contains(out, "L20\nL21\n") || strings.Contains(out, "L22\n") {
		t.Errorf("view_range:\n%s", out)
	}
	if out := read(fmt.Sprintf(`{"reads":[{"path":%q,"start_line":30,"end_line":30}]}`, path)); !strings.Contains(out, "from line 30") || !strings.Contains(out, "L30\n") || strings.Contains(out, "L31\n") {
		t.Errorf("reads item:\n%s", out)
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
	if tu := callTool(t, a, s.ID, "read_file", `{"path":"`+rel+`"}`); tu.Failed || !strings.Contains(tu.Output, "found") {
		t.Errorf("read of %q: %q", rel, tu.Output)
	}
}
