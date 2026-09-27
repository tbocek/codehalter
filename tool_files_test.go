package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeLines writes n newline-terminated lines ("L1\n".."Ln\n") to path.
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

// TestServeReadChunksAndCursor walks a multi-chunk read the way the model does:
// read_file, then continue_read from the cursor twice, ending at EOF. It pins
// the window clip (no leaked lines past readChunkLines), the partial/complete
// markers, and the cursor advancing then clearing at end of file.
func TestServeReadChunksAndCursor(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "big.txt")
	writeLines(t, path, 350)
	ctx := context.Background()

	out, failed := a.serveRead(ctx, s.ID, path, 1, readChunkLines, "tc1")
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
	if got := s.turn.readCursor[path]; got != 151 {
		t.Errorf("cursor after chunk 1 = %d, want 151", got)
	}

	out, _ = a.serveRead(ctx, s.ID, path, s.turn.readCursor[path], readChunkLines, "tc2")
	if !strings.Contains(out, "L151\n") || !strings.Contains(out, "L300\n") {
		t.Errorf("chunk 2 should be lines 151-300:\n%s", out)
	}
	if got := s.turn.readCursor[path]; got != 301 {
		t.Errorf("cursor after chunk 2 = %d, want 301", got)
	}

	out, _ = a.serveRead(ctx, s.ID, path, s.turn.readCursor[path], readChunkLines, "tc3")
	if !strings.Contains(out, "L350\n") {
		t.Errorf("final chunk missing last line:\n%s", out)
	}
	if !strings.Contains(out, "end of file") {
		t.Errorf("final chunk should be marked complete:\n%s", out)
	}
	if _, ok := s.turn.readCursor[path]; ok {
		t.Errorf("cursor should be cleared at EOF, still %d", s.turn.readCursor[path])
	}
}

// TestServeReadCompleteBoundary pins the off-by-one the line count guards:
// exactly readChunkLines lines is complete (served == max, not >), one more
// is partial.
func TestServeReadCompleteBoundary(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()

	exact := filepath.Join(s.Cwd, "exact.txt")
	writeLines(t, exact, readChunkLines)
	out, _ := a.serveRead(ctx, s.ID, exact, 1, readChunkLines, "tc")
	if !strings.Contains(out, "end of file") {
		t.Errorf("exactly readChunkLines should be complete:\n%s", out)
	}
	if _, ok := s.turn.readCursor[exact]; ok {
		t.Errorf("no cursor expected for a complete read")
	}

	over := filepath.Join(s.Cwd, "over.txt")
	writeLines(t, over, readChunkLines+1)
	out, _ = a.serveRead(ctx, s.ID, over, 1, readChunkLines, "tc")
	if !strings.Contains(out, "the file continues") {
		t.Errorf("readChunkLines+1 should be partial:\n%s", out)
	}
	if got := s.turn.readCursor[over]; got != readChunkLines+1 {
		t.Errorf("cursor = %d, want %d", got, readChunkLines+1)
	}
}

// TestServeReadDedupOnUnchangedReread pins the dedup note: re-reading the same
// window of an unchanged file still returns the bytes but leads with the
// unchanged marker runToolLoop scans for.
func TestServeReadDedupOnUnchangedReread(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()
	path := filepath.Join(s.Cwd, "f.txt")
	writeLines(t, path, 10)

	if _, failed := a.serveRead(ctx, s.ID, path, 1, readChunkLines, "tc1"); failed {
		t.Fatal("first read failed")
	}
	out, _ := a.serveRead(ctx, s.ID, path, 1, readChunkLines, "tc2")
	if !strings.Contains(out, readUnchangedMarker) {
		t.Errorf("re-read of an unchanged window should carry the unchanged marker:\n%s", out)
	}
}

// TestServeReadFreshBytesNotFlagged pins the content-based dedup: when a re-read
// of the same window returns different bytes, it is NOT redundant — even though
// the dedup entry from the prior read still exists. (Rewriting via os.WriteFile
// rather than fsWrite leaves the entry in place, so only the hash comparison
// keeps this from being a false redundant-fetch.)
func TestServeReadFreshBytesNotFlagged(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()
	path := filepath.Join(s.Cwd, "f.txt")
	writeLines(t, path, 10)

	a.serveRead(ctx, s.ID, path, 1, readChunkLines, "tc1")
	writeLines(t, path, 12) // content changes; dedup entry NOT busted
	out, _ := a.serveRead(ctx, s.ID, path, 1, readChunkLines, "tc2")
	if strings.Contains(out, readUnchangedMarker) {
		t.Errorf("a re-read returning fresh bytes must not be flagged redundant:\n%s", out)
	}
}

// TestServeReadRefusesWhenInContext pins the in-context refusal: when the exact
// bytes are already present as a prior read result in the LIVE message window, a
// re-read is REFUSED (the chunk is not re-served — the model scrolls back). A
// read that isn't in the messages (never recorded, or compacted away) is still
// served. serveRead itself doesn't record the ToolUse (runToolCall does), so the
// test seeds s.Messages to simulate the prior read being in context.
func TestServeReadRefusesWhenInContext(t *testing.T) {
	a, s := newTestAgent(t)
	ctx := context.Background()
	path := filepath.Join(s.Cwd, "f.txt")
	writeLines(t, path, 10)

	// Nothing in the message window yet → re-reads are SERVED (carry the bytes).
	out1, _ := a.serveRead(ctx, s.ID, path, 1, readChunkLines, "tc1")
	out2, _ := a.serveRead(ctx, s.ID, path, 1, readChunkLines, "tc2")
	if strings.Contains(out2, "re-read refused") {
		t.Fatalf("a read not in the message window must be served, not refused:\n%s", out2)
	}
	if !strings.Contains(out2, "L2") {
		t.Fatalf("a served read must contain the file content:\n%s", out2)
	}

	// Record the prior read in the live context, as runToolCall would.
	s.Messages = []Message{{Role: "assistant", ToolUses: []ToolUse{{Name: "read_file", Output: out1}}}}

	out3, failed := a.serveRead(ctx, s.ID, path, 1, readChunkLines, "tc3")
	if failed {
		t.Fatalf("an in-context refusal is benign — failed must be false")
	}
	if !strings.Contains(out3, "already in the context") || !strings.Contains(out3, readUnchangedMarker) {
		t.Errorf("re-read with content in context must be refused with the unchanged marker:\n%s", out3)
	}
	if strings.Contains(out3, "L2") {
		t.Errorf("the refusal must NOT re-serve the file bytes:\n%s", out3)
	}
}

// TestReadFileHonoursNumericLineAndLimit is the end-to-end regression for the
// schema/decoder mismatch: read_file declares `line` and `limit` as integers, so
// a model that obeys the schema sends JSON numbers. Decoding those into
// map[string]string used to fail the whole object and leave the numeric keys as
// "", which silently read from line 1 with the default window and reported
// success — the worst kind of wrong, because the model believes it saw line 42.
// The string forms stay accepted, since small models often quote everything.
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
		out, failed := a.executeTool(ctx, s.ID, tc)
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

	// A different window, so the dedup guard doesn't refuse this as a re-read.
	quoted := read(t, fmt.Sprintf(`{"path":%q,"line":"200","limit":"5"}`, path))
	if !strings.Contains(quoted, "L200\n") || strings.Contains(quoted, "L205\n") {
		t.Errorf("quoted line/limit not honoured:\n%s", quoted)
	}
}

// TestEditFileMissFailsAndSteers pins the edit_file recovery contract: a missed
// old_text reports failed=true (so it feeds the loop's fail cap) and steers the
// model to read the region and retry a small edit — never rewrite the whole
// file. A successful edit stays failed=false.
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
	out, failed := a.executeTool(ctx, s.ID, miss)
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
	if _, failed := a.executeTool(ctx, s.ID, hit); failed {
		t.Errorf("successful edit: failed=true, want false")
	}
}

// TestTolerantReplace covers the whitespace-tolerant edit_file fallback: it
// recovers wrong trailing whitespace and wrong indentation (re-indenting
// new_text to the file's column), stays unique-or-fail, and never matches across
// genuinely different content.
func TestTolerantReplace(t *testing.T) {
	// Trailing-whitespace mismatch: file line has a trailing space the snippet lacks.
	file := "func f() {\n\treturn 1 \n}\n"
	old := "func f() {\n\treturn 1\n}"
	out, n := tolerantReplace(file, old, "func f() {\n\treturn 2\n}")
	if n != 1 || !strings.Contains(out, "return 2") {
		t.Fatalf("trailing-ws: n=%d out=%q", n, out)
	}

	// Indentation mismatch: file indents with two tabs, snippet with none; the
	// replacement must be re-indented to the file's two-tab column.
	file = "x\n\t\tcall(a)\n\t\tcall(b)\ny\n"
	old = "call(a)\ncall(b)"
	out, n = tolerantReplace(file, old, "call(a)\ncall(c)")
	if n != 1 {
		t.Fatalf("indent: n=%d", n)
	}
	if !strings.Contains(out, "\t\tcall(c)") || strings.Contains(out, "\ncall(c)") {
		t.Errorf("indent not reapplied to new_text:\n%q", out)
	}

	// Ambiguous: the snippet (ignoring whitespace) matches two windows → no apply.
	file = "a\n  p()\nb\n  p()\nc\n"
	if _, n = tolerantReplace(file, "p()", "q()"); n != 2 {
		t.Errorf("ambiguous: want n=2, got %d", n)
	}

	// No match: genuinely absent content stays absent.
	if out, n = tolerantReplace("alpha\nbeta\n", "gamma", "x"); n != 0 || out != "" {
		t.Errorf("no-match: want n=0 empty, got n=%d out=%q", n, out)
	}
}

// TestNearMiss covers the failed-edit recovery path: when old_text matches
// neither exactly nor ignoring whitespace, find the region it was aiming at so
// the model can retry against real bytes instead of spending a read_file
// round-trip. The negative cases matter as much as the positive one — quoting
// the wrong region would send the model to edit the wrong place.
func TestNearMiss(t *testing.T) {
	file := "package main\n\nfunc load(p string) error {\n\tf, err := os.Open(p)\n\tif err != nil {\n\t\treturn err\n\t}\n\treturn nil\n}\n"

	// One line drifted (the model remembers the old parameter name). The region
	// is still recognisable, so it must be located and quoted verbatim.
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

	// Wholly unrelated text must not be mapped onto some vaguely-similar region.
	if _, _, ok := nearMiss(file, "type Server struct {\n\taddr string\n\tport int\n}"); ok {
		t.Error("unrelated snippet produced a near miss")
	}

	// Boilerplate alone must not anchor: a lone closing brace appears twice and
	// carries no information about which region was meant.
	if _, _, ok := nearMiss("a\n}\nb\n}\nc\n", "}"); ok {
		t.Error("bare boilerplate line produced a near miss")
	}

	// Below the score floor: one line out of four is not "the region you meant".
	if _, _, ok := nearMiss(file, "func load(p string) error {\n\tzzz()\n\tyyy()\n\txxx()"); ok {
		t.Error("sub-threshold overlap produced a near miss")
	}

	// Degenerate inputs must not panic or claim a match.
	for _, old := range []string{"", "\n\n", strings.Repeat("x\n", 100)} {
		if _, _, ok := nearMiss(file, old); ok {
			t.Errorf("degenerate old_text %q produced a near miss", truncate(old, 20))
		}
	}
}

// TestEditFileMissQuotesNearbyRegion pins the end-to-end payoff: a drifted
// edit_file comes back carrying the file's current bytes, and explicitly tells
// the model NOT to re-read — that saved round-trip is the whole point.
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
	out, failed := a.executeTool(ctx, s.ID, miss)
	if !failed {
		t.Error("drifted edit: failed=false, want true (must feed the fail cap)")
	}
	if !strings.Contains(out, "os.Open(p)") {
		t.Errorf("miss message did not quote the current region:\n%s", out)
	}
	if !strings.Contains(out, "Do NOT call read_file") {
		t.Errorf("miss message still sends the model back to read_file:\n%s", out)
	}
	// The file must be untouched by a failed edit.
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(after) != body {
		t.Errorf("failed edit modified the file:\n%s", after)
	}
}

// TestFsGatedOnClientCapabilities pins that a depth-0 session does its own disk
// I/O when the client never advertised the ACP filesystem. ACP forbids sending
// a client a method it didn't claim, and codehalter used to send
// fs/read_text_file to everyone — invisible against Zed, which advertises both,
// and a hard failure against any client that doesn't. a.conn is nil here, so an
// attempted wire call panics rather than silently passing.
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

	// A client advertising the capability takes the wire path instead — with a
	// nil conn that is a panic, which is exactly how we tell the two apart.
	a.clientCaps.Fs.ReadTextFile = true
	func() {
		defer func() { _ = recover() }()
		if _, err := fsRead(a, ctx, s.ID, path, nil, nil); err == nil {
			t.Error("fsRead with fs.readTextFile advertised took the disk path, want the ACP wire")
		}
	}()
}

// TestLocateSymbol pins the language-generic definition finder: braces for
// Rust and Go (a brace inside a string does not count, a Go method's
// receiver is skipped, attributes and doc comments above come along),
// indentation for Python, a prototype ending in ';' as its own block, the
// 50-line fallback for a block that never closes, and mentions when the
// name is used but never declared.
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

// TestReadFileBySymbol: read_file with `symbol` serves the whole definition
// with its line range in the note.
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
	out, failed := a.executeTool(context.Background(), s.ID, tc)
	if failed || !strings.HasPrefix(out, "[`target`: lines 3-6, block end found by braces]") || !strings.Contains(out, "two();") || strings.Contains(out, "fn b()") {
		t.Errorf("symbol read = failed %v:\n%s", failed, out)
	}
	tc.Function.Arguments = fmt.Sprintf(`{"path":%q,"symbol":"missing"}`, path)
	if out, failed := a.executeTool(context.Background(), s.ID, tc); !failed || !strings.Contains(out, "grep -rn") {
		t.Errorf("unknown symbol = failed %v: %s", failed, out)
	}
}

// TestEditFileByAnchors: a block is replaced by its first and last line's
// fragments, whole lines; an ambiguous start or a missing end changes
// nothing and says why.
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
		return a.executeTool(context.Background(), s.ID, tc)
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

// TestReadFileSeveralReads: `reads` serves each target in order under its own
// header, a failing one does not stop the rest, and more than the cap are
// named as not served.
func TestReadFileSeveralReads(t *testing.T) {
	a, s := newTestAgent(t)
	path := filepath.Join(s.Cwd, "w.rs")
	if err := os.WriteFile(path, []byte("fn a() {\n    one();\n}\n\nfn b() {\n    two();\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var tc toolCall
	tc.Function.Name = "read_file"
	tc.Function.Arguments = fmt.Sprintf(`{"reads":[{"path":%q,"symbol":"b"},{"path":%q,"symbol":"nope"},{"path":%q,"line":1,"limit":1}]}`, path, path, path)
	out, failed := a.executeTool(context.Background(), s.ID, tc)
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
	if out, _ := a.executeTool(context.Background(), s.ID, tc); !strings.Contains(out, "not served: at most") {
		t.Errorf("the cap was not reported:\n%s", out)
	}
}

// TestEditFileSeveralEdits: `edits` applies in order, each on the result of
// the one before, and writes once; one failing edit writes nothing and says
// which.
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
		return a.executeTool(context.Background(), s.ID, tc)
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
