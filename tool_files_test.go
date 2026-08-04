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
	s.Depth = 1 // force directRead (disk) instead of the ACP wire
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
	if got := s.readCursor[path]; got != 151 {
		t.Errorf("cursor after chunk 1 = %d, want 151", got)
	}

	out, _ = a.serveRead(ctx, s.ID, path, s.readCursor[path], readChunkLines, "tc2")
	if !strings.Contains(out, "L151\n") || !strings.Contains(out, "L300\n") {
		t.Errorf("chunk 2 should be lines 151-300:\n%s", out)
	}
	if got := s.readCursor[path]; got != 301 {
		t.Errorf("cursor after chunk 2 = %d, want 301", got)
	}

	out, _ = a.serveRead(ctx, s.ID, path, s.readCursor[path], readChunkLines, "tc3")
	if !strings.Contains(out, "L350\n") {
		t.Errorf("final chunk missing last line:\n%s", out)
	}
	if !strings.Contains(out, "end of file") {
		t.Errorf("final chunk should be marked complete:\n%s", out)
	}
	if _, ok := s.readCursor[path]; ok {
		t.Errorf("cursor should be cleared at EOF, still %d", s.readCursor[path])
	}
}

// TestServeReadCompleteBoundary pins the off-by-one the line count guards:
// exactly readChunkLines lines is complete (served == max, not >), one more
// is partial.
func TestServeReadCompleteBoundary(t *testing.T) {
	a, s := newTestAgent(t)
	s.Depth = 1
	ctx := context.Background()

	exact := filepath.Join(s.Cwd, "exact.txt")
	writeLines(t, exact, readChunkLines)
	out, _ := a.serveRead(ctx, s.ID, exact, 1, readChunkLines, "tc")
	if !strings.Contains(out, "end of file") {
		t.Errorf("exactly readChunkLines should be complete:\n%s", out)
	}
	if _, ok := s.readCursor[exact]; ok {
		t.Errorf("no cursor expected for a complete read")
	}

	over := filepath.Join(s.Cwd, "over.txt")
	writeLines(t, over, readChunkLines+1)
	out, _ = a.serveRead(ctx, s.ID, over, 1, readChunkLines, "tc")
	if !strings.Contains(out, "the file continues") {
		t.Errorf("readChunkLines+1 should be partial:\n%s", out)
	}
	if got := s.readCursor[over]; got != readChunkLines+1 {
		t.Errorf("cursor = %d, want %d", got, readChunkLines+1)
	}
}

// TestServeReadDedupOnUnchangedReread pins the dedup note: re-reading the same
// window of an unchanged file still returns the bytes but leads with the
// unchanged marker runToolLoop scans for.
func TestServeReadDedupOnUnchangedReread(t *testing.T) {
	a, s := newTestAgent(t)
	s.Depth = 1
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
	s.Depth = 1
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
	s.Depth = 1
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

// TestEditFileMissFailsAndSteers pins the edit_file recovery contract: a missed
// old_text reports failed=true (so it feeds the loop's fail cap) and steers the
// model to read the region and retry a small edit — never rewrite the whole
// file. A successful edit stays failed=false.
func TestEditFileMissFailsAndSteers(t *testing.T) {
	a, s := newTestAgent(t)
	s.Depth = 1 // direct disk I/O instead of the ACP wire
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
	s.Depth = 1 // direct disk I/O instead of the ACP wire
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
