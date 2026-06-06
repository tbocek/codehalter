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
