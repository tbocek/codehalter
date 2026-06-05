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
