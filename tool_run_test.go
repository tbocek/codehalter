package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestRunCmdIdleTimeout pins the idle watchdog: a command that prints nothing is
// SIGKILLed after cmdIdleTimeout instead of parking the turn until the process
// exits or the user cancels.
func TestRunCmdIdleTimeout(t *testing.T) {
	a, s := newTestAgent(t)
	old := cmdIdleTimeout
	cmdIdleTimeout = 100 * time.Millisecond
	defer func() { cmdIdleTimeout = old }()

	start := time.Now()
	out, _ := runCmdExecute(context.Background(), a, s.ID, `{"command":"sleep 10"}`)
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("silent command should be reaped fast, took %s", elapsed)
	}
	if !strings.Contains(out, "timed out") {
		t.Errorf("expected a timeout note, got: %s", out)
	}
}

// TestRunCommandReapsBackgroundChild pins the foreground-done-but-child-alive
// detection: a command that backgrounds a long-lived process returns PROMPTLY
// (not after the idle timeout), the child is reaped, and the result steers the
// model to run_background. This is the case that used to wedge the turn forever.
func TestRunCommandReapsBackgroundChild(t *testing.T) {
	a, s := newTestAgent(t)
	oldGrace, oldIdle := bgLingerGrace, cmdIdleTimeout
	bgLingerGrace = 20 * time.Millisecond
	cmdIdleTimeout = 30 * time.Second // far longer than the test: prove we don't wait for it
	defer func() { bgLingerGrace, cmdIdleTimeout = oldGrace, oldIdle }()

	start := time.Now()
	out, failed := runCmdExecute(context.Background(), a, s.ID, `{"command":"sleep 30 & echo started"}`)
	elapsed := time.Since(start)

	if failed {
		t.Fatalf("unexpected failed=true: %s", out)
	}
	if elapsed > 5*time.Second {
		t.Fatalf("did not return promptly after the foreground exited: took %s", elapsed)
	}
	if !strings.Contains(out, "started") {
		t.Errorf("expected foreground output 'started', got: %s", out)
	}
	if !strings.Contains(out, "run_background") {
		t.Errorf("expected a run_background hint after reaping the child, got: %s", out)
	}
}

// TestRunCmdCapturesFullOutput pins that the decoupled drain captures a
// high-volume command's full output. The old per-line sendUpdate path streamed
// through a synchronous io.Pipe, so a chatty command could deadlock behind a
// slow editor; the drain never blocks on the editor (nil conn here makes
// sendUpdate a no-op), so the command runs to completion and all of it is
// collected.
func TestRunCmdCapturesFullOutput(t *testing.T) {
	a, s := newTestAgent(t)
	out, failed := runCmdExecute(context.Background(), a, s.ID, `{"command":"seq 1 5000"}`)
	if failed {
		t.Fatalf("run_command returned failed=true: %.100s", out)
	}
	if !strings.HasPrefix(out, "exit 0") {
		t.Errorf("missing exit code header: %.80s", out)
	}
	if !strings.Contains(out, "\n1\n") || !strings.Contains(out, "\n5000\n") {
		t.Error("output not fully captured (want lines 1 and 5000)")
	}
	if n := strings.Count(out, "\n"); n < 5000 {
		t.Errorf("expected >= 5000 newlines, got %d", n)
	}
}

// TestBoundedOutput pins the head+tail capture: small streams come back verbatim,
// streams up to headCap+tailCap stitch without duplicating or dropping the
// overlap, and larger streams keep the head and the last tailCap with an elision
// marker for the middle.
func TestBoundedOutput(t *testing.T) {
	// cap 16 → headCap 4, tailCap 12.

	// Fits entirely (<= tailCap): verbatim, not truncated.
	b := newBoundedOutput(16)
	b.Write([]byte("abcdefghij")) // 10
	if got := b.String(); got != "abcdefghij" {
		t.Errorf("fit: got %q", got)
	}
	if b.truncated() {
		t.Error("fit: should not be truncated")
	}

	// tailCap < total <= headCap+tailCap: head+tail stitched, no marker, no dup/gap.
	b = newBoundedOutput(16)
	b.Write([]byte("abcdefghijklmn")) // 14
	if got := b.String(); got != "abcdefghijklmn" {
		t.Errorf("stitch: got %q (want full 14, no marker)", got)
	}
	if b.truncated() {
		t.Error("stitch: 14 <= headCap+tailCap, not truncated")
	}

	// Past the cap: head + marker + last tailCap, middle elided.
	b = newBoundedOutput(16)
	b.Write([]byte("abcdefghijklmnopqrst")) // 20
	if got := b.String(); got != "abcd\n[... 4 bytes omitted ...]\nijklmnopqrst" {
		t.Errorf("over: got %q", got)
	}
	if !b.truncated() {
		t.Error("over: should be truncated")
	}

	// Byte-at-a-time matches one-shot, and the ring keeps exactly the last tailCap
	// even after crossing the 2*tailCap trim point.
	b = newBoundedOutput(16)
	for _, c := range "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" { // 36 bytes
		b.Write([]byte{byte(c)})
	}
	got := b.String()
	if !strings.HasPrefix(got, "0123") || !strings.HasSuffix(got, "OPQRSTUVWXYZ") {
		t.Errorf("ring: got %q (want head 0123… tail …OPQRSTUVWXYZ)", got)
	}
	if !strings.Contains(got, "bytes omitted") {
		t.Errorf("ring: missing elision marker: %q", got)
	}
}

// TestRunCmdCapsHugeOutput pins the total-output cap: a command that streams far
// more than cmdOutputCap is captured to a bounded head+tail with an elision
// marker (so it can't poison a small-context model or grow memory without limit),
// while the command still runs to completion.
func TestRunCmdCapsHugeOutput(t *testing.T) {
	a, s := newTestAgent(t)
	oldCap, oldEd := cmdOutputCap, editorStreamCap
	cmdOutputCap, editorStreamCap = 4096, 8192
	defer func() { cmdOutputCap, editorStreamCap = oldCap, oldEd }()

	out, failed := runCmdExecute(context.Background(), a, s.ID, `{"command":"seq 1 20000"}`)
	if failed {
		t.Fatalf("failed=true: %.100s", out)
	}
	if !strings.HasPrefix(out, "exit 0") {
		t.Errorf("missing exit header: %.80s", out)
	}
	if !strings.Contains(out, "bytes omitted") {
		t.Errorf("over-cap output should carry an elision marker: %.200s", out)
	}
	if len(out) > cmdOutputCap+1024 {
		t.Errorf("captured output not bounded: %d bytes (cap %d)", len(out), cmdOutputCap)
	}
	if !strings.Contains(out, "\n1\n") {
		t.Error("head lost: want early line 1")
	}
	if !strings.Contains(out, "\n20000\n") {
		t.Error("tail lost: want final line 20000")
	}
}
