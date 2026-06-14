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
