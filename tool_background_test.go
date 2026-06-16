package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestRunBackgroundStaysRunning pins the core contract: a long-running command
// returns promptly as a tracked "running" job (not waited on), its log exists,
// and shutdownBackground reaps it and removes the log.
func TestRunBackgroundStaysRunning(t *testing.T) {
	a, s := newTestAgent(t)
	defer a.shutdownBackground() // leak guard if an assertion aborts early
	old := bgJobGrace
	bgJobGrace = 100 * time.Millisecond
	defer func() { bgJobGrace = old }()

	res, failed := runBackgroundExecute(context.Background(), a, s.ID, `{"command":"sleep 5"}`)
	if failed {
		t.Fatalf("run_background marked the turn failed: %s", res)
	}
	if !strings.Contains(res, "running (pid") {
		t.Fatalf("expected a running job, got: %s", res)
	}

	a.bgMu.Lock()
	n := len(a.bgJobs)
	var logPath string
	for _, j := range a.bgJobs {
		logPath = j.logPath
	}
	a.bgMu.Unlock()
	if n != 1 {
		t.Fatalf("expected 1 tracked job, got %d", n)
	}
	if _, err := os.Stat(logPath); err != nil {
		t.Fatalf("log file missing: %v", err)
	}

	a.shutdownBackground()
	if _, err := os.Stat(logPath); !os.IsNotExist(err) {
		t.Errorf("shutdownBackground left the log behind: %v", err)
	}
	a.bgMu.Lock()
	left := len(a.bgJobs)
	a.bgMu.Unlock()
	if left != 0 {
		t.Errorf("shutdownBackground left %d jobs tracked", left)
	}
}

// TestRunBackgroundImmediateExit pins that a command which exits during the grace
// window is reported as a crash (with exit code + captured output) and is not left
// in the job table.
func TestRunBackgroundImmediateExit(t *testing.T) {
	a, s := newTestAgent(t)
	old := bgJobGrace
	bgJobGrace = 3 * time.Second // ample: the select returns as soon as the process exits
	defer func() { bgJobGrace = old }()

	res, failed := runBackgroundExecute(context.Background(), a, s.ID, `{"command":"echo boom; exit 3"}`)
	if failed {
		t.Fatalf("unexpected turn failure: %s", res)
	}
	if !strings.Contains(res, "exited immediately") || !strings.Contains(res, "exit 3") {
		t.Fatalf("expected immediate-exit report with exit 3, got: %s", res)
	}
	if !strings.Contains(res, "boom") {
		t.Errorf("expected captured output 'boom', got: %s", res)
	}
	a.bgMu.Lock()
	n := len(a.bgJobs)
	a.bgMu.Unlock()
	if n != 0 {
		t.Errorf("crashed job left in table: %d", n)
	}
}

func TestRunBackgroundRequiresCommand(t *testing.T) {
	a, s := newTestAgent(t)
	res, failed := runBackgroundExecute(context.Background(), a, s.ID, `{}`)
	if failed || !strings.Contains(res, "command is required") {
		t.Fatalf("expected command-required error, got: %s (failed=%v)", res, failed)
	}
}
