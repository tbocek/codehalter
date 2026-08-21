package main

import (
	"context"
	"strings"
	"testing"
)

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
	// tailCap < total <= headCap+tailCap: head+tail stitched, no marker, no dup/gap.
	b = newBoundedOutput(16)
	b.Write([]byte("abcdefghijklmn")) // 14
	if got := b.String(); got != "abcdefghijklmn" {
		t.Errorf("stitch: got %q (want full 14, no marker)", got)
	}
	// Past the cap: head + marker + last tailCap, middle elided.
	b = newBoundedOutput(16)
	b.Write([]byte("abcdefghijklmnopqrst")) // 20
	if got := b.String(); got != "abcd\n[... 4 bytes omitted ...]\nijklmnopqrst" {
		t.Errorf("over: got %q", got)
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

// TestRunCommandEndToEnd pins what run_command hands back and what it leaves in
// the card: the exit code leads the result (a probe exiting non-zero is data,
// not a failure), the output is there, and the card keeps the live terminal
// instead of a static text copy of it.
func TestRunCommandEndToEnd(t *testing.T) {
	h := newTerminalHarness(t)

	result, failed := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"echo hi; exit 2"}`)
	if failed {
		t.Fatalf("failed = true, want false (a non-zero exit must not fail the turn)")
	}
	if !strings.HasPrefix(result, "exit 2\n") {
		t.Errorf("result = %q, want it to start with the exit code", result)
	}
	if !strings.Contains(result, "hi") {
		t.Errorf("result = %q, want the command output", result)
	}

	done := h.waitForStatus("completed")
	if done == nil {
		t.Fatal("the tool call was never completed")
	}
	if _, ok := done["content"]; ok {
		t.Errorf("completing update carries content %v, want it omitted so the terminal stays visible", done["content"])
	}
	if done["title"] != "Run: echo hi; exit 2 (exit 2)" {
		t.Errorf("title = %v, want the exit code in it", done["title"])
	}
	if !h.embeddedTerminal() {
		t.Error("no update embedded the terminal; the user would see an empty card")
	}
}

// TestRunCommandIsAShellLine pins that the model's command keeps shell
// semantics. terminal/create takes an argv, so run_command has to wrap it in
// bash -c; without that, every pipe, redirect and `&&` would break.
func TestRunCommandIsAShellLine(t *testing.T) {
	h := newTerminalHarness(t)

	result, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID,
		`{"command":"echo one && echo two | tr a-z A-Z"}`)
	if !strings.Contains(result, "one") || !strings.Contains(result, "TWO") {
		t.Errorf("result = %q, want both the && and the pipe to have run", result)
	}
}

// TestRunCommandCapsHugeOutput pins the total-output cap: a command that prints
// far more than cmdOutputCap comes back as a bounded head+tail with an elision
// marker, so it can't poison a small-context model — while still running to
// completion, with both its first and last lines intact.
func TestRunCommandCapsHugeOutput(t *testing.T) {
	h := newTerminalHarness(t)
	saved := cmdOutputCap
	cmdOutputCap = 4096
	t.Cleanup(func() { cmdOutputCap = saved })

	result, failed := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"seq 1 20000"}`)
	if failed {
		t.Fatalf("failed=true: %.100s", result)
	}
	if !strings.HasPrefix(result, "exit 0") {
		t.Errorf("missing exit header: %.80s", result)
	}
	if !strings.Contains(result, "bytes omitted") {
		t.Errorf("over-cap output should carry an elision marker: %.200s", result)
	}
	if len(result) > cmdOutputCap+1024 {
		t.Errorf("captured output not bounded: %d bytes (cap %d)", len(result), cmdOutputCap)
	}
	if !strings.Contains(result, "\n1\n") {
		t.Error("head lost: want early line 1")
	}
	if !strings.Contains(result, "\n20000\n") {
		t.Error("tail lost: want final line 20000")
	}
}

// TestRunCommandRequiresCommand pins the empty-args guard, which the local
// models hit often enough to matter.
func TestRunCommandRequiresCommand(t *testing.T) {
	h := newTerminalHarness(t)
	res, failed := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{}`)
	if failed || !strings.Contains(res, "command is required") {
		t.Fatalf("expected command-required error, got: %s (failed=%v)", res, failed)
	}
}
