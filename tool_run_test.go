package main

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
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

// TestBoundedOutputKeepsUTF8 pins that eliding the middle never splits a
// character at either cut: the output is stored in the session file, where one
// invalid byte made the whole session unloadable.
func TestBoundedOutputKeepsUTF8(t *testing.T) {
	for shift := 0; shift < 3; shift++ {
		b := newBoundedOutput(16)
		b.Write([]byte(strings.Repeat("x", shift) + strings.Repeat("→", 20)))
		if out := b.String(); !utf8.ValidString(out) {
			t.Errorf("shift %d: invalid UTF-8: %q", shift, out)
		}
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

// TestRunCommandRefusesSleepForBackgroundJob: a foreground sleep while one of
// the session's jobs runs is a guessed wait for it; the job wakes the model on
// its own, so the sleep is refused and names the job. With no job running a
// sleep is an ordinary command, and a sleep inside other work is not a wait.
func TestRunCommandRefusesSleepForBackgroundJob(t *testing.T) {
	h := newTerminalHarness(t)
	res, failed := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 0.01"}`)
	if failed || strings.Contains(res, "refused") {
		t.Fatalf("sleep with no job running: %s (failed=%v)", res, failed)
	}
	h.agent.registerBgJob(&backgroundJob{id: 7, sid: h.sess.ID, cmdStr: "cd rust && just test > /tmp/t.log 2>&1"})
	for _, cmd := range []string{"sleep 120", "sleep 90 && tail -20 /tmp/t.log", "cd /workspaces/x; sleep 60", "sleep"} {
		res, failed = runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":`+strconv.Quote(cmd)+`}`)
		if !strings.Contains(res, "refused") || !strings.Contains(res, "job 7") {
			t.Errorf("%q with job 7 running: %s (failed=%v)", cmd, res, failed)
		}
	}
	for _, cmd := range []string{"tail -20 /tmp/t.log", "for i in 1; do sleep 0.01; done", "echo hi; sleep 0.01"} {
		res, failed = runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":`+strconv.Quote(cmd)+`}`)
		if failed || strings.Contains(res, "refused") {
			t.Errorf("%q must run: %s (failed=%v)", cmd, res, failed)
		}
	}
}

// TestRunCommandSignalExit pins that a signal death is not read as success:
// the wrapper reports the command's status through pipefail, so a SIGKILL
// comes back as 137, never as a zero-value 0.
func TestRunCommandSignalExit(t *testing.T) {
	h := newTerminalHarness(t)
	result, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"kill -9 $$"}`)
	if !strings.HasPrefix(result, "exit 137\n") {
		t.Errorf("result = %q, want exit 137 for a signal death", result)
	}
	if !h.sawMethod("terminal/release") {
		t.Errorf("terminal was never released; got %v", h.sentMethods())
	}
}

// TestRunCommandCancelReportsPartialOutput pins that a cancelled turn still
// hands back what the command printed, kills it, and leaves no job behind.
func TestRunCommandCancelReportsPartialOutput(t *testing.T) {
	h := newTerminalHarness(t)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(300 * time.Millisecond)
		cancel()
	}()
	result, _ := runCmdExecute(ctx, h.agent, h.sess.ID, `{"command":"echo early; sleep 30"}`)
	if !strings.Contains(result, "early") || !strings.Contains(result, "terminal error") {
		t.Errorf("result = %q, want the output produced before the cancel and the error", result)
	}
	if left := h.agent.runningBgJobs(h.sess.ID); left != "" {
		t.Errorf("a cancelled command stayed tracked: %s", left)
	}
}

// TestRunCommandHandsOverAfterWait is the point of launching every command as
// a job: one that outlives the wait is not killed. The model gets what it
// printed so far and a job id, the card says so, and the exit later arrives
// as a note with the exit code and the log tail, exactly as for run_background.
func TestRunCommandHandsOverAfterWait(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	old := cmdHandoverWait
	cmdHandoverWait = 300 * time.Millisecond
	defer func() { cmdHandoverWait = old }()
	h.sess.ctl.held.Lock() // a turn is running, so the note queues instead of starting a turn
	defer h.sess.ctl.held.Unlock()

	result, failed := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"echo started; sleep 1; echo finished; exit 3","wake_after":60}`)
	if failed {
		t.Fatal("a handover must not fail the turn")
	}
	for _, want := range []string{"still running after", "background job 1", "nothing was killed", "started", "parks the turn", "woken after 1m"} {
		if !strings.Contains(result, want) {
			t.Errorf("handover result lacks %q: %q", want, result)
		}
	}
	if jobs := h.agent.parkableJobs(h.sess.ID, time.Time{}); !strings.Contains(jobs, "job 1") {
		t.Errorf("the handed-over command is not a parkable job: %q", jobs)
	}
	if h.sawMethod("terminal/kill") {
		t.Error("the command was killed at the handover")
	}

	deadline := time.Now().Add(5 * time.Second)
	for !h.sess.hasBgNotes() {
		if time.Now().After(deadline) {
			t.Fatal("the job never reported its exit")
		}
		time.Sleep(10 * time.Millisecond)
	}
	note := h.sess.takeBgNotes()[0]
	for _, want := range []string{"exited with code 3", "finished", "codehalter-job-1.log"} {
		if !strings.Contains(note.full, want) {
			t.Errorf("exit note lacks %q: %q", want, note.full)
		}
	}
	if jobs := h.agent.parkableJobs(h.sess.ID, time.Time{}); jobs != "" {
		t.Errorf("finished job still parkable: %s", jobs)
	}
	if b, err := os.ReadFile(filepath.Join(os.TempDir(), "codehalter-job-1.log")); err != nil || !strings.Contains(string(b), "finished") {
		t.Errorf("the log the model is pointed at is not there or incomplete: %v %q", err, b)
	}
}

// TestRunCommandStallKillsHungJob: a handed-over command that shows no
// progress at all (nothing on the terminal, no growth of its log or of a file
// it redirects into) is killed after bgStallTimeout and reported as such; one
// that keeps writing into a redirect file is never touched, however quiet the
// terminal is.
func TestRunCommandStallKillsHungJob(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	oldWait, oldStall, oldPoll := cmdHandoverWait, bgStallTimeout, bgStallPoll
	cmdHandoverWait, bgStallTimeout, bgStallPoll = 100*time.Millisecond, 400*time.Millisecond, 50*time.Millisecond
	defer func() { cmdHandoverWait, bgStallTimeout, bgStallPoll = oldWait, oldStall, oldPoll }()
	h.sess.ctl.held.Lock()
	defer h.sess.ctl.held.Unlock()
	waitNote := func() bgNote {
		deadline := time.Now().Add(5 * time.Second)
		for !h.sess.hasBgNotes() {
			if time.Now().After(deadline) {
				t.Fatal("no note")
			}
			time.Sleep(10 * time.Millisecond)
		}
		return h.sess.takeBgNotes()[0]
	}

	runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"echo start; sleep 30"}`)
	if n := waitNote(); !strings.Contains(n.full, "killed after") || !strings.Contains(n.full, "without any output") {
		t.Errorf("hung command's note: %q", n.full)
	}

	log := filepath.Join(t.TempDir(), "suite.log")
	runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"(for i in 1 2 3 4 5 6 7 8 9 10; do echo tick $i; sleep 0.1; done) > `+log+` 2>&1; echo exit=$? >> `+log+`"}`)
	if n := waitNote(); !strings.Contains(n.full, "exited with code 0") {
		t.Errorf("a suite writing into its file was not left alone: %q", n.full)
	}
}
