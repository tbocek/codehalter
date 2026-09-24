package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

// TestRunBackgroundStaysRunning pins the core contract: a long-running command
// returns promptly as a tracked "running" job (not waited on), its log exists,
// and shutdownBackground reaps it and removes the scratch files.
func TestRunBackgroundStaysRunning(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground() // leak guard if an assertion aborts early
	old := bgJobGrace
	bgJobGrace = 100 * time.Millisecond
	defer func() { bgJobGrace = old }()

	res, failed := runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 5"}`)
	if failed {
		t.Fatalf("run_background marked the turn failed: %s", res)
	}
	if !strings.Contains(res, "running (pid") {
		t.Fatalf("expected a running job, got: %s", res)
	}

	h.agent.bgMu.Lock()
	n := len(h.agent.bgJobs)
	var job *backgroundJob
	for _, j := range h.agent.bgJobs {
		job = j
	}
	h.agent.bgMu.Unlock()
	if n != 1 {
		t.Fatalf("expected 1 tracked job, got %d", n)
	}
	if _, err := os.Stat(job.logPath); err != nil {
		t.Fatalf("log file missing: %v", err)
	}
	// The pid is the whole point of the wrapper script: without it the model has
	// no way to stop the job, since the client owns the process.
	if job.pid <= 0 {
		t.Errorf("job.pid = %d, want the pid the wrapper recorded", job.pid)
	}
	if !strings.Contains(res, fmt.Sprintf("kill %d", job.pid)) {
		t.Errorf("result should tell the model how to stop it, got: %s", res)
	}
	// The terminal is deliberately NOT released while the job should live —
	// releasing kills the process.
	if job.terminalId == "" {
		t.Error("job has no terminal id")
	}
	if !h.embeddedTerminal() {
		t.Error("the background terminal was not embedded; the user sees no live output")
	}

	h.agent.shutdownBackground()
	if _, err := os.Stat(job.logPath); !os.IsNotExist(err) {
		t.Errorf("shutdownBackground left the log behind: %v", err)
	}
	if _, err := os.Stat(job.pidPath); !os.IsNotExist(err) {
		t.Errorf("shutdownBackground left the pid file behind: %v", err)
	}
	if !h.sawMethod("terminal/release") {
		t.Errorf("shutdownBackground never released the terminal; got %v", h.sentMethods())
	}
	h.agent.bgMu.Lock()
	left := len(h.agent.bgJobs)
	h.agent.bgMu.Unlock()
	if left != 0 {
		t.Errorf("shutdownBackground left %d jobs tracked", left)
	}
}

// TestRunBackgroundImmediateExit pins that a command which exits during the
// grace window is reported as a crash (with exit code + captured output) and is
// not left in the job table.
func TestRunBackgroundImmediateExit(t *testing.T) {
	h := newTerminalHarness(t)
	old := bgJobGrace
	bgJobGrace = 3 * time.Second // ample: the poll returns as soon as the process exits
	defer func() { bgJobGrace = old }()

	res, failed := runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{"command":"echo boom; exit 3"}`)
	if failed {
		t.Fatalf("unexpected turn failure: %s", res)
	}
	if !strings.Contains(res, "exited immediately") || !strings.Contains(res, "exit 3") {
		t.Fatalf("expected immediate-exit report with exit 3, got: %s", res)
	}
	if !strings.Contains(res, "boom") {
		t.Errorf("expected captured output 'boom', got: %s", res)
	}
	h.agent.bgMu.Lock()
	n := len(h.agent.bgJobs)
	h.agent.bgMu.Unlock()
	if n != 0 {
		t.Errorf("crashed job left in table: %d", n)
	}
}

// TestBackgroundLogIsReadableByRunCommand pins the assumption the whole design
// rests on: the client's terminal and codehalter share a filesystem, so the log
// path handed to the model is one a later `run_command: cat` can actually read.
// If that ever stops holding, run_background silently loses its output channel.
func TestBackgroundLogIsReadableByRunCommand(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	old := bgJobGrace
	bgJobGrace = 300 * time.Millisecond
	defer func() { bgJobGrace = old }()

	res, _ := runBackgroundExecute(context.Background(), h.agent, h.sess.ID,
		`{"command":"echo listening on 8765; sleep 5"}`)
	if !strings.Contains(res, "listening on 8765") {
		t.Fatalf("startup output not folded into the result: %s", res)
	}

	h.agent.bgMu.Lock()
	var logPath string
	for _, j := range h.agent.bgJobs {
		logPath = j.logPath
	}
	h.agent.bgMu.Unlock()

	out, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"cat `+logPath+`"}`)
	if !strings.Contains(out, "listening on 8765") {
		t.Errorf("cat of the job log returned %q, want the job's output", out)
	}
}

func TestRunBackgroundRequiresCommand(t *testing.T) {
	h := newTerminalHarness(t)
	res, failed := runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{}`)
	if failed || !strings.Contains(res, "command is required") {
		t.Fatalf("expected command-required error, got: %s (failed=%v)", res, failed)
	}
}

// lastUserMessage returns the newest user message in the session, "" if none.
func lastUserMessage(s *Session) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.Messages) - 1; i >= 0; i-- {
		if s.Messages[i].Role == "user" {
			return s.Messages[i].Content
		}
	}
	return ""
}

// TestBackgroundJobReportsWhenTurnEnds pins the no-interrupt rule: a job that
// finishes while a turn is running changes nothing until that turn is over.
// Then the result is stored for the model (exit code, log path, last output)
// without any turn being started.
func TestBackgroundJobReportsWhenTurnEnds(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	old := bgJobGrace
	bgJobGrace = 50 * time.Millisecond
	defer func() { bgJobGrace = old }()

	h.sess.ctl.held.Lock() // a turn is running
	res, failed := runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 0.3; echo experiment-done; exit 4"}`)
	if failed || !strings.Contains(res, "do NOT poll") {
		t.Fatalf("launch = (%q, failed=%v), want a running job that tells the model not to poll", res, failed)
	}

	deadline := time.Now().Add(5 * time.Second)
	for !h.sess.hasBgNotes() {
		if time.Now().After(deadline) {
			t.Fatal("the finished job never queued a note")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got := lastUserMessage(h.sess); got != "" {
		t.Fatalf("a message reached the session while the turn was still running: %q", got)
	}

	h.agent.flushBgNotes(context.Background(), h.sess) // what Prompt does as the turn ends
	h.sess.ctl.held.Unlock()

	got := lastUserMessage(h.sess)
	for _, want := range []string{"background job", "exited with code 4", "experiment-done", "codehalter, not the user"} {
		if !strings.Contains(got, want) {
			t.Errorf("stored note lacks %q: %q", want, got)
		}
	}
	if h.sess.hasBgNotes() {
		t.Error("note still queued after the flush")
	}
	h.agent.bgMu.Lock()
	left := len(h.agent.bgJobs)
	h.agent.bgMu.Unlock()
	if left != 0 {
		t.Errorf("finished job still tracked: %d", left)
	}
}

// TestTurnEndNamesRunningJobs: when a turn hands the prompt back with a job
// still running, the user is told which, and that the work resumes on its own
// when it exits. With nothing running the turn ends silently.
func TestTurnEndNamesRunningJobs(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	_, release, ok := h.agent.holdTurn(context.Background(), h.sess, false)
	if !ok {
		t.Fatal("could not hold the turn")
	}
	release()
	if n := len(h.updatesOfKind(KindAgentMessage)); n != 0 {
		t.Fatalf("a turn with no jobs said %d things", n)
	}

	h.agent.registerBgJob(&backgroundJob{id: 3, sid: h.sess.ID, cmdStr: "just test > /tmp/t.log 2>&1", started: time.Now()})
	_, release, ok = h.agent.holdTurn(context.Background(), h.sess, false)
	if !ok {
		t.Fatal("could not hold the turn")
	}
	release()
	u := h.waitForKind(KindAgentMessage)
	var said string
	if c, _ := u["content"].(map[string]any); c != nil {
		said = fmt.Sprint(c["text"])
	}
	for _, want := range []string{"Still running in the background", "job 3 `just test", "carry on meanwhile"} {
		if !strings.Contains(said, want) {
			t.Errorf("turn-end line lacks %q: %q", want, said)
		}
	}
}

// TestBackgroundJobWakeAfter: a job started with wake_after wakes the model
// once at that age while it still runs, with the log tail and a note that it
// is not the exit; the exit is reported on its own afterwards. A job that
// exits before the age is reported once, at exit.
func TestBackgroundJobWakeAfter(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	old := bgJobGrace
	bgJobGrace = 50 * time.Millisecond
	defer func() { bgJobGrace = old }()
	h.sess.ctl.held.Lock() // a turn is running, so notes queue instead of starting turns
	defer h.sess.ctl.held.Unlock()

	res, _ := runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{"command":"echo serving; sleep 1.5; exit 0","wake_after":1}`)
	if !strings.Contains(res, "woken after 1.0s") {
		t.Fatalf("launch did not confirm the wake: %q", res)
	}
	waitNote := func(what string) []bgNote {
		deadline := time.Now().Add(5 * time.Second)
		for !h.sess.hasBgNotes() {
			if time.Now().After(deadline) {
				t.Fatalf("no %s note", what)
			}
			time.Sleep(10 * time.Millisecond)
		}
		return h.sess.takeBgNotes()
	}
	notes := waitNote("wake")
	h.agent.bgMu.Lock()
	still := len(h.agent.bgJobs)
	h.agent.bgMu.Unlock()
	if still != 1 {
		t.Errorf("the job should still be running at the wake: %d tracked", still)
	}
	for _, want := range []string{"still running after", "wake_after you asked for, not its exit", "serving"} {
		if !strings.Contains(notes[0].full, want) {
			t.Errorf("wake note lacks %q: %q", want, notes[0].full)
		}
	}
	notes = waitNote("exit")
	if !strings.Contains(notes[0].full, "exited with code 0") {
		t.Errorf("exit note: %q", notes[0].full)
	}

	if res, _ = runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 3","wake_after":"soon"}`); !strings.Contains(res, "error: wake_after") {
		t.Errorf("a non-numeric wake_after was accepted: %q", res)
	}
	res, _ = runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 0.3; exit 0","wake_after":1}`)
	if !strings.Contains(res, "woken after 1.0s") {
		t.Fatalf("launch: %q", res)
	}
	if notes = waitNote("exit"); !strings.Contains(notes[0].full, "exited with code 0") {
		t.Errorf("exit note: %q", notes[0].full)
	}
	time.Sleep(1200 * time.Millisecond)
	if h.sess.hasBgNotes() {
		t.Error("a job that exited before its wake age was woken for anyway")
	}
}
