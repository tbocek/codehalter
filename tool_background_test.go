package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

// Returns promptly as a tracked job with a log; shutdownBackground reaps it and
// removes the scratch files.
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
	// Without the pid the model cannot stop the job: the client owns the process.
	if job.pid <= 0 {
		t.Errorf("job.pid = %d, want the pid the wrapper recorded", job.pid)
	}
	if !strings.Contains(res, fmt.Sprintf("kill -TERM -%d; sleep 3; kill -KILL -%d", job.pid, job.pid)) {
		t.Errorf("result should tell the model how to stop it, got: %s", res)
	}
	// Not released while the job should live: releasing kills the process.
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

// Reported as a crash with exit code and output, and not left in the job table.
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

// The design rests on the client's terminal and codehalter sharing a filesystem.
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

// A job finishing mid-turn changes nothing until the turn ends; then its note
// runs a follow-up turn.
func TestBackgroundJobReportsWhenTurnEnds(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	old := bgJobGrace
	bgJobGrace = 50 * time.Millisecond
	defer func() { bgJobGrace = old }()
	mock := newMockLLM(t, sseToolCall("p1", submitPlanToolName,
		`{"clear":true,"report_only":true,"subtasks":[],"answer":"The experiment failed with code 4."}`))
	defer mock.Close()
	h.agent.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}, KeepWarm: "off"}
	// An empty SUMMARISE.md keeps the turn's summariser off the mock.
	if err := os.MkdirAll(filepath.Join(h.sess.Cwd, sessionDir), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(h.sess.Cwd, sessionDir, "SUMMARISE.md"), nil, 0o644); err != nil {
		t.Fatal(err)
	}

	h.sess.ctl.held.Lock() // a turn is running
	res, failed := runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 0.3; echo experiment-done; exit 4"}`)
	if failed || !strings.Contains(res, "do NOT poll") {
		t.Fatalf("launch = (%q, failed=%v), want a running job that tells the model not to poll", res, failed)
	}

	deadline := time.Now().Add(5 * time.Second)
	for !h.sess.hasPending() {
		if time.Now().After(deadline) {
			t.Fatal("the finished job never queued a note")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got := lastUserMessage(h.sess); got != "" {
		t.Fatalf("a message reached the session while the turn was still running: %q", got)
	}

	h.agent.drainSteer(context.Background(), h.sess) // what Prompt does as the turn ends
	h.sess.ctl.held.Unlock()

	if mock.callCount() != 1 {
		t.Fatalf("LLM calls = %d, want 1: the note should have run a turn", mock.callCount())
	}
	var prompt string
	for _, m := range h.sess.Messages {
		if m.Role == "user" && strings.Contains(m.Content, "exited with code 4") {
			prompt = m.Content
		}
	}
	for _, want := range []string{"background job", "experiment-done", "codehalter, not the user", "Continue the work that was waiting on this result"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("the follow-up turn's prompt lacks %q: %q", want, prompt)
		}
	}
	if h.sess.hasPending() {
		t.Error("note still queued after the drain")
	}
	h.agent.bgMu.Lock()
	left := len(h.agent.bgJobs)
	h.agent.bgMu.Unlock()
	if left != 0 {
		t.Errorf("finished job still tracked: %d", left)
	}
}

// With nothing new running the turn ends silently.
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
	for _, want := range []string{"Running in the background", "job 3 `just test", "carry on meanwhile"} {
		if !strings.Contains(said, want) {
			t.Errorf("turn-end line lacks %q: %q", want, said)
		}
	}
	// Named once, or a long-lived dev server would close every turn with it.
	_, release, ok = h.agent.holdTurn(context.Background(), h.sess, false)
	if !ok {
		t.Fatal("could not hold the turn")
	}
	release()
	time.Sleep(50 * time.Millisecond)
	if n := len(h.updatesOfKind(KindAgentMessage)); n != 1 {
		t.Errorf("the same job was announced again: %d messages", n)
	}
}

// The wake says it is not the exit; a job exiting before the age is reported
// once, at exit.
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
	waitNote := func(what string) []pendingInput {
		deadline := time.Now().Add(5 * time.Second)
		for !h.sess.hasPending() {
			if time.Now().After(deadline) {
				t.Fatalf("no %s note", what)
			}
			time.Sleep(10 * time.Millisecond)
		}
		return h.sess.takePending()
	}
	notes := waitNote("wake")
	h.agent.bgMu.Lock()
	still := len(h.agent.bgJobs)
	h.agent.bgMu.Unlock()
	if still != 1 {
		t.Errorf("the job should still be running at the wake: %d tracked", still)
	}
	for _, want := range []string{"still running after", "wake_after you asked for, not its exit", "serving"} {
		if !strings.Contains(notes[0].note.full, want) {
			t.Errorf("wake note lacks %q: %q", want, notes[0].note.full)
		}
	}
	notes = waitNote("exit")
	if !strings.Contains(notes[0].note.full, "exited with code 0") {
		t.Errorf("exit note: %q", notes[0].note.full)
	}

	if res, _ = runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 3","wake_after":"soon"}`); !strings.Contains(res, "error: wake_after") {
		t.Errorf("a non-numeric wake_after was accepted: %q", res)
	}
	res, _ = runBackgroundExecute(context.Background(), h.agent, h.sess.ID, `{"command":"sleep 0.3; exit 0","wake_after":1}`)
	if !strings.Contains(res, "woken after 1.0s") {
		t.Fatalf("launch: %q", res)
	}
	if notes = waitNote("exit"); !strings.Contains(notes[0].note.full, "exited with code 0") {
		t.Errorf("exit note: %q", notes[0].note.full)
	}
	time.Sleep(1200 * time.Millisecond)
	if h.sess.hasPending() {
		t.Error("a job that exited before its wake age was woken for anyway")
	}
}

// TestStopKeepsQueuedNotesForNextPrompt: notes queued after a Stop start no
// turn, a later one neither; they wait for the next prompt. A Stop while idle
// holds nothing back.
func TestStopKeepsQueuedNotesForNextPrompt(t *testing.T) {
	h := newTerminalHarness(t)
	_, release, ok := h.agent.holdTurn(context.Background(), h.sess, false)
	if !ok {
		t.Fatal("could not hold the turn")
	}
	h.sess.addBgNote(bgNote{line: "job 1 done", full: "[codehalter, not the user: job 1 done]"})
	h.sess.cancelTurn()
	release()

	h.agent.deliverBgNotesWhenIdle(h.sess)
	h.sess.addBgNote(bgNote{line: "job 2 done", full: "[codehalter, not the user: job 2 done]"})
	h.agent.deliverBgNotesWhenIdle(h.sess)
	if n := len(h.sess.takePending()); n != 2 {
		t.Fatalf("%d notes left after a Stop, want both waiting for the next prompt", n)
	}

	_, release, _ = h.agent.holdTurn(context.Background(), h.sess, false)
	release()
	h.sess.cancelTurn()
	if h.sess.stoppedIdle() {
		t.Error("a Stop while idle held back the next note")
	}
}

// At startup the jobs of a codehalter that is gone are stopped; a live
// codehalter's job, and a process that only reuses a pid, are left alone.
func TestKillOrphanedJobs(t *testing.T) {
	old := jobKillGrace
	jobKillGrace = 100 * time.Millisecond
	defer func() { jobKillGrace = old }()
	dir := t.TempDir()
	t.Setenv("TMPDIR", dir)
	gone := exec.Command("true")
	if err := gone.Run(); err != nil {
		t.Fatal(err)
	}
	dead := gone.Process.Pid // its codehalter has exited
	start := func(pidFile string, names bool) *exec.Cmd {
		t.Helper()
		script := "trap '' TERM; sleep 30 & wait"
		if names {
			script = "echo $$ > " + pidFile + "; " + script
		}
		c := exec.Command("sh", "-c", script)
		c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		if err := c.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = syscall.Kill(-c.Process.Pid, syscall.SIGKILL) })
		if !names {
			if err := os.WriteFile(pidFile, []byte(strconv.Itoa(c.Process.Pid)), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		for deadline := time.Now().Add(2 * time.Second); readPidFile(pidFile) == 0 && time.Now().Before(deadline); {
			time.Sleep(10 * time.Millisecond)
		}
		return c
	}
	orphan := start(filepath.Join(dir, fmt.Sprintf("codehalter-%d-job-1.pid", dead)), true)
	live := start(filepath.Join(dir, fmt.Sprintf("codehalter-%d-job-2.pid", os.Getppid())), true)
	reused := start(filepath.Join(dir, fmt.Sprintf("codehalter-%d-job-3.pid", dead)), false)

	killOrphanedJobs()
	exited := func(c *exec.Cmd) bool {
		done := make(chan struct{})
		go func() { _ = c.Wait(); close(done) }()
		select {
		case <-done:
			return true
		case <-time.After(2 * time.Second):
			return false
		}
	}
	if !exited(orphan) {
		t.Error("the orphaned job still runs")
	}
	if syscall.Kill(live.Process.Pid, 0) != nil {
		t.Error("a live codehalter's job was killed")
	}
	if syscall.Kill(reused.Process.Pid, 0) != nil {
		t.Error("a process that only reuses a job's pid was killed")
	}
}

// A render in the background finishes unseen, so it is refused with the way to
// run it; other jobs, and the same render through run_command, are not.
func TestRunBackgroundRefusesARender(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	for _, cmd := range []string{
		`cd /workspaces/naivepost/rust && just snapshot 07-narrate > /tmp/snap-f46i.log 2>&1; echo "exit=$?" >> /tmp/snap-f46i.log`,
		`make snapshot SCREEN=03-window`,
		`npm run snapshot -- prepare`,
		`cargo run -- --snapshot 05-cut --out shots/05-cut.png`,
	} {
		b, _ := json.Marshal(map[string]string{"command": cmd})
		if res, _ := runBackgroundExecute(context.Background(), h.agent, h.sess.ID, string(b)); !strings.HasPrefix(res, "refused: this renders a screen") || !strings.Contains(res, "run_command") {
			t.Errorf("%q was not refused: %s", cmd, res)
		}
	}
	for _, cmd := range []string{`npx jest --updateSnapshot > /tmp/j.log`, `cat snapshots.txt; sleep 5`} {
		b, _ := json.Marshal(map[string]string{"command": cmd})
		if res, _ := runBackgroundExecute(context.Background(), h.agent, h.sess.ID, string(b)); strings.HasPrefix(res, "refused: this renders") {
			t.Errorf("%q is no render but was refused", cmd)
		}
	}
	if res, _ := runCmdExecute(context.Background(), h.agent, h.sess.ID, `{"command":"echo just snapshot 07-narrate"}`); strings.HasPrefix(res, "refused") {
		t.Errorf("run_command refused a render: %s", res)
	}
}

// A loop that polls with pgrep and sleep adds a tick or two of reaped children at every
// sample; that is idle, not movement. Only CPU from bgBusyRate up counts, so the polling
// job is stopped and the one that really works is left alone.
func TestStuckWatchCountsOnlyRealCPU(t *testing.T) {
	oldRate, oldRead := bgBusyRate, readActivity
	bgBusyRate = 100 // a whole core: 2 ticks per 20 ms sample
	defer func() { bgBusyRate, readActivity = oldRate, oldRead }()

	for _, tc := range []struct {
		name      string
		perSample uint64
		stuck     bool
	}{
		{"polling adds a tick a sample", 1, true},
		{"a busy job adds five", 5, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newTerminalHarness(t)
			var ticks atomic.Uint64
			readActivity = func(int) procActivity {
				return procActivity{known: true, cpu: ticks.Add(tc.perSample)}
			}
			job := &backgroundJob{id: 1, sid: h.sess.ID, cmdStr: "while pgrep -f \"just test\"; do sleep 5; done", started: time.Now()}
			h.agent.registerBgJob(job)
			stop := make(chan struct{})
			done := make(chan struct{})
			go func() {
				h.agent.stuckWatch(job, stop, 300*time.Millisecond, 20*time.Millisecond)
				close(done)
			}()
			select {
			case <-done:
				if !tc.stuck {
					t.Fatal("a job using 25% of a core was stopped")
				}
			case <-time.After(800 * time.Millisecond):
				close(stop)
				<-done
				if tc.stuck {
					t.Fatal("a job adding one tick a sample was not stopped")
				}
			}
			h.agent.bgMu.Lock()
			reason := job.stuck
			h.agent.bgMu.Unlock()
			if tc.stuck && !strings.Contains(reason, "almost no CPU") {
				t.Errorf("reason = %q", reason)
			}
		})
	}
}

// A server is what the model starts with run_server: it comes back once its port
// answers, nothing waits for it, it is never stopped as idle, and run_server stops it.
func TestRunServer(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("no python3 for the server")
	}
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	oldStuck, oldPoll := bgStuckAfter, bgStuckPoll
	bgStuckAfter, bgStuckPoll = 200*time.Millisecond, 50*time.Millisecond
	defer func() { bgStuckAfter, bgStuckPoll = oldStuck, oldPoll }()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	run := func(args string) string {
		t.Helper()
		res, _ := runServerExecute(context.Background(), h.agent, h.sess.ID, args)
		return res
	}

	before := time.Now().Add(-time.Second)
	start := fmt.Sprintf(`{"command":"python3 -m http.server %d --bind 127.0.0.1","port":%d}`, port, port)
	res := run(start)
	var id int
	if _, err := fmt.Sscanf(res, "server job %d running", &id); err != nil || !strings.Contains(res, fmt.Sprintf("accepting connections on port %d", port)) {
		t.Fatalf("start: %s", res)
	}
	if res := run(start); !strings.Contains(res, "refused: something already accepts connections") {
		t.Errorf("a second server on a taken port: %s", res)
	}
	time.Sleep(600 * time.Millisecond) // three idle limits
	h.agent.bgMu.Lock()
	job := h.agent.bgJobs[id]
	h.agent.bgMu.Unlock()
	if job == nil || job.stuck != "" {
		t.Fatalf("the idle server was stopped: %+v", job)
	}
	if jobs := h.agent.parkableJobs(h.sess.ID, before); jobs != "" {
		t.Errorf("a step would wait for the server: %s", jobs)
	}

	if res := run(`{"stop":999}`); !strings.Contains(res, fmt.Sprintf("job %d", id)) {
		t.Errorf("stopping an unknown id names no running server: %s", res)
	}
	if res := run(fmt.Sprintf(`{"stop":%d}`, id)); !strings.Contains(res, "stopped") {
		t.Fatalf("stop: %s", res)
	}
	for deadline := time.Now().Add(5 * time.Second); portOpen(port); time.Sleep(50 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the stopped server still answers")
		}
	}

	if res := run(`{"command":"echo cannot bind; exit 3"}`); !strings.Contains(res, "exited immediately (exit 3)") || !strings.Contains(res, "cannot bind") {
		t.Errorf("a server that dies at start: %s", res)
	}
}
