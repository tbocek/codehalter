package main

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestRunTerminalCmdLifecycle pins the whole terminal lifecycle: the command
// runs, its live terminal is embedded in the card so the user watches it there,
// the exit status comes back, and the terminal is released. The release matters
// because an unreleased terminal leaks in the client.
func TestRunTerminalCmdLifecycle(t *testing.T) {
	h := newTerminalHarness(t)

	out, exit, started, err := h.agent.runTerminalCmd(context.Background(), h.sess.ID, "tc1",
		"bash", []string{"-c", "echo hello; exit 3"}, h.sess.Cwd, 0)
	if !started || err != nil {
		t.Fatalf("started=%v err=%v", started, err)
	}
	if exit.code() != 3 {
		t.Errorf("exit = %d, want 3", exit.code())
	}
	if !strings.Contains(out, "hello") {
		t.Errorf("out = %q, want the command's output", out)
	}
	if !h.embeddedTerminal() {
		t.Error("no update embedded the terminal; the user would see an empty card")
	}
	if !h.sawMethod("terminal/release") {
		t.Errorf("terminal was never released; got %v", h.sentMethods())
	}
}

// TestRunTerminalCmdSignalExit pins that a signal death reports -1 rather than
// a bogus 0: terminal/wait_for_exit returns no exitCode at all in that case,
// and a zero-value struct would read as success.
func TestRunTerminalCmdSignalExit(t *testing.T) {
	h := newTerminalHarness(t)

	_, exit, started, err := h.agent.runTerminalCmd(context.Background(), h.sess.ID, "tc1",
		"bash", []string{"-c", "kill -9 $$"}, h.sess.Cwd, 0)
	if !started || err != nil {
		t.Fatalf("started=%v err=%v", started, err)
	}
	if exit.code() != -1 {
		t.Errorf("exit = %d, want -1 for a signal death", exit.code())
	}
}

// TestTerminalIdleWatchdogKills pins the silence timeout. There is no output
// stream to watch on a client terminal, so the watchdog re-reads
// terminal/output on a timer and kills on an unchanged fingerprint — without it
// a hung command parks the turn until the user notices.
func TestTerminalIdleWatchdogKills(t *testing.T) {
	h := newTerminalHarness(t)

	start := time.Now()
	out, _, started, err := h.agent.runTerminalCmd(context.Background(), h.sess.ID, "tc1",
		"bash", []string{"-c", "echo waiting; sleep 30"}, h.sess.Cwd, 50*time.Millisecond)
	if !started || err != nil {
		t.Fatalf("started=%v err=%v", started, err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("silent command should be reaped fast, took %s", elapsed)
	}
	if !h.sawMethod("terminal/kill") {
		t.Errorf("terminal/kill was never sent; got %v", h.sentMethods())
	}
	if !strings.Contains(out, "[killed: no output for") {
		t.Errorf("out = %q, want the idle-kill notice so the model knows it timed out", out)
	}
	if !strings.Contains(out, "waiting") {
		t.Errorf("out = %q, want the output captured before the kill", out)
	}
}

// TestTerminalIdleWatchdogMeasuresRealSilence pins WHEN the kill lands, which
// is the difference between the notice being true and being a guess. Polling
// once per timeout could not kill before TWO intervals, so a command advertised
// as dying after 2m of silence really died somewhere between 2m and 4m.
// Timestamping the last change puts it at the timeout itself.
//
// The command prints nothing at all, so there is no output-arrival jitter to
// blur the measurement: silence starts when the command does.
func TestTerminalIdleWatchdogMeasuresRealSilence(t *testing.T) {
	h := newTerminalHarness(t)

	const idle = time.Second
	start := time.Now()
	_, _, started, err := h.agent.runTerminalCmd(context.Background(), h.sess.ID, "tc1",
		"bash", []string{"-c", "sleep 30"}, h.sess.Cwd, idle)
	if !started || err != nil {
		t.Fatalf("started=%v err=%v", started, err)
	}
	elapsed := time.Since(start)
	if elapsed < idle {
		t.Errorf("reaped after %s, before the %s timeout was even up", elapsed, idle)
	}
	if elapsed > idle+idle/2 {
		t.Errorf("reaped after %s, want ~%s — the kill is quantised to the poll interval again", elapsed, idle)
	}
}

// TestTerminalIdleWatchdogSpareChattyCommand pins the other half of the
// watchdog: a command that keeps printing is never reaped, however long it
// runs. The fingerprint is a hash rather than a byte count for this reason —
// output that changes without growing still counts as alive.
func TestTerminalIdleWatchdogSparesChattyCommand(t *testing.T) {
	h := newTerminalHarness(t)

	out, exit, started, err := h.agent.runTerminalCmd(context.Background(), h.sess.ID, "tc1",
		"bash", []string{"-c", "for i in 1 2 3 4 5 6; do echo tick $i; sleep 0.05; done"}, h.sess.Cwd, 400*time.Millisecond)
	if !started || err != nil {
		t.Fatalf("started=%v err=%v", started, err)
	}
	if exit.code() != 0 {
		t.Errorf("exit = %d, want 0 — a printing command must not be reaped", exit.code())
	}
	if strings.Contains(out, "killed") {
		t.Errorf("out = %q, want no kill notice", out)
	}
	if !strings.Contains(out, "tick 6") {
		t.Errorf("out = %q, want the command to have run to completion", out)
	}
}

// TestTerminalCmdCancelReportsPartialOutput pins that a cancelled turn still
// hands back what the command printed. The ctx is already dead at that point,
// so the final read has to use a fresh one or it returns nothing.
func TestTerminalCmdCancelReportsPartialOutput(t *testing.T) {
	h := newTerminalHarness(t)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(300 * time.Millisecond)
		cancel()
	}()
	out, _, started, err := h.agent.runTerminalCmd(ctx, h.sess.ID, "tc1",
		"bash", []string{"-c", "echo early; sleep 30"}, h.sess.Cwd, 0)
	if !started {
		t.Fatal("started = false, want the terminal to have come up")
	}
	if err == nil {
		t.Error("err = nil, want the cancellation surfaced")
	}
	if !strings.Contains(out, "early") {
		t.Errorf("out = %q, want the output produced before the cancel", out)
	}
}
