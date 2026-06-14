package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"sync"
	"time"
)

// discoverSandbox registers `run_command` whenever we're inside a container.
// The container itself is the sandbox: it's throwaway, the host workspace is
// bind-mounted (the LLM can already write to those files via edit_file), and
// apt-get/dpkg/pip writes are scoped to the container's lifetime. Devcontainers
// are expected to bind-mount `.git` read-only so destructive git commands fail
// at the OS layer; we no longer install a PATH shim for `git`.
func (a *agent) discoverSandbox() {
	// Only register run_command inside a container — the container IS the
	// sandbox. Outside one, ensureDevcontainer aborts the session before any
	// prompt runs, so there is no "running on host with run_command disabled"
	// state to report; just skip registration.
	if containerKind() == "" {
		slog.Info("run_command: not registered (not inside a container)")
		return
	}

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "run_command",
			"description": "Run a shell command directly inside this devcontainer. The container is the sandbox: it's throwaway, so apt-get/dpkg/pip writes persist for the container's lifetime (wiped on rebuild) and workspace writes are real but recoverable from `.git/`. Use this for: (1) PROBE — `which <tool>`, `cargo check`, `node --version`, `apt list --installed | grep <pkg>` — confirm what exists. (2) TEST INSTALL — when you're about to propose a Dockerfile edit (e.g. `RUN apt-get install <pkg>`), first run the same install via run_command, then verify it works (e.g. `<tool> --version` or re-running the failing build). If the install + verification succeed, propose the Dockerfile patch with confidence; if they fail, debug here before editing the Dockerfile. Exit code is always in the output and title — `which <tool>` exiting 1 means <tool> is missing, not that the tool failed. " +
				"For project-file edits prefer `edit_file` / `write_file` — they go through the agent's diff/approval UI, raw `>` or `sed -i` do not. " +
				"The `.git` directory is bind-mounted read-only; destructive git commands (push, reset --hard, etc.) will fail at the filesystem layer. Read-only git is fine (clone, log, ls-remote, archive).",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"command"},
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command to run under bash -c. Be precise — this is not a chat. Examples: `which <tool>`, `apt-get install -y <pkg> && <tool> --version`, `cargo check 2>&1 | tail -20`.",
					},
				},
			},
		},
	}, Execute: runCmdExecute})
}

// cmdIdleTimeout reaps a run_command that prints NOTHING for this long. It's an
// IDLE timeout, not a total one: a command that keeps producing output runs
// unbounded (a long build is fine), but a silent/hung one is SIGKILLed so it
// can't park a turn forever (the parent ctx only fires on a user Stop). A var so
// tests can shorten it.
var cmdIdleTimeout = 60 * time.Second

// cmdOutputCap bounds how many bytes of a command's output we CAPTURE for the
// tool result (handed to the model); editorStreamCap bounds the live editor
// stream. The idle watchdog only fires on SILENCE, so a steadily-printing command
// (`find /`, a chatty build, `journalctl`) would otherwise grow the buffer without
// limit and dump megabytes into a weak, small-context model. We keep a head + tail
// window and elide the middle, so a long run's start AND its trailing error both
// survive. Vars so tests can shrink them.
var (
	cmdOutputCap    = 64 * 1024
	editorStreamCap = 256 * 1024
)

// boundedOutput captures a byte stream in at most headCap+tailCap bytes: the
// first headCap as a frozen head, the most recent tailCap as a ring tail, the
// middle elided. It bounds memory for an unbounded command while keeping both
// ends (head shows how the run started; tail preserves the error verify reads).
// Not safe for concurrent use; the drain serialises writes under its own lock.
type boundedOutput struct {
	headCap, tailCap int
	head, tail       []byte
	total            int
}

func newBoundedOutput(capBytes int) *boundedOutput {
	h := capBytes / 4
	return &boundedOutput{headCap: h, tailCap: capBytes - h}
}

// Write appends p, freezing the head once full and holding the tail to at most
// 2*tailCap (trimmed back to tailCap when it crosses, so it's an amortised
// O(1)/byte ring); String trims the residual to exactly the last tailCap bytes.
func (b *boundedOutput) Write(p []byte) {
	b.total += len(p)
	if len(b.head) < b.headCap {
		n := b.headCap - len(b.head)
		if n > len(p) {
			n = len(p)
		}
		b.head = append(b.head, p[:n]...)
	}
	b.tail = append(b.tail, p...)
	if len(b.tail) > 2*b.tailCap {
		b.tail = append(b.tail[:0], b.tail[len(b.tail)-b.tailCap:]...)
	}
}

// truncated reports whether the middle was elided (more bytes seen than kept).
func (b *boundedOutput) truncated() bool { return b.total > b.headCap+b.tailCap }

// String reassembles the captured window: the whole stream when it fit, else
// head + an "[... N bytes omitted ...]" marker + the last tailCap bytes, stitched
// so the overlap case neither duplicates nor drops bytes.
func (b *boundedOutput) String() string {
	tail := b.tail
	if len(tail) > b.tailCap {
		tail = tail[len(tail)-b.tailCap:]
	}
	switch {
	case b.total <= b.tailCap:
		return string(tail) // everything fit in the tail
	case b.total <= b.headCap+b.tailCap:
		overlap := b.headCap + b.tailCap - b.total // bytes head and tail share
		return string(b.head) + string(tail[overlap:])
	default:
		omitted := b.total - b.headCap - b.tailCap
		return string(b.head) + fmt.Sprintf("\n[... %d bytes omitted ...]\n", omitted) + string(tail)
	}
}

// runStreamingCmd wires cmd's combined stdout+stderr through an io.Pipe, starts
// it, streams output to the editor in 200 ms batches, and captures a bounded
// head+tail copy for the tool result. The drain is DECOUPLED from the editor (it
// always reads the pipe), so a slow editor can never deadlock the command on an
// io.Pipe write behind the single conn write lock. If idleTimeout > 0 and nothing
// prints for that long, cmd is SIGKILLed via cancelCmd. banner is the "$ <cmd>"
// echo shown before output. started=false means cmd never launched (err is the
// start failure); otherwise text is the captured output (with any markers) and
// runErr is cmd.Wait()'s result. Shared by run_command and run_task.
func (a *agent) runStreamingCmd(ctx context.Context, sid, banner string, cmd *exec.Cmd, cancelCmd context.CancelFunc, idleTimeout time.Duration) (text string, runErr error, started bool) {
	pipeR, pipeW := io.Pipe()
	cmd.Stdout = pipeW
	cmd.Stderr = pipeW
	if err := cmd.Start(); err != nil {
		return "", err, false
	}

	waitErr := make(chan error, 1)
	go func() {
		waitErr <- cmd.Wait()
		_ = pipeW.Close()
	}()

	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "\n```\n$ " + banner + "\n"}})

	// One drain goroutine reads the pipe to completion regardless of editor speed.
	// It writes into a bounded capture (for the model) and a separate pending slice
	// (for the editor, drained every 200 ms and itself capped at editorStreamCap so
	// a flood can't balloon it). scanErr flags a line past the 1 MB scanner buffer.
	capture := newBoundedOutput(cmdOutputCap)
	var (
		mu          sync.Mutex
		pending     []byte
		editorBytes int
		scanErr     error
	)
	nl := []byte{'\n'}
	drainDone := make(chan struct{})
	go func() {
		defer close(drainDone)
		scanner := bufio.NewScanner(pipeR)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			b := scanner.Bytes()
			mu.Lock()
			capture.Write(b)
			capture.Write(nl)
			if editorBytes < editorStreamCap {
				pending = append(pending, b...)
				pending = append(pending, '\n')
				editorBytes += len(b) + 1
				if editorBytes >= editorStreamCap {
					pending = append(pending, "\n[live output truncated; full result returned at the end]\n"...)
				}
			}
			mu.Unlock()
		}
		mu.Lock()
		scanErr = scanner.Err()
		mu.Unlock()
	}()

	flush := func() {
		mu.Lock()
		chunk := pending
		pending = nil
		mu.Unlock()
		if len(chunk) > 0 {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: string(chunk)}})
		}
	}

	// Flush every 200 ms; the idle watchdog rides the same tick off the total bytes
	// seen (capture.total grows even after the editor cap, so a producing command is
	// never mistaken for idle). If that count hasn't grown for idleTimeout the
	// command is silent/hung, so SIGKILL it (cascades: Wait returns, pipeW closes,
	// the drain hits EOF, drainDone fires, this loop exits). idleTimeout <= 0
	// disables the watchdog (a build may legitimately go quiet); the scanErr kill
	// below still applies.
	ticker := time.NewTicker(200 * time.Millisecond)
	lastLen, lastGrow, idleKilled := 0, time.Now(), false
	for done := false; !done; {
		select {
		case <-ticker.C:
			flush()
			mu.Lock()
			total := capture.total
			mu.Unlock()
			if total > lastLen {
				lastLen, lastGrow = total, time.Now()
			} else if idleTimeout > 0 && !idleKilled && time.Since(lastGrow) > idleTimeout {
				idleKilled = true
				cancelCmd()
			}
		case <-drainDone:
			done = true
		}
	}
	ticker.Stop()
	flush() // final tail past the last tick

	mu.Lock()
	out := capture.String()
	switch {
	case idleKilled:
		out += fmt.Sprintf("\n[killed: no output for %s, command timed out]\n", idleTimeout)
	case scanErr != nil:
		// A line past the 1 MB buffer (or a read fault) ended the scan early; say so
		// in-band rather than presenting truncated output as complete.
		out += fmt.Sprintf("\n[output truncated: %s]\n", scanErr)
	}
	se := scanErr
	mu.Unlock()

	// A scan error stops the drain while the command may keep writing into a pipe
	// nobody reads; it then blocks on write and Wait would hang past the idle
	// watchdog (which exited on drainDone). Kill it so <-waitErr returns. (An
	// idle-kill already cancelled.)
	if se != nil {
		cancelCmd()
	}
	runErr = <-waitErr
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "```\n"}})
	return out, runErr, true
}

func runCmdExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	args := parseArgs(rawArgs)
	cmdStr := args["command"]
	if cmdStr == "" {
		return "error: command is required", false
	}
	sess := a.getSession(sid)
	if sess == nil {
		return "error: no session", false
	}

	tcId := a.StartToolCall(ctx, sid, "Run: "+cmdStr, "execute", nil)

	// Derive a cancellable ctx so the idle watchdog (and the scanErr guard) can
	// SIGKILL a silent or wedged command.
	cmdCtx, cancelCmd := context.WithCancel(ctx)
	defer cancelCmd()
	cmd := exec.CommandContext(cmdCtx, "bash", "-c", cmdStr)
	cmd.Dir = sess.Cwd

	out, runErr, started := a.runStreamingCmd(ctx, sid, cmdStr, cmd, cancelCmd, cmdIdleTimeout)
	if !started {
		a.FailToolCall(ctx, sid, tcId, runErr.Error())
		return "error starting bash: " + runErr.Error(), false
	}

	// Always surface the exit code. run_command is a probe: non-zero is data, not
	// failure. Title and result both carry "(exit N)" so the model can read either
	// and act on it. Failed is always false here: a probe exiting non-zero
	// shouldn't fail the turn.
	exitCode := 0
	if exitErr, ok := runErr.(*exec.ExitError); ok {
		exitCode = exitErr.ExitCode()
	} else if runErr != nil {
		// A start failure is handled above; here it's a kill-by-signal or the like.
		// Surface -1 plus the error text so the model can tell "command exited 1"
		// apart from "shell itself broke".
		exitCode = -1
		out += fmt.Sprintf("\n[exec error: %s]\n", runErr.Error())
	}

	result := fmt.Sprintf("exit %d\n\n%s", exitCode, out)
	a.CompleteToolCallTitled(ctx, sid, tcId,
		fmt.Sprintf("Run: %s (exit %d)", cmdStr, exitCode),
		[]ToolCallContent{TextContent(result)})
	return result, false
}
