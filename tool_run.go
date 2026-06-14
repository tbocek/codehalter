package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"strings"
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

	// Derive a cancellable ctx so the idle watchdog can SIGKILL a silent command.
	cmdCtx, cancelCmd := context.WithCancel(ctx)
	defer cancelCmd()
	cmd := exec.CommandContext(cmdCtx, "bash", "-c", cmdStr)
	cmd.Dir = sess.Cwd

	pipeR, pipeW := io.Pipe()
	cmd.Stdout = pipeW
	cmd.Stderr = pipeW

	if err := cmd.Start(); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error starting bash: " + err.Error(), false
	}

	waitErr := make(chan error, 1)
	go func() {
		waitErr <- cmd.Wait()
		_ = pipeW.Close()
	}()

	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "\n```\n$ " + cmdStr + "\n"}})

	// Drain the command's output into a buffer in a goroutine, DECOUPLED from the
	// editor, so the command can NEVER block on write() behind a slow editor. The
	// old per-line sendUpdate over a synchronous io.Pipe deadlocked a high-volume
	// command: its write stalled behind the editor write, which holds the one conn
	// write lock. The drain always reads the pipe (the command runs to completion
	// regardless of editor speed); a 200ms ticker batches the new output to the UI.
	var mu sync.Mutex
	var buf []byte
	var scanErr error
	drainDone := make(chan struct{})
	go func() {
		defer close(drainDone)
		scanner := bufio.NewScanner(pipeR)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			mu.Lock()
			buf = append(buf, scanner.Bytes()...)
			buf = append(buf, '\n')
			mu.Unlock()
		}
		mu.Lock()
		scanErr = scanner.Err()
		mu.Unlock()
	}()

	flushed := 0
	flush := func() {
		mu.Lock()
		var chunk string
		if len(buf) > flushed {
			chunk = string(buf[flushed:])
			flushed = len(buf)
		}
		mu.Unlock()
		if chunk != "" {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: chunk}})
		}
	}
	// Flush every 200ms; the idle watchdog rides the same tick. `flushed` is the
	// running byte count produced so far — if it hasn't grown for cmdIdleTimeout
	// the command is silent/hung, so SIGKILL it (which cascades: Wait returns →
	// pipeW closes → the drain hits EOF → drainDone fires → this loop exits). A
	// command that keeps printing keeps lastGrow fresh and runs unbounded.
	ticker := time.NewTicker(200 * time.Millisecond)
	lastLen, lastGrow, idleKilled := 0, time.Now(), false
	for done := false; !done; {
		select {
		case <-ticker.C:
			flush()
			if flushed > lastLen {
				lastLen, lastGrow = flushed, time.Now()
			} else if !idleKilled && time.Since(lastGrow) > cmdIdleTimeout {
				idleKilled = true
				cancelCmd()
			}
		case <-drainDone:
			done = true
		}
	}
	ticker.Stop()
	flush() // final tail past the last tick

	var collected strings.Builder
	mu.Lock()
	collected.Write(buf)
	switch {
	case idleKilled:
		fmt.Fprintf(&collected, "\n[killed: no output for %s — command timed out]\n", cmdIdleTimeout)
	case scanErr != nil:
		// A line past the 1 MB buffer (or a read fault) ended the scan early —
		// say so in-band rather than presenting truncated output as complete.
		fmt.Fprintf(&collected, "\n[output truncated: %s]\n", scanErr)
	}
	mu.Unlock()

	// A scan error (e.g. a single line past the 1 MB buffer) stops the drain while
	// the command keeps writing into a pipe nobody reads — it blocks on write and
	// Wait would hang here, past the idle watchdog (which exited on drainDone).
	// Kill it so <-waitErr returns. (An idle-kill already cancelled.)
	if scanErr != nil {
		cancelCmd()
	}
	runErr := <-waitErr
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "```\n"}})

	// Always surface the exit code. run_command is a probe: non-zero is
	// data, not failure. Title and result both carry "(exit N)" so the
	// model can read either and act on it. Failed is always false here —
	// a probe exiting non-zero shouldn't fail the turn.
	exitCode := 0
	if exitErr, ok := runErr.(*exec.ExitError); ok {
		exitCode = exitErr.ExitCode()
	} else if runErr != nil {
		// Couldn't start bash, killed by signal, etc. Surface -1 plus the
		// error text so the model can tell "command exited 1" apart from
		// "shell itself broke".
		exitCode = -1
		fmt.Fprintf(&collected, "\n[exec error: %s]\n", runErr.Error())
	}

	result := fmt.Sprintf("exit %d\n\n%s", exitCode, collected.String())
	a.CompleteToolCallTitled(ctx, sid, tcId,
		fmt.Sprintf("Run: %s (exit %d)", cmdStr, exitCode),
		[]ToolCallContent{TextContent(result)})
	return result, false
}
