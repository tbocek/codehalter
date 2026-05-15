package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"strings"
)

// discoverSandbox registers `safe_run_command` when we're inside a container
// AND bwrap can create the required namespaces. Outside a container the tool
// would let the agent write to the host's /usr, /etc, etc. — there the
// "container is throwaway" assumption doesn't hold, so we don't expose it.
// The probe runs a no-op `true` under a trivial bwrap invocation; if that
// fails (no userns, kernel restricts unprivileged bwrap), we bail.
func (a *agent) discoverSandbox() {
	if containerKind() == "" {
		slog.Info("safe_run_command: not inside a container, tool not registered")
		return
	}
	if _, err := exec.LookPath("bwrap"); err != nil {
		slog.Info("safe_run_command: bwrap not on PATH, tool not registered")
		return
	}
	probe := exec.Command("bwrap", "--ro-bind", "/", "/", "--", "true")
	if err := probe.Run(); err != nil {
		slog.Info("safe_run_command: bwrap probe failed", "err", err)
		return
	}

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "safe_run_command",
			"description": "Run a shell command inside this devcontainer with the workspace overlay-protected: writes to the workspace are discarded on exit, but writes anywhere else in the container (e.g. `apt-get install`, `dpkg -i`, `pip install`) persist for the container's lifetime — they're gone after a devcontainer rebuild, but available immediately for testing. Use this for two patterns: (1) PROBE — `which tinygo`, `cargo check`, `node --version`, `apt list --installed | grep X` — confirm what exists. (2) TEST INSTALL — when you're about to propose a Dockerfile edit (e.g. `RUN apt-get install tinygo`), first run the same install via safe_run_command, then verify it works (e.g. `tinygo version` or re-running the failing build). If the install + verification succeed, propose the Dockerfile patch with confidence; if they fail, debug here before editing the Dockerfile. Exit code is always in the output and title — `which X` exiting 1 means X is missing, not that the tool failed.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"command"},
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command to run under bash -c. Be precise — this is not a chat. Examples: `which tinygo`, `apt-get install -y tinygo && tinygo version`, `cargo check 2>&1 | tail -20`.",
					},
				},
			},
		},
	}, Execute: safeRunExecute})
}

func safeRunExecute(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
	args := parseArgs(rawArgs)
	cmdStr := args["command"]
	if cmdStr == "" {
		return "error: command is required"
	}
	sess := a.getSession(sid)
	if sess == nil {
		return "error: no session"
	}

	tcId := a.StartToolCall(ctx, sid, "Safe-run: "+cmdStr, "execute", nil)

	// bwrap layout (container-only, see discoverSandbox):
	//   - / is bind-mounted READ-WRITE so `apt-get install`, `dpkg -i`,
	//     `pip install`, etc. actually take effect. Persistence is bounded by
	//     the container's lifetime — a devcontainer rebuild wipes everything
	//     not in the Dockerfile, which is exactly what makes this safe.
	//   - The workspace is overlay-mounted on top: lower = real cwd, upper =
	//     anonymous tmpfs. Writes there go into the upper layer and vanish on
	//     exit so the bind-mounted host workspace stays untouched.
	//   - /tmp is a fresh tmpfs so scratch writes don't escape.
	//   - /proc and /dev are minimal but real (some tools probe /proc).
	//   - --die-with-parent: when codehalter dies, the sandboxed shell dies
	//     too (no orphaned bwrap processes).
	bwrapArgs := []string{
		"--bind", "/", "/",
		"--tmpfs", "/tmp",
		"--proc", "/proc",
		"--dev", "/dev",
		"--overlay-src", sess.Cwd,
		"--tmp-overlay", sess.Cwd,
		"--chdir", sess.Cwd,
		"--die-with-parent",
		"--", "bash", "-c", cmdStr,
	}
	cmd := exec.CommandContext(ctx, "bwrap", bwrapArgs...)

	pipeR, pipeW := io.Pipe()
	cmd.Stdout = pipeW
	cmd.Stderr = pipeW

	if err := cmd.Start(); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error starting bwrap: " + err.Error()
	}

	waitErr := make(chan error, 1)
	go func() {
		waitErr <- cmd.Wait()
		_ = pipeW.Close()
	}()

	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("\n```\n$ "+cmdStr+"\n")))

	var collected strings.Builder
	scanner := bufio.NewScanner(pipeR)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text() + "\n"
		collected.WriteString(line)
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(line)))
	}
	runErr := <-waitErr
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("```\n")))

	// Always surface the exit code. safe_run_command is a probe: non-zero
	// is data, not failure. Title and result both carry "(exit N)" so the
	// model can read either and act on it.
	exitCode := 0
	if exitErr, ok := runErr.(*exec.ExitError); ok {
		exitCode = exitErr.ExitCode()
	} else if runErr != nil {
		// Couldn't start bwrap, killed by signal, etc. Surface -1 plus the
		// error text so the model can tell "command exited 1" apart from
		// "sandbox itself broke".
		exitCode = -1
		collected.WriteString("\n[bwrap error: " + runErr.Error() + "]\n")
	}

	result := fmt.Sprintf("exit %d\n\n%s", exitCode, collected.String())
	a.CompleteToolCallTitled(ctx, sid, tcId,
		fmt.Sprintf("Safe-run: %s (exit %d)", cmdStr, exitCode),
		[]ToolCallContent{TextContent(result)})
	return result
}
