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

// discoverSandbox registers `run_command` whenever we're inside a container.
// The container itself is the sandbox: it's throwaway, the host workspace is
// bind-mounted (the LLM can already write to those files via edit_file), and
// apt-get/dpkg/pip writes are scoped to the container's lifetime. Devcontainers
// are expected to bind-mount `.git` read-only so destructive git commands fail
// at the OS layer; we no longer install a PATH shim for `git`.
func (a *agent) discoverSandbox() {
	if containerKind() == "" {
		a.runCmdStatus = "not inside a container (codehalter runs on host)"
		slog.Info("run_command: " + a.runCmdStatus)
		return
	}
	a.runCmdStatus = "available"

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

	cmd := exec.CommandContext(ctx, "bash", "-c", cmdStr)
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

	var collected strings.Builder
	scanner := bufio.NewScanner(pipeR)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text() + "\n"
		collected.WriteString(line)
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: line}})
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
		collected.WriteString("\n[exec error: " + runErr.Error() + "]\n")
	}

	result := fmt.Sprintf("exit %d\n\n%s", exitCode, collected.String())
	a.CompleteToolCallTitled(ctx, sid, tcId,
		fmt.Sprintf("Run: %s (exit %d)", cmdStr, exitCode),
		[]ToolCallContent{TextContent(result)})
	return result, false
}
