package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// runCmdShimDir holds a per-process temp directory that contains a `git`
// stub which exits non-zero. When run_command executes a shell command, we
// prepend this dir to PATH so any bare `git ...` invocation resolves to the
// stub instead of the real git. This is how we keep recoverable history safe
// without parsing shell strings: the LLM can spell `git` any way it likes
// (`git`, `command git`, `exec git`, `$(which git) push`) and bash's own
// PATH lookup lands on our stub. Absolute paths (`/usr/bin/git`) still
// bypass PATH — we rely on the tool description telling the model not to use
// them. File deletes outside `.git/` are recoverable from git; `rm -rf .git`
// is NOT blocked here, only `git` itself.
var runCmdShimDir string

// installGitShim creates a temp dir with an executable `git` stub. Called
// once per process from discoverSandbox. The dir leaks until process exit,
// which is fine — one small directory.
func installGitShim() (string, error) {
	dir, err := os.MkdirTemp("", "codehalter-shim-")
	if err != nil {
		return "", err
	}
	stub := filepath.Join(dir, "git")
	content := `#!/bin/sh
echo "run_command: git is blocked — read code with read_file; the user runs git themselves" >&2
exit 127
`
	if err := os.WriteFile(stub, []byte(content), 0o755); err != nil {
		os.RemoveAll(dir)
		return "", err
	}
	return dir, nil
}

// envWithGitShim returns os.Environ with PATH rewritten so runCmdShimDir
// comes first. Other PATH entries are preserved in order so `which`,
// `apt-get`, etc. still resolve normally. Callers must have set
// runCmdShimDir; if it's empty we return env unchanged (and git would be
// reachable — but discoverSandbox refuses to register the tool in that
// case, so this path shouldn't fire).
func envWithGitShim() []string {
	env := os.Environ()
	if runCmdShimDir == "" {
		return env
	}
	out := make([]string, 0, len(env))
	var existing string
	for _, e := range env {
		if strings.HasPrefix(e, "PATH=") {
			existing = strings.TrimPrefix(e, "PATH=")
			continue
		}
		out = append(out, e)
	}
	out = append(out, "PATH="+runCmdShimDir+":"+existing)
	return out
}

// discoverSandbox registers `run_command` whenever we're inside a container.
// The container itself is the sandbox: it's throwaway, the host workspace is
// bind-mounted (the LLM can already write to those files via edit_file), and
// apt-get/dpkg/pip writes are scoped to the container's lifetime. To keep
// recoverable history intact we install a PATH shim so `git` resolves to a
// stub that errors out — that way the LLM can't run history-destroying
// commands like `git reset --hard` or `git push --force`.
func (a *agent) discoverSandbox() {
	if containerKind() == "" {
		a.runCmdStatus = "not inside a container (codehalter runs on host)"
		slog.Info("run_command: " + a.runCmdStatus)
		return
	}
	if runCmdShimDir == "" {
		dir, err := installGitShim()
		if err != nil {
			a.runCmdStatus = "failed to install git shim: " + err.Error()
			slog.Error("run_command: " + a.runCmdStatus)
			return
		}
		runCmdShimDir = dir
	}
	a.runCmdStatus = "available"

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "run_command",
			"description": "Run a shell command directly inside this devcontainer. The container is the sandbox: it's throwaway, so apt-get/dpkg/pip writes persist for the container's lifetime (wiped on rebuild) and workspace writes are real but recoverable from `.git/`. Use this for: (1) PROBE — `which tinygo`, `cargo check`, `node --version`, `apt list --installed | grep X` — confirm what exists. (2) TEST INSTALL — when you're about to propose a Dockerfile edit (e.g. `RUN apt-get install tinygo`), first run the same install via run_command, then verify it works (e.g. `tinygo version` or re-running the failing build). If the install + verification succeed, propose the Dockerfile patch with confidence; if they fail, debug here before editing the Dockerfile. Exit code is always in the output and title — `which X` exiting 1 means X is missing, not that the tool failed. " +
				"DO NOT USE for file inspection or editing — use the dedicated tools instead: " +
				"`cat file` / `head` / `tail` / `sed -n 'A,Bp'` → `read_file` (with `line` + `limit`); " +
				"`grep -n PATTERN file` → `read_file` with `grep`/`before`/`after`; " +
				"`grep -rn PATTERN .` → `search_text`; " +
				"`sed -i 's/X/Y/g' file` → `sed_file` (regex substitute or line delete); " +
				"slicing a prior tool's output → `view_output`. These route through the agent's diff/cache layer; raw shell duplicates them and burns LLM turns. " +
				"BLOCKED: `git` resolves to a stub that exits 127, so history-rewriting commands (reset --hard, push --force, branch -D, filter-branch) cannot destroy recoverable history. Read code with read_file, not `git show` / `git log`. The user runs git themselves.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"command"},
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command to run under bash -c. Be precise — this is not a chat. Examples: `which tinygo`, `apt-get install -y tinygo && tinygo version`, `cargo check 2>&1 | tail -20`. Do not invoke git — it is shimmed to exit 127. Do not use this for cat/head/tail/grep/sed against project files — use read_file / search_text / sed_file / view_output instead.",
					},
				},
			},
		},
	}, Execute: runCmdExecute})
}

func runCmdExecute(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
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
	cmd.Env = envWithGitShim()

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
