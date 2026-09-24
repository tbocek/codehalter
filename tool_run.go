package main

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// discoverSandbox registers `run_command` whenever we're inside a container.
// The container itself is the sandbox: it's throwaway, the host workspace is
// bind-mounted (the LLM can already write to those files via edit_file), and
// apt-get/dpkg/pip writes are scoped to the container's lifetime. Devcontainers
// are expected to bind-mount `.git` read-only, so destructive git commands fail
// at the OS layer.
func (a *agent) discoverSandbox() {
	// Only register run_command inside a container — the container IS the
	// sandbox. Outside one, ensureDevcontainer aborts the session before any
	// prompt runs, so there is no "running on host with run_command disabled"
	// state to report; just skip registration.
	if containerKind() == "" {
		slog.Info("run_command: not registered (not inside a container)")
		return
	}

	a.tools.add(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "run_command",
			"description": "Run a shell command that EXITS ON ITS OWN inside this devcontainer and wait for it to finish. For a long-running / never-exits process (a dev server, watcher, `python3 -m http.server`, `npm run dev`) use `run_background` instead — started here it stalls the turn until the idle timeout kills it, and adding a trailing `&` is worse: the process then survives with no pid or log recorded, so nothing can read or stop it afterwards. The container is the sandbox: it's throwaway, so apt-get/dpkg/pip writes persist for the container's lifetime (wiped on rebuild) and workspace writes are real but recoverable from `.git/`. Reading code is NOT what this is for: `search_text` and `read_file` do that in one line-numbered call where `grep`/`sed`/`cat` here take two or three. Use this for: (1) PROBE — `which <tool>`, `cargo check`, `node --version`, `apt list --installed | grep <pkg>` — confirm what exists. (2) TEST INSTALL — when you're about to propose a Dockerfile edit (e.g. `RUN apt-get install <pkg>`), first run the same install via run_command, then verify it works (e.g. `<tool> --version` or re-running the failing build). If the install + verification succeed, propose the Dockerfile patch with confidence; if they fail, debug here before editing the Dockerfile. Exit code is always in the output and title — `which <tool>` exiting 1 means <tool> is missing, not that the tool failed. Output is auto-capped keeping the START and the END (only the middle is elided), so do NOT pipe to `head`/`tail` to shorten it: that throws away what the cap already keeps, and the most useful lines (errors, and search hits like `yay -Ss` / `apt search`) come LAST. Run the command raw; use `grep` only to filter for a specific match, never to trim length. " +
				"For project-file edits prefer `edit_file` / `write_file` — they go through the agent's diff/approval UI, raw `>` or `sed -i` do not. " +
				"The `.git` directory is bind-mounted read-only; destructive git commands (push, reset --hard, etc.) will fail at the filesystem layer. Read-only git is fine (clone, log, ls-remote, archive).",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"command"},
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command to run under bash -c. Be precise — this is not a chat. Run it raw: output is auto-capped (start+end kept), so do NOT append `| head` / `| tail` to limit size. Examples: `which <tool>`, `apt-get install -y <pkg> && <tool> --version`, `cargo check 2>&1`.",
					},
				},
			},
		},
	}, Execute: runCmdExecute})

	a.tools.add(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "run_background",
			"description": "Start a LONG-RUNNING / background process (a dev server, watcher, daemon) inside this devcontainer and return immediately, leaving it running. Use this INSTEAD of run_command for anything that does not exit on its own: `python3 -m http.server 8765`, `npm run dev`, `vite`, `flask run`, a file watcher. run_command WAITS for the command to finish, so starting a server there (even with a trailing `&`) hangs the turn. run_background launches the command, waits briefly to catch an immediate failure (e.g. port already in use), then returns the pid and a log-file path. The process keeps running across later tool calls, so a following run_command can probe it (e.g. `curl -s localhost:8765`). Its output streams to the log file, which you read with run_command (`cat`/`tail`). Stop it with `run_command: kill <pid>`. Also right for a LONG EXPERIMENT that does exit (a benchmark, a training run, a test suite you can keep working alongside): the moment it exits, codehalter hands you its exit code, log path and last output on its own, before your next step, so do other work meanwhile and never poll or `sleep` for it. If you cannot do anything until the result is in, it is not a background job: use run_command, which waits (a silent compile of up to two minutes is fine). Do NOT add a trailing `&` — run_background already detaches it.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"command"},
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command to run under bash -c, WITHOUT a trailing `&`. Examples: `python3 -m http.server 8765`, `npm run dev`, `flask --app app run --port 5000`.",
					},
				},
			},
		},
	}, Execute: runBackgroundExecute})
}

// cmdIdleTimeout reaps a run_command that prints NOTHING for this long: no
// output, no IO, nothing on the console. It is an IDLE timeout and deliberately
// not a total one. A command that keeps producing output is making progress and
// runs unbounded, however long that takes: a `grep -rln` over an archived
// lecture site (thousands of files, gigabytes of video) took 9 minutes here and
// exited 0 with the 14 matches that answered the question. A total cap would
// have killed that at the 5 minute mark and thrown the answer away, while the
// case it is supposed to catch, a hung command, is silent and this already
// catches it. Only a user Stop overrides.
//
// 120s rather than 60s because "quiet" is not "hung": a compile step, a
// download that buffers, or a walk over a slow bind mount can all go a full
// minute without printing.
var cmdIdleTimeout = 120 * time.Second

// cmdOutputCap bounds how many bytes of a command's output we hand the model.
// The idle watchdog only fires on SILENCE, so a steadily-printing command
// (`find /`, a chatty build, `journalctl`) would otherwise dump megabytes into a
// weak, small-context model. We keep a head + tail window and elide the middle,
// so a long run's start AND its trailing error both survive. A var so tests can
// shrink it.
var cmdOutputCap = 64 * 1024

// boundedOutput captures a byte stream in at most headCap+tailCap bytes: the
// first headCap as a frozen head, the most recent tailCap as a ring tail, the
// middle elided. It bounds memory for an unbounded command while keeping both
// ends (head shows how the run started; tail preserves the error verify reads).
// Not safe for concurrent use.
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
		// Both cuts are byte offsets and may split a character; drop the halves,
		// which would otherwise be invalid UTF-8 in the session file.
		omitted := b.total - b.headCap - b.tailCap
		head := strings.ToValidUTF8(string(b.head), "")
		return head + fmt.Sprintf("\n[... %d bytes omitted ...]\n", omitted) + strings.ToValidUTF8(string(tail), "")
	}
}

func runCmdExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	args := parseArgs(rawArgs)
	cmdStr := args.str("command")
	if cmdStr == "" {
		return "error: command is required", false
	}
	sess := a.getSession(sid)
	if sess == nil {
		return "error: no session", false
	}

	tcId := a.StartToolCall(ctx, sid, "Run: "+cmdStr, "execute", nil)

	out, exit, started, err := a.runTerminalCmd(ctx, sid, tcId, "bash", []string{"-c", cmdStr}, sess.Cwd, cmdIdleTimeout)
	if !started {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error starting terminal: " + err.Error(), false
	}

	// Always surface the exit code. run_command is a probe: non-zero is data, not
	// failure. Title and result both carry "(exit N)" so the model can read either
	// and act on it. Failed is always false here: a probe exiting non-zero
	// shouldn't fail the turn.
	exitCode := exit.code()
	if err != nil {
		// Killed by the idle watchdog, cancelled by the user, or the client broke
		// mid-command. -1 plus the error text lets the model tell "the command
		// exited 1" apart from "the command never got to finish".
		exitCode = -1
		out += fmt.Sprintf("\n[terminal error: %s]\n", err)
	}

	result := fmt.Sprintf("exit %d\n\n%s", exitCode, out)
	// The card already holds the terminal, which the client keeps rendering after
	// release. Sending text content here would replace that live view with a
	// static copy, so retitle only: a nil Content is omitted from the update, and
	// an absent field leaves the existing content alone.
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: tcId,
		Title:      fmt.Sprintf("Run: %s (exit %d)", cmdStr, exitCode),
		Status:     "completed",
	})
	return result, false
}
