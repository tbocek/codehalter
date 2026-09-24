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
			"description": "Run a shell command inside this devcontainer and wait for it, up to two minutes. Exits in time (the normal case): you get its exit code and output. Still running after two minutes: it is NOT killed, it continues as a background job, you get what it printed so far, and the moment it exits codehalter hands you its exit code and last output by itself, before your next step; if you have nothing else to do, call `respond` saying you are waiting for it, which parks the turn (it does not end it) until the job reports. Never `sleep` or poll for it. If you already know a command runs longer than two minutes, start it with `run_background` instead and skip the wait. For a process that never exits (a dev server, a watcher, `npm run dev`) always use `run_background`, and never add a trailing `&` here: the process would survive with no pid or log recorded. The container is the sandbox: it's throwaway, so apt-get/dpkg/pip writes persist for the container's lifetime (wiped on rebuild) and workspace writes are real but recoverable from `.git/`. To FIND something in the tree: `grep -rn -C3 -F --exclude-dir=target '<text>' <path>` here, one call that returns line numbers, the match and its context and skips the build directory (`-E` for a regex). To READ a region you already know: `read_file` with a `line` range, not `cat`/`sed -n`. Use this for: (1) PROBE — `which <tool>`, `cargo check`, `node --version`, `apt list --installed | grep <pkg>` — confirm what exists. (2) TEST INSTALL — when you're about to propose a Dockerfile edit (e.g. `RUN apt-get install <pkg>`), first run the same install via run_command, then verify it works (e.g. `<tool> --version` or re-running the failing build). If the install + verification succeed, propose the Dockerfile patch with confidence; if they fail, debug here before editing the Dockerfile. Exit code is always in the output and title — `which <tool>` exiting 1 means <tool> is missing, not that the tool failed. Output is auto-capped keeping the START and the END (only the middle is elided), so do NOT pipe to `head`/`tail` to shorten it: that throws away what the cap already keeps, and the most useful lines (errors, and search hits like `yay -Ss` / `apt search`) come LAST. Run the command raw; use `grep` only to filter for a specific match, never to trim length. " +
				"For project-file edits prefer `edit_file` / `write_file` — they go through the agent's diff/approval UI, raw `>` or `sed -i` do not. " +
				"The `.git` directory is bind-mounted read-only; destructive git commands (push, reset --hard, etc.) will fail at the filesystem layer. Read-only git is fine (clone, log, ls-remote, archive).",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"command"},
				"properties": map[string]any{
					"wake_after": map[string]any{
						"type":        "integer",
						"description": "Optional, seconds. If the command is still running after the two-minute wait and continues in the background, wake me once at this age with its log tail (the exit is reported separately, whenever it comes).",
					},
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
			"description": "Start a process inside this devcontainer and return immediately, leaving it running: a dev server, a watcher, a daemon, or any command you already know takes longer than two minutes (a full test suite, a benchmark), so that run_command's wait is not spent on it. Use this INSTEAD of run_command for anything that does not exit on its own: `python3 -m http.server 8765`, `npm run dev`, `vite`, `flask run`, a file watcher. run_command WAITS for the command to finish, so starting a server there (even with a trailing `&`) hangs the turn. run_background launches the command, waits briefly to catch an immediate failure (e.g. port already in use), then returns the pid and a log-file path. The process keeps running across later tool calls, so a following run_command can probe it (e.g. `curl -s localhost:8765`). Its output streams to the log file, which you read with run_command (`cat`/`tail`). Stop it with `run_command: kill <pid>`. Also right for a LONG EXPERIMENT that does exit (a benchmark, a training run, a test suite you can keep working alongside): the moment it exits, codehalter hands you its exit code, log path and last output on its own, before your next step, so do other work meanwhile, or simply end your turn saying the job is running: you are resumed with the result when it exits, whether or not the user has typed anything in between. Never poll and never `sleep` for it (a foreground `sleep` is refused while a job runs); to look at a job that does not exit, give `wake_after` and you are woken at that age with its log tail. Do NOT add a trailing `&` — run_background already detaches it.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"command"},
				"properties": map[string]any{
					"wake_after": map[string]any{
						"type":        "integer",
						"description": "Optional, seconds. Wake me once at this age with the log tail even if the job is still running (for a server or watcher that never exits, or a long run you want a mid-way look at). The exit is reported separately, whenever it comes. Without it you are woken only at exit.",
					},
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command to run under bash -c, WITHOUT a trailing `&`. Examples: `python3 -m http.server 8765`, `npm run dev`, `flask --app app run --port 5000`.",
					},
				},
			},
		},
	}, Execute: runBackgroundExecute})
}

// isSleepCmd reports a command whose only point is to pass time: `sleep N`,
// possibly followed by more (`sleep 90 && tail /tmp/t.log`), or `sleep` after a
// `cd`. A `sleep` inside a for-loop or after another command is not a wait
// for a job and passes.
func isSleepCmd(cmd string) bool {
	first := strings.TrimSpace(cmd)
	if i := strings.IndexAny(first, ";&|\n"); i >= 0 {
		head := strings.TrimSpace(first[:i])
		if strings.HasPrefix(head, "cd ") {
			first = strings.TrimSpace(first[i+1:])
			first = strings.TrimLeft(first, "&|; ")
		}
	}
	return first == "sleep" || strings.HasPrefix(first, "sleep ")
}

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

// cmdHandoverWait is how long run_command stays with a command before handing
// it to the background. Nothing is killed at this point: the command keeps
// running as a job, the model gets what it printed so far and is woken when
// it exits (or at its wake_after), and a turn that ends on `respond`
// meanwhile is parked, not finished. Two minutes because that is what a probe,
// a build or a short suite needs, and past it the model is better off doing
// something else. A var so tests can shorten it.
var cmdHandoverWait = 120 * time.Second

// boundedCapture puts a finished terminal's output through the head+tail window
// that bounds what any command can hand a small-context model.
func boundedCapture(out string) string {
	b := newBoundedOutput(cmdOutputCap)
	b.Write([]byte(out))
	return b.String()
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
	secs, ok := args.num("wake_after")
	if args.has("wake_after") && (!ok || secs < 0) {
		return "error: wake_after must be a number of seconds, 0 or absent for none", false
	}
	wakeAfter := time.Duration(secs) * time.Second
	// A foreground sleep while one of this session's background jobs runs is
	// the model waiting for that job by guessing a number: in one afternoon 10
	// of 22 background suites were followed by a 90-150 s sleep, and one of them
	// idled 36 s past the job's exit because a running shell cannot be
	// interrupted with the note. The job wakes the model on its own, so the
	// sleep is refused with the reason, and the turn either does other work or
	// parks on `respond` until the job reports (Claude Code refuses a
	// foreground sleep for the same reason).
	if isSleepCmd(cmdStr) {
		if jobs := a.runningBgJobs(sid); len(jobs) > 0 {
			return fmt.Sprintf("refused: do not sleep for a background job. %s still running; the moment it exits codehalter hands you its exit code and last output on its own. "+
				"Continue with other work, or call `respond` saying you are waiting: the turn is parked, not ended, and continues here when the job reports. "+
				"For a look at a job before it exits, give `wake_after` when you start it.", jobs), false
		}
	}

	tcId := a.StartToolCall(ctx, sid, "Run: "+cmdStr, "execute", nil)
	job, err := a.launchJob(ctx, sid, tcId, cmdStr, sess.Cwd, wakeAfter, true)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error starting terminal: " + err.Error(), false
	}

	select {
	case res := <-job.exited:
		out, _, _, oerr := a.terminalOutput(ctx, sid, job.terminalId)
		a.terminalRelease(sid, job.terminalId)
		a.forgetBgJob(job)
		if res.err != nil {
			// The client broke mid-command. -1 plus the error text lets the
			// model tell "the command exited 1" apart from "the command never
			// got to finish".
			a.FailToolCall(ctx, sid, tcId, res.err.Error())
			return fmt.Sprintf("exit -1\n\n%s\n[terminal error: %s]\n", boundedCapture(out), res.err), false
		}
		if oerr != nil {
			slog.Debug("run_command: output read failed", "job", job.id, "err", oerr)
		}
		// Always surface the exit code. run_command is a probe: non-zero is
		// data, not failure. Title and result both carry "(exit N)" so the
		// model can read either and act on it. Failed is always false here: a
		// probe exiting non-zero shouldn't fail the turn.
		exitCode := res.exit.code()
		// The card already holds the terminal, which the client keeps rendering
		// after release. Sending text content here would replace that live
		// view with a static copy, so retitle only.
		a.sendUpdate(ctx, sid, toolCallUpdate{
			Kind:       "tool_call_update",
			ToolCallId: tcId,
			Title:      fmt.Sprintf("Run: %s (exit %d)", cmdStr, exitCode),
			Status:     "completed",
		})
		return fmt.Sprintf("exit %d\n\n%s", exitCode, boundedCapture(out)), false
	case <-ctx.Done():
		// The user hit Stop, or the turn was cancelled. Release kills the
		// command; report what it managed to print.
		out, _, _, _ := a.terminalOutput(context.Background(), sid, job.terminalId)
		a.killJob(job)
		a.terminalRelease(sid, job.terminalId)
		a.forgetBgJob(job)
		a.FailToolCall(ctx, sid, tcId, ctx.Err().Error())
		return fmt.Sprintf("exit -1\n\n%s\n[terminal error: %s]\n", boundedCapture(out), ctx.Err()), false
	case <-time.After(cmdHandoverWait):
	}

	// Still running: it becomes a background job. Nothing is lost, the model
	// gets what it has so far, and the exit finds it wherever it is: before
	// its next step, parked on `respond`, or idle between turns.
	job.pid = readPidFile(job.pidPath)
	go a.watchBgJob(job)
	if wakeAfter > 0 {
		go a.wakeForBgJob(job)
	}
	waited := humanDuration(cmdHandoverWait.Milliseconds())
	wake := ""
	if wakeAfter > 0 {
		wake = fmt.Sprintf(" You asked to be woken after %s if it is still running by then.", humanDuration(wakeAfter.Milliseconds()))
	}
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: tcId,
		Title:      fmt.Sprintf("Run: %s (still running after %s, continues as job %d)", cmdStr, waited, job.id),
		Status:     "in_progress",
	})
	return fmt.Sprintf("still running after %s: it continues as background job %d (pid %d), nothing was killed. "+
		"When it exits, codehalter hands you its exit code and last output by itself, before your next step.%s "+
		"Do other work meanwhile if there is any; if not, call `respond` saying you are waiting for job %d: that parks the turn, it does not end it, and you continue here the moment the job reports. "+
		"Never sleep or poll for it. Read its output any time with `run_command: cat %s`; stop it with `run_command: kill %d`. Output so far:\n\n%s",
		waited, job.id, job.pid, wake, job.id, job.logPath, job.pid, readLogTail(job.logPath, bgLogTailCap)), false
}
