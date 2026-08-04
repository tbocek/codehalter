package main

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log/slog"
	"time"
)

// ACP terminals are how codehalter runs commands: the client starts the process
// and owns it, and we drive it by id. There is no in-process exec path — a
// client that doesn't advertise clientCapabilities.terminal is refused at
// bootstrap (see ensureTerminals), rather than silently falling back to running
// processes ourselves.
//
// Handing the process to the client is not a sandbox escape. bootstrap.go
// directs the user to Zed's "Connect Dev Container", where Zed's remote server
// runs INSIDE the container and spawns codehalter there, so a terminal it
// creates is in the same container as everything else we do. run_background
// relies on the stronger form of that: the client's terminal and codehalter see
// the same filesystem, so a log file written by a terminal is readable here.
//
// Two things differ from running the process ourselves:
//
//   - We never see a pid, so there is no process group to own and no way to ask
//     whether a command left a child behind (`npm run dev &`). That case is out
//     of run_command's contract: it runs commands that exit on their own, and a
//     never-exits process belongs in run_background. run_command finishes
//     regardless, and the deferred terminal/release kills whatever the terminal
//     still holds. The steer to run_background lives in the tool description.
//   - The idle watchdog is a poll, not a stream. terminal/wait_for_exit blocks
//     and there is no output stream to watch, so silence is detected by
//     re-reading terminal/output on a timer. Same timer the exec path used;
//     only kill latency is quantised to the poll interval. See runTerminalCmd.

// terminalOutputLimit is the byte cap we ask the client to enforce. The client
// truncates from the FRONT (it keeps the tail), which loses the head of a long
// build — so we set it well above cmdOutputCap and let boundedOutput do the
// head+tail elision on the result instead. Only a command that outputs this
// much loses its opening lines.
const terminalOutputLimit = 4 * 1024 * 1024

// terminalExit is the exit status of a finished terminal. ExitCode is nil when
// the command was killed by a signal, which is why this isn't a bare int.
type terminalExit struct {
	ExitCode *int   `json:"exitCode"`
	Signal   string `json:"signal"`
}

// code flattens an exit status the way run_command reports it: the real code,
// or -1 for a signal death (which is also what a failure to exec reported).
func (e terminalExit) code() int {
	if e.ExitCode != nil {
		return *e.ExitCode
	}
	return -1
}

// clientSessionID maps a session id to one the client actually knows. Subagent
// sessions are minted locally as "sub_<parent>_…" and never announced, so a
// terminal request naming one is rejected exactly as fs/read_text_file is;
// their commands run on the parent's session instead. The loop (rather than a
// single hop) is for depth-2 subagents, whose parent is itself a sub_ session.
func (a *agent) clientSessionID(sid string) string {
	for i := 0; i < maxSubagentDepth+1; i++ {
		sess := a.getSession(sid)
		if sess == nil || sess.ParentID == "" {
			return sid
		}
		sid = sess.ParentID
	}
	return sid
}

// terminalCreate starts command+args in a client terminal and returns its id.
// terminal/create takes an argv, not a shell line, so a caller with a shell
// line passes ("bash", []string{"-c", line}) — that's what keeps pipes,
// redirects and `&&` working.
func (a *agent) terminalCreate(ctx context.Context, sid, command string, args []string, cwd string) (string, error) {
	if args == nil {
		args = []string{}
	}
	raw, err := a.conn.sendRequest(ctx, "terminal/create", map[string]any{
		"sessionId":       a.clientSessionID(sid),
		"command":         command,
		"args":            args,
		"cwd":             cwd,
		"outputByteLimit": terminalOutputLimit,
	})
	if err != nil {
		return "", err
	}
	var resp struct {
		TerminalId string `json:"terminalId"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", err
	}
	if resp.TerminalId == "" {
		return "", fmt.Errorf("terminal/create returned no terminalId")
	}
	return resp.TerminalId, nil
}

// terminalOutput fetches everything the terminal has produced so far. exit is
// nil while the command is still running.
func (a *agent) terminalOutput(ctx context.Context, sid, tid string) (out string, truncated bool, exit *terminalExit, err error) {
	raw, err := a.conn.sendRequest(ctx, "terminal/output", map[string]any{"sessionId": a.clientSessionID(sid), "terminalId": tid})
	if err != nil {
		return "", false, nil, err
	}
	var resp struct {
		Output     string        `json:"output"`
		Truncated  bool          `json:"truncated"`
		ExitStatus *terminalExit `json:"exitStatus"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", false, nil, err
	}
	return resp.Output, resp.Truncated, resp.ExitStatus, nil
}

// terminalWaitForExit blocks until the command finishes.
func (a *agent) terminalWaitForExit(ctx context.Context, sid, tid string) (terminalExit, error) {
	raw, err := a.conn.sendRequest(ctx, "terminal/wait_for_exit", map[string]any{"sessionId": a.clientSessionID(sid), "terminalId": tid})
	if err != nil {
		return terminalExit{}, err
	}
	var exit terminalExit
	if err := json.Unmarshal(raw, &exit); err != nil {
		return terminalExit{}, err
	}
	return exit, nil
}

// terminalKill kills the command but keeps the terminal id valid, so output
// produced before the kill can still be read.
func (a *agent) terminalKill(ctx context.Context, sid, tid string) error {
	_, err := a.conn.sendRequest(ctx, "terminal/kill", map[string]any{"sessionId": a.clientSessionID(sid), "terminalId": tid})
	return err
}

// terminalRelease kills the command if it is still running and invalidates the
// id — which is why a background job's terminal is held until shutdown and
// released only there. Best-effort by design: it runs on the way out of a tool
// call, including paths where the turn is already being torn down, and there is
// nothing useful to do with a failure beyond logging it.
func (a *agent) terminalRelease(sid, tid string) {
	// Deliberately not the caller's ctx: release is the cleanup for a cancelled
	// context, so reusing it would skip the kill exactly when it matters most.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := a.conn.sendRequest(ctx, "terminal/release", map[string]any{"sessionId": a.clientSessionID(sid), "terminalId": tid}); err != nil {
		slog.Debug("terminal/release failed", "terminal", tid, "err", err)
	}
}

// runTerminalCmd runs command+args to completion in a client terminal: create,
// show it live in the tool call's card, wait for exit, then read the full
// output back. idle > 0 kills a command that produces nothing new for that
// long; idle == 0 waits indefinitely (a build may legitimately go quiet while
// linking or fetching deps).
//
// The returned text is the captured output, already head+tail bounded.
// started=false means the terminal never came up, and err is the reason.
func (a *agent) runTerminalCmd(ctx context.Context, sid, tcId, command string, args []string, cwd string, idle time.Duration) (text string, exit terminalExit, started bool, err error) {
	tid, err := a.terminalCreate(ctx, sid, command, args, cwd)
	if err != nil {
		return "", terminalExit{}, false, err
	}
	defer a.terminalRelease(sid, tid)

	// Put the live terminal in the card before anything can release it — after
	// release the client keeps displaying it, but it won't accept the embed.
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: tcId,
		Status:     "in_progress",
		Content:    []ToolCallContent{TerminalContent(tid)},
	})

	type waitResult struct {
		exit terminalExit
		err  error
	}
	exited := make(chan waitResult, 1)
	go func() {
		e, werr := a.terminalWaitForExit(ctx, sid, tid)
		exited <- waitResult{e, werr}
	}()

	idleKilled := false
	if idle > 0 {
		// One poll per idle interval, not per tick: each poll drags the whole
		// accumulated output across the wire, so a chatty build would otherwise
		// re-transfer megabytes. The cost is that a silent command dies somewhere
		// between one and two intervals instead of at exactly one.
		ticker := time.NewTicker(idle)
		defer ticker.Stop()
		var lastHash uint64
		first := true
	wait:
		for {
			select {
			case res := <-exited:
				if res.err != nil {
					return "", terminalExit{}, true, res.err
				}
				exit = res.exit
				break wait
			case <-ctx.Done():
				// The user hit Stop, or the turn was cancelled. The deferred release
				// kills the command; report what it managed to print.
				out, _, _, _ := a.terminalOutput(context.Background(), sid, tid)
				return boundedCapture(out), terminalExit{}, true, ctx.Err()
			case <-ticker.C:
				out, _, status, oerr := a.terminalOutput(ctx, sid, tid)
				if oerr != nil {
					// Polling is only the watchdog; a failed poll must not fail the
					// command. Skip this round and let wait_for_exit decide.
					slog.Debug("terminal/output poll failed", "terminal", tid, "err", oerr)
					continue
				}
				if status != nil {
					continue // finished between ticks; wait_for_exit is about to return
				}
				h := hashOutput(out)
				if first || h != lastHash {
					first, lastHash = false, h
					continue
				}
				if idleKilled {
					continue // already killed; still waiting for wait_for_exit to land
				}
				// Nothing new for a whole interval: silent or hung. Kill it, but do
				// NOT release yet — we still want to read what it printed.
				idleKilled = true
				if kerr := a.terminalKill(ctx, sid, tid); kerr != nil {
					slog.Debug("terminal/kill failed", "terminal", tid, "err", kerr)
				}
			}
		}
	} else {
		select {
		case res := <-exited:
			if res.err != nil {
				return "", terminalExit{}, true, res.err
			}
			exit = res.exit
		case <-ctx.Done():
			out, _, _, _ := a.terminalOutput(context.Background(), sid, tid)
			return boundedCapture(out), terminalExit{}, true, ctx.Err()
		}
	}

	out, truncated, _, oerr := a.terminalOutput(ctx, sid, tid)
	if oerr != nil {
		return "", exit, true, oerr
	}
	text = boundedCapture(out)
	if truncated {
		text = "[... earlier output truncated by the client ...]\n" + text
	}
	if idleKilled {
		text += fmt.Sprintf("\n[killed: no output for %s, command timed out]\n", idle)
	}
	return text, exit, true, nil
}

// boundedCapture puts a finished terminal's output through the head+tail window
// that bounds what any command can hand a small-context model.
func boundedCapture(out string) string {
	b := newBoundedOutput(cmdOutputCap)
	b.Write([]byte(out))
	return b.String()
}

// hashOutput fingerprints accumulated output for the idle check. A hash rather
// than a length because once the client's byte cap is reached the output stops
// growing while the command is still very much alive, and a length comparison
// would read that as silence.
func hashOutput(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}
