package main

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// ACP terminals are how codehalter runs commands: the client starts the process
// and owns it, and we drive it by id. There is no in-process exec path; a
// client without clientCapabilities.terminal is refused at bootstrap.
//
// This is not a sandbox escape: Zed's remote server runs INSIDE the container
// and spawns codehalter there, so its terminals share our container and
// filesystem (run_background relies on reading a terminal's log file here).
//
// Two things differ from running the process ourselves. We never see a pid, so
// a child a command leaves behind (`npm run dev &`) is out of run_command's
// contract: never-exiting processes belong in run_background, and the deferred
// terminal/release kills whatever remains. And the idle watchdog is a poll:
// there is no output stream, so silence is detected by re-reading
// terminal/output on a timer, and kill latency is bounded by that interval.

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

// terminalCreate starts command+args in a client terminal and returns its id.
// terminal/create takes an argv, not a shell line, so a caller with a shell
// line passes ("bash", []string{"-c", line}) — that's what keeps pipes,
// redirects and `&&` working.
func (a *agent) terminalCreate(ctx context.Context, sid, command string, args []string, cwd string) (string, error) {
	if args == nil {
		args = []string{}
	}
	raw, err := a.conn.sendRequest(ctx, "terminal/create", map[string]any{
		"sessionId":       sid,
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
	raw, err := a.conn.sendRequest(ctx, "terminal/output", map[string]any{"sessionId": sid, "terminalId": tid})
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
	raw, err := a.conn.sendRequest(ctx, "terminal/wait_for_exit", map[string]any{"sessionId": sid, "terminalId": tid})
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
	_, err := a.conn.sendRequest(ctx, "terminal/kill", map[string]any{"sessionId": sid, "terminalId": tid})
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
	if _, err := a.conn.sendRequest(ctx, "terminal/release", map[string]any{"sessionId": sid, "terminalId": tid}); err != nil {
		slog.Debug("terminal/release failed", "terminal", tid, "err", err)
	}
}

// redirectRe finds the files a shell line writes into: `> f`, `>> f`, `&> f`,
// with an optional descriptor in front (`2> f`). `2>&1` names no file.
var redirectRe = regexp.MustCompile(`(?:^|[^&])(?:\d?>>?|&>)\s*([^\s;&|()<>]+)`)

// redirectTargets returns the files a `bash -c` command redirects into,
// resolved against cwd, so the watchdog can watch them grow. Anything that is
// not a plain path (a `$var`, a quote) is skipped: a wrong guess only means
// the file is not watched, which is where every command was before.
func redirectTargets(command string, args []string, cwd string) []string {
	if command != "bash" || len(args) != 2 || args[0] != "-c" {
		return nil
	}
	var files []string
	for _, m := range redirectRe.FindAllStringSubmatch(args[1], -1) {
		f := m[1]
		if strings.ContainsAny(f, "$\"'`*") || f == "/dev/null" {
			continue
		}
		if !filepath.IsAbs(f) {
			f = filepath.Join(cwd, f)
		}
		files = append(files, f)
	}
	return files
}

// progressSignature is what the watchdog compares between polls: the
// terminal's output, and the size of every file the command redirects into.
func progressSignature(out string, files []string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(out))
	for _, f := range files {
		size := int64(-1)
		if st, err := os.Stat(f); err == nil {
			size = st.Size()
		}
		_, _ = fmt.Fprintf(h, "\x00%s=%d", f, size)
	}
	return h.Sum64()
}
