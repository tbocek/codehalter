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

// Zed's remote server spawns codehalter inside the container, so the client's
// ACP terminals share our filesystem; they expose no pid and no output stream.

// terminalOutputLimit: the client truncates from the FRONT, so this sits far
// above cmdOutputCap and boundedCapture does the head+tail elision instead.
const terminalOutputLimit = 4 * 1024 * 1024

type terminalExit struct {
	ExitCode *int   `json:"exitCode"`
	Signal   string `json:"signal"`
}

func (e terminalExit) code() int {
	if e.ExitCode != nil {
		return *e.ExitCode
	}
	return -1
}

// terminal/create takes an argv, not a shell line; pass ("bash", {"-c", line})
// for pipes, redirects and `&&`.
func (a *agent) terminalCreate(ctx context.Context, sid, command string, args []string, cwd string) (string, error) {
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

func (a *agent) terminalOutput(ctx context.Context, sid, tid string) (string, error) {
	raw, err := a.conn.sendRequest(ctx, "terminal/output", map[string]any{"sessionId": sid, "terminalId": tid})
	if err != nil {
		return "", err
	}
	var resp struct {
		Output string `json:"output"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", err
	}
	return resp.Output, nil
}

// terminalKill keeps the terminal id valid, so output from before the kill can
// still be read.
func (a *agent) terminalKill(ctx context.Context, sid, tid string) error {
	_, err := a.conn.sendRequest(ctx, "terminal/kill", map[string]any{"sessionId": sid, "terminalId": tid})
	return err
}

// terminalRelease kills a still-running command and invalidates the id.
func (a *agent) terminalRelease(sid, tid string) {
	// Not the caller's ctx: release is the cleanup for a cancelled context.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := a.conn.sendRequest(ctx, "terminal/release", map[string]any{"sessionId": sid, "terminalId": tid}); err != nil {
		slog.Debug("terminal/release failed", "terminal", tid, "err", err)
	}
}

// redirectRe: `> f`, `>> f`, `&> f`, `2> f`; `2>&1` names no file.
var redirectRe = regexp.MustCompile(`(?:^|[^&])(?:\d?>>?|&>)\s*([^\s;&|()<>]+)`)

// shellHeredocRe opens a shell here-document: `<<EOF`, `<< 'eof'`, `<<-"END"`. An
// unquoted word is upper-case, as heredocRe's, so `1<<3` and Python's `1 << shift`
// in a `-c` string open none; `<<<` (a here-string) is ruled out by the caller.
var shellHeredocRe = regexp.MustCompile(`<<-?[ \t]*(?:'([A-Za-z_]\w*)'|"([A-Za-z_]\w*)"|\\?([A-Z_][A-Z0-9_]*)\b)`)

// stripHeredocBodies drops the bodies of a command's here-documents, which are
// another program's text: `while step>=1:` in a Python heredoc is no redirect to
// a file named `=1`. A body without its closing word runs to the end, as in bash.
func stripHeredocBodies(cmd string) string {
	if !strings.Contains(cmd, "<<") {
		return cmd
	}
	var out []string
	var pending []string // closing words still to come, in order
	for _, ln := range strings.Split(cmd, "\n") {
		if len(pending) > 0 {
			if strings.TrimLeft(ln, "\t") == pending[0] {
				pending = pending[1:]
			}
			continue
		}
		out = append(out, ln)
		for _, m := range shellHeredocRe.FindAllStringSubmatchIndex(ln, -1) {
			if m[0] > 0 && ln[m[0]-1] == '<' || m[1] < len(ln) && ln[m[1]] == '<' {
				continue // `<<<`
			}
			for g := 2; g < len(m); g += 2 {
				if m[g] >= 0 {
					pending = append(pending, ln[m[g]:m[g+1]])
				}
			}
		}
	}
	return strings.Join(out, "\n")
}

// redirectTargets skips anything that is not a plain path (a `$var`, a quote):
// a missed file is only not watched.
func redirectTargets(line, cwd string) []string {
	var files []string
	add := func(f string) {
		if strings.ContainsAny(f, "$\"'`*") || strings.HasPrefix(f, "/dev/") {
			return
		}
		if !filepath.IsAbs(f) {
			f = filepath.Join(cwd, f)
		}
		files = append(files, f)
	}
	line = stripHeredocBodies(line)
	for _, m := range redirectRe.FindAllStringSubmatch(line, -1) {
		add(m[1])
	}
	for _, m := range teeRe.FindAllStringSubmatch(line, -1) {
		for _, f := range strings.Fields(m[1]) {
			if !strings.HasPrefix(f, "-") {
				add(f)
			}
		}
	}
	return files
}

// teeRe: the files of a `| tee [-a] f…` in a pipeline.
var teeRe = regexp.MustCompile(`\|\s*tee((?:\s+[^\s;&|()<>]+)*)`)

func progressSignature(files []string) uint64 {
	h := fnv.New64a()
	for _, f := range files {
		size := int64(-1)
		if st, err := os.Stat(f); err == nil {
			size = st.Size()
		}
		fmt.Fprintf(h, "\x00%s=%d", f, size)
	}
	return h.Sum64()
}
