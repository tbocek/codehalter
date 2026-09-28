package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"slices"
	"strings"
	"time"
)

// discoverSandbox registers run_command only inside a container, which is the
// sandbox; devcontainers are expected to bind-mount .git read-only.
func (a *agent) discoverSandbox() {
	if containerKind() == "" {
		slog.Info("run_command: not registered (not inside a container)")
		return
	}

	a.tools.add(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "run_command",
			"description": "Run a shell command inside this devcontainer and wait for it, up to two minutes. Exits in time (the normal case): you get its exit code and output. Still running after two minutes: it is NOT killed, it continues as a background job, you get what it printed so far, and the moment it exits codehalter hands you its exit code and last output by itself, before your next step; if you have nothing else to do, call `respond` saying you are waiting for it, which parks the turn (it does not end it) until the job reports. Never `sleep` or poll for it. If you already know a command runs longer than two minutes, start it with `run_background` instead and skip the wait. For a process that never exits (a dev server, a watcher, `npm run dev`) always use `run_background`, and never add a trailing `&` here: the process would survive with no pid or log recorded. The container is the sandbox: it's throwaway, so apt-get/dpkg/pip writes persist for the container's lifetime (wiped on rebuild) and workspace writes are real but recoverable from `.git/`. To FIND something in the tree: `grep -rn -C3 -F '<text>' src tests` here, naming the source directories to search rather than excluding the build one, one call that returns line numbers, the match and its context (`-E` for a regex). Keep to short options: a BusyBox grep (Alpine) rejects `--include`, `--exclude` and `--exclude-dir` and prints its usage instead. To READ a region you already know: `read_file` with a `line` range, or with `symbol` for a whole function or type, not `cat`/`sed -n`. To CHANGE a file: `edit_file` (with `start`/`end` for a whole block), never a Python, sed or awk script here: a script is not checked, not shown as a diff, and a wrong anchor rewrites the wrong span silently. Use this for: (1) PROBE — `which <tool>`, `cargo check`, `node --version`, `apt list --installed | grep <pkg>` — confirm what exists. (2) TEST INSTALL — when you're about to propose a Dockerfile edit (e.g. `RUN apt-get install <pkg>`), first run the same install via run_command, then verify it works (e.g. `<tool> --version` or re-running the failing build). If the install + verification succeed, propose the Dockerfile patch with confidence; if they fail, debug here before editing the Dockerfile. Exit code is always in the output and title — `which <tool>` exiting 1 means <tool> is missing, not that the tool failed. Output is auto-capped keeping the START and the END (only the middle is elided), so do NOT pipe to `head`/`tail` to shorten it: that throws away what the cap already keeps, and the most useful lines (errors, and search hits like `yay -Ss` / `apt search`) come LAST. Run the command raw; use `grep` only to filter for a specific match, never to trim length. " +
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

// isSleepCmd matches a line that starts with `sleep` (optionally after `cd`); a
// sleep later in the line is not a wait for a job.
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

// shellSegments splits a command line at ; && || & and newlines outside quotes,
// and at | too when pipes is set; without it a pipeline stays one command.
func shellSegments(cmd string, pipes bool) []string {
	var segs []string
	var cur strings.Builder
	var quote byte
	flush := func() {
		if s := strings.TrimSpace(cur.String()); s != "" {
			segs = append(segs, s)
		}
		cur.Reset()
	}
	for i := 0; i < len(cmd); i++ {
		c := cmd[i]
		switch {
		case quote != 0:
			if c == '\\' && quote == '"' && i+1 < len(cmd) {
				cur.WriteByte(c)
				i++
				c = cmd[i]
			} else if c == quote {
				quote = 0
			}
		case c == '\\' && i+1 < len(cmd):
			cur.WriteByte(c)
			i++
			c = cmd[i]
		case c == '\'' || c == '"':
			quote = c
		// A redirect's & (2>&1, &>f, |&) joins, it does not split.
		case c == '&' && (i > 0 && (cmd[i-1] == '>' || cmd[i-1] == '|') || i+1 < len(cmd) && cmd[i+1] == '>'):
		case c == '|' && !pipes && (i+1 >= len(cmd) || cmd[i+1] != '|') && (i == 0 || cmd[i-1] != '|'):
		case c == ';' || c == '\n' || c == '|' || c == '&':
			flush()
			continue
		}
		cur.WriteByte(c)
	}
	flush()
	return segs
}

// readOnlyTools only read files or print; git and find are checked further below.
var readOnlyTools = map[string]bool{
	"cat": true, "head": true, "tail": true, "sed": true, "awk": true, "grep": true, "egrep": true,
	"fgrep": true, "rg": true, "nl": true, "wc": true, "ls": true, "find": true, "git": true,
	"tree": true, "stat": true, "diff": true, "sort": true, "uniq": true, "cut": true, "tr": true,
	"cd": false, "echo": false, "printf": false, "pwd": false, "true": false,
}

var readOnlyGit = map[string]bool{
	"log": true, "show": true, "diff": true, "status": true, "ls-files": true, "grep": true,
	"blame": true, "rev-parse": true, "shortlog": true, "cat-file": true, "ls-tree": true,
}

// unquoted blanks each quoted span to a pair of quotes, so a > or a flag inside an
// argument is not seen as one.
func unquoted(seg string) string {
	var b strings.Builder
	var quote byte
	for i := 0; i < len(seg); i++ {
		c := seg[i]
		switch {
		case quote != 0:
			if c == '\\' && quote == '"' {
				i++
			} else if c == quote {
				quote = 0
				b.WriteString("''")
			}
		case c == '\'' || c == '"':
			quote = c
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// awkWritesRe: an awk program that runs a command or prints into a file or a pipe.
var awkWritesRe = regexp.MustCompile(`system\(|\bprintf?\b[^;}]*(?:>|\|)\s*"`)

// onlyReads: every segment is a file read, a search or a print, and nothing is
// written. Such a command is served like read_file, not clipped to head and tail.
func onlyReads(cmd string) bool {
	if strings.Contains(cmd, "$(") || strings.Contains(cmd, "`") || strings.Contains(cmd, "<<") {
		return false
	}
	reads := false
	for _, seg := range shellSegments(cmd, true) {
		fields := strings.Fields(unquoted(seg))
		if len(fields) == 0 {
			continue
		}
		tool := fields[0]
		reader, known := readOnlyTools[tool]
		if !known {
			return false
		}
		reads = reads || reader
		sub := ""
		for i, f := range fields[1:] {
			switch {
			case strings.Contains(f, ">") && f != "2>&1" && f != "2>/dev/null" && f != ">/dev/null" && f != "&>/dev/null":
				return false
			case tool == "sed" && (strings.HasPrefix(f, "--in-place") || !strings.HasPrefix(f, "--") && strings.HasPrefix(f, "-") && strings.Contains(f, "i")),
				tool == "awk" && f == "-i",
				tool == "find" && (strings.HasPrefix(f, "-exec") || strings.HasPrefix(f, "-ok") || strings.HasPrefix(f, "-fprint") || f == "-fls" || f == "-delete"),
				tool == "sort" && (strings.HasPrefix(f, "--output") || !strings.HasPrefix(f, "--") && strings.HasPrefix(f, "-") && strings.Contains(f, "o")),
				tool == "tree" && f == "-o",
				strings.HasPrefix(f, "--output"):
				return false
			case tool == "git" && sub == "" && !strings.HasPrefix(f, "-") && (i == 0 || fields[i] != "-C" && fields[i] != "-c"):
				sub = f
			}
		}
		if tool == "git" && !readOnlyGit[sub] || tool == "awk" && awkWritesRe.MatchString(seg) {
			return false
		}
		// `uniq in out` writes its second name.
		if tool == "uniq" && len(slices.DeleteFunc(slices.Clone(fields[1:]), func(f string) bool { return strings.HasPrefix(f, "-") })) > 1 {
			return false
		}
	}
	return reads
}

// cmdOutputCap: the idle watchdog fires only on silence, so a steadily printing
// command needs a byte cap. A var so tests can shrink it.
var cmdOutputCap = 64 * 1024

// cmdHandoverWait is when run_command hands a still-running command to the
// background; nothing is killed. A var so tests can shorten it.
var cmdHandoverWait = 120 * time.Second

// boundedCapture keeps the first quarter and the last three quarters; the byte
// cuts may split a rune, which ToValidUTF8 drops.
func boundedCapture(out string) string {
	if len(out) <= cmdOutputCap {
		return out
	}
	head := cmdOutputCap / 4
	tail := cmdOutputCap - head
	return strings.ToValidUTF8(out[:head], "") +
		fmt.Sprintf("\n[... %d bytes omitted ...]\n", len(out)-head-tail) +
		strings.ToValidUTF8(out[len(out)-tail:], "")
}

func runCmdExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	job, refusal := a.launchJob(ctx, sid, rawArgs, true)
	if job == nil {
		return refusal, false
	}
	hint := func() string {
		note, told := toolHints(job.cmdStr)
		if told != "" {
			a.say(ctx, sid, told+"\n")
		}
		return note
	}

	select {
	case res := <-job.exited:
		out, oerr := a.terminalOutput(ctx, sid, job.terminalId)
		a.terminalRelease(sid, job.terminalId)
		a.forgetBgJob(job)
		if res.err != nil {
			// -1 tells "never got to finish" apart from a real exit 1.
			a.FailToolCall(ctx, sid, job.tcId, res.err.Error())
			return fmt.Sprintf("exit -1\n\n%s\n[terminal error: %s]\n", boundedCapture(out), res.err), false
		}
		if oerr != nil {
			slog.Debug("run_command: output read failed", "job", job.id, "err", oerr)
		}
		// A non-zero exit is data, not failure, so failed stays false.
		exitCode := res.exit.code()
		a.retitleToolCall(ctx, sid, job.tcId, fmt.Sprintf("Run: %s (exit %d)", job.cmdStr, exitCode), "completed")
		return fmt.Sprintf("exit %d\n\n%s", exitCode, boundedCapture(out)) + hint(), false
	case <-ctx.Done():
		out, oerr := a.terminalOutput(context.Background(), sid, job.terminalId)
		if oerr != nil {
			slog.Debug("run_command: output read after cancel failed", "job", job.id, "err", oerr)
		}
		a.killJob(job)
		a.terminalRelease(sid, job.terminalId)
		a.forgetBgJob(job)
		a.FailToolCall(ctx, sid, job.tcId, ctx.Err().Error())
		return fmt.Sprintf("exit -1\n\n%s\n[terminal error: %s]\n", boundedCapture(out), ctx.Err()), false
	case <-time.After(cmdHandoverWait):
	}

	wake := a.handOver(job)
	waited := humanDuration(cmdHandoverWait.Milliseconds())
	a.retitleToolCall(ctx, sid, job.tcId, fmt.Sprintf("Run: %s (still running after %s, continues as job %d)", job.cmdStr, waited, job.id), "in_progress")
	return hint() + "\n" + fmt.Sprintf("still running after %s: it continues as background job %d (pid %d), nothing was killed. "+
		"When it exits, codehalter hands you its exit code and last output by itself, before your next step.%s "+
		"Do other work meanwhile if there is any; if not, call `respond` saying you are waiting for job %d: that parks the turn, it does not end it, and you continue here the moment the job reports. "+
		"Never sleep or poll for it. Read its output any time with `run_command: cat %s`; stop it with `run_command: kill %d`. Output so far:\n\n%s",
		waited, job.id, job.pid, wake, job.id, job.logPath, job.pid, readLogTail(job.logPath, bgLogTailCap)), false
}

// toolHints returns (note for the model, chat line for the user). There are
// deliberately no notes after grep or sed reads: the executor ignored them.
func toolHints(cmd string) (string, string) {
	if note, told := bgSleepHint(cmd); note != "" {
		return note, told
	}
	target, ok := scriptEditTarget(cmd)
	if !ok {
		return "", ""
	}
	return "\n[codehalter: that script edited " + target + ". " + scriptEditPreview(cmd, target) +
			" edit_file checks that old_text (or start) matches exactly one place, applies the change, shows the user a diff, and answers `file written successfully` or says exactly why not; a script does none of that. Use edit_file for the next change.]",
		"💡 told the model: edit_file instead of a script on " + target
}

var pyReplaceRe = regexp.MustCompile(`\.replace\(\s*(` + pyStr + `)\s*,\s*(` + pyStr + `)\s*[,)]`)

const pyStr = `"""(?s:.*?)"""|'''(?s:.*?)'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'`

func scriptEditPreview(cmd, target string) string {
	ms := pyReplaceRe.FindAllStringSubmatch(cmd, -1)
	if len(ms) == 0 {
		return fmt.Sprintf("edit_file does the same as a block: {\"path\": %q, \"start\": \"<fragment of the block's first line>\", \"end\": \"<fragment of its last line>\", \"new_text\": \"<the new block>\"}, or several changes at once as {\"path\": %q, \"edits\": [...]}.", target, target)
	}
	// quote decodes a Python literal well enough for a preview, not exactly.
	quote := func(lit string) string {
		s := lit
		switch {
		case len(lit) >= 6 && (strings.HasPrefix(lit, `"""`) && strings.HasSuffix(lit, `"""`) || strings.HasPrefix(lit, `'''`) && strings.HasSuffix(lit, `'''`)):
			s = lit[3 : len(lit)-3]
		case len(lit) >= 2:
			s = strings.NewReplacer(`\n`, "\n", `\t`, "\t", `\"`, `"`, `\'`, `'`, `\\`, `\`).Replace(lit[1 : len(lit)-1])
		}
		b, _ := json.Marshal(truncate(s, 160))
		return string(b)
	}
	pair := fmt.Sprintf(`"old_text": %s, "new_text": %s`, quote(ms[0][1]), quote(ms[0][2]))
	if len(ms) == 1 {
		return fmt.Sprintf(`Its replace is exactly this edit_file call: {"path": %q, %s}.`, target, pair)
	}
	return fmt.Sprintf(`Its %d replaces are ONE edit_file call with an edits list: {"path": %q, "edits": [{%s}, ...the others likewise]}.`, len(ms), target, pair)
}

// A Python heredoc or -c that splices a file and writes it back. Scripts that
// only compute or generate get no note.
var (
	scriptEditRe   = regexp.MustCompile(`python3?\s+(-\s*<<|-c\b)`)
	scriptSpliceRe = regexp.MustCompile(`\.replace\(|\.index\(|\.find\(`)
	scriptWriteRe  = regexp.MustCompile(`open\([^)]*['"][wa]\+?['"]|\.write_text\(`)
	scriptPathRe   = regexp.MustCompile(`['"]([\w./-]+\.(rs|go|py|ts|tsx|js|jsx|c|cc|cpp|h|hpp|java|kt|swift|rb|vue|svelte|css|scss|html|toml|yaml|yml|md))['"]`)
)

func scriptEditTarget(cmd string) (string, bool) {
	if !scriptEditRe.MatchString(cmd) || !scriptSpliceRe.MatchString(cmd) || !scriptWriteRe.MatchString(cmd) {
		return "", false
	}
	if m := scriptPathRe.FindStringSubmatch(cmd); m != nil {
		return m[1], true
	}
	return "", false
}

func wrongToolHint(args toolArgs) string {
	switch {
	case args.has("reads") || args.has("symbol") || (args.has("path") && (args.has("line") || args.has("limit"))):
		return ": these arguments are read_file's. Call read_file with them, not run_command."
	case args.has("edits") || args.has("old_text") || args.has("start"):
		return ": these arguments are edit_file's. Call edit_file with them, not run_command."
	}
	return ""
}

// bgSleepRe finds `cmd & sleep N` inside a longer line, which isSleepCmd does
// not see. Group 1 is a backgrounded (subshell), group 2 the seconds.
var bgSleepRe = regexp.MustCompile(`(?:\(([^()]*)\)\s*&|(?:^|[^&])&)\s*(?:;\s*)?sleep\s+(\d+)`)

func bgSleepHint(cmd string) (string, string) {
	m := bgSleepRe.FindStringSubmatch(cmd)
	if m == nil {
		return "", ""
	}
	job := strings.TrimSpace(m[1])
	if job == "" {
		job = "<the command you put in the background>"
	}
	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(map[string]any{"command": job})
	return "\n[codehalter: that line put a command in the background with `&` and slept " + m[2] + " s to wait for it, which is a guess, and the backgrounded command also kept this terminal open. " +
			"run_background does it properly, and tells you the exit code and the last output the moment it finishes, with nothing to poll: " + strings.TrimSpace(buf.String()) +
			" (add \"wake_after\": seconds only if you want a look before it finishes). Meanwhile do other work, or call `respond` saying you are waiting.]",
		"💡 told the model: run_background instead of `& sleep`"
}
