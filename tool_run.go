package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"slices"
	"strconv"
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
			"name":        "run_command",
			"description": "ONLY for builds, tests, installs and git. NEVER for project files: find with `search` (not grep, rg, find), read with `read_file` (not cat, head, tail, sed -n, awk), change with `edit_file` (not sed -i, awk or a script), create with `write_file` (not echo > or cat <<EOF). Runs in this devcontainer and waits up to two minutes for the exit code and output. Still running then: it is NOT killed but goes on as a background job; you get its output so far, and its exit code and last output arrive by themselves before your next step. Nothing else to do: call `respond` saying you wait for it, which parks the turn until the job reports. Never `sleep` or poll for it. Known to take over two minutes: use `run_background`. Never ends (dev server, watcher, `npm run dev`): use `run_server`, and never a trailing `&` here (it would survive with no pid or log). The container is a throwaway sandbox: installs last until a rebuild, workspace writes are real and recoverable from git. Before proposing a Dockerfile change, run the same install here and verify it (`<tool> --version`, or the failing build again); propose the patch once that works. The exit code is in the output and title: `which <tool>` exiting 1 means the tool is missing. Output is capped keeping its START and END, so run commands raw: a pipe to `head`/`tail` throws away what the cap keeps, and errors come last. Use `grep` to filter for a match, not to trim length.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"command"},
				"properties": map[string]any{
					"wake_after": map[string]any{
						"type":        "integer",
						"description": "Optional, seconds: if it still runs in the background at this age, wake me once with its log tail. Its exit is reported separately.",
					},
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command, run under bash -c. Examples: `which <tool>`, `apt-get install -y <pkg> && <tool> --version`, `cargo check 2>&1`.",
					},
				},
			},
		},
	}, Execute: runCmdExecute})

	a.tools.add(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "run_background",
			"description": "Start a command that ENDS on its own but takes over two minutes (a full test suite, a long build, a benchmark) and return at once. It runs across later tool calls; when it exits, its exit code, log path and last output arrive by themselves before your next step. Meanwhile do other work, or call `respond` saying you wait: the step is parked, not ended. Never poll or `sleep` for it (a foreground `sleep` is refused while a job runs); for a look before it exits, give `wake_after`. A job using almost no CPU and writing no output for 5 minutes counts as stuck and is stopped. Never ends on its own (dev server, watcher): use `run_server`. Returns the pid and a log path to read with run_command (`tail`, `grep`). Stop it early with `run_command: kill -TERM -<pid>; sleep 3; kill -KILL -<pid> 2>/dev/null; true` (the whole job; a plain `kill <pid>` may be ignored here). No trailing `&`: it is detached already.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"command"},
				"properties": map[string]any{
					"wake_after": map[string]any{
						"type":        "integer",
						"description": "Optional, seconds: wake me once at this age with the log tail if it still runs (a mid-way look). The exit is reported separately.",
					},
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command, run under bash -c, no trailing `&`. Examples: `cd rust && just test > /tmp/test.log 2>&1`, `cargo build --release`.",
					},
				},
			},
		},
	}, Execute: runBackgroundExecute})

	a.tools.add(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "run_server",
			"description": "Start a process that never ends on its own: a dev server (`npm run dev`, `python3 -m http.server 8765`), a watcher, a daemon, a fake service a manual check talks to. It runs until `{\"stop\": <its job id>}` or the session ends; nothing waits for it and it is never stopped as idle. With `port`, the call returns once that port accepts connections (up to 60 s): do not sleep or poll for it. A command that ends (a test suite, a build) is no server: use run_command, or run_background past two minutes. No trailing `&`.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"command": map[string]any{
						"type":        "string",
						"description": "Shell command, run under bash -c, no trailing `&`. Examples: `python3 -m http.server 8765`, `npm run dev`.",
					},
					"port": map[string]any{
						"type":        "integer",
						"description": "Optional: the TCP port it listens on; the call waits until it accepts connections.",
					},
					"stop": map[string]any{
						"type":        "integer",
						"description": "Instead of command: the job id of a server you started, to stop it.",
					},
				},
			},
		},
	}, Execute: runServerExecute})
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

// processPollRe finds a loop whose condition asks about a process, the shape of a
// hand-made wait: `while pgrep -f x; do sleep 5; done`, `until ! kill -0 $pid; ...`.
var processPollRe = regexp.MustCompile(`\b(?:while|until)\b[^;\n]*?(?:\bpgrep\b|\bpidof\b|\bps\b|\bkill\s+-0\b|/proc/)`)

// pgrepRe finds `pgrep FLAGS PATTERN` and `pkill FLAGS PATTERN`: group 1 the flags
// (`-f`, `-af`, `-a -f`), then the pattern quoted or bare.
var pgrepRe = regexp.MustCompile(`\b(?:pgrep|pkill)((?:\s+-\w+)+)\s+(?:"([^"]*)"|'([^']*)'|([^\s;|&)]+))`)

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

// cmdOutputCap bounds what run_command returns; the terminal itself keeps up to
// terminalOutputLimit. A var so tests can shrink it.
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
	job, refusal := a.launchJob(ctx, sid, rawArgs, jobHandedOver)
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
		a.endJob(job)
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
		a.endJob(job)
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
		"Never sleep or poll for it. Read its output any time with `run_command: cat %s`; %s. Output so far:\n\n%s",
		waited, job.id, job.pid, wake, job.id, job.logPath, stopHint(job.pid), readLogTail(job.logPath, bgLogTailCap)), false
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
	return "\n[codehalter: that line put a command in the background with `&` and slept " + m[2] + " s to wait for it, which is a guess, and the backgrounded command also kept this terminal open. " +
			"run_background does it properly, and tells you the exit code and the last output the moment it finishes, with nothing to poll: {\"command\": " + strconv.Quote(job) + "}" +
			" (add \"wake_after\": seconds only if you want a look before it finishes). Meanwhile do other work, or call `respond` saying you are waiting.]",
		"💡 told the model: run_background instead of `& sleep`"
}
