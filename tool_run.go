package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
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
		note, told := sess.toolHints(cmdStr, out)
		if told != "" {
			a.say(ctx, sid, told+"\n")
		}
		return fmt.Sprintf("exit %d\n\n%s", exitCode, boundedCapture(out)) + note, false
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

// toolHints is the note a finished run_command result gets when the model
// took a shell route that a file tool does better: the read_file symbol call
// for a grep that found a definition, the edit_file call for a script that
// spliced a source file, the read_file call for a range read. A note, never
// a refusal: the command ran. Every time, with the concrete call filled in,
// because once per session did not move this model: after the first note it
// made 17 more range reads. The second return is the line the user sees in
// the chat, so a nudge is never invisible.
func (s *Session) toolHints(cmd, output string) (string, string) {
	var out string
	var seen []string
	if path, name := grepDefinitionHit(cmd, output, s.Cwd); name != "" {
		out += fmt.Sprintf("\n[codehalter: that hit is where `%s` is defined. To read the whole definition, read_file does it in one call, in any language: {\"path\": %q, \"symbol\": %q}%s; several at once as {\"reads\": [{\"path\": ..., \"symbol\": ...}, ...]}.]", name, path, name, symbolPreview(s.Cwd, path, name))
		seen = append(seen, "read_file symbol="+name+" instead of grep")
	}
	if target, ok := scriptEditTarget(cmd); ok {
		out += "\n[codehalter: that script edited " + target + ". " + scriptEditPreview(cmd, target) +
			" edit_file checks that old_text (or start) matches exactly one place, applies the change, shows the user a diff, and answers `file written successfully` or says exactly why not; a script does none of that. Use edit_file for the next change.]"
		seen = append(seen, "edit_file instead of a script on "+target)
	}
	if h := rangeReadHint(cmd, s.Cwd); h != "" {
		out += h
		seen = append(seen, "read_file instead of sed/awk line ranges")
	}
	if len(seen) == 0 {
		return "", ""
	}
	return out, "💡 told the model: " + strings.Join(seen, "; ")
}

// symbolPreview says what a read_file symbol call would return, from the
// file as it is now: the line range, its length and its first line, so the
// note shows the result instead of describing it. Empty when the file cannot
// be read or the definition not delimited.
func symbolPreview(cwd, path, name string) string {
	data, err := os.ReadFile(filepath.Join(cwd, path))
	if err != nil {
		return ""
	}
	loc := locateSymbol(string(data), name)
	if loc.start == 0 {
		return ""
	}
	lines := strings.Split(string(data), "\n")
	first := ""
	for i := loc.start - 1; i < loc.end && i < len(lines); i++ {
		if t := strings.TrimSpace(lines[i]); t != "" && !strings.HasPrefix(t, "//") && !strings.HasPrefix(t, "#") && !strings.HasPrefix(t, "@") && !strings.HasPrefix(t, "*") && !strings.HasPrefix(t, "/*") {
			first = t
			break
		}
	}
	return fmt.Sprintf(", which returns lines %d-%d (%d lines, the whole block, found by %s), starting `%s`", loc.start, loc.end, loc.end-loc.start+1, loc.how, truncate(first, 100))
}

// grepHitRe is one line of grep -n output: "path:line:text" when grep names
// files, "line:text" for a single file (context lines use '-' and are not hits).
var grepHitRe = regexp.MustCompile(`^(?:([^:\s][^:]*):)?(\d+):(.*)$`)

// identRe is a name as most languages spell one.
var identRe = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*`)

// grepDefinitionHit finds, in a grep's output, the first hit whose text
// declares the name the grep searched for (declKeywordRe, the same detection
// read_file's symbol mode uses), and returns its file relative to the
// project and the declared name. A single-file grep's path is its last argument. Empty when the
// command is not a grep or no hit declares anything.
func grepDefinitionHit(cmd, output, cwd string) (string, string) {
	if !regexp.MustCompile(`\b(grep|rg)\b`).MatchString(cmd) {
		return "", ""
	}
	dir := ""
	if m := cdPrefixRe.FindStringSubmatch(cmd); m != nil {
		dir = m[1]
	}
	single := ""
	if f := strings.Fields(regexp.MustCompile(`[;|&]`).Split(cmd[strings.LastIndex(cmd, "grep"):], 2)[0]); len(f) > 0 {
		single = strings.Trim(f[len(f)-1], `'"`)
	}
	for _, ln := range strings.Split(output, "\n") {
		m := grepHitRe.FindStringSubmatch(ln)
		if m == nil {
			continue
		}
		file, text := m[1], m[3]
		if file == "" {
			file = single
		}
		if file == "" || strings.HasPrefix(file, "-") {
			continue
		}
		loc := declKeywordRe.FindStringIndex(text + " ")
		if loc == nil {
			continue
		}
		name := identRe.FindString(text[min(loc[1], len(text)):])
		// The hit must declare what the model was looking for: `let col =
		// cut_form_column(…)` declares col, which nobody grepped for.
		if name == "" || len(name) < 3 || !regexp.MustCompile(`\b`+regexp.QuoteMeta(name)+`\b`).MatchString(cmd) {
			continue
		}
		p := file
		if dir != "" && !filepath.IsAbs(p) {
			p = filepath.Join(dir, p)
		}
		if filepath.IsAbs(p) {
			if rel, err := filepath.Rel(cwd, p); err == nil && !strings.HasPrefix(rel, "..") {
				p = rel
			}
		}
		return filepath.ToSlash(p), name
	}
	return "", ""
}

// pyReplaceRe finds a Python `.replace(A, B)` with two string literals:
// triple-quoted, double- or single-quoted.
var pyReplaceRe = regexp.MustCompile(`\.replace\(\s*(` + pyStr + `)\s*,\s*(` + pyStr + `)\s*[,)]`)

const pyStr = `"""(?s:.*?)"""|'''(?s:.*?)'''|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'`

// pyLiteral decodes a Python string literal well enough for a preview: the
// quotes go, and in a plain literal the common escapes are resolved.
func pyLiteral(lit string) string {
	for _, q := range []string{`"""`, `'''`} {
		if strings.HasPrefix(lit, q) && strings.HasSuffix(lit, q) && len(lit) >= 6 {
			return lit[3 : len(lit)-3]
		}
	}
	if len(lit) < 2 {
		return lit
	}
	body := lit[1 : len(lit)-1]
	return strings.NewReplacer(`\n`, "\n", `\t`, "\t", `\"`, `"`, `\'`, `'`, `\\`, `\`).Replace(body)
}

// scriptEditPreview spells out the edit_file call a splice script amounts
// to: for a `.replace(A, B)` the exact old_text/new_text call (several become
// one edits list), else the block form with the path filled in.
func scriptEditPreview(cmd, target string) string {
	ms := pyReplaceRe.FindAllStringSubmatch(cmd, -1)
	if len(ms) == 0 {
		return fmt.Sprintf("edit_file does the same as a block: {\"path\": %q, \"start\": \"<fragment of the block's first line>\", \"end\": \"<fragment of its last line>\", \"new_text\": \"<the new block>\"}, or several changes at once as {\"path\": %q, \"edits\": [...]}.", target, target)
	}
	quote := func(s string) string {
		b, _ := json.Marshal(truncate(s, 160))
		return string(b)
	}
	pair := fmt.Sprintf(`"old_text": %s, "new_text": %s`, quote(pyLiteral(ms[0][1])), quote(pyLiteral(ms[0][2])))
	if len(ms) == 1 {
		return fmt.Sprintf(`Its replace is exactly this edit_file call: {"path": %q, %s}.`, target, pair)
	}
	return fmt.Sprintf(`Its %d replaces are ONE edit_file call with an edits list: {"path": %q, "edits": [{%s}, ...the others likewise]}.`, len(ms), target, pair)
}

// scriptEditRe spots the script edits edit_file can do: a Python heredoc or
// -c that reads a file, replaces text in it (str.replace, or slicing between
// two index/find anchors) and writes it back. Other scripts, which compute,
// convert or generate, are not edits of that kind and get no note.
var (
	scriptEditRe   = regexp.MustCompile(`python3?\s+(-\s*<<|-c\b)`)
	scriptSpliceRe = regexp.MustCompile(`\.replace\(|\.index\(|\.find\(`)
	scriptWriteRe  = regexp.MustCompile(`open\([^)]*['"][wa]\+?['"]|\.write_text\(`)
	scriptPathRe   = regexp.MustCompile(`['"]([\w./-]+\.(rs|go|py|ts|tsx|js|jsx|c|cc|cpp|h|hpp|java|kt|swift|rb|vue|svelte|css|scss|html|toml|yaml|yml|md))['"]`)
)

// scriptEditTarget reports the source file a shell line rewrites the way
// edit_file would, if it does: the first quoted path with a source extension.
func scriptEditTarget(cmd string) (string, bool) {
	// A Python script that both splices text and writes it back, in either
	// order in its source.
	if !scriptEditRe.MatchString(cmd) || !scriptSpliceRe.MatchString(cmd) || !scriptWriteRe.MatchString(cmd) {
		return "", false
	}
	if m := scriptPathRe.FindStringSubmatch(cmd); m != nil {
		return m[1], true
	}
	return "", false
}

// rangeReadRe matches the shell's line-range reads: `sed -n 'a,bp' file`
// and `awk 'NR>=a && NR<=b ...' file`.
var (
	sedRangeRe = regexp.MustCompile(`sed -n '?(\d+),(\d+)p'? +([^\s;|&]+)`)
	awkRangeRe = regexp.MustCompile(`awk '[^']*NR *>= *(\d+)[^']*NR *<= *(\d+)[^']*' +([^\s;|&]+)`)
	cdPrefixRe = regexp.MustCompile(`^\s*cd +([^\s;&]+) *(&&|;)`)
)

// rangeReadHint: after a shell line that read files by line range, the
// read_file call that does the same, spelled out (see toolHints for when).
func rangeReadHint(cmd, cwd string) string {
	dir := ""
	if m := cdPrefixRe.FindStringSubmatch(cmd); m != nil {
		dir = m[1]
	}
	var reads []string
	add := func(from, to, file string) {
		a, _ := strconv.Atoi(from)
		b, _ := strconv.Atoi(to)
		if b < a || len(reads) == maxReadsPerCall {
			return
		}
		p := file
		if dir != "" && !filepath.IsAbs(p) {
			p = filepath.Join(dir, p)
		}
		if filepath.IsAbs(p) {
			if rel, err := filepath.Rel(cwd, p); err == nil && !strings.HasPrefix(rel, "..") {
				p = rel
			}
		}
		reads = append(reads, fmt.Sprintf(`{"path": %q, "line": %d, "limit": %d}`, filepath.ToSlash(p), a, b-a+1))
	}
	for _, m := range sedRangeRe.FindAllStringSubmatch(cmd, -1) {
		add(m[1], m[2], m[3])
	}
	for _, m := range awkRangeRe.FindAllStringSubmatch(cmd, -1) {
		add(m[1], m[2], m[3])
	}
	switch len(reads) {
	case 0:
		return ""
	case 1:
		return "\n[codehalter: read_file does this without the shell: " + reads[0] + "; for a whole function use \"symbol\" instead of line numbers.]"
	}
	return "\n[codehalter: read_file does all of these in ONE call without the shell: {\"reads\": [" + strings.Join(reads, ", ") + "]}; for a whole function use \"symbol\" instead of line numbers.]"
}
