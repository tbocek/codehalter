package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// bgJobGrace is how long run_background waits after launch before returning, to
// capture startup output and catch a command that exits straight away (a server
// that can't bind, a typo). A process still running after this keeps running
// across later tool calls. A var so tests can shorten it.
var bgJobGrace = 700 * time.Millisecond

// bgLogTailCap bounds how many trailing bytes of a job's log we fold into the
// tool result; the model reads the full log via run_command.
const bgLogTailCap = 8 * 1024

// backgroundJob tracks one long-lived job so the agent can release its terminal
// at shutdown (releasing kills the process, which is exactly what we want then,
// and skipping it would leave a dev server holding a port for the container's
// life) and so the grace-period check can tell "still running" from "exited
// immediately".
type backgroundJob struct {
	id         int
	sid        string // the session that launched it; terminal calls are addressed by it
	cmdStr     string
	pid        int
	logPath    string
	pidPath    string
	terminalId string
	started    time.Time
	wakeAfter  time.Duration // >0: wake the model once at this age even if still running
	announced  bool          // named to the user at a turn end already (under bgMu)
}

// nextBgID reserves the next sequential job id (used to name the log file before
// the job is registered).
func (a *agent) nextBgID() int {
	a.bgMu.Lock()
	defer a.bgMu.Unlock()
	a.bgSeq++
	return a.bgSeq
}

func (a *agent) registerBgJob(job *backgroundJob) {
	a.bgMu.Lock()
	defer a.bgMu.Unlock()
	if a.bgJobs == nil {
		a.bgJobs = make(map[int]*backgroundJob)
	}
	a.bgJobs[job.id] = job
}

// runningBgJobs names the session's jobs still tracked, with how long each has
// run, for the sleep refusal and the turn-end line.
func (a *agent) runningBgJobs(sid string) string {
	a.bgMu.Lock()
	defer a.bgMu.Unlock()
	var names []string
	for _, j := range a.bgJobs {
		if j.sid == sid {
			names = append(names, fmt.Sprintf("job %d `%s` (%s so far)", j.id, truncate(j.cmdStr, 60), humanDuration(time.Since(j.started).Milliseconds())))
		}
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// sayRunningBgJobs is the last line of a turn that leaves a new job behind:
// the user is about to get the prompt back and should know what is still
// running, that the model picks the work up by itself when it exits, and that
// they need not wait for it. Each job is named once; a dev server that lives
// for hours would otherwise close every turn with the same line.
func (a *agent) sayRunningBgJobs(sess *Session) {
	a.bgMu.Lock()
	var names []string
	for _, j := range a.bgJobs {
		if j.sid == sess.ID && !j.announced {
			j.announced = true
			names = append(names, fmt.Sprintf("job %d `%s`", j.id, truncate(j.cmdStr, 60)))
		}
	}
	a.bgMu.Unlock()
	if len(names) == 0 {
		return
	}
	sort.Strings(names)
	a.say(context.Background(), sess.ID, "\n⏳ Running in the background: "+strings.Join(names, ", ")+". When it exits I pick up the work that was waiting on it and tell you; you can carry on meanwhile.\n")
}

// forgetBgJob drops a job from the table and removes its scratch files. Used
// both for a job that never stayed up and, via shutdownBackground, at exit.
func (a *agent) forgetBgJob(job *backgroundJob) {
	a.bgMu.Lock()
	delete(a.bgJobs, job.id)
	a.bgMu.Unlock()
	_ = os.Remove(job.logPath)
	_ = os.Remove(job.pidPath)
}

// shutdownBackground releases every still-tracked job's terminal on app exit —
// which kills the process, since that is what release does — and removes their
// scratch files. Called from main after the connection closes, alongside
// shutdownMCP.
func (a *agent) shutdownBackground() {
	a.bgMu.Lock()
	jobs := make([]*backgroundJob, 0, len(a.bgJobs))
	for _, j := range a.bgJobs {
		jobs = append(jobs, j)
	}
	a.bgJobs = nil
	a.bgMu.Unlock()
	for _, j := range jobs {
		if j.terminalId != "" && a.conn != nil {
			a.terminalRelease(j.sid, j.terminalId)
		}
		_ = os.Remove(j.logPath)
		_ = os.Remove(j.pidPath)
	}
}

// readLogTail returns the last max bytes of a job log (the whole file when
// smaller), prefixed with a marker when truncated. "" if the log can't be read.
func readLogTail(path string, max int) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	if len(data) > max {
		return "[... earlier output truncated ...]\n" + tailUTF8(string(data), max)
	}
	return string(data)
}

// readPidFile returns the pid bgScript recorded, or 0 if it isn't there yet.
func readPidFile(path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return 0
	}
	return pid
}

// runBackgroundExecute starts a long-running command in a client terminal that
// outlives this tool call, waits a short grace period to catch an immediate
// crash, then returns the pid + log path while the process keeps running.
// Cleanup happens at session exit (shutdownBackground) or when the model kills
// the pid via run_command.
func runBackgroundExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
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

	tcId := a.StartToolCall(ctx, sid, "Background: "+cmdStr, "execute", nil)

	id := a.nextBgID()
	logPath := filepath.Join(os.TempDir(), fmt.Sprintf("codehalter-bg-%d.log", id))
	pidPath := filepath.Join(os.TempDir(), fmt.Sprintf("codehalter-bg-%d.pid", id))
	job := &backgroundJob{id: id, sid: sid, cmdStr: cmdStr, logPath: logPath, pidPath: pidPath, started: time.Now(), wakeAfter: wakeAfter}

	// bgScript wraps the model's command so the job keeps the two handles the model
	// already knows how to use: a log file it reads with `run_command: cat …` and a
	// pid it stops with `run_command: kill …`. Neither is available from an ACP
	// terminal — the client owns the process and only ever tells us a terminal id —
	// so the shell inside the terminal produces them instead.
	//
	// This works because the client's terminal and codehalter share a filesystem
	// (Zed's dev-container flow runs both inside the container), which is the same
	// assumption run_command makes when the model cats a file a command just wrote.
	//
	// `cmd &` then `$!` rather than `$$` + exec: for a simple command bash execs it
	// directly into the backgrounded child, so `$!` is the real process, and for a
	// compound one (`cd x && npm run dev`) it's the subshell — the same pid the
	// old exec.Command path recorded. `wait` keeps the shell alive so the terminal
	// exits when the job does, which is what the grace check reads.
	script := fmt.Sprintf("exec > %s 2>&1\n%s &\necho $! > %s\nwait $!", logPath, cmdStr, pidPath)
	tid, err := a.terminalCreate(ctx, sid, "bash", []string{"-c", script}, sess.Cwd)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error starting terminal: " + err.Error(), false
	}
	job.terminalId = tid
	// Registered before the grace wait so a shutdown racing the launch still
	// releases the terminal instead of orphaning the process.
	a.registerBgJob(job)

	// The terminal is NOT released here: release kills the process, and this job
	// is meant to outlive the tool call. It stays embedded in the card, so the
	// user watches the dev server's output live for as long as it runs.
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: tcId,
		Status:     "in_progress",
		Content:    []ToolCallContent{TerminalContent(tid)},
	})

	// Grace window: catch an immediate exit (failed bind, bad command) before
	// reporting the job as running. Polled rather than a flat sleep so a fast
	// crash is reported without making every healthy launch wait it out.
	var exit *terminalExit
	deadline := time.Now().Add(bgJobGrace)
	for {
		_, _, status, oerr := a.terminalOutput(ctx, sid, tid)
		if oerr != nil {
			a.forgetBgJob(job)
			a.terminalRelease(sid, tid)
			a.FailToolCall(ctx, sid, tcId, oerr.Error())
			return "error reading terminal: " + oerr.Error(), false
		}
		if status != nil {
			exit = status
			break
		}
		if remaining := time.Until(deadline); remaining <= 0 {
			break
		} else if remaining > 50*time.Millisecond {
			time.Sleep(50 * time.Millisecond)
		} else {
			time.Sleep(remaining)
		}
	}

	tail := readLogTail(logPath, bgLogTailCap)
	if exit != nil {
		// It already exited, so there's nothing to keep alive: release the terminal
		// and drop the job (its output is in the result below).
		a.terminalRelease(sid, tid)
		a.forgetBgJob(job)
		result := fmt.Sprintf("background job %d exited immediately (exit %d) — it did not stay running. Likely a startup error (port already in use, bad command, missing file). Output:\n\n%s", id, exit.code(), tail)
		a.CompleteToolCallTitled(ctx, sid, tcId, fmt.Sprintf("Background: %s (exited %d)", cmdStr, exit.code()), []ToolCallContent{TextContent(result)})
		return result, false
	}

	job.pid = readPidFile(pidPath)
	stop := fmt.Sprintf("stop it with `run_command: kill %d`", job.pid)
	if job.pid == 0 {
		// bgScript writes the pid before anything slow happens, so an empty file
		// here means the shell died oddly. Say so rather than printing "kill 0",
		// which would signal the whole process group.
		stop = "its pid was not recorded, so it can only be stopped by ending the session"
	}
	wake := ""
	if wakeAfter > 0 {
		wake = fmt.Sprintf(" You also asked to be woken after %s if it is still running by then.", humanDuration(wakeAfter.Milliseconds()))
	}
	result := fmt.Sprintf("background job %d running (pid %d). It keeps running across tool calls and across turns. When it exits, codehalter reports the exit code and the last output by itself, so do NOT poll or sleep waiting for it: carry on with other work, or if there is none, end the turn with `respond` saying the job is running.%s Read its output any time with `run_command: cat %s` (or tail/grep it); %s. Output so far:\n\n%s",
		id, job.pid, wake, logPath, stop, tail)
	go a.watchBgJob(job)
	if wakeAfter > 0 {
		go a.wakeForBgJob(job)
	}
	// Retitle only: the card is holding the live terminal, and text content here
	// would replace it with a static snapshot taken at second one of a dev server.
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: tcId,
		Title:      fmt.Sprintf("Background: %s (pid %d)", cmdStr, job.pid),
		Status:     "completed",
	})
	return result, false
}

// bgNoteTailCap bounds the log tail carried in a job's finish note. Smaller than
// bgLogTailCap: the note lands in the conversation for good, and the full log
// stays on disk for the model to read.
const bgNoteTailCap = 3 * 1024

// bgWakeRetry is how often a finished job re-checks for a quiet moment while a
// turn is running.
const bgWakeRetry = 2 * time.Second

// watchBgJob waits for a running job to exit and hands its result to the
// session. This is what lets a long experiment run while the chat stays usable:
// the model ends its turn instead of polling, and the result comes back on its
// own. The terminal is released once the job is gone (nothing left to show live
// that the log doesn't have) and the log file is kept for the model to read.
func (a *agent) watchBgJob(job *backgroundJob) {
	exit, err := a.terminalWaitForExit(context.Background(), job.sid, job.terminalId)
	// Shutdown reaps every job and drops the table: a job that "exits" because
	// we killed it on the way out has nothing to report and nobody to report to.
	a.bgMu.Lock()
	_, tracked := a.bgJobs[job.id]
	delete(a.bgJobs, job.id)
	a.bgMu.Unlock()
	if !tracked {
		return
	}
	a.terminalRelease(job.sid, job.terminalId)
	_ = os.Remove(job.pidPath)
	sess := a.getSession(job.sid)
	if sess == nil {
		return
	}
	outcome := fmt.Sprintf("exited with code %d", exit.code())
	if err != nil {
		outcome = "was lost (" + err.Error() + ")"
	}
	took := humanDuration(time.Since(job.started).Milliseconds())
	sess.addBgNote(bgNote{
		line: fmt.Sprintf("background job %d `%s` %s after %s", job.id, truncate(job.cmdStr, 80), outcome, took),
		full: fmt.Sprintf("[codehalter, not the user: background job %d `%s` %s after %s. Full log: %s. Last output:]\n\n%s",
			job.id, job.cmdStr, outcome, took, job.logPath, readLogTail(job.logPath, bgNoteTailCap)),
	})
	a.deliverBgNotesWhenIdle(sess)
}

// wakeForBgJob is the timed half of a job started with wake_after: once, at
// that age, if the job is still tracked, the model gets the log tail as it
// would at exit, with the job left running. This is how a server that never
// exits gets looked at without the model sleeping in the foreground: the wait
// belongs to codehalter, so the user keeps the prompt meanwhile and a job that
// exits first is reported first.
func (a *agent) wakeForBgJob(job *backgroundJob) {
	time.Sleep(job.wakeAfter)
	a.bgMu.Lock()
	_, tracked := a.bgJobs[job.id]
	a.bgMu.Unlock()
	if !tracked {
		return // it exited (and was reported) or the session is gone
	}
	sess := a.getSession(job.sid)
	if sess == nil {
		return
	}
	age := humanDuration(time.Since(job.started).Milliseconds())
	sess.addBgNote(bgNote{
		line: fmt.Sprintf("background job %d `%s` still running after %s, as asked", job.id, truncate(job.cmdStr, 80), age),
		full: fmt.Sprintf("[codehalter, not the user: background job %d `%s` is still running after %s; this is the wake_after you asked for, not its exit. It is reported again when it exits. Full log: %s. Last output:]\n\n%s",
			job.id, job.cmdStr, age, job.logPath, readLogTail(job.logPath, bgNoteTailCap)),
	})
	a.deliverBgNotesWhenIdle(sess)
}

// deliverBgNotesWhenIdle reports finished jobs at the next quiet point and
// never interrupts a turn. Idle: codehalter runs a turn of its own
// so the model looks at the result and tells the user. A turn is running: the
// notes stay queued and Prompt delivers them when that turn ends
// (flushBgNotes); this loop only exists to catch the window where the turn has
// passed its flush but not yet released the lock.
func (a *agent) deliverBgNotesWhenIdle(sess *Session) {
	for sess.hasBgNotes() {
		// Never interrupts: a prompt the user sends while this runs replaces it,
		// as it would any other turn.
		ctx, release, ok := a.holdTurn(context.Background(), sess, false)
		if !ok {
			time.Sleep(bgWakeRetry)
			continue
		}
		notes := sess.takeBgNotes()
		if len(notes) == 0 {
			release()
			return
		}
		var full []string
		for _, n := range notes {
			a.say(ctx, sess.ID, "\n🔔 "+n.line+"\n\n")
			full = append(full, n.full)
		}
		prompt := strings.Join(full, "\n\n") + "\n\nContinue the work that was waiting on this result, if any was; otherwise tell the user what the result means. Do not start new work the user did not ask for."
		if err := a.runPromptTurn(ctx, sess, prompt); err != nil && !isCancelled(err) {
			slog.Warn("background job report turn failed", "sid", sess.ID, "err", err)
			a.say(context.Background(), sess.ID, "⚠ Could not report on the finished background job: "+err.Error()+"\n")
		}
		release()
		return
	}
}

// flushBgNotes is the mid-conversation half: called by Prompt when its turn has
// ended. Each finished job gets one line on screen, and its full note is stored
// as a message so the model sees it at the start of the next turn. No turn is
// started here: the user is reading an answer they asked for.
func (a *agent) flushBgNotes(ctx context.Context, sess *Session) {
	notes := sess.takeBgNotes()
	if len(notes) == 0 {
		return
	}
	var full []string
	for _, n := range notes {
		a.say(ctx, sess.ID, "\n🔔 "+n.line+". The log is at hand; ask about it whenever you like.\n")
		full = append(full, n.full)
	}
	sess.AddUser(strings.Join(full, "\n\n"))
	sess.saveOrLog()
}
