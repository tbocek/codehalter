package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// bgJobGrace catches a command that exits straight away (a server that cannot
// bind, a typo). A var so tests can shorten it.
var bgJobGrace = 700 * time.Millisecond

const bgLogTailCap = 8 * 1024

// backgroundJob is tracked so shutdown can release, and thereby kill, its
// terminal; otherwise a dev server would hold its port for the container's life.
type backgroundJob struct {
	id         int
	sid        string
	tcId       string
	cmdStr     string
	pid        int
	logPath    string
	pidPath    string
	terminalId string
	started    time.Time
	wakeAfter  time.Duration
	announced  bool // under bgMu
	// expectExit: a handed-over run_command, not a run_background job.
	expectExit bool
	// waitable: a `respond` parks for it, and the stall watchdog may kill it. A
	// handed-over run_command, or a run_background test, build or lint run; never
	// a server, which does not exit and is quiet on purpose.
	waitable  bool
	redirects []string      // growth counts as progress
	stalled   time.Duration // >0: killed by the stall watchdog (under bgMu)
	exited    chan jobExit
}

// jobExit is delivered once per job, to run_command's wait or, after handover,
// to watchBgJob.
type jobExit struct {
	exit terminalExit
	err  error
}

// ACP terminals expose no pid or log, so the wrapper writes its pid (a process
// group under a pty, killed whole by its trap) and tees the output to a log.
func (a *agent) launchJob(ctx context.Context, sid, rawArgs string, expectExit bool) (*backgroundJob, string) {
	args := parseArgs(rawArgs)
	cmdStr := args.str("command")
	cmds := shellSegments(cmdStr, false)
	if len(cmds) == 0 {
		return nil, "error: command is required" + wrongToolHint(args)
	}
	sess := a.getSession(sid)
	if sess == nil {
		return nil, "error: no session"
	}
	secs, ok := args.num("wake_after")
	if args.has("wake_after") && (!ok || secs < 0) {
		return nil, "error: wake_after must be a number of seconds, 0 or absent for none"
	}
	// An inline script on a picture file shows the model nothing: one step wrote 40
	// PIL crops of a snapshot and never looked at one.
	if pic := pictureRe.FindString(cmdStr); pic != "" && pictureScriptRe.MatchString(cmdStr) {
		return nil, fmt.Sprintf("refused: this works on a picture with a script, and a script cannot show it to you: only `screenshot` puts a picture in front of you. "+
			"Look at it with `screenshot` on the file; for a close look at one part, give `region` [x, y, width, height] in its pixels and that part comes back enlarged: "+
			`{"path": %q, "region": [0, 0, 800, 400]}. The reply for the whole picture states its size.`, strings.Trim(pic, `'" `))
	}
	// A foreground sleep while a job runs is a guessed wait: the job wakes the
	// model itself, and a running shell cannot be interrupted with the note.
	if expectExit && isSleepCmd(cmdStr) {
		if jobs := a.runningBgJobs(sid, nil); jobs != "" {
			return nil, fmt.Sprintf("refused: do not sleep for a background job. %s still running; the moment it exits codehalter hands you its exit code and last output on its own. "+
				"Continue with other work, or call `respond` saying you are waiting: the turn is parked, not ended, and continues here when the job reports. "+
				"For a look at a job before it exits, give `wake_after` when you start it.", jobs)
		}
	}
	verb := "Background"
	if expectExit {
		verb = "Run"
	}
	tcId := a.StartToolCall(ctx, sid, verb+": "+cmdStr, "execute", nil)

	// The last command decides: `cargo build && ./target/release/api` ends in a server.
	last := cmds[len(cmds)-1]
	waitable := expectExit || finiteRunRe.MatchString(last) && !serverRunRe.MatchString(last)

	a.bgMu.Lock()
	a.bgSeq++
	id := a.bgSeq
	a.bgMu.Unlock()
	job := &backgroundJob{
		id: id, sid: sid, tcId: tcId, cmdStr: cmdStr, started: time.Now(),
		// Our pid in the name: ids restart at 1 in every codehalter process.
		logPath:    filepath.Join(os.TempDir(), fmt.Sprintf("codehalter-%d-job-%d.log", os.Getpid(), id)),
		pidPath:    filepath.Join(os.TempDir(), fmt.Sprintf("codehalter-%d-job-%d.pid", os.Getpid(), id)),
		wakeAfter:  time.Duration(secs) * time.Second,
		expectExit: expectExit,
		waitable:   waitable,
		redirects:  redirectTargets(cmdStr, sess.Cwd),
		exited:     make(chan jobExit, 1),
	}
	// pipefail + wait: the exit status is the command's, not tee's.
	quoted := "'" + strings.ReplaceAll(cmdStr, "'", `'\''`) + "'"
	script := fmt.Sprintf("echo $$ > %s\nset -o pipefail\ntrap 'trap - TERM INT HUP; kill -- -$$ 2>/dev/null; exit 143' TERM INT HUP\n( exec bash -c %s ) 2>&1 | tee %s &\nwait $!", job.pidPath, quoted, job.logPath)
	tid, err := a.terminalCreate(ctx, sid, "bash", []string{"-c", script}, sess.Cwd)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return nil, "error starting terminal: " + err.Error()
	}
	job.terminalId = tid
	// Registered first so a shutdown racing the launch still releases the
	// terminal instead of orphaning the process.
	a.registerBgJob(job)
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: tcId,
		Status:     "in_progress",
		Content:    []ToolCallContent{{Type: "terminal", TerminalId: tid}},
	})
	go func() {
		var e terminalExit
		raw, err := a.conn.sendRequest(context.Background(), "terminal/wait_for_exit", map[string]any{"sessionId": sid, "terminalId": tid})
		if err == nil {
			err = json.Unmarshal(raw, &e)
		}
		job.exited <- jobExit{e, err}
	}()
	return job, ""
}

// handOver returns the wake_after sentence for the result, or "".
func (a *agent) handOver(job *backgroundJob) string {
	job.pid = readPidFile(job.pidPath)
	stop := make(chan struct{})
	if job.waitable {
		// Limits read here, not in the watcher, which may outlive a test that shortened them.
		go a.stallWatch(job, stop, bgStallTimeout, bgStallPoll)
	}
	go a.watchBgJob(job, stop)
	if job.wakeAfter <= 0 {
		return ""
	}
	go a.wakeForBgJob(job)
	return fmt.Sprintf(" You asked to be woken after %s if it is still running by then.", humanDuration(job.wakeAfter.Milliseconds()))
}

func (a *agent) registerBgJob(job *backgroundJob) {
	a.bgMu.Lock()
	defer a.bgMu.Unlock()
	if a.bgJobs == nil {
		a.bgJobs = make(map[int]*backgroundJob)
	}
	a.bgJobs[job.id] = job
}

func (a *agent) runningBgJobs(sid string, keep func(*backgroundJob) bool) string {
	a.bgMu.Lock()
	defer a.bgMu.Unlock()
	var names []string
	for _, j := range a.bgJobs {
		if j.sid == sid && (keep == nil || keep(j)) {
			names = append(names, fmt.Sprintf("job %d `%s` (%s so far)", j.id, truncate(j.cmdStr, 60), humanDuration(time.Since(j.started).Milliseconds())))
		}
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// sayRunningBgJobs names each job once, or a long-lived dev server would close
// every turn with the same line.
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

// A run_background gate ended the subtask on its "waiting" respond 13 times in
// the logs, and the result arrived in another subtask's context.
// A picture file, and an inline script or an image tool that would crop or
// measure it. A project's own script file (`python3 tools/icons.py a.png`) is not
// inline and stays allowed.
var (
	pictureRe       = regexp.MustCompile(`[\w./-]+\.(?i:png|jpe?g|webp|gif|bmp)\b`)
	pictureScriptRe = regexp.MustCompile(`\b(?:python3?|node|perl|ruby)\s+(?:-c\b|-e\b|-\s*<<)|\b(?:convert|magick|mogrify|pngtopnm|pnmcut|ffmpeg)\s`)
)

// A finite run is a test, build or lint tool, by its command and subcommand
// (after a wrapper like xvfb-run): a word in a path or a flag (`--build`,
// `build/libs/app.jar`) says nothing.
var (
	finiteRunRe = regexp.MustCompile(`^(?:(?:xvfb-run|timeout|time|nice|env)\b[^|;&]*?\s+)?(?:just\s+(?:test|check|build|lint|ci)\b|cargo\s+(?:test|build|check|clippy|bench|nextest)\b|go\s+(?:test|build|vet)\b|(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|build|lint|check)\b|make\b|pytest\b|python3?\s+-m\s+(?:pytest|unittest)\b|tox\b|nox\b|jest\b|vitest\s+run\b|mocha\b|rspec\b|ctest\b|mvn\s+(?:test|verify|package)\b|(?:\./)?gradlew?\s+(?:test|build|check)\b|dotnet\s+(?:test|build)\b|swift\s+(?:test|build)\b|mix\s+test\b)`)
	serverRunRe = regexp.MustCompile(`(?i)--watch|\bwatch\b|\bmake\s+(?:run|serve|dev|start)\b`)
)

// parkableJobs: a job from an earlier turn, or one that may never exit, never
// parks a turn.
func (a *agent) parkableJobs(sid string, since time.Time) string {
	return a.runningBgJobs(sid, func(j *backgroundJob) bool { return j.waitable && j.started.After(since) })
}

var parkPoll = 500 * time.Millisecond

// parkForJobs returns what resumes a parked turn: the queued job notes and the
// user's text, in arrival order.
func (a *agent) parkForJobs(ctx context.Context, sid, jobs string) (string, error) {
	sess := a.getSession(sid)
	if sess == nil {
		return "", fmt.Errorf("no session")
	}
	a.say(ctx, sid, "\n⏸ Waiting for "+jobs+". This turn continues when it reports; type and Send Now to interject.\n")
	for {
		if items := sess.takePending(); len(items) > 0 {
			text, hasNote := a.sayPending(ctx, sid, items, true)
			if hasNote {
				text += "\n\nContinue the work that was waiting on this."
			}
			return text, nil
		}
		if a.parkableJobs(sid, time.Time{}) == "" && !sess.hasPending() {
			// Every job is gone without a note (a session teardown took them):
			// resume rather than hang.
			return "[codehalter, not the user: the background job you were waiting for is no longer running; check its log.]", nil
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(parkPoll):
		}
	}
}

func (a *agent) forgetBgJob(job *backgroundJob) {
	a.bgMu.Lock()
	delete(a.bgJobs, job.id)
	a.bgMu.Unlock()
	_ = os.Remove(job.logPath)
	_ = os.Remove(job.pidPath)
}

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
			a.killJob(j)
			a.terminalRelease(j.sid, j.terminalId)
		}
		_ = os.Remove(j.logPath)
		_ = os.Remove(j.pidPath)
	}
}

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

func runBackgroundExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	job, refusal := a.launchJob(ctx, sid, rawArgs, false)
	if job == nil {
		return refusal, false
	}

	select {
	case res := <-job.exited:
		tail := readLogTail(job.logPath, bgLogTailCap)
		a.terminalRelease(sid, job.terminalId)
		a.forgetBgJob(job)
		if res.err != nil {
			a.FailToolCall(ctx, sid, job.tcId, res.err.Error())
			return "error reading terminal: " + res.err.Error(), false
		}
		result := fmt.Sprintf("background job %d exited immediately (exit %d) — it did not stay running. Likely a startup error (port already in use, bad command, missing file). Output:\n\n%s", job.id, res.exit.code(), tail)
		a.CompleteToolCallTitled(ctx, sid, job.tcId, fmt.Sprintf("Background: %s (exited %d)", job.cmdStr, res.exit.code()), []ToolCallContent{TextContent(result)})
		return result, false
	case <-time.After(bgJobGrace):
	}

	wake := a.handOver(job)
	stop := fmt.Sprintf("stop it with `run_command: kill %d`", job.pid)
	if job.pid == 0 {
		// Never print `kill 0`: it would signal the whole process group.
		stop = "its pid was not recorded, so it can only be stopped by ending the session"
	}
	idle := "end the turn with `respond` saying the job is running"
	if job.waitable {
		idle = fmt.Sprintf("call `respond` saying you are waiting for job %d: that parks this step, it does not end it, and you continue here the moment the job reports", job.id)
	}
	result := fmt.Sprintf("background job %d running (pid %d). It keeps running across tool calls and across turns. When it exits, codehalter reports the exit code and the last output by itself, so do NOT poll or sleep waiting for it: carry on with other work, or if there is none, %s.%s Read its output any time with `run_command: cat %s` (or tail/grep it); %s. Output so far:\n\n%s",
		job.id, job.pid, idle, wake, job.logPath, stop, readLogTail(job.logPath, bgLogTailCap))
	a.retitleToolCall(ctx, sid, job.tcId, fmt.Sprintf("Background: %s (pid %d)", job.cmdStr, job.pid), "completed")
	return result, false
}

// bgNoteTailCap is smaller than bgLogTailCap: the note stays in the
// conversation for good.
const bgNoteTailCap = 3 * 1024

const bgWakeRetry = 2 * time.Second

// bgStallTimeout is long on purpose: it only has to catch a command that will
// never finish (a hang, a prompt for input). Vars so tests can shorten them.
var (
	bgStallTimeout = 10 * time.Minute
	bgStallPoll    = 30 * time.Second
)

// watchBgJob closes stop, ending the job's stall watcher, once the job exits.
func (a *agent) watchBgJob(job *backgroundJob, stop chan struct{}) {
	defer close(stop)
	res := <-job.exited
	exit, err := res.exit, res.err
	// Untracked means shutdown killed it: nothing to report, nobody to report to.
	a.bgMu.Lock()
	_, tracked := a.bgJobs[job.id]
	stalled := job.stalled
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
	if stalled == 0 && err == nil {
		sess.recordJobRun(jobRun{cmd: job.cmdStr, code: exit.code(), started: job.started, ended: time.Now()})
	}
	switch {
	case stalled > 0:
		outcome = fmt.Sprintf("was killed after %s without any output (hung, or waiting for input)", humanDuration(stalled.Milliseconds()))
	case err != nil:
		outcome = "was lost (" + err.Error() + ")"
	}
	kind := "Background"
	if job.expectExit {
		kind = "Run"
	}
	a.retitleToolCall(context.Background(), job.sid, job.tcId, fmt.Sprintf("%s: %s (job %d %s)", kind, job.cmdStr, job.id, outcome), "completed")
	took := humanDuration(time.Since(job.started).Milliseconds())
	sess.addBgNote(bgNote{
		line: fmt.Sprintf("background job %d `%s` %s after %s", job.id, truncate(job.cmdStr, 80), outcome, took),
		full: fmt.Sprintf("[codehalter, not the user: background job %d `%s` %s after %s. Full log: %s. Last output:]\n\n%s",
			job.id, job.cmdStr, outcome, took, job.logPath, readLogTail(job.logPath, bgNoteTailCap)),
	})
	a.deliverBgNotesWhenIdle(sess)
}

// killJob signals the process group directly, since codehalter shares the
// container, rather than relying on what the client's kill reaches.
func (a *agent) killJob(job *backgroundJob) {
	if job.pid == 0 {
		job.pid = readPidFile(job.pidPath)
	}
	if job.pid > 0 {
		if err := syscall.Kill(-job.pid, syscall.SIGTERM); err != nil {
			_ = syscall.Kill(job.pid, syscall.SIGTERM)
		}
	}
	if err := a.terminalKill(context.Background(), job.sid, job.terminalId); err != nil {
		slog.Debug("terminal kill failed", "job", job.id, "err", err)
	}
}

// stallWatch counts any growth of the log or of a redirect target as progress,
// so a quiet but working suite is never killed.
func (a *agent) stallWatch(job *backgroundJob, stop <-chan struct{}, timeout, poll time.Duration) {
	files := append([]string{job.logPath}, job.redirects...)
	last, lastChange := progressSignature(files), time.Now()
	ticker := time.NewTicker(poll)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			if sig := progressSignature(files); sig != last {
				last, lastChange = sig, time.Now()
				continue
			}
			if time.Since(lastChange) < timeout {
				continue
			}
			a.bgMu.Lock()
			job.stalled = timeout
			a.bgMu.Unlock()
			a.killJob(job)
			return
		}
	}
}

func (a *agent) wakeForBgJob(job *backgroundJob) {
	time.Sleep(job.wakeAfter)
	a.bgMu.Lock()
	_, tracked := a.bgJobs[job.id]
	a.bgMu.Unlock()
	if !tracked {
		return
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

// deliverBgNotesWhenIdle never interrupts a turn: a running one takes the queue
// between rounds and drains it at its end; the retry covers a turn past its drain,
// or stopped before it, that still holds the lock.
func (a *agent) deliverBgNotesWhenIdle(sess *Session) {
	for sess.hasPending() {
		// After a Stop, no turn starts on its own: the note rides the next prompt.
		if sess.stoppedIdle() {
			a.say(context.Background(), sess.ID, "\n🔔 a background job reported; it reaches the model with your next message.\n")
			return
		}
		ctx, release, ok := a.holdTurn(context.Background(), sess, false)
		if !ok {
			time.Sleep(bgWakeRetry)
			continue
		}
		a.drainSteer(ctx, sess)
		release()
		return
	}
}
