package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"slices"
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
	announced  bool     // under bgMu
	expectExit bool     // a handed-over run_command, not a run_background job
	redirects  []string // growth counts as progress
	// Under bgMu, written by stuckWatch: stuck is why it stopped the job, "" while it runs;
	// still is when its processes last moved; listening, one of them holds a listening socket.
	stuck     string
	still     time.Time
	listening bool
	exited    chan jobExit
}

// jobExit is delivered once per job, to run_command's wait or, after handover,
// to watchBgJob.
type jobExit struct {
	exit terminalExit
	err  error
}

// jobSweepFn signals every process in the wrapper's process group, and in its
// session when it leads one; it succeeds when it signalled something. /proc, not
// pkill: BusyBox and GNU differ, /proc/<pid>/stat does not.
const jobSweepFn = `sweep() {
  local sig=$1 me=$$ own p line n=0
  read -r line < /proc/$me/stat; set -- ${line##*) }; own=$4
  for p in /proc/[0-9]*; do
    p=${p#/proc/}; [ "$p" = "$me" ] && continue
    read -r line < /proc/$p/stat 2>/dev/null || continue
    set -- ${line##*) }
    if [ "$3" = "$me" ] || { [ "$own" = "$me" ] && [ "$4" = "$me" ]; }; then kill -$sig $p 2>/dev/null && n=$((n+1)); fi
  done
  [ $n -gt 0 ]
}`

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
	// Only a foreground render attaches its picture, so only it counts as a look.
	if !expectExit && renderCmdRe.MatchString(cmdStr) {
		return nil, "refused: this renders a screen, which takes seconds. Run it with `run_command` instead: only a render through run_command attaches the picture to your view, and only then does the look count. A render in the background finishes unseen."
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
		redirects:  redirectTargets(cmdStr, sess.Cwd),
		exited:     make(chan jobExit, 1),
	}
	// pipefail + wait: the exit status is the command's, not tee's. Leftovers are
	// swept on every exit, by group and by session (timeout moves its command out
	// of the group; an xvfb-run suite left its Xvfb behind).
	quoted := "'" + strings.ReplaceAll(cmdStr, "'", `'\''`) + "'"
	script := fmt.Sprintf("echo $$ > %s\nset -o pipefail\n%s\ntrap 'trap - TERM INT HUP; sweep TERM; exit 143' TERM INT HUP\n( exec bash -c %s ) 2>&1 | tee %s &\nwait $!\nrc=$?\ntrap '' TERM\nif sweep TERM; then sleep 3; sweep KILL; fi\nexit $rc",
		job.pidPath, jobSweepFn, quoted, job.logPath)
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
	// Limits read here, not in the watcher, which may outlive a test that shortened them.
	go a.stuckWatch(job, stop, bgStuckAfter, bgStuckPoll)
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
			state := fmt.Sprintf("%s so far", humanDuration(time.Since(j.started).Milliseconds()))
			if quiet := time.Since(j.still); !j.still.IsZero() && quiet >= time.Minute {
				state += fmt.Sprintf(", no CPU use or output for %s", humanDuration(quiet.Milliseconds()))
			}
			names = append(names, fmt.Sprintf("job %d `%s` (%s)", j.id, truncate(j.cmdStr, 60), state))
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
var ()

// parkableJobs: every job this step started that will end; a server (it listens on a
// port) and a job from an earlier turn never park it.
func (a *agent) parkableJobs(sid string, since time.Time) string {
	return a.runningBgJobs(sid, func(j *backgroundJob) bool { return j.started.After(since) && !j.listening })
}

var parkPoll = 500 * time.Millisecond

// parkForJobs returns what resumes a parked turn: the queued job notes and the
// user's text, in arrival order. Without parking, a gate that ended on a
// "waiting" respond got its result in another subtask.
func (a *agent) parkForJobs(ctx context.Context, sid string, since time.Time) (string, error) {
	sess := a.getSession(sid)
	if sess == nil {
		return "", fmt.Errorf("no session")
	}
	if items := sess.takePending(); len(items) > 0 {
		return a.resumeWith(ctx, sid, items), nil
	}
	a.say(ctx, sid, "\n⏸ Waiting for "+a.parkableJobs(sid, since)+". This turn continues when it reports; type and Send Now to interject.\n")
	for {
		if items := sess.takePending(); len(items) > 0 {
			return a.resumeWith(ctx, sid, items), nil
		}
		if a.parkableJobs(sid, since) == "" && !sess.hasPending() {
			// Every job is gone without a note (a session teardown took them):
			// resume rather than hang.
			return "[codehalter, not the user: the background job you were waiting for is no longer running, or turned out to be a server (it listens on a port, so no step waits for it); check its log.]", nil
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(parkPoll):
		}
	}
}

func (a *agent) resumeWith(ctx context.Context, sid string, items []pendingInput) string {
	text, hasNote := a.sayPending(ctx, sid, items, true)
	if hasNote {
		text += "\n\nContinue the work that was waiting on this."
	}
	return text
}

// endJob releases a finished or stopped job's terminal and forgets it.
func (a *agent) endJob(job *backgroundJob) {
	a.terminalRelease(job.sid, job.terminalId)
	a.bgMu.Lock()
	delete(a.bgJobs, job.id)
	a.bgMu.Unlock()
	removeJobFiles(job.logPath, job.pidPath)
}

func removeJobFiles(files ...string) {
	for _, f := range files {
		if err := os.Remove(f); err != nil && !os.IsNotExist(err) {
			slog.Debug("background job: file not removed", "file", f, "err", err)
		}
	}
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
		removeJobFiles(j.logPath, j.pidPath)
	}
}

func readLogTail(path string, max int) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	out := collapseStackTraces(string(data))
	if len(out) > max {
		return "[... earlier output truncated ...]\n" + tailUTF8(out, max)
	}
	return out
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
		a.endJob(job)
		if res.err != nil {
			a.FailToolCall(ctx, sid, job.tcId, res.err.Error())
			return "error reading terminal: " + res.err.Error(), false
		}
		result := fmt.Sprintf("background job %d exited immediately (exit %d): it did not stay running. Likely a startup error (port already in use, bad command, missing file). Output:\n\n%s", job.id, res.exit.code(), tail)
		a.CompleteToolCallTitled(ctx, sid, job.tcId, fmt.Sprintf("Background: %s (exited %d)", job.cmdStr, res.exit.code()), []ToolCallContent{TextContent(result)})
		return result, false
	case <-time.After(bgJobGrace):
	}

	wake := a.handOver(job)
	stop := stopHint(job.pid)
	idle := fmt.Sprintf("call `respond` saying you are waiting for job %d: that parks this step, it does not end it, and you continue here the moment the job reports. "+
		"A job whose processes neither use CPU nor write output for %s is stuck and gets stopped; a server, which listens on a port, is left running, so stop one you no longer need", job.id, humanDuration(bgStuckAfter.Milliseconds()))
	result := fmt.Sprintf("background job %d running (pid %d). It keeps running across tool calls and across turns. When it exits, codehalter reports the exit code and the last output by itself, so do NOT poll or sleep waiting for it: carry on with other work, or if there is none, %s.%s Read its output any time with `run_command: cat %s` (or tail/grep it); %s. Output so far:\n\n%s",
		job.id, job.pid, idle, wake, job.logPath, stop, readLogTail(job.logPath, bgLogTailCap))
	a.retitleToolCall(ctx, sid, job.tcId, fmt.Sprintf("Background: %s (pid %d)", job.cmdStr, job.pid), "completed")
	return result, false
}

// bgNoteTailCap is smaller than bgLogTailCap: the note stays in the
// conversation for good.
const bgNoteTailCap = 3 * 1024

const bgWakeRetry = 2 * time.Second

// bgStuckAfter: a job whose processes used no CPU and wrote no output this long is
// stuck (a deadlock, a prompt for input, a wait for something that never comes). A
// silent build still uses CPU, and a server listens, so neither is taken for stuck.
// Vars so tests can shorten them.
var (
	bgStuckAfter = 5 * time.Minute
	bgStuckPoll  = 30 * time.Second
)

// watchBgJob closes stop, ending the job's stuck watcher, once the job exits.
func (a *agent) watchBgJob(job *backgroundJob, stop chan struct{}) {
	defer close(stop)
	res := <-job.exited
	exit, err := res.exit, res.err
	// Untracked means shutdown killed it: nothing to report, nobody to report to.
	a.bgMu.Lock()
	_, tracked := a.bgJobs[job.id]
	stuck := job.stuck
	a.bgMu.Unlock()
	if !tracked {
		return
	}
	// Forgotten only once its note is queued: a parked step that saw neither would
	// resume without the exit code.
	forget := func() {
		a.bgMu.Lock()
		delete(a.bgJobs, job.id)
		a.bgMu.Unlock()
	}
	a.terminalRelease(job.sid, job.terminalId)
	removeJobFiles(job.pidPath) // the log stays: the note points the model at it
	sess := a.getSession(job.sid)
	if sess == nil {
		forget()
		return
	}
	outcome := fmt.Sprintf("exited with code %d", exit.code())
	if stuck == "" && err == nil {
		sess.recordJobRun(jobRun{cmd: job.cmdStr, code: exit.code(), started: job.started, ended: time.Now()})
	}
	switch {
	case stuck != "":
		outcome = "was stopped as stuck: " + stuck
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
	forget()
	a.deliverBgNotesWhenIdle(sess)
}

// killJob signals the process group directly, since codehalter shares the
// container, rather than relying on what the client's kill reaches.
func (a *agent) killJob(job *backgroundJob) {
	if job.pid == 0 {
		job.pid = readPidFile(job.pidPath)
	}
	if job.pid > 0 {
		killGroup(job.pid)
	}
	if err := a.terminalKill(context.Background(), job.sid, job.terminalId); err != nil {
		slog.Debug("terminal kill failed", "job", job.id, "err", err)
	}
}

// renderCmdRe is a run of the project's snapshot recipe, the render the setup
// round adds (SPEC-SETUP.md), or the app's own --snapshot flag.
var renderCmdRe = regexp.MustCompile(`(?:^|[\s;&|(])(?:just|make|task)\s+snapshot\b|\b(?:npm|pnpm|yarn)\s+(?:run\s+)?snapshot\b|\s--snapshot\b`)

// stopHint tells the model how to stop a job: its whole group, SIGKILL if
// SIGTERM is ignored. Never `kill -0`, which signals the caller's own group.
func stopHint(pid int) string {
	if pid <= 0 {
		return "its pid was not recorded, so it can only be stopped by ending the session"
	}
	return fmt.Sprintf("stop it with `run_command: kill -TERM -%d; sleep 3; kill -KILL -%d 2>/dev/null; true` (the whole job; here a plain kill can be ignored)", pid, pid)
}

// jobKillGrace: a job still there this long after SIGTERM gets SIGKILL. Zed's
// terminals start commands with SIGTERM ignored.
var jobKillGrace = 3 * time.Second

// killGroup: SIGTERM to pid's group (the pid alone if it leads none), SIGKILL
// after jobKillGrace; it returns at once.
func killGroup(pid int) {
	target := -pid
	if err := syscall.Kill(target, syscall.SIGTERM); err != nil {
		target = pid
		if err := syscall.Kill(target, syscall.SIGTERM); err != nil {
			slog.Debug("killGroup: SIGTERM", "pid", pid, "err", err)
			return // nothing left to stop
		}
	}
	// A stopped process (state T) holds the SIGTERM until it is continued.
	if err := syscall.Kill(target, syscall.SIGCONT); err != nil {
		slog.Debug("killGroup: SIGCONT", "pid", pid, "err", err)
	}
	grace := jobKillGrace
	go func() {
		time.Sleep(grace)
		if syscall.Kill(target, 0) != nil {
			return // gone after the SIGTERM
		}
		if err := syscall.Kill(target, syscall.SIGKILL); err != nil {
			slog.Debug("killGroup: SIGKILL", "pid", pid, "err", err)
		}
	}()
}

// killOrphanedJobs stops the jobs a dead codehalter left running. Only a wrapper
// whose command line names its own pid file is taken, so a reused pid is safe.
func killOrphanedJobs() {
	files, err := filepath.Glob(filepath.Join(os.TempDir(), "codehalter-*-job-*.pid"))
	if err != nil {
		slog.Debug("killOrphanedJobs: glob", "err", err)
		return
	}
	for _, f := range files {
		var owner, id int
		if _, err := fmt.Sscanf(filepath.Base(f), "codehalter-%d-job-%d.pid", &owner, &id); err != nil ||
			owner == os.Getpid() || syscall.Kill(owner, 0) == nil {
			continue // not ours to judge, or its codehalter still runs
		}
		if pid := readPidFile(f); pid > 0 {
			cmdline, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
			if err == nil && strings.Contains(string(cmdline), filepath.Base(f)) {
				slog.Info("killing a job an earlier codehalter left running", "pidfile", f, "pid", pid)
				killGroup(pid)
			}
		}
		if err := os.Remove(f); err != nil {
			slog.Debug("killOrphanedJobs: remove", "file", f, "err", err)
		}
	}
}

// stuckWatch stops a job once nothing in it moves for `after`: its processes use no CPU,
// write no output (its log or a redirect target), and wait for no lock whose holder is
// busy. A server (it listens on a socket) is never stuck, since it waits for requests by
// design; a stopped process (state T) is, at once, since only a SIGCONT would let it go
// on. Where /proc cannot say (not Linux), output alone decides.
func (a *agent) stuckWatch(job *backgroundJob, stop <-chan struct{}, after, poll time.Duration) {
	files := append([]string{job.logPath}, job.redirects...)
	out, prev := progressSignature(files), jobActivity(job.pid)
	a.bgMu.Lock()
	job.still = time.Now()
	a.bgMu.Unlock()
	ticker := time.NewTicker(poll)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
		}
		o, now := progressSignature(files), jobActivity(job.pid)
		moved := o != out || now.known && (now.cpu != prev.cpu || now.locked && now.lockCPU != prev.lockCPU)
		out, prev = o, now
		a.bgMu.Lock()
		job.listening = now.listening
		if moved || now.listening {
			job.still = time.Now()
		}
		if _, tracked := a.bgJobs[job.id]; tracked {
			switch {
			case now.stopped && !moved:
				job.stuck = "one of its processes was stopped (state T, by SIGSTOP or a read from the terminal) and could not go on without a SIGCONT"
			case time.Since(job.still) >= after:
				job.stuck = fmt.Sprintf("its processes used no CPU and wrote no output for %s (a deadlock, a prompt for input, or a wait for something that never comes)", humanDuration(after.Milliseconds()))
			}
		}
		stuck := job.stuck != ""
		a.bgMu.Unlock()
		if stuck {
			a.killJob(job)
			return
		}
	}
}

// procActivity is what /proc says about a job's process tree at one moment.
type procActivity struct {
	known     bool   // false where /proc cannot say
	cpu       uint64 // ticks the tree used so far, children it reaped included
	listening bool   // a process in it holds a listening socket
	stopped   bool   // a process in it is stopped (state T) and cannot go on without SIGCONT
	locked    bool   // a process in it waits for a file lock held outside the tree
	lockCPU   uint64 // ticks the holders of those locks, with their children, used so far
}

type proc struct {
	ppid  int
	state string
	cpu   uint64
}

// procTable reads every process's parent, state and CPU ticks (its reaped children's included).
func procTable() map[int]proc {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil
	}
	procs := map[int]proc{}
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		data, err := os.ReadFile("/proc/" + e.Name() + "/stat")
		if err != nil {
			continue
		}
		// The fields after the command, which may itself hold spaces and parentheses.
		f := strings.Fields(string(data[strings.LastIndexByte(string(data), ')')+1:]))
		if len(f) < 15 {
			continue
		}
		ppid, _ := strconv.Atoi(f[1])
		var ticks uint64
		for _, field := range f[11:15] { // utime, stime, cutime, cstime
			n, _ := strconv.ParseUint(field, 10, 64)
			ticks += n
		}
		procs[pid] = proc{ppid, f[0], ticks}
	}
	return procs
}

// treeUnder: root and every descendant in procs.
func treeUnder(procs map[int]proc, root int) map[int]bool {
	tree := map[int]bool{root: true}
	for grew := true; grew; {
		grew = false
		for pid, p := range procs {
			if !tree[pid] && tree[p.ppid] {
				tree[pid], grew = true, true
			}
		}
	}
	return tree
}

func jobActivity(root int) procActivity {
	procs := procTable()
	if _, alive := procs[root]; !alive || root <= 0 {
		return procActivity{}
	}
	tree := treeUnder(procs, root)
	act := procActivity{known: true}
	sockets := map[string]bool{}
	for pid := range tree {
		act.cpu += procs[pid].cpu
		act.stopped = act.stopped || procs[pid].state == "T"
		fds, _ := os.ReadDir(fmt.Sprintf("/proc/%d/fd", pid))
		for _, fd := range fds {
			link, err := os.Readlink(fmt.Sprintf("/proc/%d/fd/%s", pid, fd.Name()))
			if inode, ok := strings.CutPrefix(link, "socket:["); err == nil && ok {
				sockets[strings.TrimSuffix(inode, "]")] = true
			}
		}
	}
	act.listening = len(sockets) > 0 && listensOn(root, sockets)
	// A holder's work may run in a child (flock's does).
	for _, holder := range lockHolders(tree) {
		act.locked = true
		for pid := range treeUnder(procs, holder) {
			act.lockCPU += procs[pid].cpu
		}
	}
	return act
}

// lockHolders: the processes outside the tree holding a file lock (flock, fcntl) a
// process in the tree waits for. In /proc/locks a waiter's line carries "->" and the
// same lock number as its holder's.
func lockHolders(tree map[int]bool) []int {
	data, err := os.ReadFile("/proc/locks")
	if err != nil {
		return nil
	}
	holder := map[string]int{}
	var waits []string
	for _, line := range strings.Split(string(data), "\n") {
		f := strings.Fields(line)
		switch {
		case len(f) > 5 && f[1] == "->":
			if pid, _ := strconv.Atoi(f[5]); tree[pid] {
				waits = append(waits, f[0])
			}
		case len(f) > 4:
			if pid, _ := strconv.Atoi(f[4]); pid > 0 {
				holder[f[0]] = pid
			}
		}
	}
	var out []int
	for _, lock := range waits {
		if pid := holder[lock]; pid > 0 && !tree[pid] && !slices.Contains(out, pid) {
			out = append(out, pid)
		}
	}
	return out
}

// listensOn: one of the socket inodes is a listening TCP socket or a listening unix socket.
func listensOn(pid int, sockets map[string]bool) bool {
	for _, table := range []string{"tcp", "tcp6"} {
		data, _ := os.ReadFile(fmt.Sprintf("/proc/%d/net/%s", pid, table))
		for _, line := range strings.Split(string(data), "\n") {
			// sl local remote st ... inode: state 0A is LISTEN.
			if f := strings.Fields(line); len(f) > 9 && f[3] == "0A" && sockets[f[9]] {
				return true
			}
		}
	}
	data, _ := os.ReadFile(fmt.Sprintf("/proc/%d/net/unix", pid))
	for _, line := range strings.Split(string(data), "\n") {
		// Num RefCount Protocol Flags Type St Inode: flag 00010000 is a listening socket.
		if f := strings.Fields(line); len(f) > 6 && f[3] == "00010000" && sockets[f[6]] {
			return true
		}
	}
	return false
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
