package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
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

// backgroundJob tracks one detached run_background process so the agent can reap
// its whole process group at shutdown (it owns no context, so it would otherwise
// orphan and keep holding a port) and so the grace-period check can tell "still
// running" from "exited immediately".
type backgroundJob struct {
	id      int
	cmdStr  string
	pid     int
	logPath string
	cmd     *exec.Cmd
	done    chan struct{} // closed when cmd.Wait() returns
	exitErr error         // valid only after done is closed
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

// shutdownBackground SIGKILLs every still-running run_background process group on
// app exit and removes their log files. Called from main after the connection
// closes, alongside shutdownMCP.
func (a *agent) shutdownBackground() {
	a.bgMu.Lock()
	jobs := make([]*backgroundJob, 0, len(a.bgJobs))
	for _, j := range a.bgJobs {
		jobs = append(jobs, j)
	}
	a.bgJobs = nil
	a.bgMu.Unlock()
	for _, j := range jobs {
		select {
		case <-j.done: // already exited; nothing to kill
		default:
			if j.cmd.Process != nil {
				_ = syscall.Kill(-j.cmd.Process.Pid, syscall.SIGKILL)
			}
		}
		_ = os.Remove(j.logPath)
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
		return "[... earlier output truncated ...]\n" + string(data[len(data)-max:])
	}
	return string(data)
}

// runBackgroundExecute starts a long-running command detached from this tool
// call, captures its output to a log file, waits a short grace period to catch an
// immediate crash, then returns the pid + log path while the process keeps
// running. Cleanup happens at session exit (shutdownBackground) or when the model
// kills the pid via run_command.
func runBackgroundExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	args := parseArgs(rawArgs)
	cmdStr := args["command"]
	if cmdStr == "" {
		return "error: command is required", false
	}
	sess := a.getSession(sid)
	if sess == nil {
		return "error: no session", false
	}

	tcId := a.StartToolCall(ctx, sid, "Background: "+cmdStr, "execute", nil)

	id := a.nextBgID()
	logPath := filepath.Join(os.TempDir(), fmt.Sprintf("codehalter-bg-%d.log", id))
	logFile, err := os.Create(logPath)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error creating log file: " + err.Error(), false
	}

	// Detached on purpose: no owning context (it must outlive this tool call so a
	// later run_command can probe it) and its own process group so shutdown can
	// SIGKILL the whole tree. Output goes to a log the model tails via run_command.
	cmd := exec.Command("bash", "-c", cmdStr)
	cmd.Dir = sess.Cwd
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		logFile.Close()
		_ = os.Remove(logPath)
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error starting bash: " + err.Error(), false
	}

	job := &backgroundJob{id: id, cmdStr: cmdStr, pid: cmd.Process.Pid, logPath: logPath, cmd: cmd, done: make(chan struct{})}
	go func() {
		job.exitErr = cmd.Wait()
		logFile.Close()
		close(job.done)
	}()
	a.registerBgJob(job)

	// Grace window: catch an immediate exit (failed bind, bad command) before
	// reporting the job as running.
	crashed := false
	select {
	case <-job.done:
		crashed = true
	case <-time.After(bgJobGrace):
	}

	tail := readLogTail(logPath, bgLogTailCap)
	if crashed {
		exitCode := 0
		if ee, ok := job.exitErr.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else if job.exitErr != nil {
			exitCode = -1
		}
		// It already exited, so there's nothing to reap: drop it from the table and
		// remove the log (its output is in the result below).
		a.bgMu.Lock()
		delete(a.bgJobs, id)
		a.bgMu.Unlock()
		_ = os.Remove(logPath)
		result := fmt.Sprintf("background job %d exited immediately (exit %d) — it did not stay running. Likely a startup error (port already in use, bad command, missing file). Output:\n\n%s", id, exitCode, tail)
		a.CompleteToolCallTitled(ctx, sid, tcId, fmt.Sprintf("Background: %s (exited %d)", cmdStr, exitCode), []ToolCallContent{TextContent(result)})
		return result, false
	}

	result := fmt.Sprintf("background job %d running (pid %d). It keeps running across tool calls. Read its output with `run_command: cat %s` (or tail/grep it); stop it with `run_command: kill %d`. Output so far:\n\n%s",
		id, job.pid, logPath, job.pid, tail)
	a.CompleteToolCallTitled(ctx, sid, tcId, fmt.Sprintf("Background: %s (pid %d)", cmdStr, job.pid), []ToolCallContent{TextContent(result)})
	return result, false
}
