//go:build unix

package main

import (
	"os/exec"
	"syscall"
)

// detachGroup puts cmd in its own process group and rewires context-cancellation
// to SIGKILL the WHOLE group, not just the direct child. Without it, a command
// that backgrounds a process (`server & ; echo done`) leaves the grandchild
// holding the inherited stdout pipe open after the shell exits: cmd.Wait() then
// blocks on the copy goroutine forever, and a kill that targets only the
// already-dead shell can't free it. Killing the group reaps the daemon too, the
// pipe closes, and Wait returns. Callers still get the timeout/cancel they asked
// for; this only widens what the kill reaches.
func detachGroup(cmd *exec.Cmd) {
	setPgid(cmd)
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		// ESRCH (already gone) is fine — exec ignores Cancel's error once the
		// process has exited.
		return killGroup(cmd.Process.Pid)
	}
}

// setPgid puts cmd in its own process group (so killGroup can later reap its
// whole tree) without touching cmd.Cancel. Used by run_background, which owns no
// context and reaps its jobs explicitly at shutdown.
func setPgid(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// killGroup SIGKILLs the entire process group led by pid. The negative pid
// targets the group; with Setpgid the leader's pid is the pgid.
func killGroup(pid int) error {
	return syscall.Kill(-pid, syscall.SIGKILL)
}

// groupAlive reports whether process group pgid still has a live member. Used
// after the foreground process exits to detect a child it backgrounded
// (`server &`) that is still running. kill(-pgid, 0) sends no signal but performs
// the existence check: nil => at least one member, ESRCH => none.
func groupAlive(pgid int) bool {
	if pgid <= 1 {
		return false
	}
	return syscall.Kill(-pgid, 0) == nil
}
