//go:build unix

package main

import (
	"log/slog"
	"os/exec"
	"syscall"
	"time"
)

// killGroup: SIGTERM to pid's group (the pid alone if it leads none), SIGKILL
// after jobKillGrace; it returns at once.
func killGroup(pid int) {
	// The tree first, before any signal: a parent that dies reparents its children to init,
	// and one in its own session or group (an Xvfb, a database a test started) outlived its
	// job by hours.
	var tree []int
	for p := range treeUnder(procTable(), pid) {
		if p != pid {
			tree = append(tree, p)
		}
	}
	for _, p := range tree {
		if err := syscall.Kill(p, syscall.SIGTERM); err != nil {
			slog.Debug("killGroup: SIGTERM to a descendant", "pid", p, "err", err)
			continue
		}
		if err := syscall.Kill(p, syscall.SIGCONT); err != nil { // one that is stopped holds the SIGTERM
			slog.Debug("killGroup: SIGCONT to a descendant", "pid", p, "err", err)
		}
	}
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
		for _, p := range tree {
			if syscall.Kill(p, 0) != nil {
				continue // gone after the SIGTERM
			}
			if err := syscall.Kill(p, syscall.SIGKILL); err != nil {
				slog.Debug("killGroup: SIGKILL to a descendant", "pid", p, "err", err)
			}
		}
		if syscall.Kill(target, 0) != nil {
			return // gone after the SIGTERM
		}
		if err := syscall.Kill(target, syscall.SIGKILL); err != nil {
			slog.Debug("killGroup: SIGKILL", "pid", pid, "err", err)
		}
	}()
}

// endProcessGroup stops what is left of c's group once c has ended.
func endProcessGroup(c *exec.Cmd) {
	if c.Process != nil {
		killGroup(c.Process.Pid)
	}
}

// newProcessGroup starts c as the leader of a group of its own.
func newProcessGroup(c *exec.Cmd) {
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// ownProcessGroup starts c in a group of its own, which a cancel stops whole, descendants
// included, instead of the shell alone.
func ownProcessGroup(c *exec.Cmd) {
	newProcessGroup(c)
	c.Cancel = func() error {
		killGroup(c.Process.Pid)
		return nil
	}
	c.WaitDelay = jobKillGrace + time.Second
}

// processAlive: signal 0 checks a pid exists without touching it.
func processAlive(pid int) bool {
	return syscall.Kill(pid, 0) == nil
}
