//go:build unix

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// A job that ignores SIGTERM, as commands in Zed's terminals do, is gone shortly
// after the grace; one that has a live codehalter is not an orphan.
func TestKillGroupEscalates(t *testing.T) {
	old := jobKillGrace
	jobKillGrace = 200 * time.Millisecond
	defer func() { jobKillGrace = old }()
	c := exec.Command("sh", "-c", "trap '' TERM; sleep 30 & wait")
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- c.Wait() }()
	killGroup(c.Process.Pid)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		_ = syscall.Kill(-c.Process.Pid, syscall.SIGKILL)
		t.Fatal("a group that ignores SIGTERM outlived the grace")
	}
}

// A descendant in a session and group of its own (an Xvfb, a database a test started) is
// stopped with its job: the group signal never reached one, which outlived its job by hours.
func TestKillGroupReachesDescendantsInOtherGroups(t *testing.T) {
	if _, err := exec.LookPath("setsid"); err != nil {
		t.Skip("no setsid")
	}
	old := jobKillGrace
	jobKillGrace = 200 * time.Millisecond
	defer func() { jobKillGrace = old }()
	// The pid goes through a file: a pipe the child holds open changes how it ends.
	pidFile := filepath.Join(t.TempDir(), "pid")
	c := exec.Command("sh", "-c", "setsid sleep 600 & echo $! > "+pidFile+"; wait")
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}
	go c.Wait() //nolint:errcheck // reaps sh; it is killed, as meant
	var child int
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		if b, err := os.ReadFile(pidFile); err == nil {
			if _, err := fmt.Sscan(string(b), &child); err == nil {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatal("the child never started")
		}
	}
	killGroup(c.Process.Pid)
	for deadline := time.Now().Add(5 * time.Second); syscall.Kill(child, 0) == nil; time.Sleep(20 * time.Millisecond) {
		if time.Now().After(deadline) {
			_ = syscall.Kill(child, syscall.SIGKILL)
			t.Fatal("the descendant in its own session outlived its job")
		}
	}
}
