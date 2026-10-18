//go:build unix

package main

import (
	"os/exec"
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
