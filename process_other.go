//go:build !unix

package main

import (
	"log/slog"
	"os"
	"os/exec"
)

// Jobs run in the Linux devcontainer; on another host only the CLI's launcher runs, so
// these exist to build. Windows has no process groups to signal: the process is killed.
func killGroup(pid int) {
	p, err := os.FindProcess(pid)
	if err == nil {
		err = p.Kill()
	}
	if err != nil {
		slog.Debug("killGroup", "pid", pid, "err", err)
	}
}

// endProcessGroup: no process group to sweep here.
func endProcessGroup(c *exec.Cmd) {}

// newProcessGroup: no process groups to start in here.
func newProcessGroup(c *exec.Cmd) {}

// ownProcessGroup: no process groups to start in here; a cancel kills the process.
func ownProcessGroup(c *exec.Cmd) {}

// processAlive: FindProcess succeeds for a live process only on Windows.
func processAlive(pid int) bool {
	_, err := os.FindProcess(pid)
	return err == nil
}
