//go:build unix

package main

import (
	"log/slog"
	"syscall"
	"time"
)

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

// processAlive: signal 0 checks a pid exists without touching it.
func processAlive(pid int) bool {
	return syscall.Kill(pid, 0) == nil
}
