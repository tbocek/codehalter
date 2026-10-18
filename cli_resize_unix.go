//go:build unix

package main

import (
	"os"
	"os/signal"
	"syscall"
)

// watchResize re-reads the size when the terminal changes it. A /spec loop is one turn
// that runs for hours: read only per turn, a narrowed or shortened window was drawn at the
// old size all along, and a live region taller than the screen left rows behind.
func (u *cliUI) watchResize() {
	if !u.tty {
		return
	}
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGWINCH)
	go func() {
		for range ch {
			u.refreshSize()
			u.mu.Lock()
			u.drawLive()
			u.mu.Unlock()
		}
	}()
}
