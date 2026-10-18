//go:build !unix

package main

// watchResize: no SIGWINCH here, so the size is read per turn only.
func (u *cliUI) watchResize() {}
