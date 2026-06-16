//go:build !unix

package main

import "os/exec"

// Non-Unix stubs so cross-platform builds compile. There's no portable
// process-group concept to use here, so detachGroup leaves exec's default
// context-kill (Process.Kill) in place and the rest are no-ops; groupAlive
// reporting false just means run_command's background-child detection is skipped.
// codehalter only runs inside a Linux devcontainer (it aborts otherwise), so this
// path is never actually executed at runtime.
func detachGroup(cmd *exec.Cmd) {}

func setPgid(cmd *exec.Cmd) {}

func killGroup(pid int) error { return nil }

func groupAlive(pgid int) bool { return false }
