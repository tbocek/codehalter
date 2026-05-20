package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// gitCommitFile is the path (relative to cwd) where the background updater
// stores the current draft commit message. EXECUTE.md tells the agent to
// hand this path to the user via `git commit -F` rather than committing
// itself. .codehalter/ is gitignored on first bootstrap so the file never
// gets accidentally staged.
const gitCommitFile = ".codehalter/.git_commit"

const gitCommitPrompt = "You are drafting a git commit message for the user's CURRENT uncommitted changes. " +
	"Reply with ONLY the message text — no preamble, no code fences.\n\n" +
	"Format:\n" +
	"- Line 1: short imperative subject, ≤72 chars (e.g. \"add ...\", \"fix ...\", \"refactor ...\").\n" +
	"- Blank line.\n" +
	"- Body: 1-3 short bullets or sentences covering WHY the change is being made, not a restatement of the diff.\n"

// gitStatusPorcelain runs `git status --porcelain` in cwd. Empty output means
// the working tree (including the index) is clean. Returns ("", err) when
// cwd isn't a git checkout or git isn't on PATH — callers treat that as
// "no work to do" without surfacing the error to the user.
func gitStatusPorcelain(cwd string) (string, error) {
	out, err := exec.Command("git", "-C", cwd, "status", "--porcelain").Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// gitDiffHead returns `git diff HEAD` in cwd — staged + unstaged tracked
// changes against HEAD. Untracked files are NOT included here; the porcelain
// status fed alongside this output is what lists those.
func gitDiffHead(cwd string) (string, error) {
	out, err := exec.Command("git", "-C", cwd, "diff", "HEAD").Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// cleanupGitCommitIfClean is called from checkSettings on every user prompt.
// Waits for any in-flight backgroundGitCommit to finish (otherwise its late
// write would resurrect a file we are about to delete), then checks
// `git status --porcelain`. Empty status means everything was committed
// externally — delete .codehalter/.git_commit so the next round starts
// fresh from the next uncommitted change.
func (a *agent) cleanupGitCommitIfClean(cwd string, sid SessionId) {
	if !dirExists(filepath.Join(cwd, ".git")) {
		return
	}
	if sess := a.getSession(sid); sess != nil {
		sess.gitCommitPending.Wait()
	}
	status, err := gitStatusPorcelain(cwd)
	if err != nil {
		return
	}
	if strings.TrimSpace(status) != "" {
		return
	}
	_ = os.Remove(filepath.Join(cwd, gitCommitFile))
}

// pickGitCommitConn selects an LLM slot for the background git-commit updater.
//
// Rules:
//   - Never LLM[0] — that's the foreground KV-cache slot for the user's next
//     turn.
//   - Prefer LLM[2..X]: those are free of the per-turn shadow summariser
//     (which runs on LLM[1]), so we can fan out in parallel.
//   - Fall back to LLM[1] only when no LLM[2+] is reachable. In that case the
//     caller must wait on sess.shadowPending first so we don't pile two LLM
//     calls on the same conn back-to-back and starve its semaphore.
//   - Subagent sessions and <2-slot configurations get nil — feature disabled.
//
// Returns (conn, waitForShadow). waitForShadow=true means "wait on
// sess.shadowPending before issuing the call"; false means "go now".
func (a *agent) pickGitCommitConn(sid SessionId) (*LLMConnection, bool) {
	sess := a.getSession(sid)
	if sess == nil || sess.Depth > 0 {
		return nil, false
	}
	if len(a.settings.LLM) < 2 {
		return nil, false
	}
	for i := 2; i < len(a.settings.LLM); i++ {
		c := &a.settings.LLM[i]
		if len(a.connReachable) > 0 && !a.connReachable[connKey(c)] {
			continue
		}
		return a.settings.ConnAt(i, "execute"), false
	}
	c := &a.settings.LLM[1]
	if len(a.connReachable) > 0 && !a.connReachable[connKey(c)] {
		return nil, false
	}
	return a.settings.ConnAt(1, "execute"), true
}

// backgroundGitCommit fires after every assistant turn. Snapshots the current
// `git diff HEAD` + `git status --porcelain` and asks the LLM to (re)write
// .codehalter/.git_commit so it always matches the latest uncommitted state.
// Self-skips when:
//   - cwd has no .git directory (not a checkout, or .git not mounted),
//   - the working tree is clean (nothing to summarise),
//   - no eligible LLM[1+] slot is available (see pickGitCommitConn).
//
// Multiple in-flight calls are allowed — they race on the file with
// last-write-wins, which is fine because each LLM call's snapshot is point-in-
// time and the freshest wins. The pre-write status re-check guards against
// the narrow race where the user commits during the LLM call.
func (a *agent) backgroundGitCommit(sess *Session) {
	if sess == nil {
		return
	}
	if !dirExists(filepath.Join(sess.Cwd, ".git")) {
		return
	}
	conn, waitForShadow := a.pickGitCommitConn(sess.ID)
	if conn == nil {
		return
	}

	status, err := gitStatusPorcelain(sess.Cwd)
	if err != nil || strings.TrimSpace(status) == "" {
		return
	}
	diff, _ := gitDiffHead(sess.Cwd)

	sess.gitCommitPending.Add(1)
	go func() {
		defer sess.gitCommitPending.Done()
		if waitForShadow {
			sess.shadowPending.Wait()
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		var buf strings.Builder
		buf.WriteString(gitCommitPrompt)
		buf.WriteString("\n--- git status --porcelain ---\n")
		buf.WriteString(status)
		if strings.TrimSpace(diff) != "" {
			buf.WriteString("\n--- git diff HEAD ---\n")
			buf.WriteString(clipBytes(diff, maxShadowInputBytes))
		}

		out, err := a.llmSimple(ctx, sess.ID, conn, []llmMessage{{Role: "user", Content: buf.String()}})
		if err != nil {
			return
		}

		// Race guard: re-check status before write. If the user committed
		// during the LLM call, skip — otherwise we'd resurrect a stale file
		// that cleanupGitCommitIfClean has not yet had a chance to delete.
		if s2, err := gitStatusPorcelain(sess.Cwd); err == nil && strings.TrimSpace(s2) == "" {
			return
		}

		path := filepath.Join(sess.Cwd, gitCommitFile)
		_ = os.MkdirAll(filepath.Dir(path), 0o755)
		_ = os.WriteFile(path, []byte(strings.TrimSpace(out)+"\n"), 0o644)
	}()
}
