package main

import (
	"context"
	"crypto/sha256"
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

// cleanupGitCommitIfClean is called from checkMCP on every user prompt.
// Waits for any in-flight backgroundGitCommit to finish (otherwise its late
// write would resurrect a file we are about to delete), then checks
// `git status --porcelain`. Empty status means everything was committed
// externally — delete .codehalter/.git_commit so the next round starts
// fresh from the next uncommitted change.
func (a *agent) cleanupGitCommitIfClean(cwd string, sid string) {
	if info, err := os.Stat(filepath.Join(cwd, ".git")); err != nil || !info.IsDir() {
		return
	}
	if sess := a.getSession(sid); sess != nil {
		sess.gitCommitJob.Wait()
	}
	status, err := gitStatusPorcelain(cwd)
	if err != nil {
		return
	}
	if strings.TrimSpace(status) != "" {
		return
	}
	_ = os.Remove(filepath.Join(cwd, gitCommitFile))
	// Reset hash so the next non-empty status regenerates the file, even
	// if (rarely) the new status+diff hashes identical to the prior one.
	if sess := a.getSession(sid); sess != nil {
		sess.gitCommitMu.Lock()
		sess.gitCommitLastHash = [32]byte{}
		sess.gitCommitMu.Unlock()
	}
}

// backgroundGitCommit fires after every assistant turn. Snapshots the current
// `git diff HEAD` + `git status --porcelain` and asks the LLM to (re)write
// .codehalter/.git_commit so it always matches the latest uncommitted state.
// Self-skips when:
//   - cwd has no .git directory (not a checkout, or .git not mounted),
//   - the working tree is clean (nothing to summarise),
//   - no eligible background slot is available (see pickBackgroundLLM).
//
// Multiple in-flight calls are allowed — they race on the file with
// last-write-wins, which is fine because each LLM call's snapshot is point-in-
// time and the freshest wins. The pre-write status re-check guards against
// the narrow race where the user commits during the LLM call. When this and
// the shadow summariser land on the same entry, the per-conn parallel
// semaphore in llmStream serialises them naturally — no explicit join needed.
func (a *agent) backgroundGitCommit(sess *Session) {
	if sess == nil {
		return
	}
	if info, err := os.Stat(filepath.Join(sess.Cwd, ".git")); err != nil || !info.IsDir() {
		return
	}
	status, err := gitStatusPorcelain(sess.Cwd)
	if err != nil || strings.TrimSpace(status) == "" {
		return
	}
	diff, _ := gitDiffHead(sess.Cwd)
	hash := sha256.Sum256([]byte(status + "\x00" + diff))
	sess.gitCommitMu.Lock()
	unchanged := hash == sess.gitCommitLastHash
	sess.gitCommitMu.Unlock()
	if unchanged {
		return
	}

	if !sess.gitCommitJob.TryStart() {
		return
	}
	conn := a.pickBackgroundLLM(sess.ID)
	if conn == nil {
		sess.gitCommitJob.Done()
		return
	}

	go func() {
		defer sess.gitCommitJob.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		var buf strings.Builder
		buf.WriteString(gitCommitPrompt)
		buf.WriteString("\n<git_status>\n")
		buf.WriteString(status)
		buf.WriteString("</git_status>\n")
		if strings.TrimSpace(diff) != "" {
			buf.WriteString("\n<git_diff>\n")
			buf.WriteString(clipBytes(diff, maxShadowInputBytes))
			buf.WriteString("\n</git_diff>\n")
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
		if err := os.WriteFile(path, []byte(strings.TrimSpace(out)+"\n"), 0o644); err != nil {
			return
		}
		sess.gitCommitMu.Lock()
		sess.gitCommitLastHash = hash
		sess.gitCommitMu.Unlock()
	}()
}
