package main

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// parallel runs fn for each index [0, n) with up to `cap` concurrent
// goroutines. Callers pass an explicit upper bound matched to the work-list
// (e.g. launch_subagent's SubagentPinOrder length, probeAllLLMs's len(conns))
// so excess work queues instead of contending for slots.
func parallel(n, cap int, fn func(i int)) {
	if cap > n {
		cap = n
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, cap)
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			fn(i)
		}(i)
	}
	wg.Wait()
}

// failureSimilarityThreshold is the Jaccard ratio above which two failed-
// subtask reason bags are considered "the same problem." Tuned empirically
// for short LLM-generated strings: 0.6 catches "missing import" /
// "import is missing" (Jaccard 0.67) without collapsing genuinely different
// files. The orchestrator uses this to escalate the replan context — same
// failure recurring N times tells the planner the prior fix didn't work
// and a structurally different approach is needed.
const failureSimilarityThreshold = 0.6

// issueBag tokenises a list of issue strings into a single set of distinct
// lowercase alphanumeric words. Punctuation, casing and ordering are all
// discarded so two attempts reporting the same root cause in different
// phrasing collapse to comparable bags.
func issueBag(issues []string) map[string]bool {
	bag := make(map[string]bool)
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			bag[cur.String()] = true
			cur.Reset()
		}
	}
	for _, iss := range issues {
		for _, r := range strings.ToLower(iss) {
			switch {
			case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
				cur.WriteRune(r)
			default:
				flush()
			}
		}
		flush()
	}
	return bag
}

// jaccard returns |A ∩ B| / |A ∪ B| for two word sets. 1.0 = identical,
// 0.0 = disjoint. Two empty bags are treated as identical.
func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	inter := 0
	for w := range a {
		if b[w] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// trimJSON extracts a JSON object from an LLM response. Small models often
// wrap the JSON in prose ("Sure, here's the JSON: { … } Let me know!") or
// markdown fences; we just locate the first `{` and the matching `}` and
// keep that slice. Brace counting respects strings + escapes so braces inside
// string values don't confuse the scan. Returns the trimmed input unchanged
// if no balanced object is found — caller surfaces the parse error.
func trimJSON(s string) string {
	s = strings.TrimSpace(s)
	start := strings.IndexByte(s, '{')
	if start < 0 {
		return s
	}
	depth := 0
	inStr := false
	esc := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}
	return s
}

// cwdOrDefault resolves the session's working directory to a clean absolute
// path. Clients may pass "." (bench harness) or any relative path; resolvePath
// then prefix-checks against sess.Cwd, and the check breaks when Cwd isn't
// absolute because filepath.Clean drops the leading "./" — read_file("go.mod")
// would resolve to "go.mod" and fail the "outside project directory" check
// even though it's inside the project.
func cwdOrDefault(cwd string) string {
	if cwd == "" {
		cwd, _ = os.Getwd()
	}
	if abs, err := filepath.Abs(cwd); err == nil {
		return abs
	}
	return cwd
}

// cwdAvailable reports whether the client-supplied workspace root actually
// exists as a directory in this environment. Zed restores agent threads with
// the cwd they were created under; when that workspace isn't mounted here
// (e.g. a thread from another project's devcontainer), every later step —
// scaffolding .codehalter, reading the session .toml — fails with a confusing
// low-level mkdir/permission error against a path the user never chose. Gate
// session/new and session/load on this up front so they refuse with a clear
// message instead of trying to create .codehalter under an unavailable root.
func cwdAvailable(cwd string) error {
	info, err := os.Stat(cwd)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("workspace %s is not available in this environment (mount it, or open the project from a path that exists here)", cwd)
		}
		return fmt.Errorf("workspace %s: %w", cwd, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("workspace %s is not a directory", cwd)
	}
	return nil
}

// usableCwd resolves the client-supplied req.Cwd to an absolute workspace root
// that actually exists in this environment. It prefers the requested path, but
// when that isn't mounted here (e.g. Zed restoring a thread created under
// another project's devcontainer — /workspaces/codehalter while the user has
// since switched to preveltekit) it falls back to the directory the agent
// process was launched in rather than refusing to start. The substitution is
// logged for diagnosis. The bool is true when a fallback was substituted, so
// the caller can tell the user this is a fresh session rather than a restored
// one. Returns an error only when neither the requested cwd nor the process cwd
// is a usable directory.
func usableCwd(reqCwd string) (string, bool, error) {
	cwd := cwdOrDefault(reqCwd)
	err := cwdAvailable(cwd)
	if err == nil {
		return cwd, false, nil
	}
	fallback := cwdOrDefault("")
	if cwdAvailable(fallback) != nil {
		return "", false, err
	}
	slog.Info("workspace unavailable; opening a new session under the process cwd",
		"requested", cwd, "fallback", fallback, "err", err)
	return fallback, true, nil
}

// truncate shortens s to maxLen runes-ish (bytes) with an ellipsis suffix when
// it overflows; shorter strings pass through unchanged.
func truncate(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}

// maxLLMInputBytes caps the bytes of any single text payload handed to an LLM
// outside the main tool loop: a clipped per-turn summary / git diff for a
// background call (backgroundSummarise, backgroundGitCommit, via clipBytes), or
// a whole-file attachment inlined into the foreground prompt (readLinkedResource).
// Without it a megabyte blob — a huge run_command dump, a giant attached file —
// would blow through the LLM's context window.
const maxLLMInputBytes = 20 * 1024

// clipBytes truncates s to at most max bytes, leaving a marker in the middle
// when it had to cut. Used to bound any single payload's contribution to a
// background LLM call (turn summaries via backgroundSummarise, git diffs via
// backgroundGitCommit) so one giant tool output can't blow the summariser's
// own context window.
func clipBytes(s string, max int) string {
	if len(s) <= max {
		return s
	}
	half := max / 2
	return s[:half] + fmt.Sprintf("\n[... %d bytes truncated ...]\n", len(s)-max) + s[len(s)-half:]
}
