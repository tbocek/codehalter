package main

import (
	"fmt"
	"hash/fnv"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"unicode/utf8"
)

// fnvHash returns the 64-bit FNV-1a hash of s — a cheap, deterministic identity
// for "is this byte-for-byte the content I saw before?" (read dedup, the tool
// loop's repeat detection). Not cryptographic; collisions are irrelevant here.
func fnvHash(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// parallel runs fn for each index [0, n) with up to `cap` concurrent
// goroutines. Callers pass an explicit upper bound matched to the work-list
// (e.g. probeAllLLMs's len(conns))
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

// 0.6 catches rephrasings of short reasons ("missing import" / "import is
// missing" = 0.67) without merging genuinely different failures.
const failureSimilarityThreshold = 0.6

// Catches reruns whose output differs only by a timestamp or pid. Much higher than
// failureSimilarityThreshold because long tool outputs share vocabulary easily.
const stuckOutputSimilarity = 0.9

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

// trimJSON extracts the first balanced JSON object: models wrap JSON in prose or
// fences. Without one it returns s trimmed and the caller reports the parse error.
func trimJSON(s string) string { return trimBalanced(s, '{', '}') }

func trimJSONArray(s string) string { return trimBalanced(s, '[', ']') }

func trimBalanced(s string, open, close byte) string {
	s = strings.TrimSpace(s)
	start := strings.IndexByte(s, open)
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
		case open:
			depth++
		case close:
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}
	return s
}

// Must return an absolute path: resolvePath prefix-checks against sess.Cwd, which
// breaks for a relative root like "." because filepath.Clean drops the "./".
func cwdOrDefault(cwd string) string {
	if cwd == "" {
		cwd, _ = os.Getwd()
	}
	if abs, err := filepath.Abs(cwd); err == nil {
		return abs
	}
	return cwd
}

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

// usableCwd falls back to the process cwd when the requested root isn't mounted
// here (Zed restores threads from other devcontainers); the bool reports a fallback.
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

func truncate(s string, maxLen int) string {
	if len(s) > maxLen {
		return clipUTF8(s, maxLen) + "..."
	}
	return s
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return s
}

func clipUTF8(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// Every cut of text that can reach the session file must use clipUTF8 or tailUTF8:
// invalid UTF-8 in it makes the session unloadable (see loadSession).
func tailUTF8(s string, n int) string {
	if len(s) <= n {
		return s
	}
	i := len(s) - n
	for i < len(s) && !utf8.RuneStart(s[i]) {
		i++
	}
	return s[i:]
}

// maxLLMInputBytes caps a single payload sent to an LLM outside the main tool loop.
const maxLLMInputBytes = 20 * 1024

// clipBytes keeps the head and tail halves with a truncation marker between.
func clipBytes(s string, max int) string {
	if len(s) <= max {
		return s
	}
	half := max / 2
	return clipUTF8(s, half) + fmt.Sprintf("\n[... %d bytes truncated ...]\n", len(s)-max) + tailUTF8(s, half)
}

// writeFileAtomic replaces path in one step: a temp file beside it, synced, then
// renamed over it. A crash leaves the old file or the new one, never half of
// either. The machine went down twice in one day while the session file, the
// /spec ledger and QUESTIONS.md were being rewritten in place.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	// Hidden and without the target's extension, so no listing takes it for the file.
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	fail := func(err error) error {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if _, err := f.Write(data); err != nil {
		return fail(err)
	}
	if err := f.Chmod(perm); err != nil {
		return fail(err)
	}
	if err := f.Sync(); err != nil {
		return fail(err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}

// stackFrameRe matches one line of a stack trace in the common languages: a
// numbered Rust frame, an `at` line (Rust, Java, JavaScript, C#), Python's
// `File "…", line N`, Go's `/path/file.go:N +0x…`.
var stackFrameRe = regexp.MustCompile(`^\s*\d+:\s+(0x[0-9a-f]+ - )?\S|^\s+at \S|^\s+File ".*", line \d+|^\s+\S+\.go:\d+( \+0x[0-9a-f]+)?$`)

// stackCollapseMin: a shorter trace stays, it is the failure's own location.
const stackCollapseMin = 6

// collapseStackTraces shortens each long stack trace to one line, so cutting a
// failed test run's output to its end keeps the failure's message: a Rust
// panic's backtrace alone filled the last 4,000 bytes, and the assertion above
// it was cut off for the next round and for the question to the user.
func collapseStackTraces(out string) string {
	lines := strings.Split(out, "\n")
	kept := make([]string, 0, len(lines))
	for i := 0; i < len(lines); {
		if !stackFrameRe.MatchString(lines[i]) {
			kept = append(kept, lines[i])
			i++
			continue
		}
		// A trace runs over frame lines, each perhaps followed by one other line:
		// Python's source line, Go's function line.
		j, frames := i, 0
		for j < len(lines) {
			switch {
			case stackFrameRe.MatchString(lines[j]):
				frames++
				j++
				continue
			case j+1 < len(lines) && stackFrameRe.MatchString(lines[j+1]):
				j++
				continue
			}
			break
		}
		if frames >= stackCollapseMin {
			kept = append(kept, fmt.Sprintf("[… %d stack-trace lines left out …]", j-i))
		} else {
			kept = append(kept, lines[i:j]...)
		}
		i = j
	}
	return strings.Join(kept, "\n")
}
