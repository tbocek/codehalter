package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	maxSearchResults   = 100
	searchContextLines = 2 // lines of context shown on each side of a match
)

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "search_text",
			"description": fmt.Sprintf("Search for text or a regex across all files in the project. Returns up to %d matches, each as `file:line` plus the matched line and %d lines of context on each side (the match line marked with `>`) — so you can often act on a hit without opening the file. Case-sensitive by default — use `(?i)` inline flag in regex mode for case-insensitive. Line-oriented by default; set multiline=true so a regex can match across newlines.", maxSearchResults, searchContextLines),
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"query"},
				"properties": map[string]any{
					"query":     map[string]any{"type": "string", "description": "Text to search for. Literal substring by default; Go RE2 regex when regex=true."},
					"path":      map[string]any{"type": "string", "description": "Subdirectory to search in (relative to project root, empty for all)"},
					"regex":     map[string]any{"type": "boolean", "description": "If true, interpret query as a Go RE2 regular expression. Default: false (literal substring)."},
					"multiline": map[string]any{"type": "boolean", "description": "If true, match against the whole file at once so patterns can span newlines (e.g. `foo\\nbar`). Implies regex=true. Default: false (line-by-line)."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		sess := a.getSession(sid)
		if sess == nil {
			return "error: no session", false
		}
		query := args["query"]
		if query == "" {
			return "error: query is empty", false
		}
		multiline := args["multiline"] == "true"
		useRegex := multiline || args["regex"] == "true"

		var re *regexp.Regexp
		matcher := func(s string) bool { return strings.Contains(s, query) }
		if useRegex {
			compiled, err := regexp.Compile(query)
			if err != nil {
				return "error: invalid regex: " + err.Error(), false
			}
			re = compiled
			matcher = re.MatchString
		}

		root := sess.Cwd
		dir := root
		subdir := args["path"]
		if subdir != "" {
			resolved, err := a.resolvePath(sid, subdir)
			if err != nil {
				return "error: " + err.Error(), false
			}
			dir = resolved
		}

		// label carries the path so two searches with the same query but different
		// scopes (e.g. `res` vs the whole repo) read distinctly in the UI — a "0
		// matches in res" then "17 matches" is otherwise indistinguishable.
		label := query
		if subdir != "" {
			label += " in " + subdir
		}
		tcId := a.StartToolCall(ctx, sid, "Searching: "+label, "search", nil)

		var results []string
		dirCounts := map[string]int{} // matches per path bucket, for the cap-hit hint
		files := listProjectFiles(dir)
		for _, relPath := range files {
			if len(results) >= maxSearchResults {
				break
			}
			absPath := filepath.Join(dir, relPath)
			var matches []int
			if multiline {
				matches = searchInFileMultiline(absPath, re, maxSearchResults-len(results))
			} else {
				matches = searchInFile(absPath, matcher, maxSearchResults-len(results))
			}
			if len(matches) == 0 {
				continue
			}
			dirCounts[searchBucket(relPath)] += len(matches)
			// Read the file once to pull the lines around each hit.
			var lines []string
			if data, err := os.ReadFile(absPath); err == nil {
				lines = strings.Split(string(data), "\n")
				if n := len(lines); n > 0 && lines[n-1] == "" {
					lines = lines[:n-1]
				}
			}
			for _, lineNum := range matches {
				results = append(results, formatMatchBlock(relPath, lines, lineNum, searchContextLines))
			}
		}

		out := "no matches found"
		summary := "no matches"
		if len(results) > 0 {
			summary = fmt.Sprintf("%d matches", len(results))
			out = strings.Join(results, "\n")
			if len(results) >= maxSearchResults {
				summary += ", limit reached"
				// Hit the cap: more matches exist than shown, and the file-walk
				// order may have spent the whole budget on one noisy tree (e.g.
				// gitignored build/bench logs). Point the model at where the shown
				// matches cluster so it can re-scope with `path` or ignore that
				// tree itself, instead of re-running the same flooded search.
				if hot := topBuckets(dirCounts, 5); hot != "" {
					out = fmt.Sprintf("[note: hit the %d-match cap — more matches exist than shown, so this result is partial. The matches shown cluster under: %s. Re-run with a narrower `path`, or ignore those paths.]\n\n%s", maxSearchResults, hot, out)
				}
			}
		}

		// Literal-repeat dedup: the same query+path+flags returning the same
		// results this turn gets a note so the model reuses the earlier result
		// instead of re-running. Mirrors read_file's readUnchangedMarker dedup;
		// loop.go scans the output for the marker to also count it as a stuck
		// round (the prepended note changes the bytes, so callOutHash misses it).
		dedupKey := query + "\x00" + subdir + "\x00" + fmt.Sprintf("%t%t", useRegex, multiline)
		sum := fnvHash(out)
		var dedupNote string
		sess.searchDedupMu.Lock()
		if prev, ok := sess.searchDedup[dedupKey]; ok && prev == sum {
			dedupNote = fmt.Sprintf("[note: %s — you already ran this exact search (%s) earlier this turn and the results are UNCHANGED. Re-searching makes no progress; reuse the earlier result.]", readUnchangedMarker, label)
		}
		if sess.searchDedup == nil {
			sess.searchDedup = map[string]uint64{}
		}
		sess.searchDedup[dedupKey] = sum
		sess.searchDedupMu.Unlock()

		title := "Searching: " + label + " (" + summary + ")"
		if dedupNote != "" {
			title += " (repeat)"
		}
		a.CompleteToolCallTitled(ctx, sid, tcId, title, []ToolCallContent{TextContent(summary)})
		if dedupNote != "" {
			out = dedupNote + "\n" + out
		}
		return out, false
	}})
}

// formatMatchBlock renders one search hit as `file:matchLine` followed by the
// matched line and up to ctx lines of context on each side, line-numbered, the
// match marked with ">". Falls back to a bare file:line when the file couldn't
// be read for context (lines nil/empty).
func formatMatchBlock(relPath string, lines []string, matchLine, ctx int) string {
	if matchLine < 1 || matchLine > len(lines) {
		return fmt.Sprintf("%s:%d", relPath, matchLine)
	}
	lo := matchLine - ctx
	if lo < 1 {
		lo = 1
	}
	hi := matchLine + ctx
	if hi > len(lines) {
		hi = len(lines)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s:%d\n", relPath, matchLine)
	for i := lo; i <= hi; i++ {
		marker := "  "
		if i == matchLine {
			marker = "> "
		}
		fmt.Fprintf(&b, "%s%d| %s\n", marker, i, lines[i-1])
	}
	return b.String()
}

// searchBucket groups a matched file under the first up-to-2 path segments
// (bench/results/2026.../x.log → "bench/results"), the granularity the model
// can hand straight back as a `path` scope. Root-level files bucket as "(root)".
func searchBucket(relPath string) string {
	d := filepath.ToSlash(filepath.Dir(relPath))
	if d == "" || d == "." {
		return "(root)"
	}
	parts := strings.Split(d, "/")
	if len(parts) > 2 {
		parts = parts[:2]
	}
	return strings.Join(parts, "/")
}

// topBuckets renders the highest-count buckets as "path (n) · path (n)",
// highest first, ties broken by name, up to n entries. Empty when counts is.
func topBuckets(counts map[string]int, n int) string {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if counts[keys[i]] != counts[keys[j]] {
			return counts[keys[i]] > counts[keys[j]]
		}
		return keys[i] < keys[j]
	})
	parts := make([]string, 0, n)
	for _, k := range keys {
		if len(parts) >= n {
			break
		}
		parts = append(parts, fmt.Sprintf("%s (%d)", k, counts[k]))
	}
	return strings.Join(parts, " · ")
}

func searchInFile(path string, matcher func(string) bool, limit int) []int {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, binarySniffLen)
	if head, _ := r.Peek(binarySniffLen); looksBinary(head) {
		return nil // binary file — never scan as text (its bytes poison the context)
	}
	var matches []int
	scanner := bufio.NewScanner(r)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		if matcher(scanner.Text()) {
			matches = append(matches, lineNum)
			if len(matches) >= limit {
				return matches
			}
		}
	}
	if err := scanner.Err(); err != nil {
		// A line longer than bufio's 64 KB default ends the scan early; the
		// match list is then partial. Log it rather than returning a silently
		// truncated result as if the whole file was searched.
		slog.Debug("searchInFile: scan ended early (long line?)", "path", path, "err", err)
	}
	return matches
}

// searchInFileMultiline runs the regex against the whole file so patterns can
// straddle newlines. Match positions come back as byte offsets; we convert the
// start offset of each match to a 1-indexed line number by counting `\n` in
// the prefix. The conversion is single-pass over the file.
func searchInFileMultiline(path string, re *regexp.Regexp, limit int) []int {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	if looksBinary(data) {
		return nil // binary file — never scan as text (its bytes poison the context)
	}
	idx := re.FindAllIndex(data, limit)
	if len(idx) == 0 {
		return nil
	}
	matches := make([]int, 0, len(idx))
	line := 1
	pos := 0
	for _, m := range idx {
		for pos < m[0] {
			if data[pos] == '\n' {
				line++
			}
			pos++
		}
		matches = append(matches, line)
	}
	return matches
}
