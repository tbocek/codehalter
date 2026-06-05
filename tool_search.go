package main

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
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
		if subdir := args["path"]; subdir != "" {
			resolved, err := a.resolvePath(sid, subdir)
			if err != nil {
				return "error: " + err.Error(), false
			}
			dir = resolved
		}

		tcId := a.StartToolCall(ctx, sid, "Searching: "+query, "search", nil)

		var results []string
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

		if len(results) == 0 {
			a.CompleteToolCallTitled(ctx, sid, tcId,
				"Searching: "+query+" (no matches)",
				[]ToolCallContent{TextContent("no matches found")})
			return "no matches found", false
		}

		summary := fmt.Sprintf("%d matches", len(results))
		if len(results) >= maxSearchResults {
			summary += ", limit reached"
		}
		a.CompleteToolCallTitled(ctx, sid, tcId,
			fmt.Sprintf("Searching: %s (%s)", query, summary),
			[]ToolCallContent{TextContent(summary)})
		return strings.Join(results, "\n"), false
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
