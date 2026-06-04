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

const maxSearchResults = 100

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "search_text",
			"description": "Search for text or a regex across all files in the project. Returns up to 100 matches as file:line pairs. Case-sensitive by default — use `(?i)` inline flag in regex mode for case-insensitive. Line-oriented by default; set multiline=true so a regex can match across newlines.",
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
			for _, lineNum := range matches {
				results = append(results, fmt.Sprintf("%s:%d", relPath, lineNum))
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

func searchInFile(path string, matcher func(string) bool, limit int) []int {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	var matches []int
	scanner := bufio.NewScanner(f)
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
