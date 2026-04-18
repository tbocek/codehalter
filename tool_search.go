package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const maxSearchResults = 100

func init() {
	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "search_text",
			"description": "Search for text or a regex across all files in the project. Returns up to 100 matches as file:line pairs. Case-sensitive by default — use `(?i)` inline flag in regex mode for case-insensitive.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"query"},
				"properties": map[string]any{
					"query": map[string]any{"type": "string", "description": "Text to search for. Literal substring by default; Go RE2 regex when regex=true."},
					"path":  map[string]any{"type": "string", "description": "Subdirectory to search in (relative to project root, empty for all)"},
					"regex": map[string]any{"type": "boolean", "description": "If true, interpret query as a Go RE2 regular expression. Default: false (literal substring)."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
			args := parseArgs(rawArgs)
		sess := a.getSession(sid)
		if sess == nil {
			return "error: no session"
		}
		query := args["query"]
		if query == "" {
			return "error: query is empty"
		}

		matcher := func(s string) bool { return strings.Contains(s, query) }
		if args["regex"] == "true" {
			re, err := regexp.Compile(query)
			if err != nil {
				return "error: invalid regex: " + err.Error()
			}
			matcher = re.MatchString
		}

		root := sess.Cwd
		dir := root
		if subdir := args["path"]; subdir != "" {
			resolved, err := a.resolvePath(sid, subdir)
			if err != nil {
				return "error: " + err.Error()
			}
			dir = resolved
		}

		tcId := a.StartToolCall(ctx, sid, "Searching for: "+query, "search", nil)

		var results []string
		files := listProjectFiles(dir)
		for _, relPath := range files {
			if len(results) >= maxSearchResults {
				break
			}
			absPath := filepath.Join(root, relPath)
			matches := searchInFile(absPath, matcher, maxSearchResults-len(results))
			for _, lineNum := range matches {
				results = append(results, fmt.Sprintf("%s:%d", relPath, lineNum))
			}
		}

		if len(results) == 0 {
			a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("no matches found")})
			return "no matches found"
		}

		summary := fmt.Sprintf("%d matches", len(results))
		if len(results) >= maxSearchResults {
			summary += " (limit reached)"
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(summary)})
		return strings.Join(results, "\n")
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
				break
			}
		}
	}
	return matches
}
