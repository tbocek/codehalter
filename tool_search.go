package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const maxSearchResults = 100

func init() {
	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "search_text",
			"description": "Search for text across all files in the project. Returns up to 100 matches as file:line pairs.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"query"},
				"properties": map[string]any{
					"query": map[string]any{"type": "string", "description": "Text to search for (case-sensitive substring match)"},
					"path":  map[string]any{"type": "string", "description": "Subdirectory to search in (relative to project root, empty for all)"},
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
			matches := searchInFile(absPath, query, maxSearchResults-len(results))
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

func searchInFile(path, query string, limit int) []int {
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
		if strings.Contains(scanner.Text(), query) {
			matches = append(matches, lineNum)
			if len(matches) >= limit {
				break
			}
		}
	}
	return matches
}
