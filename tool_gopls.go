package main

import (
	"context"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
)

// ensureGopls starts the shared gopls client on first use, rooted at cwd.
// Subsequent calls return the existing client (or the recorded start error).
func (a *agent) ensureGopls(ctx context.Context, cwd string) (*Gopls, error) {
	a.goplsOnce.Do(func() {
		g, err := StartGopls(ctx, cwd)
		if err != nil {
			a.goplsErr = err
			return
		}
		a.gopls = g
	})
	if a.goplsErr != nil {
		return nil, a.goplsErr
	}
	return a.gopls, nil
}

func init() {
	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "go_symbols",
			"description": "Find Go symbols (functions, types, methods, variables) by name across the workspace. Backed by gopls workspace/symbol. Returns kind, name, and file:line:col. Use this when you have a symbol name and want to jump to its declaration.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"query"},
				"properties": map[string]any{
					"query": map[string]any{"type": "string", "description": "Symbol name or fragment (case-insensitive substring match)."},
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

		tcId := a.StartToolCall(ctx, sid, "go_symbols: "+query, "search", nil)
		g, err := a.ensureGopls(ctx, sess.Cwd)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}
		syms, err := g.WorkspaceSymbol(ctx, query)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}
		if len(syms) == 0 {
			a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("no symbols found")})
			return "no symbols found"
		}

		var b strings.Builder
		for _, s := range syms {
			rel, err := filepath.Rel(sess.Cwd, s.File)
			if err != nil {
				rel = s.File
			}
			fmt.Fprintf(&b, "%s %s %s:%d:%d\n", s.Kind, s.Name, rel, s.Line, s.Col)
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("%d symbols", len(syms)))})
		return strings.TrimRight(b.String(), "\n")
	}})

	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "go_references",
			"description": "Find all references to the Go symbol at a given file position. Backed by gopls textDocument/references. Use go_symbols first to get file:line:col, then pass those here. Line and column are 1-indexed.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"file", "line", "col"},
				"properties": map[string]any{
					"file": map[string]any{"type": "string", "description": "Path to the Go file (absolute or relative to project root)."},
					"line": map[string]any{"type": "string", "description": "1-indexed line number where the symbol appears."},
					"col":  map[string]any{"type": "string", "description": "1-indexed column where the symbol name starts."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
		args := parseArgs(rawArgs)
		sess := a.getSession(sid)
		if sess == nil {
			return "error: no session"
		}
		file := args["file"]
		if file == "" {
			return "error: file is empty"
		}
		line, err := strconv.Atoi(args["line"])
		if err != nil || line < 1 {
			return "error: invalid line"
		}
		col, err := strconv.Atoi(args["col"])
		if err != nil || col < 1 {
			return "error: invalid col"
		}
		absPath, err := a.resolvePath(sid, file)
		if err != nil {
			return "error: " + err.Error()
		}

		title := fmt.Sprintf("go_references: %s:%d:%d", file, line, col)
		tcId := a.StartToolCall(ctx, sid, title, "search", nil)
		g, err := a.ensureGopls(ctx, sess.Cwd)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}
		refs, err := g.References(ctx, absPath, line, col)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}
		if len(refs) == 0 {
			a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("no references found")})
			return "no references found"
		}

		var b strings.Builder
		for _, r := range refs {
			rel, err := filepath.Rel(sess.Cwd, r.File)
			if err != nil {
				rel = r.File
			}
			fmt.Fprintf(&b, "%s:%d:%d\n", rel, r.Line, r.Col)
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("%d references", len(refs)))})
		return strings.TrimRight(b.String(), "\n")
	}})
}
