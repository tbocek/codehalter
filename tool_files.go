package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

var skipDirs = map[string]bool{
	".git": true, ".codehalter": true, "node_modules": true,
	"__pycache__": true, ".venv": true, "vendor": true,
	".idea": true, ".vscode": true, "target": true, "dist": true, "build": true,
}

// Read-size caps guard the LLM context against pathological files. When the
// caller doesn't pass `limit`, read_file reads defaultReadLines and warns if
// truncated. An explicit `limit` is still capped at maxReadLines, and the
// final content is further capped by maxReadBytes to stop a minified blob
// from blowing through the byte budget even under the line limit.
const (
	defaultReadLines = 2000
	maxReadLines     = 5000
	maxReadBytes     = 200 * 1024
)

// capReadContent trims over-long reads and returns a note describing what was
// dropped. Returns empty note when nothing was trimmed.
func capReadContent(content string, explicitLimit bool, startLine string) (string, string) {
	var notes []string
	lineCount := strings.Count(content, "\n")
	if !strings.HasSuffix(content, "\n") && content != "" {
		lineCount++
	}

	// Only emit a truncation hint when we capped something the caller didn't
	// opt into. If they asked for limit=X explicitly, hitting X is expected.
	if !explicitLimit && lineCount >= defaultReadLines {
		start := "1"
		if startLine != "" {
			start = startLine
		}
		notes = append(notes, fmt.Sprintf("[truncated at %d lines starting at line %s; call read_file again with a later `line` to see the rest]", defaultReadLines, start))
	}

	if len(content) > maxReadBytes {
		content = content[:maxReadBytes]
		notes = append(notes, fmt.Sprintf("[truncated at %d bytes — file has long lines; use `line`+`limit` to narrow]", maxReadBytes))
	}

	return content, strings.Join(notes, " ")
}

// listProjectFiles returns relative paths of all files under root, skipping common junk dirs.
func listProjectFiles(root string) []string {
	var files []string
	filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		files = append(files, rel)
		return nil
	})
	return files
}

func init() {
	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "list_files",
			"description": "List files in the project directory. Returns a newline-separated list of relative paths.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{"type": "string", "description": "Subdirectory to list (relative to project root, empty for root)"},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
			args := parseArgs(rawArgs)
		sess := a.getSession(sid)
		if sess == nil {
			return "error: no session"
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

		tcId := a.StartToolCall(ctx, sid, "Listing files", "search", nil)
		files := listProjectFiles(dir)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("%d files", len(files)))})
		return strings.Join(files, "\n")
	}})

	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "read_file",
			"description": fmt.Sprintf("Read the contents of a file from the project. When neither line nor limit is given, reads up to %d lines (or %d KB) — if the file is larger the response is truncated with a note giving the total line count, and you can re-read specific ranges with line+limit.", defaultReadLines, maxReadBytes/1024),
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path"},
				"properties": map[string]any{
					"path":  map[string]any{"type": "string", "description": "Path to the file (absolute or relative to project root)"},
					"line":  map[string]any{"type": "integer", "description": "Start line (1-based). Omit to read from the beginning."},
					"limit": map[string]any{"type": "integer", "description": fmt.Sprintf("Max number of lines to read (hard cap %d). Omit for the default window.", maxReadLines)},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
			args := parseArgs(rawArgs)
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error()
		}

		title := "Reading " + path
		if args["line"] != "" {
			title = fmt.Sprintf("Reading %s:%s", path, args["line"])
		}
		tcId := a.StartToolCall(ctx, sid, title, "read", []ToolCallLocation{{Path: path}})

		effectiveLimit := args["limit"]
		explicitLimit := effectiveLimit != ""
		if !explicitLimit {
			effectiveLimit = strconv.Itoa(defaultReadLines)
		} else if n, err := strconv.Atoi(effectiveLimit); err == nil && n > maxReadLines {
			effectiveLimit = strconv.Itoa(maxReadLines)
		}

		content, err := fsReadRange(a.conn.RPC(), ctx, sid, path, args["line"], effectiveLimit)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}

		content, truncNote := capReadContent(content, explicitLimit, args["line"])
		if truncNote != "" {
			content += "\n" + truncNote
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(content)})
		return content
	}})

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "write_file",
			"description": "Write content to a file in the project.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path", "content"},
				"properties": map[string]any{
					"path":    map[string]any{"type": "string", "description": "Path to the file (absolute or relative to project root)"},
					"content": map[string]any{"type": "string", "description": "The new file content"},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
			args := parseArgs(rawArgs)
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error()
		}
		newContent := args["content"]
		tcId := a.StartToolCall(ctx, sid, "Writing "+path, "edit", []ToolCallLocation{{Path: path}})

		oldContent, _ := fsRead(a.conn.RPC(), ctx, sid, path)

		if err := fsWrite(a.conn.RPC(), ctx, sid, path, newContent); err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error writing file: " + err.Error()
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &oldContent, newContent)})

		a.mu.Lock()
		a.pendingRefs = append(a.pendingRefs, MakeFileRef(path))
		a.mu.Unlock()

		return "file written successfully"
	}})

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "edit_file",
			"description": "Edit a file by replacing a specific text snippet. Use read_file first to see the current content. The old_text must match exactly (including whitespace).",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path", "old_text", "new_text"},
				"properties": map[string]any{
					"path":     map[string]any{"type": "string", "description": "Path to the file (absolute or relative to project root)"},
					"old_text": map[string]any{"type": "string", "description": "The exact text to find and replace"},
					"new_text": map[string]any{"type": "string", "description": "The replacement text"},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
			args := parseArgs(rawArgs)
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error()
		}
		oldText := args["old_text"]
		newText := args["new_text"]

		tcId := a.StartToolCall(ctx, sid, "Editing "+path, "edit", []ToolCallLocation{{Path: path}})

		content, err := fsRead(a.conn.RPC(), ctx, sid, path)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error reading file: " + err.Error()
		}

		count := strings.Count(content, oldText)
		if count == 0 {
			a.FailToolCall(ctx, sid, tcId, "old_text not found in file")
			return "error: old_text not found in file"
		}
		if count > 1 {
			a.FailToolCall(ctx, sid, tcId, fmt.Sprintf("old_text matches %d times, must be unique", count))
			return fmt.Sprintf("error: old_text matches %d times, must be unique", count)
		}

		newContent := strings.Replace(content, oldText, newText, 1)

		if err := fsWrite(a.conn.RPC(), ctx, sid, path, newContent); err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error writing file: " + err.Error()
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &content, newContent)})

		a.mu.Lock()
		a.pendingRefs = append(a.pendingRefs, MakeFileRef(path))
		a.mu.Unlock()

		return "file written successfully"
	}})
}


func fsReadRange(c *Connection, ctx context.Context, sid SessionId, path, line, limit string) (string, error) {
	type req struct {
		SessionId SessionId `json:"sessionId"`
		Path      string    `json:"path"`
		Line      *int      `json:"line,omitempty"`
		Limit     *int      `json:"limit,omitempty"`
	}
	r := req{SessionId: sid, Path: path}
	if line != "" {
		if v, err := strconv.Atoi(line); err == nil {
			r.Line = &v
		}
	}
	if limit != "" {
		if v, err := strconv.Atoi(limit); err == nil {
			r.Limit = &v
		}
	}
	resp, err := SendRequest[struct {
		Content string `json:"content"`
	}](c, ctx, "fs/read_text_file", r)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

func fsRead(c *Connection, ctx context.Context, sid SessionId, path string) (string, error) {
	resp, err := SendRequest[struct {
		Content string `json:"content"`
	}](c, ctx, "fs/read_text_file", struct {
		SessionId SessionId `json:"sessionId"`
		Path      string    `json:"path"`
	}{SessionId: sid, Path: path})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

func fsWrite(c *Connection, ctx context.Context, sid SessionId, path, content string) error {
	_, err := SendRequest[struct{}](c, ctx, "fs/write_text_file", struct {
		SessionId SessionId `json:"sessionId"`
		Path      string    `json:"path"`
		Content   string    `json:"content"`
	}{SessionId: sid, Path: path, Content: content})
	return err
}
