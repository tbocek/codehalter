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
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, args map[string]string) string {
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
			"description": "Read the contents of a file from the project. Use line and limit to read a specific range.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path"},
				"properties": map[string]any{
					"path":  map[string]any{"type": "string", "description": "Path to the file (absolute or relative to project root)"},
					"line":  map[string]any{"type": "integer", "description": "Start line (1-based). Omit to read from the beginning."},
					"limit": map[string]any{"type": "integer", "description": "Max number of lines to read. Omit to read the whole file."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, args map[string]string) string {
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error()
		}

		title := "Reading " + path
		if args["line"] != "" {
			title = fmt.Sprintf("Reading %s:%s", path, args["line"])
		}
		tcId := a.StartToolCall(ctx, sid, title, "read", []ToolCallLocation{{Path: path}})

		content, err := fsReadRange(a.conn.RPC(), ctx, sid, path, args["line"], args["limit"])
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(content)})
		return content
	}})

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "write_file",
			"description": "Write content to a file in the project. The user will be asked to approve the change.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path", "content"},
				"properties": map[string]any{
					"path":    map[string]any{"type": "string", "description": "Path to the file (absolute or relative to project root)"},
					"content": map[string]any{"type": "string", "description": "The new file content"},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, args map[string]string) string {
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error()
		}
		newContent := args["content"]
		tcId := a.StartToolCall(ctx, sid, "Editing "+path, "edit", []ToolCallLocation{{Path: path}})

		oldContent, _ := fsRead(a.conn.RPC(), ctx, sid, path)

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &oldContent, newContent)})

		return approveAndWrite(ctx, a, sid, tcId, path, newContent)
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
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, args map[string]string) string {
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
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &content, newContent)})

		return approveAndWrite(ctx, a, sid, tcId, path, newContent)
	}})
}

// approveAndWrite handles the permission prompt and file write, shared by write_file and edit_file.
func approveAndWrite(ctx context.Context, a *agent, sid SessionId, tcId, path, newContent string) string {
	a.mu.Lock()
	allowed := a.allowWrites
	a.mu.Unlock()

	if allowed == "" {
		choice, err := a.conn.AskWritePermission(ctx, sid, tcId)
		if err != nil {
			return "error asking user: " + err.Error()
		}
		switch choice {
		case "reject":
			return "user rejected the changes"
		case "allow_turn":
			a.mu.Lock()
			a.allowWrites = "turn"
			a.mu.Unlock()
		}
	}

	if err := fsWrite(a.conn.RPC(), ctx, sid, path, newContent); err != nil {
		return "error writing file: " + err.Error()
	}

	// Record a code ref for history tracking.
	a.mu.Lock()
	a.pendingRefs = append(a.pendingRefs, MakeFileRef(path))
	a.mu.Unlock()

	return "file written successfully"
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
