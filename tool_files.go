package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
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

// grepLines filters content to lines matching re, optionally with before/after
// context. Each emitted line is prefixed with its 1-based line number (within
// the original file, offset by startLine to account for line-windowed reads).
// Returns (output, matchCount, totalLines). Adjacent or overlapping context
// windows are merged with a "--" separator between disjoint groups, mirroring
// grep -A -B behaviour.
func grepLines(content string, re *regexp.Regexp, before, after, startLine int) (string, int, int) {
	if startLine < 1 {
		startLine = 1
	}
	if before < 0 {
		before = 0
	}
	if after < 0 {
		after = 0
	}
	lines := strings.Split(strings.TrimRight(content, "\n"), "\n")
	if content == "" {
		lines = nil
	}
	keep := make([]bool, len(lines))
	matches := 0
	for i, ln := range lines {
		if re.MatchString(ln) {
			matches++
			lo := i - before
			if lo < 0 {
				lo = 0
			}
			hi := i + after
			if hi >= len(lines) {
				hi = len(lines) - 1
			}
			for j := lo; j <= hi; j++ {
				keep[j] = true
			}
		}
	}
	var b strings.Builder
	lastEmitted := -2
	for i, ln := range lines {
		if !keep[i] {
			continue
		}
		if lastEmitted >= 0 && i > lastEmitted+1 {
			b.WriteString("--\n")
		}
		fmt.Fprintf(&b, "%d: %s\n", startLine+i, ln)
		lastEmitted = i
	}
	if matches == 0 {
		return fmt.Sprintf("[no matches in %d lines]", len(lines)), 0, len(lines)
	}
	return b.String(), matches, len(lines)
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
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "list_files",
			"description": "List project files. Returns relative paths, newline-separated. Skips .git/.codehalter/node_modules/vendor and similar junk dirs automatically.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{"type": "string", "description": "Subdirectory relative to project root. Empty = list from root."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		sess := a.getSession(sid)
		if sess == nil {
			return "error: no session", false
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

		tcId := a.StartToolCall(ctx, sid, "Listing: "+dir, "search", []ToolCallLocation{{Path: dir}})
		files := listProjectFiles(dir)
		a.CompleteToolCallTitled(ctx, sid, tcId,
			fmt.Sprintf("Listing: %s (%d files)", dir, len(files)),
			[]ToolCallContent{TextContent(fmt.Sprintf("%d files", len(files)))})
		return strings.Join(files, "\n"), false
	}})

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "read_file",
			"description": fmt.Sprintf("Read a text file. Output is truncated to %d lines or %d KB — a truncation note will tell you to re-call with line+limit to continue. Do NOT re-read a file whose contents you already have in this turn's tool history AND which you have not modified since; scroll back instead. Once you call edit_file or write_file on a path, re-reading IS allowed (and expected). Use the `grep` parameter to extract only matching lines with line numbers — replaces `run_command grep -n …`. Path accepts absolute (/workspaces/foo/bar.go) or project-relative (bar.go) — both are resolved.", defaultReadLines, maxReadBytes/1024),
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path"},
				"properties": map[string]any{
					"path":   map[string]any{"type": "string", "description": "Absolute path or path relative to the project root. A relative path that looks absolute-but-missing-leading-slash (e.g. `workspaces/foo`) will also be tried with `/` prepended."},
					"line":   map[string]any{"type": "integer", "description": "1-based start line. Omit to read from the beginning."},
					"limit":  map[string]any{"type": "integer", "description": fmt.Sprintf("Max lines to read (hard cap %d). Omit for the default %d-line window.", maxReadLines, defaultReadLines)},
					"grep":   map[string]any{"type": "string", "description": "Optional Go RE2 regex. When set, returns only matching lines (each prefixed with `lineno: `), not the full file. Use `(?i)` for case-insensitive. Combine with line/limit to grep within a window."},
					"before": map[string]any{"type": "integer", "description": "Context lines before each grep match (like `grep -B`). Ignored when `grep` is unset."},
					"after":  map[string]any{"type": "integer", "description": "Context lines after each grep match (like `grep -A`). Ignored when `grep` is unset."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error(), false
		}

		title := "Reading: " + path
		if args["grep"] != "" {
			title = fmt.Sprintf("Grepping: %s for %q", path, args["grep"])
		} else if args["line"] != "" {
			title = fmt.Sprintf("Reading: %s:%s", path, args["line"])
		}
		tcId := a.StartToolCall(ctx, sid, title, "read", []ToolCallLocation{{Path: path}})

		var linePtr *int
		if v, err := strconv.Atoi(args["line"]); err == nil && v > 0 {
			linePtr = &v
		}

		explicitLimit := args["limit"] != ""
		limit := defaultReadLines
		if v, err := strconv.Atoi(args["limit"]); err == nil && v > 0 {
			limit = v
			if limit > maxReadLines {
				limit = maxReadLines
			}
		}

		content, err := fsRead(a, ctx, sid, path, linePtr, &limit)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error(), false
		}

		// Grep mode: filter to matching lines with optional context, return
		// early without applying the full-file truncation rules.
		if pattern := args["grep"]; pattern != "" {
			re, gerr := regexp.Compile(pattern)
			if gerr != nil {
				a.FailToolCall(ctx, sid, tcId, "invalid regex: "+gerr.Error())
				return "error: invalid grep regex: " + gerr.Error(), false
			}
			before, _ := strconv.Atoi(args["before"])
			after, _ := strconv.Atoi(args["after"])
			startLine := 1
			if linePtr != nil {
				startLine = *linePtr
			}
			out, matches, total := grepLines(content, re, before, after, startLine)
			resultTitle := fmt.Sprintf("Grepping: %s for %q (%d/%d)", path, pattern, matches, total)
			a.CompleteToolCallTitled(ctx, sid, tcId, resultTitle, []ToolCallContent{TextContent(out)})
			return out, false
		}

		content, truncNote := capReadContent(content, explicitLimit, args["line"])

		// Line count of the actual returned slice (before any truncation note
		// is appended), so the title preview reflects what the LLM/user got.
		lineCount := 0
		if content != "" {
			lineCount = strings.Count(content, "\n")
			if !strings.HasSuffix(content, "\n") {
				lineCount++
			}
		}
		start := 1
		if v, err := strconv.Atoi(args["line"]); err == nil && v > 0 {
			start = v
		}
		end := start
		if lineCount > 0 {
			end = start + lineCount - 1
		}
		resultTitle := fmt.Sprintf("Reading: %s (%d-%d)", path, start, end)
		if truncNote != "" {
			resultTitle += " (truncated)"
		}

		if truncNote != "" {
			content += "\n" + truncNote
		}

		a.CompleteToolCallTitled(ctx, sid, tcId, resultTitle, []ToolCallContent{TextContent(content)})
		return content, false
	}})

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "write_file",
			"description": "Create a new file or completely overwrite an existing one. For targeted changes to an existing file use edit_file — it preserves context that write_file would clobber.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path", "content"},
				"properties": map[string]any{
					"path":    map[string]any{"type": "string", "description": "Absolute path or path relative to the project root. A relative path that looks absolute-but-missing-leading-slash (e.g. `workspaces/foo`) will also be tried with `/` prepended."},
					"content": map[string]any{"type": "string", "description": "Full file content. Will replace the file entirely."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error(), false
		}
		newContent := args["content"]
		tcId := a.StartToolCall(ctx, sid, "Writing: "+path, "edit", []ToolCallLocation{{Path: path}})

		oldContent, _ := fsRead(a, ctx, sid, path, nil, nil)

		if err := fsWrite(a, ctx, sid, path, newContent); err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error writing file: " + err.Error(), false
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &oldContent, newContent)})

		return "file written successfully", false
	}})

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "edit_file",
			"description": "Replace one exact text snippet in a file. Use read_file first to see the current contents. Returns an error when old_text is missing, matches more than once, or the file can't be written — fix and retry.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path", "old_text", "new_text"},
				"properties": map[string]any{
					"path":     map[string]any{"type": "string", "description": "Absolute path or path relative to the project root. A relative path that looks absolute-but-missing-leading-slash (e.g. `workspaces/foo`) will also be tried with `/` prepended."},
					"old_text": map[string]any{"type": "string", "description": "Exact text to find. MUST match the file byte-for-byte (whitespace, indentation, trailing newlines included) AND must be unique in the file — include enough surrounding context to disambiguate."},
					"new_text": map[string]any{"type": "string", "description": "Replacement text. Pass an empty string to delete old_text."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error(), false
		}
		oldText := args["old_text"]
		newText := args["new_text"]

		tcId := a.StartToolCall(ctx, sid, "Editing: "+path, "edit", []ToolCallLocation{{Path: path}})

		content, err := fsRead(a, ctx, sid, path, nil, nil)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error reading file: " + err.Error(), false
		}

		count := strings.Count(content, oldText)
		if count == 0 {
			a.FailToolCall(ctx, sid, tcId, "old_text not found in file")
			return "error: old_text not found in file", false
		}
		if count > 1 {
			a.FailToolCall(ctx, sid, tcId, fmt.Sprintf("old_text matches %d times, must be unique", count))
			return fmt.Sprintf("error: old_text matches %d times, must be unique", count), false
		}

		newContent := strings.Replace(content, oldText, newText, 1)

		if err := fsWrite(a, ctx, sid, path, newContent); err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error writing file: " + err.Error(), false
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &content, newContent)})

		return "file written successfully", false
	}})
}


// fsRead reads a text file. For top-level sessions known to the ACP client
// (Zed), the call goes over the wire so the editor can render diffs and
// honour unsaved buffer state. Subagent sessions were never announced to
// Zed (newSubagentSession just mints an id locally), so an ACP read would
// hit -32603 Internal error — we fall back to direct disk I/O for them.
// line/limit are optional: pass nil for both to read the whole file, or
// non-nil pointers to bound the response to a 1-indexed line window.
func fsRead(a *agent, ctx context.Context, sid SessionId, path string, line, limit *int) (string, error) {
	if sess := a.getSession(sid); sess != nil && sess.Depth > 0 {
		return directRead(path, line, limit)
	}
	resp, err := SendRequest[struct {
		Content string `json:"content"`
	}](a.conn.RPC(), ctx, "fs/read_text_file", struct {
		SessionId SessionId `json:"sessionId"`
		Path      string    `json:"path"`
		Line      *int      `json:"line,omitempty"`
		Limit     *int      `json:"limit,omitempty"`
	}{SessionId: sid, Path: path, Line: line, Limit: limit})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// fsWrite writes a text file. Same subagent fallback as fsRead — Zed has no
// record of a sub_* session id, so ACP writes are dead and we go straight
// to disk.
func fsWrite(a *agent, ctx context.Context, sid SessionId, path, content string) error {
	if sess := a.getSession(sid); sess != nil && sess.Depth > 0 {
		return os.WriteFile(path, []byte(content), 0644)
	}
	_, err := SendRequest[struct{}](a.conn.RPC(), ctx, "fs/write_text_file", struct {
		SessionId SessionId `json:"sessionId"`
		Path      string    `json:"path"`
		Content   string    `json:"content"`
	}{SessionId: sid, Path: path, Content: content})
	return err
}

// directRead is the subagent-path equivalent of an ACP fs/read_text_file:
// reads the file from disk and applies the 1-indexed line/limit window so
// the returned slice matches the shape the ACP path would have produced.
// SplitAfter keeps trailing newlines on each line so the join is lossless.
func directRead(path string, line, limit *int) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if line == nil && limit == nil {
		return string(data), nil
	}
	lines := strings.SplitAfter(string(data), "\n")
	start := 0
	if line != nil && *line > 0 {
		start = *line - 1
	}
	if start >= len(lines) {
		return "", nil
	}
	end := len(lines)
	if limit != nil && *limit > 0 && start+*limit < end {
		end = start + *limit
	}
	return strings.Join(lines[start:end], ""), nil
}
