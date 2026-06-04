package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
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

// readDedupEntry remembers a successful read_file outcome so a literal-repeat
// call can short-circuit. We compare the file's current mtime+size against
// what we saw last time; when they match, the model already has this content
// in its tool-call history and should not pay for another read.
type readDedupEntry struct {
	mtime time.Time
	size  int64
}

// readDedupKey canonicalises (path, line, limit) into a single map key so two
// equivalent calls collide. Empty line/limit normalise to 0 (whole file).
func readDedupKey(path string, line, limit int) string {
	return fmt.Sprintf("%s|%d|%d", path, line, limit)
}

// readUnchangedMarker is a stable phrase the dedup note carries when a read is a
// literal repeat of unchanged content. runToolLoop scans tool output for it to
// inject a corrective on the FIRST redundant fetch — catching the interleaved
// re-read pattern (read, search, read, read) that the consecutive-repeat nudge
// misses. Only present when the mtime+size guard confirmed no change, so a
// legitimate post-edit re-read (dedup busted, fresh content) never trips it.
const readUnchangedMarker = "already in your context and unchanged"

// listProjectFiles returns relative paths of all files under root, skipping
// common junk dirs. The skipDirs filter only applies to descendants — if the
// caller explicitly points us at e.g. `.codehalter`, they want its contents,
// not an empty result because the dir name matches the junk list.
func listProjectFiles(root string) []string {
	var files []string
	// The walk fn swallows per-entry errors (returns nil) by design — one
	// unreadable file shouldn't abort the listing. WalkDir itself only errors
	// if root can't be walked at all; log that rather than returning an
	// empty list with no trace.
	if err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if path != root && skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		files = append(files, rel)
		return nil
	}); err != nil {
		slog.Debug("listProjectFiles: walk failed", "root", root, "err", err)
	}
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
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
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
			"description": fmt.Sprintf("Read a text file. Output is truncated to %d lines or %d KB — a truncation note will tell you to re-call with line+limit to continue. If instead the output ends with an end-of-file marker, you have the WHOLE file; there is nothing more to read, so do not re-read or shell out to sed/cat for the same content. Do NOT re-read a file whose contents you already have in this turn's tool history AND which you have not modified since; scroll back instead. A literal-repeat read (same path, same line/limit, file unchanged) is rejected with a pointer to the prior result. Once you call edit_file or write_file on a path, re-reading IS allowed (and expected). Path accepts absolute (/workspaces/foo/bar.go) or project-relative (bar.go) — both are resolved.", defaultReadLines, maxReadBytes/1024),
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path"},
				"properties": map[string]any{
					"path":  map[string]any{"type": "string", "description": "Absolute path or path relative to the project root. A relative path that looks absolute-but-missing-leading-slash (e.g. `workspaces/foo`) will also be tried with `/` prepended."},
					"line":  map[string]any{"type": "integer", "description": "1-based start line. Omit to read from the beginning."},
					"limit": map[string]any{"type": "integer", "description": fmt.Sprintf("Max lines to read (hard cap %d). Omit for the default %d-line window.", maxReadLines, defaultReadLines)},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error(), false
		}

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

		// Dedup repeat reads. Same path + same window + file's mtime/size
		// unchanged since the last read in this turn → return the same content
		// again with a leading note so the model sees the bytes it asked for
		// (small MoE models at high temp don't reliably follow a "scroll back"
		// pointer). The disk re-read is cheap and the file is guaranteed
		// unchanged by the mtime+size guard. Reset at Prompt() boundary (see
		// Prompt() in prompt.go), and busted whenever edit_file / write_file
		// touches the path so a post-edit re-read still goes through.
		sess := a.getSession(sid)
		lineKey := 0
		if linePtr != nil {
			lineKey = *linePtr
		}
		limitKey := 0
		if explicitLimit {
			limitKey = limit
		}
		dedupKey := readDedupKey(path, lineKey, limitKey)
		var dedupNote string
		if sess != nil {
			sess.readDedupMu.Lock()
			if prev, ok := sess.readDedup[dedupKey]; ok {
				if st, statErr := os.Stat(path); statErr == nil && st.ModTime().Equal(prev.mtime) && st.Size() == prev.size {
					dedupNote = fmt.Sprintf("[note: %s (line=%d, limit=%d) is %s — you read it earlier this turn and it has NOT changed. The same bytes follow, but reading it again makes no progress: use the copy already in your tool history. To see a different part, pass a different `line`/`limit`.]", path, lineKey, limitKey, readUnchangedMarker)
				}
			}
			sess.readDedupMu.Unlock()
		}

		title := "Reading: " + path
		if args["line"] != "" {
			title = fmt.Sprintf("Reading: %s:%s", path, args["line"])
		}
		tcId := a.StartToolCall(ctx, sid, title, "read", []ToolCallLocation{{Path: path}})

		content, err := fsRead(a, ctx, sid, path, linePtr, &limit)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error(), false
		}

		// Record success in the dedup cache after a clean read. Use the
		// post-read stat so the recorded mtime/size match what the LLM just
		// saw — a write between read and stat would still bust the cache on
		// the next call because the new stat won't match this one.
		if sess != nil {
			if st, statErr := os.Stat(path); statErr == nil {
				sess.readDedupMu.Lock()
				if sess.readDedup == nil {
					sess.readDedup = make(map[string]readDedupEntry)
				}
				sess.readDedup[dedupKey] = readDedupEntry{mtime: st.ModTime(), size: st.Size()}
				sess.readDedupMu.Unlock()
			}
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
		// A read is "partial" when capReadContent trimmed it OR runToolCall's
		// truncateForLLM will byte-clip it for the model (read_file uses
		// readTruncateThreshold). Either way the model is NOT seeing the whole
		// file, so the UI says "partial" and we withhold the "complete, do not
		// re-read" marker — claiming complete while the body is clipped is the
		// contradiction that made the planner re-read whole files in a loop.
		// Without a positive complete marker on a small file the model can't tell
		// a complete short file from a truncated one and re-reads, so we DO emit
		// it whenever the whole file is actually visible.
		candidate := fmt.Sprintf("[end of file — line %d is the last line; you have the complete file from line %d, do not re-read]", end, start)
		partial := truncNote != "" || len(dedupNote)+len(content)+len(candidate)+2 > readTruncateThreshold

		eofNote := ""
		switch {
		case truncNote != "":
			// capReadContent already truncated; its note tells the model to paginate.
		case content == "":
			eofNote = "[file is empty]"
		case !partial:
			eofNote = candidate
		}

		resultTitle := fmt.Sprintf("Reading: %s (%d-%d)", path, start, end)
		if dedupNote != "" {
			resultTitle += " (re-read)"
		}
		if partial {
			resultTitle += " (partial)"
		} else {
			resultTitle += " (complete)"
		}

		if truncNote != "" {
			content += "\n" + truncNote
		}
		if eofNote != "" {
			content += "\n" + eofNote
		}
		if dedupNote != "" {
			content = dedupNote + "\n" + content
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
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
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
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
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
func fsRead(a *agent, ctx context.Context, sid string, path string, line, limit *int) (string, error) {
	if sess := a.getSession(sid); sess != nil && sess.Depth > 0 {
		return directRead(path, line, limit)
	}
	raw, err := a.conn.sendRequest(ctx, "fs/read_text_file", struct {
		SessionId string `json:"sessionId"`
		Path      string `json:"path"`
		Line      *int   `json:"line,omitempty"`
		Limit     *int   `json:"limit,omitempty"`
	}{sid, path, line, limit})
	if err != nil {
		return "", err
	}
	var resp struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", err
	}
	return resp.Content, nil
}

// fsWrite writes a text file. Same subagent fallback as fsRead — Zed has no
// record of a sub_* session id, so ACP writes are dead and we go straight
// to disk. Any cached read-dedup entries for this path are dropped here
// because the file just changed — a subsequent read_file must run.
func fsWrite(a *agent, ctx context.Context, sid string, path, content string) error {
	if sess := a.getSession(sid); sess != nil {
		sess.readDedupMu.Lock()
		for k := range sess.readDedup {
			if strings.HasPrefix(k, path+"|") {
				delete(sess.readDedup, k)
			}
		}
		sess.readDedupMu.Unlock()
		if sess.Depth > 0 {
			return os.WriteFile(path, []byte(content), 0644)
		}
	}
	_, err := a.conn.sendRequest(ctx, "fs/write_text_file", struct {
		SessionId string `json:"sessionId"`
		Path      string `json:"path"`
		Content   string `json:"content"`
	}{sid, path, content})
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
