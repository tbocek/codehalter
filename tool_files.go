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
)

var skipDirs = map[string]bool{
	".git": true, ".codehalter": true, "node_modules": true,
	"__pycache__": true, ".venv": true, "vendor": true,
	".idea": true, ".vscode": true, "target": true, "dist": true, "build": true,
}

// Read-size caps guard the LLM context. read_file / continue_read serve at most
// readChunkLines whole lines per call (a sequential window the model pages
// through via continue_read); an explicit `limit` is capped at maxReadLines, and
// maxReadBytes bounds a minified / long-line blob even under the line limit.
const (
	readChunkLines = 150        // default lines per read_file / continue_read chunk
	maxReadLines   = 5000       // hard cap when the caller passes an explicit limit
	maxReadBytes   = 200 * 1024 // byte safety for minified / very long lines
)

// serveRead is the shared body of read_file and continue_read: it reads up to
// maxLines whole lines of path from 1-based `start`, advances or clears the
// per-path continue_read cursor, and returns the model-visible output — the
// chunk plus a note that points to continue_read when the file continues, or
// marks EOF when it doesn't. read_file/continue_read are exempt from the
// downstream byte-clip (truncateForLLM), so this output is exactly what the
// model sees. tcId is the already-started tool-call card to complete/fail.
func (a *agent) serveRead(ctx context.Context, sid, path string, start, maxLines int, tcId string) (string, bool) {
	sess := a.getSession(sid)
	// Key format is contractual: fsWrite busts entries by `path+"|"` prefix.
	dedupKey := fmt.Sprintf("%s|%d|%d", path, start, maxLines)

	// Read one line past the window so we can tell whether the file continues.
	fetch := maxLines + 1
	startCopy := start
	content, err := fsRead(a, ctx, sid, path, &startCopy, &fetch)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error: " + err.Error(), false
	}

	// Served line count (a trailing partial line with no final newline counts),
	// then clip to maxLines (newlines preserved) when the file ran past the window.
	served := strings.Count(content, "\n")
	if content != "" && !strings.HasSuffix(content, "\n") {
		served++
	}
	more := served > maxLines
	if more {
		content = strings.Join(strings.SplitAfter(content, "\n")[:maxLines], "")
		served = maxLines
	}
	byteNote := ""
	if len(content) > maxReadBytes {
		content = content[:maxReadBytes]
		more = true
		byteNote = fmt.Sprintf("[truncated at %d bytes — long lines; use search_text or read_file line+limit to narrow] ", maxReadBytes)
	}
	end := start
	if served > 0 {
		end = start + served - 1
	}

	// Dedup on the ACTUAL served bytes, not a stat proxy: only flag a re-read as
	// redundant when the content is byte-identical to what this same window
	// served earlier this turn. A re-read that returns new bytes (an unsaved Zed
	// buffer, a coarse-mtime filesystem) is NOT redundant and must not trip the
	// loop's redundant-fetch guard. Still return the bytes (small models ignore
	// "scroll back"), but lead with a note steering to continue_read. fsWrite
	// clears a path's entries on write, so a post-edit re-read starts fresh.
	var dedupNote string
	if sess != nil {
		sum := fnvHash(content)
		sess.readDedupMu.Lock()
		if prev, ok := sess.readDedup[dedupKey]; ok && prev.hash == sum {
			dedupNote = fmt.Sprintf("[note: %s — you read %s from line %d earlier this turn and it has NOT changed. Re-reading the same window makes no progress; for MORE of the file call continue_read path=%q.]", readUnchangedMarker, path, start, path)
		}
		if sess.readDedup == nil {
			sess.readDedup = map[string]readDedupEntry{}
		}
		sess.readDedup[dedupKey] = readDedupEntry{hash: sum}
		sess.readDedupMu.Unlock()
	}

	// Advance the cursor while the file continues; clear it at EOF.
	if sess != nil {
		sess.readCursorMu.Lock()
		if sess.readCursor == nil {
			sess.readCursor = map[string]int{}
		}
		if more {
			sess.readCursor[path] = end + 1
		} else {
			delete(sess.readCursor, path)
		}
		sess.readCursorMu.Unlock()
	}

	var note string
	switch {
	case content == "":
		note = "[file is empty or past end of file]"
	case more:
		note = fmt.Sprintf("[showing lines %d-%d — the file continues. Call continue_read path=%q for the next ~%d lines (or read_file line=%d / search_text for a specific part). Do NOT re-read the whole file.]", start, end, path, readChunkLines, end+1)
	default:
		note = fmt.Sprintf("[end of file — line %d is the last; you have the file through line %d, do not re-read]", end, end)
	}

	out := content
	if byteNote != "" {
		out += "\n" + byteNote
	}
	out += "\n" + note
	if dedupNote != "" {
		out = dedupNote + "\n" + out
	}

	title := fmt.Sprintf("Reading: %s (%d-%d)", path, start, end)
	if dedupNote != "" {
		title += " (re-read)"
	}
	if more {
		title += " (partial)"
	} else {
		title += " (complete)"
	}
	a.CompleteToolCallTitled(ctx, sid, tcId, title, []ToolCallContent{TextContent(out)})
	return out, false
}

// readDedupEntry holds the fnv hash of the bytes a read_file/continue_read
// window last served. A later read of the same window whose content hashes
// identically is a true literal repeat — the model already has those exact
// bytes in its tool-call history.
type readDedupEntry struct {
	hash uint64
}

// readUnchangedMarker is a stable phrase the dedup note carries when a read is a
// literal repeat of unchanged content. runToolLoop scans tool output for it to
// inject a corrective on the FIRST redundant fetch — catching the interleaved
// re-read pattern (read, search, read, read) that the consecutive-repeat nudge
// misses. Only present when the served bytes hashed identically to a prior read
// of the same window, so a re-read that returns fresh content never trips it.
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
			"description": fmt.Sprintf("Read a text file from the top (or from `line`). Serves up to %d lines per call. If the file continues past that, the output is marked partial and ends with a pointer to call continue_read for the next chunk (it remembers where you left off, so no line math). When the output ends with an end-of-file marker you have the file through that point, so do not re-read. A literal-repeat read (same window, file unchanged) is rejected with a pointer to continue_read. After edit_file/write_file on a path, re-reading IS expected. Path accepts absolute (/workspaces/foo/bar.go) or project-relative (bar.go).", readChunkLines),
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path"},
				"properties": map[string]any{
					"path":  map[string]any{"type": "string", "description": "Absolute path or path relative to the project root. A relative path that looks absolute-but-missing-leading-slash (e.g. `workspaces/foo`) will also be tried with `/` prepended."},
					"line":  map[string]any{"type": "integer", "description": "1-based start line. Omit to read from the beginning."},
					"limit": map[string]any{"type": "integer", "description": fmt.Sprintf("Max lines to read (hard cap %d). Omit for the default %d-line chunk, then use continue_read for more.", maxReadLines, readChunkLines)},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error(), false
		}
		start := 1
		if v, e := strconv.Atoi(args["line"]); e == nil && v > 0 {
			start = v
		}
		maxLines := readChunkLines
		if v, e := strconv.Atoi(args["limit"]); e == nil && v > 0 {
			maxLines = v
			if maxLines > maxReadLines {
				maxLines = maxReadLines
			}
		}
		title := "Reading: " + path
		if args["line"] != "" {
			title = fmt.Sprintf("Reading: %s:%s", path, args["line"])
		}
		tcId := a.StartToolCall(ctx, sid, title, "read", []ToolCallLocation{{Path: path}})
		return a.serveRead(ctx, sid, path, start, maxLines, tcId)
	}})

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "continue_read",
			"description": "Read the NEXT chunk of a file you have already partially read. It picks up exactly where the last read_file/continue_read left off, so you never compute line numbers. Use this (not another read_file) whenever a read came back marked partial. Returns the next lines and stops at end of file.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path"},
				"properties": map[string]any{
					"path": map[string]any{"type": "string", "description": "The file to keep reading: the same path you read before."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		path, err := a.resolvePath(sid, args["path"])
		if err != nil {
			return "error: " + err.Error(), false
		}
		start := 1
		if sess := a.getSession(sid); sess != nil {
			sess.readCursorMu.Lock()
			if c, ok := sess.readCursor[path]; ok {
				start = c
			}
			sess.readCursorMu.Unlock()
		}
		tcId := a.StartToolCall(ctx, sid, fmt.Sprintf("Continuing: %s:%d", path, start), "read", []ToolCallLocation{{Path: path}})
		return a.serveRead(ctx, sid, path, start, readChunkLines, tcId)
	}})

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "write_file",
			"description": "Create a NEW file (or fully regenerate a small/generated one). Do NOT use write_file to change a file you've been reading — reproducing a large existing file from memory loses content; use edit_file for targeted changes.",
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
		newContent = a.formatGuarded(sid, path, oldContent, newContent)

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
			"description": "Replace one exact text snippet in an EXISTING file — always prefer this over write_file for changing a file that already exists. old_text must match the file byte-for-byte AND be unique; copy it from a fresh read_file and keep it small (a few lines, not a whole function). Change a large region as SEVERAL small edits. If old_text isn't found, the file differs from what you remember — read_file that region and retry on a small snippet; never rewrite the whole file. Errors (not found / not unique / unwritable) come back as messages — fix and retry.",
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
			// Failed=true feeds the loop's fail cap (a model spraying wrong edits
			// gives up instead of looping to the iteration backstop); the verdict
			// authority excludes edit_file, so a recovered miss never condemns.
			return "error: old_text not found — the file differs from what you remember (reformatting, or an earlier edit). Call read_file with line= at the region you're changing for its CURRENT exact text, then retry edit_file on a SMALL unique snippet. Do NOT re-read from the top, and do NOT rewrite the whole file with write_file.", true
		}
		if count > 1 {
			a.FailToolCall(ctx, sid, tcId, fmt.Sprintf("old_text matches %d times, must be unique", count))
			return fmt.Sprintf("error: old_text matches %d places — it must be unique. Add a few more exact lines of surrounding context (copied from a fresh read_file) so it pins exactly one spot; don't split the edit in a way that loses uniqueness.", count), true
		}

		newContent := strings.Replace(content, oldText, newText, 1)
		newContent = a.formatGuarded(sid, path, content, newContent)

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
		// The file changed — drop any continue_read cursor so the next read
		// starts fresh rather than continuing from a now-stale line.
		sess.readCursorMu.Lock()
		delete(sess.readCursor, path)
		sess.readCursorMu.Unlock()
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
