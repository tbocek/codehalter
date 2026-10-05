package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// A NUL in the first binarySniffLen bytes marks a file binary (the git/grep
// heuristic); rendering binary bytes poisons the model's context.
const binarySniffLen = 8192

func looksBinary(b []byte) bool {
	if len(b) > binarySniffLen {
		b = b[:binarySniffLen]
	}
	return bytes.IndexByte(b, 0) >= 0
}

func trimBlankEdges(lines []string) []string {
	for len(lines) > 0 && strings.TrimSpace(lines[0]) == "" {
		lines = lines[1:]
	}
	for len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func leadingWS(s string) string { return s[:len(s)-len(strings.TrimLeft(s, " \t"))] }

func reindent(text, oldIndent, fileIndent string) string {
	if oldIndent == fileIndent {
		return text
	}
	lines := strings.Split(text, "\n")
	switch {
	case strings.HasPrefix(fileIndent, oldIndent):
		extra := fileIndent[len(oldIndent):]
		for i, ln := range lines {
			if strings.TrimSpace(ln) != "" {
				lines[i] = extra + ln
			}
		}
	case strings.HasPrefix(oldIndent, fileIndent):
		extra := oldIndent[len(fileIndent):]
		for i, ln := range lines {
			lines[i] = strings.TrimPrefix(ln, extra)
		}
	}
	return strings.Join(lines, "\n")
}

// tolerantReplace ignores trailing whitespace, then indentation; its content is
// valid only when the returned match count is 1.
func tolerantReplace(content, oldText, newText string) (string, int) {
	fileLines := strings.Split(content, "\n")
	oldLines := trimBlankEdges(strings.Split(oldText, "\n"))
	if len(oldLines) == 0 {
		return "", 0
	}
	for _, ignoreIndent := range []bool{false, true} {
		norm := func(s string) string {
			if ignoreIndent {
				return strings.TrimSpace(s)
			}
			return strings.TrimRight(s, " \t")
		}
		var hits []int
		for i := 0; i+len(oldLines) <= len(fileLines); i++ {
			match := true
			for j := range oldLines {
				if norm(fileLines[i+j]) != norm(oldLines[j]) {
					match = false
					break
				}
			}
			if match {
				hits = append(hits, i)
			}
		}
		if len(hits) > 1 {
			return "", len(hits) // ambiguous: do not loosen further
		}
		if len(hits) == 1 {
			start := hits[0]
			repl := newText
			if ignoreIndent {
				repl = reindent(newText, leadingWS(oldLines[0]), leadingWS(fileLines[start]))
			}
			out := append([]string(nil), fileLines[:start]...)
			out = append(out, strings.Split(repl, "\n")...)
			out = append(out, fileLines[start+len(oldLines):]...)
			return strings.Join(out, "\n"), 1
		}
	}
	return "", 0
}

const nearMissMinScore = 0.5

const nearMissMaxFileLines = 20_000

const nearMissTieEpsilon = 1e-9

const nearMissSnippetCap = 1500

// nearMiss scores lines by shared prefix and suffix, since a renamed identifier
// changes the whole line. A tie refuses: the wrong region is worse than none.
func nearMiss(content, oldText string) (startLine int, snippet string, ok bool) {
	fileLines := strings.Split(content, "\n")
	oldLines := trimBlankEdges(strings.Split(oldText, "\n"))
	if len(oldLines) == 0 || len(fileLines) > nearMissMaxFileLines || len(oldLines) > len(fileLines) {
		return 0, "", false
	}

	// -1, not 0, so zero-scoring windows do not tie with the initial state.
	bestStart, bestScore, bestCount := -1, -1.0, 0
	for i := 0; i+len(oldLines) <= len(fileLines); i++ {
		sum := 0.0
		for j := range oldLines {
			sum += lineSimilarity(oldLines[j], fileLines[i+j])
		}
		score := sum / float64(len(oldLines))
		switch {
		case score > bestScore+nearMissTieEpsilon:
			bestStart, bestScore, bestCount = i, score, 1
		case score > bestScore-nearMissTieEpsilon:
			bestCount++
		}
	}
	if bestStart < 0 || bestScore < nearMissMinScore || bestCount > 1 {
		return 0, "", false
	}
	return bestStart + 1, strings.Join(fileLines[bestStart:bestStart+len(oldLines)], "\n"), true
}

// lineSimilarity is byte-wise, not rune-wise: identifiers are ASCII and a split
// rune only costs a fraction of a point.
func lineSimilarity(a, b string) float64 {
	a, b = strings.TrimSpace(a), strings.TrimSpace(b)
	switch {
	case a == b:
		return 1
	case a == "" || b == "":
		return 0
	}
	shorter := min(len(a), len(b))
	p := 0
	for p < shorter && a[p] == b[p] {
		p++
	}
	// Cap the suffix scan so prefix and suffix cannot count the same bytes twice
	// ("abc" vs "abcabc" would score above 1).
	s := 0
	for s < shorter-p && a[len(a)-1-s] == b[len(b)-1-s] {
		s++
	}
	return float64(p+s) / float64(max(len(a), len(b)))
}

// readByteBudget leaves room under liveExemptCap for the notes after a read, so
// liveToolOutput never clips them off.
const (
	readChunkLines = 150
	maxReadLines   = 5000
	readByteBudget = liveExemptCap - 2*1024
)

const maxReadsPerCall = 8

func (a *agent) readTarget(ctx context.Context, sid string, args toolArgs) (string, bool) {
	path, err := a.resolvePath(sid, args.str("path"))
	if err != nil {
		return "error: " + err.Error(), false
	}
	if sym := strings.TrimSpace(args.str("symbol")); sym != "" {
		content, err := fsRead(a, ctx, sid, path, nil, nil)
		if err != nil {
			return "error reading file: " + err.Error(), false
		}
		loc := locateSymbol(content, sym)
		if loc.start == 0 {
			msg := fmt.Sprintf("error: no definition of `%s` found in %s.", sym, path)
			if len(loc.mentions) > 0 {
				msg += fmt.Sprintf(" The name appears at lines %s; read one of those with line=, or pass the exact declared name.", joinInts(loc.mentions))
			} else {
				msg += " The name does not appear in this file at all; find the right file with `grep -rn` first."
			}
			return msg, true
		}
		n := loc.end - loc.start + 1
		if n > maxReadLines {
			n = maxReadLines
		}
		tcId := a.StartToolCall(ctx, sid, fmt.Sprintf("Reading: %s (%s)", path, sym), "read", []ToolCallLocation{{Path: path, Line: &loc.start}})
		out, failed := a.serveRead(ctx, sid, path, loc.start, n, tcId, args.flag("numbered"))
		head := fmt.Sprintf("[`%s`: lines %d-%d, block end found by %s", sym, loc.start, loc.end, loc.how)
		if len(loc.others) > 0 {
			head += fmt.Sprintf("; also declared at lines %s", joinInts(loc.others))
		}
		return head + "]\n" + out, failed
	}
	start := 1
	line, haveLine := args.num("line")
	if haveLine && line > 0 {
		start = line
	}
	maxLines := readChunkLines
	if v, ok := args.num("limit"); ok && v > 0 {
		maxLines = v
	}
	// start_line/end_line (the `sed -n '130,205p'` shape) and view_range (Qwen's
	// own file tool) are both inclusive.
	from, to := 0, 0
	if n, ok := args.num("start_line"); ok {
		from = n
		if m, ok := args.num("end_line"); ok {
			to = m
		}
	}
	if vr, ok := args["view_range"].([]any); ok && len(vr) == 2 {
		ta := toolArgs{"a": vr[0], "b": vr[1]}
		from, _ = ta.num("a")
		to, _ = ta.num("b")
	}
	if from > 0 {
		start, line, haveLine = from, from, true
		if to >= from {
			maxLines = to - from + 1
		}
	}
	if maxLines > maxReadLines {
		maxLines = maxReadLines
	}
	title := "Reading: " + path
	if haveLine {
		title = fmt.Sprintf("Reading: %s:%d", path, line)
	}
	tcId := a.StartToolCall(ctx, sid, title, "read", []ToolCallLocation{{Path: path}})
	return a.serveRead(ctx, sid, path, start, maxLines, tcId, args.flag("numbered"))
}

func numberLines(content string, start int) string {
	lines := strings.SplitAfter(content, "\n")
	var b strings.Builder
	for i, ln := range lines {
		if ln == "" {
			continue
		}
		fmt.Fprintf(&b, "%d|%s", start+i, ln)
	}
	return b.String()
}

// serveRead's output stays under readByteBudget plus its notes, so it is exactly
// what the model sees. tcId is an already-started tool-call card.
func (a *agent) serveRead(ctx context.Context, sid, path string, start, maxLines int, tcId string, numbered bool) (string, bool) {
	sess := a.getSession(sid)

	// One line past the window tells whether the file continues.
	fetch := maxLines + 1
	startCopy := start
	content, err := fsRead(a, ctx, sid, path, &startCopy, &fetch)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error: " + err.Error(), false
	}
	if looksBinary([]byte(content)) {
		msg := fmt.Sprintf("%s is a binary file (NUL bytes), not shown. Reading it as text would corrupt the context. Use a shell tool to inspect its bytes if you must.", path)
		a.CompleteToolCallTitled(ctx, sid, tcId, "Read (binary, skipped): "+path, []ToolCallContent{TextContent(msg)})
		return msg, false
	}

	served := strings.Count(content, "\n")
	if content != "" && !strings.HasSuffix(content, "\n") {
		served++
	}
	more := served > maxLines
	if more {
		content = strings.Join(strings.SplitAfter(content, "\n")[:maxLines], "")
		served = maxLines
	}
	out := content
	if numbered {
		out = numberLines(content, start)
	}
	byteNote := ""
	if len(out) > readByteBudget {
		more = true
		if cut := strings.LastIndexByte(out[:readByteBudget], '\n'); cut >= 0 {
			out = out[:cut+1]
			served = strings.Count(out, "\n")
			byteNote = fmt.Sprintf("[stopped at %d KB, the most one read returns.]", readByteBudget/1024)
		} else {
			out = clipUTF8(out, readByteBudget)
			served = 1
			byteNote = fmt.Sprintf("[line %d is longer than %d KB and was cut there. Find the part you need with `grep -n` through run_command.]", start, readByteBudget/1024)
		}
	}
	end := start
	if served > 0 {
		end = start + served - 1
	}

	// A window from line 1 that ran out of file IS the whole file. fsRead checks
	// only unwindowed reads, and every read_file is windowed.
	if sess != nil && start == 1 && !more {
		sess.checkExternalChange(path, content)
	}

	var note string
	switch {
	case content == "":
		note = "[file is empty or past end of file]"
	case more:
		note = fmt.Sprintf("[showing lines %d-%d, the file continues. "+
			"MORE of it: read_file {\"path\": %q, \"start_line\": %d} returns the next %d lines. "+
			"LESS of it: `grep -n -C5 -F '<what you are looking for>' %s` through run_command returns only the lines around each hit. "+
			"Do NOT re-read the whole file.]", start, end, path, end+1, readChunkLines, path)
	default:
		note = fmt.Sprintf("[end of file: line %d is the last; you have the file through line %d, do not re-read]", end, end)
	}

	if byteNote != "" {
		out += "\n" + byteNote
	}
	out += "\n" + note
	if sess != nil {
		out += sess.takeDriftNote(path)
	}

	title := fmt.Sprintf("Reading: %s (%d-%d)", path, start, end)
	if more {
		title += " (partial)"
	} else {
		title += " (complete)"
	}
	a.CompleteToolCallTitled(ctx, sid, tcId, title, []ToolCallContent{TextContent(out)})
	return out, false
}

// hasProjectFiles applies skipWalkDir only below root, so a hidden root counts.
func hasProjectFiles(root string, n int) bool {
	files := 0
	// Per-entry errors are skipped, but a root error is propagated: a missing
	// root would otherwise read as an empty dir.
	if err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if path == root {
				return err
			}
			return nil
		}
		if d.IsDir() {
			if path != root && skipWalkDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		files++
		if files >= n {
			return filepath.SkipAll
		}
		return nil
	}); err != nil {
		slog.Debug("hasProjectFiles: walk failed", "root", root, "err", err)
	}
	return files >= n
}

var fileTools = []Tool{
	{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "read_file",
			"description": fmt.Sprintf("Read files. THREE WAYS, pick one per call:\n"+
				"(1) A line range, both ends inclusive, like `sed -n '120,179p'`: {\"path\": \"src/cut.rs\", \"start_line\": 120, \"end_line\": 179}.\n"+
				"(2) One definition by name, the whole function, type, class or test, in any language: {\"path\": \"src/ui/window.rs\", \"symbol\": \"cut_form_column\"}. Use this instead of `grep -n` followed by `sed -n`: one call, the whole block, its line range in the note.\n"+
				"Add \"numbered\": true to any of them for `N|text` lines with line numbers, as the awk printf NR idiom gives.\n"+
				"(3) SEVERAL reads at once, the way you put several commands in one shell line: {\"reads\": [{\"path\": \"src/ui/window.rs\", \"symbol\": \"wire_zoom\"}, {\"path\": \"src/fx_zoom.rs\", \"symbol\": \"zoom_at\"}, {\"path\": \"tests/zoom_widgets.rs\", \"start_line\": 1, \"end_line\": 40}]}. Each comes back under its own \"=== read N of M ===\" header. When you know you need two or three things, ask for them in ONE call like this, not one call each.\n"+
				"Details: up to %d lines per read. The text comes back PLAIN, exactly as in the file, with no line-number prefixes: a snippet can be copied straight into edit_file's old_text. The note under it states which lines were served (\"showing lines 120-165\"), so you know where you are without numbering anything yourself. Prefer this to `cat`, `sed -n` or `awk 'NR>=a && NR<=b'` through run_command for a region you already know: one call, no shell quoting, and edit_file needs the text, never the numbers. Use `grep -n` through run_command only to FIND a region, not to read one. If the file continues past that, the output is marked partial and its note names the read_file call for the next part, {\"path\": ..., \"start_line\": N}, so no line math. When the output ends with an end-of-file marker you have the file through that point, so do not re-read. After edit_file/write_file on a path, re-reading IS expected. Path accepts absolute (/workspaces/foo/bar.go) or project-relative (bar.go).", readChunkLines),
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"reads": map[string]any{
						"type":        "array",
						"description": fmt.Sprintf("SEVERAL reads in ONE call, instead of the fields below: a list, each item its own {path, symbol} or {path, line, limit}. Up to %d. Example: [{\"path\": \"src/ui/window.rs\", \"symbol\": \"wire_zoom\"}, {\"path\": \"tests/zoom_widgets.rs\", \"start_line\": 1, \"end_line\": 40}].", maxReadsPerCall),
						"items": map[string]any{
							"type":     "object",
							"required": []string{"path"},
							"properties": map[string]any{
								"path":       map[string]any{"type": "string"},
								"symbol":     map[string]any{"type": "string"},
								"start_line": map[string]any{"type": "integer"},
								"end_line":   map[string]any{"type": "integer"},
								"line":       map[string]any{"type": "integer"},
								"limit":      map[string]any{"type": "integer"},
								"numbered":   map[string]any{"type": "boolean"},
							},
						},
					},
					"path":       map[string]any{"type": "string", "description": "Absolute path or path relative to the project root. A relative path that looks absolute-but-missing-leading-slash (e.g. `workspaces/foo`) will also be tried with `/` prepended."},
					"start_line": map[string]any{"type": "integer", "description": "First line to read, 1-based, with end_line the last, both inclusive: `sed -n '130,205p'` is start_line 130, end_line 205."},
					"end_line":   map[string]any{"type": "integer", "description": "Last line to read, inclusive (see start_line)."},
					"line":       map[string]any{"type": "integer", "description": "1-based start line. Omit to read from the beginning."},
					"limit":      map[string]any{"type": "integer", "description": fmt.Sprintf("Max lines to read (hard cap %d). Omit for the default %d-line chunk; a partial read's note names the call for the next part.", maxReadLines, readChunkLines)},
					"numbered":   map[string]any{"type": "boolean", "description": "true: every line comes back as `N|text` with its line number, like `awk '{printf \"%d|%s\\n\", NR, $0}'`. Works with line windows, symbol and each item of reads. For edit_file's old_text copy the text without the `N|` (edit_file strips it if you do not)."},
					"symbol":     map[string]any{"type": "string", "description": "Read one definition instead of a line range: a function, method, type, class, trait or impl by name (`cut_form_column`, or `fn cut_form_column`). Any language: the block ends where its braces close, or where the indentation returns (Python), or after 50 lines when neither can be found (broken code). Comments and attributes directly above it come along. Replaces grep -n followed by a sed range: one call, the whole definition, its line range in the note."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		list, isList := args["reads"].([]any)
		if !isList {
			return a.readTarget(ctx, sid, args)
		}
		if len(list) == 0 {
			return "error: `reads` is empty. Give one or more reads, each {\"path\": ..., \"symbol\": ...} or {\"path\": ..., \"line\": N, \"limit\": M}.", true
		}
		var b strings.Builder
		failedAll := true
		for i, item := range list {
			if i == maxReadsPerCall {
				fmt.Fprintf(&b, "=== reads %d-%d not served: at most %d per call; ask for them in the next call ===\n", i+1, len(list), maxReadsPerCall)
				break
			}
			t, ok := item.(map[string]any)
			if !ok {
				fmt.Fprintf(&b, "=== read %d: not an object; each read is {\"path\": ..., \"symbol\" or \"line\"/\"limit\"} ===\n\n", i+1)
				continue
			}
			ta := toolArgs(t)
			what := ta.str("symbol")
			if what == "" {
				if n, ok := ta.num("start_line"); ok {
					what = fmt.Sprintf("from line %d", n)
				} else if l, ok := ta.num("line"); ok {
					what = fmt.Sprintf("from line %d", l)
				} else {
					what = "from the top"
				}
			}
			out, failed := a.readTarget(ctx, sid, ta)
			if !failed {
				failedAll = false
			}
			fmt.Fprintf(&b, "=== read %d of %d: %s %s ===\n%s\n\n", i+1, len(list), ta.str("path"), what, strings.TrimRight(out, "\n"))
		}
		return strings.TrimRight(b.String(), "\n") + "\n", failedAll
	}},

	{Def: map[string]any{
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
		if args.wrongType("content") {
			return "error: `content` must be a JSON string. You sent a non-string value, which would be coerced to an empty string and ERASE the file. Resend with the full file content as a quoted string.", false
		}
		path, err := a.resolvePath(sid, args.str("path"))
		if err != nil {
			return "error: " + err.Error(), false
		}
		if refusal := a.specFenceRefusal(sid, path); refusal != "" {
			return refusal, true
		}
		newContent := args.str("content")
		tcId := a.StartToolCall(ctx, sid, "Writing: "+path, "edit", []ToolCallLocation{{Path: path}})

		// The ACP read path cannot tell a missing file from a read fault, so any
		// error means a new file.
		oldContent, rerr := fsRead(a, ctx, sid, path, nil, nil)
		if rerr != nil {
			slog.Debug("write_file: pre-edit read returned an error; treating as new file", "path", path, "err", rerr)
		}
		// Before the write: fsWrite resets the path's drift state, and the model
		// must still learn its remembered copy went stale.
		drift := ""
		if sess := a.getSession(sid); sess != nil {
			drift = sess.takeDriftNote(path)
		}
		if msg, failed := a.commitWrite(ctx, sid, tcId, path, oldContent, newContent, drift); msg != "" {
			return msg, failed
		}
		return "file written successfully" + drift, false
	}},

	{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "edit_file",
			"description": "Change an EXISTING file. Always prefer this over write_file for a file that exists, and ALWAYS over a Python, sed or awk script through run_command: this edit is checked, shown to the user as a diff, and counted as an edit; a script is none of that, and a wrong anchor in it silently rewrites the wrong span. THREE WAYS, pick one per call:\n" +
				"(1) A small exact change: {\"path\": \"src/cut.rs\", \"old_text\": \"let zoom = 1.0;\", \"new_text\": \"let zoom = params::ZOOM;\"}. old_text must be unique in the file; copy it from a fresh read.\n" +
				"(2) A whole BLOCK (a function, a test, a match arm, a widget section): {\"path\": \"src/ui/window.rs\", \"start\": \"fn wire_zoom(\", \"end\": \"} // wire_zoom\", \"new_text\": \"fn wire_zoom(...) {\\n    ...\\n}\"}. `start` is a fragment of the block's FIRST line, unique in the file; `end` a fragment of its LAST line (the first line containing it at or after start). Every line from start through end is replaced by new_text. Use this instead of copying forty lines into old_text.\n" +
				"(3) SEVERAL changes to the same file at once: {\"path\": \"src/ui/window.rs\", \"edits\": [{\"old_text\": \"let zoom = 1.0;\", \"new_text\": \"let zoom = params::ZOOM;\"}, {\"start\": \"fn wire_zoom(\", \"end\": \"} // wire_zoom\", \"new_text\": \"...\"}]}. Applied in order, all or none; an error names the edit that failed. When you have two or three changes to one file, send them in ONE call like this.\n" +
				"Errors (not found / not unique / unwritable) come back as messages: fix and retry.",

			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path"},
				"properties": map[string]any{
					"edits": map[string]any{
						"type":        "array",
						"description": "SEVERAL changes to this ONE file in ONE call, instead of old_text/start/end/new_text below: a list applied in order, each {old_text, new_text} or {start, end, new_text}. All apply or none does; an error names the one that failed.",
						"items": map[string]any{
							"type":     "object",
							"required": []string{"new_text"},
							"properties": map[string]any{
								"old_text": map[string]any{"type": "string"},
								"start":    map[string]any{"type": "string"},
								"end":      map[string]any{"type": "string"},
								"new_text": map[string]any{"type": "string"},
							},
						},
					},
					"path":     map[string]any{"type": "string", "description": "Absolute path or path relative to the project root. A relative path that looks absolute-but-missing-leading-slash (e.g. `workspaces/foo`) will also be tried with `/` prepended."},
					"old_text": map[string]any{"type": "string", "description": "Way (1): exact text to find. MUST match the file byte-for-byte (whitespace, indentation, trailing newlines included) AND must be unique in the file — include enough surrounding context to disambiguate."},
					"start":    map[string]any{"type": "string", "description": "Way (2): a fragment of the block's FIRST line, unique in the file (for example `fn cut_form_column(`)."},
					"end":      map[string]any{"type": "string", "description": "Way (2): a fragment of the block's LAST line; the first line at or after `start` that contains it ends the block (for example the closing `}` line's text, or a comment on it). Omit to replace the start line alone."},
					"new_text": map[string]any{"type": "string", "description": "Replacement text: for (1) it replaces old_text, for (2) it replaces the lines from start through end, whole lines. Pass an empty string to delete."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		if args.wrongType("old_text") || args.wrongType("new_text") {
			return "error: `old_text` and `new_text` must be JSON strings; a non-string value is coerced to \"\" and would mis-edit the file. Resend them quoted (use \"\" only to intentionally delete old_text).", false
		}
		path, err := a.resolvePath(sid, args.str("path"))
		if err != nil {
			return "error: " + err.Error(), false
		}
		if refusal := a.specFenceRefusal(sid, path); refusal != "" {
			return refusal, true
		}
		type editSpec struct{ oldText, start, end, newText string }
		var edits []editSpec
		if list, ok := args["edits"].([]any); ok {
			for i, item := range list {
				m, ok := item.(map[string]any)
				if !ok {
					return fmt.Sprintf("error: edit %d of %d is not an object. Each edit is {\"old_text\": ..., \"new_text\": ...} or {\"start\": ..., \"end\": ..., \"new_text\": ...}. Nothing was written.", i+1, len(list)), true
				}
				e := toolArgs(m)
				if e.wrongType("old_text") || e.wrongType("new_text") {
					return fmt.Sprintf("error: edit %d of %d: `old_text` and `new_text` must be JSON strings. Nothing was written.", i+1, len(list)), false
				}
				edits = append(edits, editSpec{e.str("old_text"), e.str("start"), e.str("end"), e.str("new_text")})
			}
			if len(edits) == 0 {
				return "error: `edits` is empty; nothing to change.", true
			}
		} else {
			edits = []editSpec{{args.str("old_text"), args.str("start"), args.str("end"), args.str("new_text")}}
		}
		for i, e := range edits {
			if e.oldText == "" && e.start == "" {
				msg := "error: give either `old_text` (an exact snippet) or `start` (and `end`) for a block; with neither there is nothing to replace."
				if len(edits) > 1 {
					msg = fmt.Sprintf("error: edit %d of %d has neither `old_text` nor `start`. Nothing was written.", i+1, len(edits))
				}
				return msg, true
			}
		}

		tcId := a.StartToolCall(ctx, sid, "Editing: "+path, "edit", []ToolCallLocation{{Path: path}})

		content, err := fsRead(a, ctx, sid, path, nil, nil)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error reading file: " + err.Error(), false
		}
		// Taken before the write, which resets drift state. On a failed match it
		// explains why old_text no longer matches.
		drift := ""
		if sess := a.getSession(sid); sess != nil {
			drift = sess.takeDriftNote(path)
		}

		cur := content
		var notes []string
		for i, e := range edits {
			next, note, msg := applyEdit(path, cur, e.oldText, e.start, e.end, e.newText)
			if msg != "" {
				a.FailToolCall(ctx, sid, tcId, firstLine(msg))
				if len(edits) > 1 {
					msg = fmt.Sprintf("error: edit %d of %d failed, so NOTHING was written (the %d before it are not applied either): %s", i+1, len(edits), i, strings.TrimPrefix(msg, "error: "))
				}
				return msg + drift, true
			}
			cur = next
			notes = append(notes, note)
		}
		if msg, failed := a.commitWrite(ctx, sid, tcId, path, content, cur, drift); msg != "" {
			return msg, failed
		}
		if len(edits) == 1 {
			return "file written successfully" + notes[0] + drift, false
		}
		return fmt.Sprintf("file written successfully: all %d edits applied in order%s", len(edits), strings.Join(notes, "")) + drift, false
	}},
}

// commitWrite is the end write_file and edit_file share: format, the brief's
// guard, the write, the diff card. A non-empty msg is what the tool returns.
func (a *agent) commitWrite(ctx context.Context, sid, tcId, path, old, next, drift string) (msg string, failed bool) {
	next = a.formatGuarded(sid, path, old, next)
	if refusal := a.agentsFileRefusal(sid, path, old, next); refusal != "" {
		a.FailToolCall(ctx, sid, tcId, firstLine(refusal))
		return refusal + drift, true
	}
	if err := fsWrite(a, ctx, sid, path, next); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error writing file: " + err.Error(), false
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &old, next)})
	return "", false
}

// start includes leading comments and attributes; mentions is set only when no
// declaration was found.
type symbolLoc struct {
	start, end int
	how        string
	others     []int
	mentions   []int
}

const symbolFallbackLines = 50

// declKeywordRe matches a declaration keyword, then an optional Go receiver or
// Rust generics, right before the name.
var declKeywordRe = regexp.MustCompile(`(?:^|[\s(])(?:fn|func|def|class|struct|enum|trait|impl|interface|type|mod|module|macro_rules!|function|union|record|object|protocol|extension|let|const|var|val)\s+(?:\([^)]*\)\s*)?(?:<[^>]*>\s*)?`)

func locateSymbol(content, symbol string) symbolLoc {
	name := symbol
	if f := strings.Fields(strings.TrimRight(strings.TrimSpace(symbol), "(){}:")); len(f) > 0 {
		name = f[len(f)-1]
	}
	name = strings.TrimRight(name, "(){}:")
	lines := strings.Split(content, "\n")
	nameRe := regexp.MustCompile(`\b` + regexp.QuoteMeta(name) + `\b`)
	var loc symbolLoc
	var decls []int
	for i, ln := range lines {
		idx := nameRe.FindStringIndex(ln)
		if idx == nil {
			continue
		}
		before := ln[:idx[0]]
		if m := declKeywordRe.FindAllStringIndex(before+" ", -1); len(m) > 0 && m[len(m)-1][1] >= len(before) {
			decls = append(decls, i)
		} else if len(loc.mentions) < 8 {
			loc.mentions = append(loc.mentions, i+1)
		}
	}
	if len(decls) == 0 {
		return loc
	}
	d := decls[0]
	for _, o := range decls[1:] {
		if len(loc.others) < 5 {
			loc.others = append(loc.others, o+1)
		}
	}
	loc.mentions = nil
	top := d
	for top > 0 && d-top < 20 {
		t := strings.TrimSpace(lines[top-1])
		if strings.HasPrefix(t, "//") || strings.HasPrefix(t, "#[") || strings.HasPrefix(t, "#!") || strings.HasPrefix(t, "@") ||
			strings.HasPrefix(t, "/*") || strings.HasPrefix(t, "*") || strings.HasPrefix(t, "\"\"\"") || strings.HasPrefix(t, "--") {
			top--
			continue
		}
		break
	}
	loc.start = top + 1
	if end, ok := braceBlockEnd(lines, d); ok {
		loc.end, loc.how = end+1, "braces"
		return loc
	}
	if strings.HasSuffix(strings.TrimSpace(stripLineComment(lines[d])), ":") {
		loc.end, loc.how = indentBlockEnd(lines, d)+1, "indentation"
		return loc
	}
	loc.end = d + symbolFallbackLines
	if loc.end > len(lines) {
		loc.end = len(lines)
	}
	loc.how = fmt.Sprintf("the %d-line fallback (no block end found)", symbolFallbackLines)
	return loc
}

// braceBlockEnd: a ';' before any '{' ends the block (a prototype), and the '{'
// must open within the first few lines.
func braceBlockEnd(lines []string, d int) (int, bool) {
	depth, opened := 0, false
	for i := d; i < len(lines); i++ {
		ln := stripLineComment(lines[i])
		inStr := byte(0)
		for j := 0; j < len(ln); j++ {
			c := ln[j]
			if inStr != 0 {
				if c == '\\' {
					j++
				} else if c == inStr {
					inStr = 0
				}
				continue
			}
			switch c {
			case '"', '`':
				inStr = c
			case '{':
				depth++
				opened = true
			case '}':
				depth--
				if opened && depth == 0 {
					return i, true
				}
			case ';':
				if !opened {
					return i, true
				}
			}
		}
		if !opened && i-d >= 3 {
			return 0, false
		}
	}
	return 0, false
}

func indentBlockEnd(lines []string, d int) int {
	base := len(leadingWS(lines[d]))
	end := d
	for i := d + 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "" {
			continue
		}
		if len(leadingWS(lines[i])) <= base {
			break
		}
		end = i
	}
	return end
}

// stripLineComment is good enough for block scanning; it is not a parser.
func stripLineComment(ln string) string {
	inStr := byte(0)
	for j := 0; j < len(ln); j++ {
		c := ln[j]
		if inStr != 0 {
			if c == '\\' {
				j++
			} else if c == inStr {
				inStr = 0
			}
			continue
		}
		switch {
		case c == '"':
			inStr = c
		case c == '/' && j+1 < len(ln) && ln[j+1] == '/':
			return ln[:j]
		case c == '#' && (j == 0 || ln[j-1] == ' ' || ln[j-1] == '\t') && !(j+1 < len(ln) && ln[j+1] == '['):
			return ln[:j]
		}
	}
	return ln
}

// applyEdit returns (content, success note, error message); a non-empty message
// means nothing was replaced.
func applyEdit(path, content, oldText, start, end, newText string) (string, string, string) {
	if oldText == "" {
		next, span, msg := replaceBlock(content, start, end, newText)
		if msg != "" {
			return "", "", "error: " + msg
		}
		return next, fmt.Sprintf(" (lines %s replaced)", span), ""
	}
	switch count := strings.Count(content, oldText); {
	case count > 1:
		return "", "", fmt.Sprintf("error: old_text matches %d places; it must be unique. Add a few more exact lines of surrounding context (copied from a fresh read_file) so it pins exactly one spot; don't split the edit in a way that loses uniqueness.", count)
	case count == 1:
		return strings.Replace(content, oldText, newText, 1), "", ""
	}
	tol, n := tolerantReplace(content, oldText, newText)
	switch {
	case n == 1:
		return tol, " (old_text matched ignoring whitespace/indentation)", ""
	case n > 1:
		return "", "", fmt.Sprintf("error: old_text isn't a byte-for-byte match, and ignoring whitespace it matches %d places; add a couple more lines of surrounding context (from a fresh read_file) to pin exactly one spot.", n)
	}
	// A snippet copied from a numbered read still carries its `N|` prefixes.
	if stripped, ok := stripLineNumbers(oldText); ok {
		nt := newText
		if s2, ok := stripLineNumbers(newText); ok {
			nt = s2
		}
		if next, note, msg := applyEdit(path, content, stripped, "", "", nt); msg == "" {
			return next, note + " (old_text matched after removing its line-number prefixes)", ""
		}
	}
	if line, snippet, found := nearMiss(content, oldText); found {
		return "", "", fmt.Sprintf("error: old_text not found: the file has drifted from what you remember. The closest region is %s lines %d-%d, which CURRENTLY reads:\n\n%s\n\n"+
			"Retry edit_file with old_text copied byte-for-byte from that block (a SMALL unique part of it is enough). Do NOT call read_file first: the text above is the file's current content. Do NOT rewrite the whole file with write_file.",
			path, line, line+strings.Count(snippet, "\n"), truncate(snippet, nearMissSnippetCap))
	}
	return "", "", "error: old_text not found: the file differs from what you remember (reformatting, or an earlier edit), and no similar region was found either, so it may be the wrong file. Call read_file with line= at the region you're changing for its CURRENT exact text, then retry edit_file on a SMALL unique snippet. Do NOT re-read from the top, and do NOT rewrite the whole file with write_file."
}

var lineNumberPrefixRe = regexp.MustCompile(`^\s*\d+\|`)

// stripLineNumbers acts only when every non-empty line has `N|`, so real code
// starting with a number and a pipe is never touched.
func stripLineNumbers(s string) (string, bool) {
	lines := strings.Split(s, "\n")
	found := false
	for _, ln := range lines {
		if strings.TrimSpace(ln) == "" {
			continue
		}
		if !lineNumberPrefixRe.MatchString(ln) {
			return s, false
		}
		found = true
	}
	if !found {
		return s, false
	}
	for i, ln := range lines {
		lines[i] = lineNumberPrefixRe.ReplaceAllString(ln, "")
	}
	return strings.Join(lines, "\n"), true
}

// replaceBlock returns (content, replaced span, error message). end matches the
// first line at or after start; an empty end replaces start's line alone.
func replaceBlock(content, start, end, newText string) (string, string, string) {
	lines := strings.Split(content, "\n")
	var hits []int
	for i, ln := range lines {
		if strings.Contains(ln, start) {
			hits = append(hits, i)
		}
	}
	switch {
	case len(hits) == 0:
		return "", "", fmt.Sprintf("`start` %q is on no line of the file. Copy a fragment of the block's first line from a fresh read_file (read_file with `symbol` shows the whole block).", start)
	case len(hits) > 1:
		var at []int
		for _, h := range hits {
			at = append(at, h+1)
		}
		return "", "", fmt.Sprintf("`start` %q is on %d lines (%s); make it longer so it is on exactly one.", start, len(hits), joinInts(at))
	}
	s, e := hits[0], hits[0]
	if end != "" {
		e = -1
		for i := s; i < len(lines); i++ {
			if strings.Contains(lines[i], end) {
				e = i
				break
			}
		}
		if e < 0 {
			return "", "", fmt.Sprintf("`end` %q is on no line at or after line %d, where `start` is.", end, s+1)
		}
	}
	repl := strings.Split(strings.TrimSuffix(newText, "\n"), "\n")
	if newText == "" {
		repl = nil
	}
	out := append(append(append([]string{}, lines[:s]...), repl...), lines[e+1:]...)
	return strings.Join(out, "\n"), fmt.Sprintf("%d-%d", s+1, e+1), ""
}

func joinInts(ns []int) string {
	parts := make([]string, len(ns))
	for i, n := range ns {
		parts[i] = strconv.Itoa(n)
	}
	return strings.Join(parts, ", ")
}

// fsRead goes over ACP so the editor's unsaved buffers count; a client that did
// not advertise fs.readTextFile gets disk I/O, as ACP forbids unclaimed methods.
func fsRead(a *agent, ctx context.Context, sid string, path string, line, limit *int) (string, error) {
	sess := a.getSession(sid)
	content, err := func() (string, error) {
		if !a.clientCan("read") {
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
	}()
	// Only a whole read says whether the file still matches what we wrote.
	if err == nil && line == nil && limit == nil && sess != nil {
		sess.checkExternalChange(path, content)
	}
	return content, err
}

func fsWrite(a *agent, ctx context.Context, sid string, path, content string) error {
	var err error
	if !a.clientCan("write") {
		err = os.WriteFile(path, []byte(content), 0644)
	} else {
		_, err = a.conn.sendRequest(ctx, "fs/write_text_file", struct {
			SessionId string `json:"sessionId"`
			Path      string `json:"path"`
			Content   string `json:"content"`
		}{sid, path, content})
	}
	// Recorded so a later whole read can tell a misremembering model from an
	// external rewrite.
	if sess := a.getSession(sid); err == nil && sess != nil {
		sess.recordWrite(path, content)
	}
	return err
}

// directRead mirrors fs/read_text_file's 1-indexed line/limit window.
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

// batchNoteLead starts every batching note; the repetition ladder compares
// outputs without it.
const batchNoteLead = "\n[codehalter: that was your second "

// batchHint returns a note for the model and a chat line for the user when two
// single calls in a row could have been one. Never after a failed call.
func (s *Session) batchHint(name, args string, failed bool) (string, string) {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	prev := s.rt.prevCall
	s.rt.prevCall = toolCallBrief{name: name, args: args, failed: failed}
	first := s.rt.replyStart
	s.rt.replyStart = false
	// A second call in the SAME reply means the model batched, so only a reply's
	// first call is compared with the call before it.
	// Identical calls are a repeat, not a batch: leave them to the repetition ladder.
	if !first || failed || prev.failed || prev.name != name || prev.args == args {
		return "", ""
	}
	cur, before := parseArgs(args), parseArgs(prev.args)
	keep := func(a toolArgs, keys ...string) string {
		m := map[string]any{}
		for _, k := range keys {
			if v, ok := a[k]; ok && v != "" && v != nil {
				if str, ok := v.(string); ok {
					v = truncate(str, 80)
				}
				m[k] = v
			}
		}
		b, _ := json.Marshal(m)
		return string(b)
	}
	switch name {
	case "read_file":
		if cur.has("reads") || before.has("reads") {
			return "", ""
		}
		return batchNoteLead + "read_file in a row, and each is a model call. Independent reads go in ONE call, like several commands in one shell line; these two as one: {\"reads\": [" +
			keep(before, "path", "symbol", "start_line", "end_line", "line", "limit") + ", " + keep(cur, "path", "symbol", "start_line", "end_line", "line", "limit") + "]}. Up to 8 per call.]", "💡 told the model: several reads go in one read_file call"
	case "edit_file":
		if cur.has("edits") || before.has("edits") || cur.str("path") != before.str("path") || cur.str("path") == "" {
			return "", ""
		}
		return fmt.Sprintf(batchNoteLead+"edit to %s in a row, and each is a model call. Several changes to one file go in ONE edit_file call, applied in order, all or none; these two as one: {\"path\": %q, \"edits\": [%s, %s]}.]",
			cur.str("path"), cur.str("path"), keep(before, "old_text", "start", "end", "new_text"), keep(cur, "old_text", "start", "end", "new_text")), "💡 told the model: several edits to " + cur.str("path") + " go in one edit_file call"
	}
	return "", ""
}
