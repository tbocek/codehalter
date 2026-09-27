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

// binarySniffLen is how many leading bytes we sniff for a NUL to classify a file
// as binary (the git/grep heuristic). Binary files (zips, images) must never be
// rendered by read_file — their bytes poison the
// context (the model emits garbage and stalls).
const binarySniffLen = 8192

func looksBinary(b []byte) bool {
	if len(b) > binarySniffLen {
		b = b[:binarySniffLen]
	}
	return bytes.IndexByte(b, 0) >= 0
}

// trimBlankEdges drops leading/trailing all-whitespace lines (a trailing
// newline's empty element, or a stray blank line) so they don't skew matching.
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

// reindent shifts text by the indent delta between the snippet's indent and the
// file's, so a block matched ignoring indentation lands at the file's column.
// Handles the common "off by a consistent prefix" case; otherwise leaves it.
func reindent(text, oldIndent, fileIndent string) string {
	if oldIndent == fileIndent {
		return text
	}
	lines := strings.Split(text, "\n")
	switch {
	case strings.HasPrefix(fileIndent, oldIndent): // file deeper — add the extra
		extra := fileIndent[len(oldIndent):]
		for i, ln := range lines {
			if strings.TrimSpace(ln) != "" {
				lines[i] = extra + ln
			}
		}
	case strings.HasPrefix(oldIndent, fileIndent): // file shallower — strip it
		extra := oldIndent[len(fileIndent):]
		for i, ln := range lines {
			lines[i] = strings.TrimPrefix(ln, extra)
		}
	}
	return strings.Join(lines, "\n")
}

// tolerantReplace recovers an edit whose old_text matches except for per-line
// whitespace (the dominant edit_file failure for small models): whole-line match
// ignoring trailing whitespace, then leading indentation (re-indenting new_text).
// Returns the rewritten content and how many windows matched — the caller applies
// it only when exactly one did. A fallback AFTER an exact match misses.
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
			return "", len(hits) // ambiguous — report; don't loosen further
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

// nearMissMinScore is the fraction of old_text's lines that must match a file
// window before we're willing to call it "the region you meant". Below this the
// candidate is more likely to mislead than help, and the model is better served
// by the plain "go read it" message.
const nearMissMinScore = 0.5

// nearMissMaxFileLines skips the scan on very large files. The search is
// O(fileLines × oldLines) string compares; on a 20k-line file with a 10-line
// snippet that is still only ~200k trivial comparisons, but past that the cost
// stops being free and the payoff (a model editing a file that big from memory)
// is dubious anyway.
const nearMissMaxFileLines = 20_000

// nearMissTieEpsilon is the margin within which two candidate windows count as
// equally good. Float scores rarely land exactly equal, so a bare `==` would
// miss the ambiguity this guards against.
const nearMissTieEpsilon = 1e-9

// nearMissSnippetCap bounds the bytes of file text quoted back in a failed-edit
// message. Enough for the handful of lines a well-formed old_text should be,
// and a hard stop on a model that passed half the file as old_text.
const nearMissSnippetCap = 1500

// nearMiss finds the region old_text most likely MEANT to match, when both the
// exact and the whitespace-tolerant match failed. For a small model this is
// the recovery that matters: the failure is rarely an invented snippet, it is a
// region reproduced from a read four calls ago in which something drifted.
// Returning the region's current bytes lets it retry without a read_file.
//
// Lines are compared positionally, each by shared prefix and suffix rather
// than equality: a renamed identifier changes its whole line, so equality would
// score a two-line snippet with one drifted line at the floor. Returns the
// 1-based start line and the real text, or ok=false when nothing clears
// nearMissMinScore or the best score is a tie (unique-or-refuse, like
// tolerantReplace: the wrong region is worse than none).
func nearMiss(content, oldText string) (startLine int, snippet string, ok bool) {
	fileLines := strings.Split(content, "\n")
	oldLines := trimBlankEdges(strings.Split(oldText, "\n"))
	if len(oldLines) == 0 || len(fileLines) > nearMissMaxFileLines || len(oldLines) > len(fileLines) {
		return 0, "", false
	}

	// bestScore starts below every attainable score (which are all ≥ 0) so the
	// tie test below can't match the initial state — otherwise zero-scoring
	// windows would count as ties with it.
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
			// Indistinguishable from the current best: remember that it happened.
			bestCount++
		}
	}
	// A perfect score is unreachable by construction — an all-lines match would
	// have been caught by tolerantReplace before we were called — so a high score
	// here genuinely means "this region, something drifted".
	if bestStart < 0 || bestScore < nearMissMinScore || bestCount > 1 {
		return 0, "", false
	}
	return bestStart + 1, strings.Join(fileLines[bestStart:bestStart+len(oldLines)], "\n"), true
}

// lineSimilarity scores two lines in [0,1] by how much of the longer one is
// covered by a shared prefix plus a shared suffix, ignoring indentation. Cheap
// (two byte scans, no allocation) and well-shaped for source code, where a
// drifted line is almost always "same line with something swapped in the
// middle". Byte-wise rather than rune-wise: a split multi-byte rune only ever
// costs a fraction of a point, and identifiers in code are ASCII.
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
	// Cap the suffix scan so prefix and suffix can't count the same bytes twice
	// (e.g. "abc" vs "abcabc" would otherwise score above 1).
	s := 0
	for s < shorter-p && a[len(a)-1-s] == b[len(b)-1-s] {
		s++
	}
	return float64(p+s) / float64(max(len(a), len(b)))
}

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
// maxReadsPerCall bounds a read_file `reads` list.
const maxReadsPerCall = 8

// readTarget serves one read: a definition by symbol, or a line window.
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
		out, failed := a.serveRead(ctx, sid, path, loc.start, n, tcId)
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
		if maxLines > maxReadLines {
			maxLines = maxReadLines
		}
	}
	title := "Reading: " + path
	if haveLine {
		title = fmt.Sprintf("Reading: %s:%d", path, line)
	}
	tcId := a.StartToolCall(ctx, sid, title, "read", []ToolCallLocation{{Path: path}})
	return a.serveRead(ctx, sid, path, start, maxLines, tcId)
}

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
	if looksBinary([]byte(content)) {
		msg := fmt.Sprintf("%s is a binary file (NUL bytes) — not shown. Reading it as text would corrupt the context. Use a shell tool to inspect its bytes if you must.", path)
		a.CompleteToolCallTitled(ctx, sid, tcId, "Read (binary, skipped): "+path, []ToolCallContent{TextContent(msg)})
		return msg, false
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
		byteNote = fmt.Sprintf("[truncated at %d bytes — long lines; use read_file line+limit, or grep -n through run_command, to narrow] ", maxReadBytes)
	}
	end := start
	if served > 0 {
		end = start + served - 1
	}

	// A window that begins at line 1 and ran out of file IS the whole file, so it
	// can be compared against what we last wrote to that path (fsRead only checks
	// unwindowed reads, and every read_file is windowed). This is the earliest
	// point the model can be told a file was rewritten behind it.
	if sess != nil && start == 1 && !more && byteNote == "" {
		sess.checkExternalChange(path, content)
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
		if sess.repeatedResult(dedupKey, fnvHash(content)) {
			dedupNote = fmt.Sprintf("[note: %s — you read %s from line %d earlier this turn and it has NOT changed. Re-reading the same window makes no progress; for MORE of the file call continue_read path=%q.]", readUnchangedMarker, path, start, path)
		}
	}

	// Advance the cursor while the file continues; clear it at EOF.
	if sess != nil {
		sess.turnMu.Lock()
		if sess.turn.readCursor == nil {
			sess.turn.readCursor = map[string]int{}
		}
		if more {
			sess.turn.readCursor[path] = end + 1
		} else {
			delete(sess.turn.readCursor, path)
		}
		sess.turnMu.Unlock()
	}

	var note string
	switch {
	case content == "":
		note = "[file is empty or past end of file]"
	case more:
		note = fmt.Sprintf("[showing lines %d-%d, the file continues. "+
			"MORE of it: continue_read path=%q returns the next ~%d lines (or read_file line=%d to jump). "+
			"LESS of it: `grep -n -C5 -F '<what you are looking for>' %s` through run_command returns only the lines around each hit. "+
			"Do NOT re-read the whole file.]", start, end, path, readChunkLines, end+1, path)
	default:
		note = fmt.Sprintf("[end of file — line %d is the last; you have the file through line %d, do not re-read]", end, end)
	}

	// Already in the live context verbatim: refuse instead of re-serving. The
	// model can scroll back to the copy it already has, and re-reading would just
	// duplicate the whole chunk in the prompt — a small model that loops on
	// read_file otherwise keeps inflating n_ctx with identical bytes. Scoped to
	// content that fits whole in context (≤ liveExemptCap, so it wasn't
	// byte-clipped) and is genuinely still present — verified against the live
	// messages, not a per-turn hash, so a compacted-away read IS re-served.
	// readUnchangedMarker keeps runToolLoop's repetition ladder counting it.
	// Exception: if edit_file just failed for this path, the model needs a fresh
	// look to get the exact old_text for a retry — bypass the guard once.
	editFailed := sess != nil && sess.clearEditFailed(path)
	if !editFailed && sess != nil && len(content) > 0 && len(content) <= liveExemptCap && sess.readContentInContext(content) {
		ptr := " You already have these lines above — scroll back to that output instead of re-reading."
		if more {
			ptr = fmt.Sprintf(" You already have lines %d-%d above; for the rest of the file call continue_read path=%q (or read_file line=%d, or grep -n -C for a specific part).", start, end, path, end+1)
		}
		refusal := fmt.Sprintf("This file is already in the context — %s. You read %s lines %d-%d earlier this turn and it has not changed; re-read refused.%s",
			readUnchangedMarker, path, start, end, ptr)
		a.CompleteToolCallTitled(ctx, sid, tcId, fmt.Sprintf("Read (already in context): %s (%d-%d)", path, start, end), []ToolCallContent{TextContent(refusal)})
		return refusal, false
	}

	out := content
	if byteNote != "" {
		out += "\n" + byteNote
	}
	out += "\n" + note
	if dedupNote != "" {
		out = dedupNote + "\n" + out
	}
	if sess != nil {
		out += sess.takeDriftNote(path)
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
	// The walk fn swallows per-entry errors (one unreadable file shouldn't abort
	// the listing) but propagates the error on root itself: a missing or
	// unreadable root would otherwise return an empty slice indistinguishable
	// from a real empty dir, with no trace.
	if err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if path == root {
				return err
			}
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

var fileTools = []Tool{
	{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "read_file",
			"description": fmt.Sprintf("Read files. THREE WAYS, pick one per call:\n"+
				"(1) A line window: {\"path\": \"src/cut.rs\", \"line\": 120, \"limit\": 60}.\n"+
				"(2) One definition by name, the whole function, type, class or test, in any language: {\"path\": \"src/ui/window.rs\", \"symbol\": \"cut_form_column\"}. Use this instead of `grep -n` followed by `sed -n`: one call, the whole block, its line range in the note.\n"+
				"(3) SEVERAL reads at once, the way you put several commands in one shell line: {\"reads\": [{\"path\": \"src/ui/window.rs\", \"symbol\": \"wire_zoom\"}, {\"path\": \"src/fx_zoom.rs\", \"symbol\": \"zoom_at\"}, {\"path\": \"tests/zoom_widgets.rs\", \"line\": 1, \"limit\": 40}]}. Each comes back under its own \"=== read N of M ===\" header. When you know you need two or three things, ask for them in ONE call like this, not one call each.\n"+
				"Details: up to %d lines per read. The text comes back PLAIN, exactly as in the file, with no line-number prefixes: a snippet can be copied straight into edit_file's old_text. The note under it states which lines were served (\"showing lines 120-165\"), so you know where you are without numbering anything yourself. Prefer this to `cat`, `sed -n` or `awk 'NR>=a && NR<=b'` through run_command for a region you already know: one call, no shell quoting, and edit_file needs the text, never the numbers. Use `grep -n` through run_command only to FIND a region, not to read one. If the file continues past that, the output is marked partial and ends with a pointer to call continue_read for the next chunk (it remembers where you left off, so no line math). When the output ends with an end-of-file marker you have the file through that point, so do not re-read. A repeat read whose exact content is still in this conversation is refused (scroll back to it, or call continue_read for the next part); once it has scrolled out of context it is re-served. After edit_file/write_file on a path, re-reading IS expected. Path accepts absolute (/workspaces/foo/bar.go) or project-relative (bar.go).", readChunkLines),
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"reads": map[string]any{
						"type":        "array",
						"description": fmt.Sprintf("SEVERAL reads in ONE call, instead of the fields below: a list, each item its own {path, symbol} or {path, line, limit}. Up to %d. Example: [{\"path\": \"src/ui/window.rs\", \"symbol\": \"wire_zoom\"}, {\"path\": \"tests/zoom_widgets.rs\", \"line\": 1, \"limit\": 40}].", maxReadsPerCall),
						"items": map[string]any{
							"type":     "object",
							"required": []string{"path"},
							"properties": map[string]any{
								"path":   map[string]any{"type": "string"},
								"symbol": map[string]any{"type": "string"},
								"line":   map[string]any{"type": "integer"},
								"limit":  map[string]any{"type": "integer"},
							},
						},
					},
					"path":   map[string]any{"type": "string", "description": "Absolute path or path relative to the project root. A relative path that looks absolute-but-missing-leading-slash (e.g. `workspaces/foo`) will also be tried with `/` prepended."},
					"line":   map[string]any{"type": "integer", "description": "1-based start line. Omit to read from the beginning."},
					"limit":  map[string]any{"type": "integer", "description": fmt.Sprintf("Max lines to read (hard cap %d). Omit for the default %d-line chunk, then use continue_read for more.", maxReadLines, readChunkLines)},
					"symbol": map[string]any{"type": "string", "description": "Read one definition instead of a line range: a function, method, type, class, trait or impl by name (`cut_form_column`, or `fn cut_form_column`). Any language: the block ends where its braces close, or where the indentation returns (Python), or after 50 lines when neither can be found (broken code). Comments and attributes directly above it come along. Replaces grep -n followed by a sed range: one call, the whole definition, its line range in the note."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		list, isList := args["reads"].([]any)
		if !isList {
			return a.readTarget(ctx, sid, args)
		}
		// Several reads in one call, served in order, each under its own
		// header. The model packs several commands into one shell line all
		// the time and almost never sends two tool calls in one reply; this
		// is that habit for reads.
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
				if l, ok := ta.num("line"); ok {
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
		path, err := a.resolvePath(sid, args.str("path"))
		if err != nil {
			return "error: " + err.Error(), false
		}
		start := 1
		if sess := a.getSession(sid); sess != nil {
			sess.turnMu.Lock()
			if c, ok := sess.turn.readCursor[path]; ok {
				start = c
			}
			sess.turnMu.Unlock()
		}
		tcId := a.StartToolCall(ctx, sid, fmt.Sprintf("Continuing: %s:%d", path, start), "read", []ToolCallLocation{{Path: path}})
		return a.serveRead(ctx, sid, path, start, readChunkLines, tcId)
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

		// Pre-edit read for the diff card + formatGuarded's dry run. A missing file
		// (the common new-file case) and a read fault both surface as an error here,
		// and the ACP read path can't reliably tell them apart, so proceed as a new
		// file (oldContent ""). Log it rather than dropping it to `_`, so a genuine
		// read fault on an existing file still leaves a trail.
		oldContent, rerr := fsRead(a, ctx, sid, path, nil, nil)
		if rerr != nil {
			slog.Debug("write_file: pre-edit read returned an error; treating as new file", "path", path, "err", rerr)
		}
		// Taken BEFORE the write: our own fsWrite resets the path's drift state,
		// and the model still needs to know its remembered copy went stale — the
		// content it just composed may have been written against it.
		drift := ""
		if sess := a.getSession(sid); sess != nil {
			drift = sess.takeDriftNote(path)
		}
		newContent = a.formatGuarded(sid, path, oldContent, newContent)

		if err := fsWrite(a, ctx, sid, path, newContent); err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error writing file: " + err.Error(), false
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &oldContent, newContent)})

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
		// Rides along on every outcome below. On a failed match it is the ANSWER:
		// old_text was copied from a read that something else has since rewritten,
		// and without this the model can only guess (one session spent minutes
		// diffing against git to work out what had happened). Taken before the
		// write, which resets the path's drift state.
		drift := ""
		if sess := a.getSession(sid); sess != nil {
			drift = sess.takeDriftNote(path)
		}

		// All or nothing: each edit applies to the result of the one before,
		// in memory; the file is written once, as one diff, or not at all.
		cur := content
		var notes []string
		for i, e := range edits {
			next, note, msg, notFound := applyEdit(path, cur, e.oldText, e.start, e.end, e.newText)
			if msg != "" {
				a.FailToolCall(ctx, sid, tcId, firstLine(msg))
				if notFound {
					if sess := a.getSession(sid); sess != nil {
						sess.markEditFailed(path)
					}
				}
				if len(edits) > 1 {
					msg = fmt.Sprintf("error: edit %d of %d failed, so NOTHING was written (the %d before it are not applied either): %s", i+1, len(edits), i, strings.TrimPrefix(msg, "error: "))
				}
				return msg + drift, true
			}
			cur = next
			notes = append(notes, note)
		}
		newContent := a.formatGuarded(sid, path, content, cur)

		if err := fsWrite(a, ctx, sid, path, newContent); err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error writing file: " + err.Error(), false
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &content, newContent)})

		if len(edits) == 1 {
			return "file written successfully" + notes[0] + drift, false
		}
		return fmt.Sprintf("file written successfully: all %d edits applied in order%s", len(edits), strings.Join(notes, "")) + drift, false
	}},
}

// symbolLoc is where locateSymbol found a definition: its first line
// (leading comments and attributes included) and last line, how the end was
// found, other declarations of the same name, and, when there is none, the
// lines that merely mention it.
type symbolLoc struct {
	start, end int
	how        string
	others     []int
	mentions   []int
}

// symbolFallbackLines is how much a definition read serves when its block
// end cannot be found (broken code, an unusual syntax).
const symbolFallbackLines = 50

// declKeywordRe: the words that introduce a definition across the common
// languages, with the modifiers that may precede them. The name follows,
// possibly after a Go method receiver or Rust generics.
var declKeywordRe = regexp.MustCompile(`(?:^|[\s(])(?:fn|func|def|class|struct|enum|trait|impl|interface|type|mod|module|macro_rules!|function|union|record|object|protocol|extension|let|const|var|val)\s+(?:\([^)]*\)\s*)?(?:<[^>]*>\s*)?`)

// locateSymbol finds the definition of symbol in content, for any language:
// a declaration keyword before the name, then the block's end by braces,
// by indentation for a line ending in ':', or a fixed fallback.
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
	// Comments and attributes directly above belong to the definition.
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

// braceBlockEnd scans from the declaration line for the brace block that
// opens within its first few lines and returns the line where it closes. A
// declaration that ends in ';' before any brace (a prototype, a trait
// method) is its own block. Strings and line comments are skipped so a
// brace inside them does not count; a block that never closes is not found.
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

// indentBlockEnd: the last line of an indentation block started by line d,
// that is, the last non-blank line indented deeper than d.
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

// stripLineComment drops a `//` or `#` comment tail outside strings, well
// enough for block scanning; it is not a parser.
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

// applyEdit applies one edit to content: an exact old_text, the same ignoring
// whitespace, or a block between start and end anchors. It returns the new
// content and a note for the success message, or the message saying why
// nothing was replaced (notFound when old_text matched nowhere, which the
// caller records against the path).
func applyEdit(path, content, oldText, start, end, newText string) (string, string, string, bool) {
	if oldText == "" {
		next, span, msg := replaceBlock(content, start, end, newText)
		if msg != "" {
			return "", "", "error: " + msg, false
		}
		return next, fmt.Sprintf(" (lines %s replaced)", span), "", false
	}
	switch count := strings.Count(content, oldText); {
	case count > 1:
		return "", "", fmt.Sprintf("error: old_text matches %d places — it must be unique. Add a few more exact lines of surrounding context (copied from a fresh read_file) so it pins exactly one spot; don't split the edit in a way that loses uniqueness.", count), false
	case count == 1:
		return strings.Replace(content, oldText, newText, 1), "", "", false
	}
	// Exact match failed. Small models routinely mis-reproduce indentation
	// or trailing whitespace from a read_file, so retry ignoring per-line
	// whitespace (still unique-or-fail) before sending them back to re-read.
	tol, n := tolerantReplace(content, oldText, newText)
	switch {
	case n == 1:
		return tol, " (old_text matched ignoring whitespace/indentation)", "", false
	case n > 1:
		return "", "", fmt.Sprintf("error: old_text isn't a byte-for-byte match, and ignoring whitespace it matches %d places — add a couple more lines of surrounding context (from a fresh read_file) to pin exactly one spot.", n), false
	}
	// Quote the region old_text was probably aiming at, when there is one:
	// the model retries against text it can see instead of spending a read.
	if line, snippet, found := nearMiss(content, oldText); found {
		return "", "", fmt.Sprintf("error: old_text not found — the file has drifted from what you remember. The closest region is %s lines %d-%d, which CURRENTLY reads:\n\n%s\n\n"+
			"Retry edit_file with old_text copied byte-for-byte from that block (a SMALL unique part of it is enough). Do NOT call read_file first — the text above is the file's current content. Do NOT rewrite the whole file with write_file.",
			path, line, line+strings.Count(snippet, "\n"), truncate(snippet, nearMissSnippetCap)), true
	}
	return "", "", "error: old_text not found — the file differs from what you remember (reformatting, or an earlier edit), and no similar region was found either, so it may be the wrong file. Call read_file with line= at the region you're changing for its CURRENT exact text, then retry edit_file on a SMALL unique snippet. Do NOT re-read from the top, and do NOT rewrite the whole file with write_file.", true
}

// replaceBlock replaces whole lines from the unique line containing start
// through the first line at or after it containing end (start's line alone
// when end is empty) with newText. It returns the new content and the
// replaced span, or a message saying why nothing was replaced.
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

// joinInts renders line numbers as "12, 40, 97".
func joinInts(ns []int) string {
	parts := make([]string, len(ns))
	for i, n := range ns {
		parts[i] = strconv.Itoa(n)
	}
	return strings.Join(parts, ", ")
}

// fsRead reads a text file. For top-level sessions known to the ACP client
// (Zed), the call goes over the wire so the editor can render diffs and
// honour unsaved buffer state. A client that did not advertise
// fs.readTextFile gets direct disk I/O instead: ACP forbids sending it a
// method it never claimed to implement.
// line/limit are optional: pass nil for both to read the whole file, or
// non-nil pointers to bound the response to a 1-indexed line window.
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
	// Drift check on every WHOLE read, wherever it came from: a windowed read is
	// a slice of the file and says nothing about whether the file as a whole
	// still matches what we wrote. Detection lives here, at the one point every
	// read passes through; the note it arms is delivered by whichever tool is
	// returning this content (see takeDriftNote).
	if err == nil && line == nil && limit == nil && sess != nil {
		sess.checkExternalChange(path, content)
	}
	return content, err
}

// fsWrite writes a text file, with the same capability fallback as fsRead.
// Any cached read-dedup entries for this path
// are dropped here because the file just changed — a subsequent read_file
// must run. That invalidation happens before either fallback, so it holds
// for every path through this function.
func fsWrite(a *agent, ctx context.Context, sid string, path, content string) error {
	direct := !a.clientCan("write")
	sess := a.getSession(sid)
	if sess != nil {
		// The file changed: drop its read windows and any continue_read cursor,
		// so the next read runs and starts fresh rather than from a stale line.
		sess.turnMu.Lock()
		for k := range sess.turn.seen {
			if strings.HasPrefix(k, path+"|") {
				delete(sess.turn.seen, k)
			}
		}
		delete(sess.turn.readCursor, path)
		sess.turnMu.Unlock()
	}
	var err error
	if direct {
		err = os.WriteFile(path, []byte(content), 0644)
	} else {
		_, err = a.conn.sendRequest(ctx, "fs/write_text_file", struct {
			SessionId string `json:"sessionId"`
			Path      string `json:"path"`
			Content   string `json:"content"`
		}{sid, path, content})
	}
	// Remember exactly what we put there, so the next whole read of this path can
	// tell "the model misremembers the file" from "something rewrote the file
	// under us" (fsRead → checkExternalChange).
	if err == nil && sess != nil {
		sess.recordWrite(path, content)
	}
	return err
}

// directRead is the disk equivalent of an ACP fs/read_text_file:
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
