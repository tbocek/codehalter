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
	"slices"
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
			return strings.Join(slices.Concat(fileLines[:start], strings.Split(repl, "\n"), fileLines[start+len(oldLines):]), "\n"), 1
		}
	}
	return "", 0
}

const (
	nearMissMinScore     = 0.5
	nearMissMaxFileLines = 20_000
	nearMissTieEpsilon   = 1e-9
	nearMissSnippetCap   = 1500
)

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
			"name":        "write_file",
			"description": "Create a NEW file, or fully regenerate a small generated one. Not for changing a file you read: rewriting a large file from memory loses content; use edit_file.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path", "content"},
				"properties": map[string]any{
					"path":    map[string]any{"type": "string", "description": "Absolute or project-relative. `workspaces/foo` (no leading slash) is also tried as `/workspaces/foo`."},
					"content": map[string]any{"type": "string", "description": "The whole file content; it replaces the file."},
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
			"name":        "edit_file",
			"description": "Change an EXISTING file. Use it over write_file for any existing file, and ALWAYS over a Python, sed or awk script through run_command: this edit is checked, shown to the user as a diff, and counted; a script is none of that, and a wrong anchor silently rewrites the wrong span. Pick one form per call:\n(1) A small exact change: {\"path\": \"src/cut.rs\", \"old_text\": \"let zoom = 1.0;\", \"new_text\": \"let zoom = params::ZOOM;\"}. old_text must be unique in the file; copy it from a fresh read.\n(2) A whole BLOCK (a function, a test, a match arm, a widget section): {\"path\": \"src/ui/window.rs\", \"start\": \"fn wire_zoom(\", \"end\": \"}\", \"new_text\": \"...\"}. Every line from the start line through the end line is replaced; an `end` of only a closing bracket means the block's own closing line, found by its braces (or indentation). Use it instead of copying forty lines into old_text.\n(3) SEVERAL changes to one file: {\"path\": \"src/ui/window.rs\", \"edits\": [{\"old_text\": \"...\", \"new_text\": \"...\"}, {\"start\": \"fn wire_zoom(\", \"end\": \"}\", \"new_text\": \"...\"}]}, applied in order, all or none. Two or three changes to one file: ONE call like this.\nErrors (not found, not unique, unwritable) come back as messages: fix and retry.",

			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path"},
				"properties": map[string]any{
					"edits": map[string]any{
						"type":        "array",
						"description": "SEVERAL changes to this file in ONE call, instead of the fields below: each {old_text, new_text} or {start, end, new_text}, in order, all or none; an error names the one that failed.",
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
					"path":     map[string]any{"type": "string", "description": "Absolute or project-relative. `workspaces/foo` (no leading slash) is also tried as `/workspaces/foo`."},
					"old_text": map[string]any{"type": "string", "description": "Form (1): the exact text, byte for byte (whitespace, indentation, newlines) and unique in the file: include enough context to tell it apart."},
					"start":    map[string]any{"type": "string", "description": "Form (2): a fragment of the block's FIRST line, unique in the file (`fn cut_form_column(`)."},
					"end":      map[string]any{"type": "string", "description": "Form (2): a fragment of the block's LAST line; the first line at or after `start` containing it ends the block. Only a closing bracket (`}`) means the block's own closing line, found by its braces. Omit to replace the start line alone."},
					"new_text": map[string]any{"type": "string", "description": "Replacement: for (1) of old_text, for (2) of the whole lines start through end. An empty string deletes."},
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
		// A whole list sent as a string is decoded before the tool runs (fitArgs): one still a string
		// here was cut off, and a list of edits is all or none.
		if _, cut := args["edits"].(string); cut {
			return "error: `edits` came as a string holding a list that is cut off. Nothing was written: send `edits` as a JSON array.", true
		}
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

// commentLineRe: a line that is only a comment, in the common comment styles.
var commentLineRe = regexp.MustCompile(`^\s*(?://|#|/\*|\*|--|;|"""|'''|<!--)`)

// braceBlockEnd: a ';' before any '{' ends the block (a prototype), and the '{'
// must open within the first few lines.
// braceBlockEnd: where the brace block that opens at or after line d closes, and the line
// its `{` is on. Strings ("…", `…`, '…' but not a Rust lifetime 'a) and comments (//, /* */,
// `# `) do not count. Not ok when no `{` opens within three lines; a `;` ending line d first ends it there.
func braceBlockEnd(lines []string, d int) (end, open int, ok bool) {
	depth, opened, inBlock := 0, false, false
	for i := d; i < len(lines); i++ {
		ln := lines[i]
		inStr := byte(0)
		for j := 0; j < len(ln); j++ {
			c := ln[j]
			switch {
			case inBlock:
				if c == '*' && j+1 < len(ln) && ln[j+1] == '/' {
					inBlock = false
					j++
				}
				continue
			case inStr != 0:
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
			case '\'':
				// A lifetime ('a, 'static) opens nothing; a char literal or a single-quoted string does.
				if !lifetimeRe.MatchString(ln[j:]) {
					inStr = c
				}
			case '/':
				if j+1 < len(ln) && ln[j+1] == '/' {
					j = len(ln)
				} else if j+1 < len(ln) && ln[j+1] == '*' {
					inBlock = true
					j++
				}
			case '#':
				// A shell, Python, Ruby or PHP comment: `# …` after whitespace. Not a CSS
				// colour (#fff), a Rust attribute (#[test]), `$#` or `#define`.
				if (j == 0 || ln[j-1] == ' ' || ln[j-1] == '\t') && (j+1 == len(ln) || ln[j+1] == ' ' || ln[j+1] == '\t') {
					j = len(ln)
				}
			case '{':
				if !opened {
					open = i
				}
				depth++
				opened = true
			case '}':
				depth--
				// `} else {`, `} catch (e) {`, `}).then(() => {`: the chain goes on.
				if opened && depth == 0 && !strings.Contains(stripLineComment(ln[j+1:]), "{") {
					return i, open, true
				}
			case ';':
				// Only a `;` that ends the line ends a declaration: not shell's `; then`, nor
				// the ones inside a C `for (…; …; …) {`.
				if !opened && strings.TrimSpace(stripLineComment(ln[j+1:])) == "" {
					return i, i, true
				}
			}
		}
		if !opened && i-d >= 3 {
			return 0, 0, false
		}
	}
	return 0, 0, false
}

// lifetimeRe: a Rust lifetime or loop label, a quote no other quote closes.
var lifetimeRe = regexp.MustCompile(`^'[A-Za-z_]\w*(?:[^\w']|$)`)

// closerLineRe is a line that only closes a block, at the indentation the block opened at.
// A keyword closes only alone or before punctuation or a block word: `end`, `end)`,
// `end if`, `End Sub`, not `end = 5`.
var closerLineRe = regexp.MustCompile(`(?i)^(?:[)\]}]+[;,)]*|(?:end|fi|done|esac|endif|endfunction|endwhile|endfor|wend)(?:[\s;),.]*|\s+(?:if|do|function|sub|select|while|for|module|class|case|type|program|subroutine|loop|interface|procedure)\b.*))$`)

// heredocRe opens a here-document (`<<EOF`, `<<-'END'`, Ruby's `<<~SQL`), whose body ends at
// its word alone. The word is upper-case and follows directly, so `list << item` and `1<<x`
// are no here-documents.
var heredocRe = regexp.MustCompile(`<<[-~]?['"]?([A-Z_][A-Z0-9_]*)['"]?`)

// tripleQuoteRe: Python's string delimiters that span lines.
var tripleQuoteRe = regexp.MustCompile(`"""|'''`)

// continueLineRe goes on a block at its own indentation: Python's `else:`, Ruby's
// `elsif`, shell's `elif`, `except`, `rescue`.
var continueLineRe = regexp.MustCompile(`^(?:elif|else|elsif|elseif|except|finally|rescue|ensure|catch)\b`)

// blockEnd: where the block opening on line d ends, in any language. Braces decide when the
// block opens with a `{` on line d, or on a later line at d's own indentation (a signature
// spread over lines); a `{` indented deeper belongs to the body (a dict, a hash). Otherwise
// indentation does: the lines indented deeper than d, plus a closing line at d's own
// indentation that only closes (`}`, `end`, `fi`, `done`). A declaration a `;` ends is its
// own line. Not ok when no block opens on d.
func blockEnd(lines []string, d int) (int, string, bool) {
	// A decorator (`@cache`, `@Override`) heads the definition below it.
	if strings.HasPrefix(strings.TrimSpace(lines[d]), "@") {
		for i := d + 1; i < len(lines); i++ {
			if t := strings.TrimSpace(lines[i]); t != "" && !strings.HasPrefix(t, "@") {
				return blockEnd(lines, i)
			}
		}
		return 0, "", false
	}
	base := len(leadingWS(lines[d]))
	e, open, ok := braceBlockEnd(lines, d)
	switch {
	case ok && open == d && !strings.Contains(stripLineComment(lines[d]), "{"):
		return e, "its semicolon", true // a declaration without a body: `fn draw(&self);`
	case ok && (open == d || len(leadingWS(lines[open])) <= base):
		return e, "braces", true
	}
	end, deeper := d, false
	closeAt := "" // a here-document's word, or a triple quote: text up to it is opaque
	for i := d + 1; i < len(lines); i++ {
		t := strings.TrimSpace(lines[i])
		// A here-document's body or a triple-quoted string may sit at column 0 inside the block.
		if closeAt != "" {
			if t == closeAt || tripleQuoteRe.MatchString(closeAt) && strings.Contains(t, closeAt) {
				closeAt = ""
			}
			continue
		}
		if m := heredocRe.FindStringSubmatch(t); m != nil {
			closeAt = m[1]
		} else if q := tripleQuoteRe.FindAllString(t, -1); len(q)%2 == 1 {
			closeAt = q[0]
		}
		// Blank lines and comments say nothing of the block: a Python comment may sit at column 0.
		if strings.TrimSpace(stripLineComment(t)) == "" {
			continue
		}
		ind := len(leadingWS(lines[i]))
		switch {
		case ind > base:
			end, deeper = i, true
			continue
		case deeper && ind == base && continueLineRe.MatchString(t):
			end = i // `else:`, `except E:`: the same statement goes on
			continue
		case deeper && ind == base && closerLineRe.MatchString(t):
			end = i
		}
		break
	}
	return end, "indentation", deeper
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
		return next, fmt.Sprintf(" (lines %s)", span), ""
	}
	switch count := strings.Count(content, oldText); {
	case count > 1:
		return "", "", fmt.Sprintf("error: old_text matches %d places; it must be unique. Add a few more exact lines of surrounding context (copied from a fresh read of the file) so it pins exactly one spot; don't split the edit in a way that loses uniqueness.", count)
	case count == 1:
		return strings.Replace(content, oldText, newText, 1), "", ""
	}
	tol, n := tolerantReplace(content, oldText, newText)
	switch {
	case n == 1:
		return tol, " (old_text matched ignoring whitespace/indentation)", ""
	case n > 1:
		return "", "", fmt.Sprintf("error: old_text isn't a byte-for-byte match, and ignoring whitespace it matches %d places; add a couple more lines of surrounding context (from a fresh read of the file) to pin exactly one spot.", n)
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
	// Not "the file drifted": codehalter cannot know that, and mostly the model wrote old_text
	// from memory (it once never saw the 58 lines it quoted, having sent them to a file with
	// sed). The first line where its copy differs says what to fix.
	if line, snippet, found := nearMiss(content, oldText); found {
		differs := ""
		fileLines, oldLines := strings.Split(snippet, "\n"), trimBlankEdges(strings.Split(oldText, "\n"))
		for j := range min(len(fileLines), len(oldLines)) {
			if strings.TrimRight(oldLines[j], " \t\r") != strings.TrimRight(fileLines[j], " \t\r") {
				differs = fmt.Sprintf(" Your copy differs from the file first at line %d: you wrote %q, the file has %q.", line+j, truncate(oldLines[j], 120), truncate(fileLines[j], 120))
				break
			}
		}
		return "", "", fmt.Sprintf("error: old_text not found: it is not in the file as it is now.%s The closest region is %s lines %d-%d, which CURRENTLY reads:\n\n%s\n\n"+
			"Retry edit_file with old_text copied byte-for-byte from that block (a SMALL unique part of it is enough). Do NOT read the file again first: the text above is its current content. Do NOT rewrite the whole file with write_file.",
			differs, path, line, line+strings.Count(snippet, "\n"), truncate(snippet, nearMissSnippetCap))
	}
	return "", "", "error: old_text not found: it is not in the file as it is now, and no similar region was found either, so it may be the wrong file. Read the lines you are changing again (e.g. `sed -n '120,180p' <file>`) for their CURRENT exact text, then retry edit_file on a SMALL unique snippet. Do NOT re-read the whole file, and do NOT rewrite the whole file with write_file."
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
		return "", "", fmt.Sprintf("`start` %q is on no line of the file. Copy a fragment of the block's first line from a fresh read of the file.", start)
	case len(hits) > 1:
		var at []int
		for _, h := range hits {
			at = append(at, h+1)
		}
		return "", "", fmt.Sprintf("`start` %q is on %d lines (%s); make it longer so it is on exactly one.", start, len(hits), joinInts(at))
	}
	s, e, how := hits[0], hits[0], ""
	if end != "" {
		e = -1
		// A bare word matches as a word: `end` is no part of `append`.
		contains := func(ln string) bool { return strings.Contains(ln, end) }
		if bareWordRe.MatchString(end) {
			word := regexp.MustCompile(`\b` + regexp.QuoteMeta(end) + `\b`)
			contains = word.MatchString
		}
		search := func() {
			for i := s; i < len(lines); i++ {
				if contains(lines[i]) {
					e = i
					break
				}
			}
		}
		closer := closingOnlyRe.MatchString(strings.TrimSpace(end))
		if !closer {
			search()
		}
		// A function's last line is a bare `}`, which also closes every inner block, so the
		// model cannot name it. A bare closer (`}`, `end`) is documented to mean the block's own
		// closing line, found by its braces or its indentation.
		//
		// HACK: an `end` that is on no line is outside the contract (a fragment of the block's
		// last line) and should be an error. Qwen3.8-flash-next does not follow the tool's
		// directions here: in a replay it invented "} // placeholder-not-used" for every
		// whole-function edit (9 of 9 failed), and steering wording, a line-range form and a
		// symbol form changed nothing it did. So its intent is guessed: the block opening on the
		// start line. Drop the guess once the model follows the description.
		if e < 0 {
			// A statement on the start line is no block to extend to.
			if b, by, ok := blockEnd(lines, s); ok && by != "its semicolon" {
				e, how = b, ", its end found by "+by
			} else if closer {
				search() // from a line inside a block, `}` is the next line closing one
			}
		}
		if e < 0 {
			return "", "", fmt.Sprintf("`end` %q is on no line at or after line %d, where `start` is, and no block opens there.", end, s+1)
		}
	}
	repl := strings.Split(strings.TrimSuffix(newText, "\n"), "\n")
	if newText == "" {
		repl = nil
	}
	return strings.Join(slices.Concat(lines[:s], repl, lines[e+1:]), "\n"), fmt.Sprintf("%d-%d replaced%s", s+1, e+1, how), ""
}

// closingOnlyRe is an `end` that only closes (`}`, `end`, `fi`): it fits every inner block
// too, so the block's own closing line is meant.
var closingOnlyRe = regexp.MustCompile(`(?i)^(?:[)\]}]+[;,]?|end|fi|done|esac|endif|endfunction|endwhile|endfor|wend)$`)

var bareWordRe = regexp.MustCompile(`^\w+$`)

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
	case "edit_file":
		if cur.has("edits") || before.has("edits") || cur.str("path") != before.str("path") || cur.str("path") == "" {
			return "", ""
		}
		return fmt.Sprintf(batchNoteLead+"edit to %s in a row, and each is a model call. Several changes to one file go in ONE edit_file call, applied in order, all or none; these two as one: {\"path\": %q, \"edits\": [%s, %s]}.]",
			cur.str("path"), cur.str("path"), keep(before, "old_text", "start", "end", "new_text"), keep(cur, "old_text", "start", "end", "new_text")), "💡 told the model: several edits to " + cur.str("path") + " go in one edit_file call"
	}
	return "", ""
}
