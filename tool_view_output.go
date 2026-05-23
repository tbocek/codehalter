package main

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// view_output reads a portion of a prior tool call's FULL output from the
// session cache, addressed by the per-call `id` (tu_<n>) that every
// truncation hint surfaces. Lets the model retrieve any slice of a long
// `run_task` / `run_command` / `read_file` result without re-running the
// underlying tool — critical when re-running is slow (cargo build) or
// non-idempotent (apt-get install). Three modes:
//
//   - head — first N lines (default 100)
//   - tail — last  N lines (default 100)
//   - grep — every line matching `pattern` (RE2 regex), each prefixed with
//     its 1-based line number so the model can identify location
//
// Output itself is subject to the regular truncateForLLM ceiling — a grep
// that matches thousands of lines will get truncated and the model is
// nudged to narrow the pattern.
const defaultViewOutputLines = 100

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "view_output",
			"description": "View a portion of a prior tool call's full output without re-running it. " +
				"ONLY use this when the original tool result was truncated — the truncation hint shows the `id` to pass here. " +
				"If you have the original output in your tool history and it was NOT truncated, scroll back to it instead; don't call view_output. " +
				"Modes: `head` = first N lines, `tail` = last N lines, `grep` = every line matching `pattern` (RE2 regex). " +
				"For head/tail, `lines` defaults to 100. Prefer this over re-running the original tool (run_task / run_command) when the data is genuinely missing from history.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"id", "mode"},
				"properties": map[string]any{
					"id": map[string]any{
						"type":        "string",
						"description": "The tool-use id from an earlier truncation hint, e.g. `tu_42`.",
					},
					"mode": map[string]any{
						"type":        "string",
						"enum":        []string{"head", "tail", "grep"},
						"description": "Which portion to return.",
					},
					"pattern": map[string]any{
						"type":        "string",
						"description": "RE2 regex, required when mode=grep. Anchored matches (e.g. `^error`) and case insensitivity flags (`(?i)…`) supported.",
					},
					"lines": map[string]any{
						"type":        "integer",
						"description": "Number of lines to return for head/tail (default 100). Ignored for grep.",
					},
				},
			},
		},
	}, Execute: viewOutputExecute})
}

func viewOutputExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	var args struct {
		ID      string `json:"id"`
		Mode    string `json:"mode"`
		Pattern string `json:"pattern"`
		Lines   int    `json:"lines"`
	}
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		return fmt.Sprintf("view_output: invalid arguments: %v", err), true
	}
	if args.ID == "" {
		return "view_output: missing `id`. Pass the tool-use id from a prior truncation hint (e.g. tu_42).", true
	}

	sess := a.getSession(sid)
	if sess == nil {
		return "view_output: no session", true
	}
	full := sess.FindToolUseOutput(args.ID)
	if full == "" {
		return fmt.Sprintf("view_output: id %q not found in session. Valid ids are surfaced in truncation hints — check the hint for the exact id, or use a different lookup (read_file with line range, search_text, etc.).", args.ID), true
	}

	lines := strings.Split(full, "\n")
	n := args.Lines
	if n <= 0 {
		n = defaultViewOutputLines
	}

	switch args.Mode {
	case "head":
		if n > len(lines) {
			n = len(lines)
		}
		return fmt.Sprintf("[%s head: lines 1-%d of %d]\n%s", args.ID, n, len(lines), strings.Join(lines[:n], "\n")), false
	case "tail":
		if n > len(lines) {
			n = len(lines)
		}
		start := len(lines) - n
		return fmt.Sprintf("[%s tail: lines %d-%d of %d]\n%s", args.ID, start+1, len(lines), len(lines), strings.Join(lines[start:], "\n")), false
	case "grep":
		if args.Pattern == "" {
			return "view_output: missing `pattern` for mode=grep", true
		}
		re, err := regexp.Compile(args.Pattern)
		if err != nil {
			return fmt.Sprintf("view_output: invalid regex %q: %v", args.Pattern, err), true
		}
		var b strings.Builder
		matches := 0
		for i, line := range lines {
			if re.MatchString(line) {
				fmt.Fprintf(&b, "%d: %s\n", i+1, line)
				matches++
			}
		}
		if matches == 0 {
			return fmt.Sprintf("[%s grep %q: no matches in %d lines]", args.ID, args.Pattern, len(lines)), false
		}
		return fmt.Sprintf("[%s grep %q: %d matches in %d lines]\n%s", args.ID, args.Pattern, matches, len(lines), b.String()), false
	default:
		return fmt.Sprintf("view_output: unknown mode %q. Use head, tail, or grep.", args.Mode), true
	}
}
