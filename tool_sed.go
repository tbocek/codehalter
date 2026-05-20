package main

import (
	"context"
	"fmt"
	"regexp"
	"strings"
)

// sed_file applies a regex substitution to a file in place. Fills the gap
// between edit_file (single literal-text replacement, must be unique) and
// write_file (full rewrite): when you need to rename a symbol across many
// occurrences, strip a recurring suffix, or rewrite many similar lines in
// one shot, edit_file's unique-match requirement forces N separate calls.
// sed_file does all matches at once with a regex.
//
// Modes:
//   - substitute (default) — s/pattern/replacement/. Optional `count` caps
//     the number of replacements (0 = all). $1, $2 expand capture groups.
//   - delete — drop every line matching `pattern`. Replacement is ignored.
//
// Output is the unified diff that was applied, so the model can verify the
// change matched expectations. Empty match-set returns an error so the model
// doesn't think a typo'd pattern succeeded.

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "sed_file",
			"description": "Apply a regex substitution (or line delete) to a file in place. Prefer this over `run_command sed -i …` — the diff routes through the agent's edit UI, the file is bounded to one path, and the regex syntax is Go RE2 (predictable, no GNU/BSD sed dialect quirks). " +
				"Use it when edit_file's literal-text + unique-match requirement is too restrictive: renaming a symbol across many occurrences in one file, stripping a recurring prefix/suffix, rewriting many similar lines, or deleting every line matching a pattern. " +
				"For ONE precise edit of a unique snippet, still use edit_file — it's safer because a regex that matches more than expected silently rewrites the wrong code. " +
				"Always inspect the returned diff to confirm the change matched what you intended.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"path", "pattern"},
				"properties": map[string]any{
					"path":        map[string]any{"type": "string", "description": "Absolute or project-relative path."},
					"pattern":     map[string]any{"type": "string", "description": "Go RE2 regex. Use `(?m)` for multiline mode (^ and $ match line boundaries), `(?s)` for dot-matches-newline, `(?i)` for case-insensitive."},
					"replacement": map[string]any{"type": "string", "description": "Replacement text for mode=substitute. Use $1, $2 for capture groups; $$ for a literal $. Ignored when mode=delete. Empty string is allowed (deletes matched text)."},
					"mode":        map[string]any{"type": "string", "enum": []string{"substitute", "delete"}, "description": "`substitute` (default) replaces matched text with `replacement`. `delete` drops every LINE containing a match. Default: substitute."},
					"count":       map[string]any{"type": "integer", "description": "Max substitutions to make (mode=substitute only). 0 = replace all matches. Default 0."},
				},
			},
		},
	}, Execute: sedFileExecute})
}

func sedFileExecute(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
	args := parseArgs(rawArgs)
	path, err := a.resolvePath(sid, args["path"])
	if err != nil {
		return "error: " + err.Error(), false
	}
	pattern := args["pattern"]
	if pattern == "" {
		return "error: pattern is required", false
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return "error: invalid regex: " + err.Error(), false
	}
	mode := args["mode"]
	if mode == "" {
		mode = "substitute"
	}

	tcId := a.StartToolCall(ctx, sid, fmt.Sprintf("sed_file: %s (%s)", path, mode), "edit", []ToolCallLocation{{Path: path}})

	original, err := fsRead(a, ctx, sid, path, nil, nil)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error reading file: " + err.Error(), false
	}

	var updated string
	var matchCount int
	switch mode {
	case "substitute":
		replacement := args["replacement"]
		count := 0
		if v := args["count"]; v != "" {
			fmt.Sscanf(v, "%d", &count)
		}
		matchCount = len(re.FindAllStringIndex(original, -1))
		if matchCount == 0 {
			a.FailToolCall(ctx, sid, tcId, "pattern matched 0 times")
			return fmt.Sprintf("error: pattern %q matched 0 times in %s — nothing to substitute", pattern, path), false
		}
		if count <= 0 || count >= matchCount {
			updated = re.ReplaceAllString(original, replacement)
		} else {
			n := 0
			updated = re.ReplaceAllStringFunc(original, func(s string) string {
				if n >= count {
					return s
				}
				n++
				return re.ReplaceAllString(s, replacement)
			})
		}
	case "delete":
		lines := strings.SplitAfter(original, "\n")
		kept := make([]string, 0, len(lines))
		for _, ln := range lines {
			if re.MatchString(strings.TrimRight(ln, "\n")) {
				matchCount++
				continue
			}
			kept = append(kept, ln)
		}
		if matchCount == 0 {
			a.FailToolCall(ctx, sid, tcId, "pattern matched 0 lines")
			return fmt.Sprintf("error: pattern %q matched 0 lines in %s — nothing to delete", pattern, path), false
		}
		updated = strings.Join(kept, "")
	default:
		a.FailToolCall(ctx, sid, tcId, "unknown mode: "+mode)
		return fmt.Sprintf("error: unknown mode %q. Use `substitute` or `delete`.", mode), false
	}

	if updated == original {
		a.FailToolCall(ctx, sid, tcId, "pattern matched but produced no change")
		return fmt.Sprintf("error: pattern %q matched %d time(s) but replacement produced no change", pattern, matchCount), false
	}

	if err := fsWrite(a, ctx, sid, path, updated); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error writing file: " + err.Error(), false
	}

	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{DiffContent(path, &original, updated)})
	return fmt.Sprintf("sed_file: %d match(es) in %s — applied", matchCount, path), false
}
