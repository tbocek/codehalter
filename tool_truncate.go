package main

import (
	"fmt"
)

// Truncation thresholds for tool output going BACK to the LLM. Anything at or
// under truncateThreshold passes through untouched — head/tail+marker would
// cost more than the original content. Larger results are replaced with the
// first truncateHeadChars + a per-tool search hint + the last truncateTailChars
// so the cumulative prefix the model re-processes each turn stays small. The
// full output is still preserved in ToolUse.Output on disk; only the
// in-flight LLM message stream is shortened.
const (
	truncateThreshold = 1500
	truncateHeadChars = 600
	truncateTailChars = 600
)

// truncateForLLM returns content unchanged if short; otherwise emits a
// head/tail-shaped slice with a per-tool "to see more" hint in the middle.
// toolName + args let the hint name the exact follow-up call (e.g. the cached
// URL for web_read, the path+line for read_file) so the model doesn't have to
// reconstruct what it just asked about.
func truncateForLLM(toolName, args, content string) string {
	if len(content) <= truncateThreshold {
		return content
	}
	omitted := len(content) - truncateHeadChars - truncateTailChars
	head := content[:truncateHeadChars]
	tail := content[len(content)-truncateTailChars:]
	hint := truncationHint(toolName, args)
	return fmt.Sprintf("%s\n\n[... %d of %d chars omitted. %s]\n\n%s", head, omitted, len(content), hint, tail)
}

// truncationHint returns the per-tool "to see more" pointer. The hint tells
// the model exactly which follow-up call recovers the missing middle without
// burning another turn on trial-and-error. parseArgs is best-effort — when the
// args don't contain a useful key (or come in malformed) we fall back to a
// generic suggestion.
func truncationHint(toolName, args string) string {
	a := parseArgs(args)
	switch toolName {
	case "read_file":
		path := a["path"]
		if path == "" {
			return "To see more: call read_file again with line=<n> limit=<m> to view a specific range, or search_text with a pattern to jump straight to the relevant section."
		}
		return fmt.Sprintf("To see more: call read_file path=%q with line=<n> limit=<m> for a specific range, or search_text path=%q pattern=<regex> to jump to the relevant section.", path, path)
	case "web_read", "web_read_raw":
		u := a["url"]
		if u == "" {
			return "To see more: call this tool again with offset=<n> limit=<m> — the full body is cached server-side, so no re-fetch."
		}
		return fmt.Sprintf("To see more: call %s again with url=%q offset=<n> limit=<m> — the full body is cached server-side, so no re-fetch.", toolName, u)
	case "run_command", "run_task":
		return "To see more: re-run with output filtered (e.g. `cmd 2>&1 | grep <pattern>` or `... | sed -n '100,200p'`) to isolate the relevant section."
	case "list_files":
		path := a["path"]
		if path == "" {
			return "To see more: call list_files on a deeper subdirectory, or use search_text to locate a specific file by pattern."
		}
		return fmt.Sprintf("To see more: call list_files on a subdirectory of %q, or use search_text to locate a specific file by pattern.", path)
	case "search_text":
		return "To see more: re-run search_text with a more specific pattern, or narrow the path argument."
	default:
		return "To see more: call this tool again with narrower parameters."
	}
}
