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
// useID is the ToolUse handle the loop just assigned — embedded in the hint
// so the model can `view_output id=useID mode=grep pattern=…` to retrieve any
// portion of the original without re-running the underlying tool. toolName +
// args let the hint name the exact alternate follow-up call (e.g. read_file
// path+line, web_read url+offset) so the model doesn't have to reconstruct
// what it just asked about.
func truncateForLLM(useID, toolName, args, content string) string {
	if len(content) <= truncateThreshold {
		return content
	}
	omitted := len(content) - truncateHeadChars - truncateTailChars
	head := content[:truncateHeadChars]
	tail := content[len(content)-truncateTailChars:]
	hint := truncationHint(useID, toolName, args)
	return fmt.Sprintf("%s\n\n[... %d of %d chars omitted. %s]\n\n%s", head, omitted, len(content), hint, tail)
}

// truncationHint returns the per-tool "to see more" pointer. Every hint
// includes a `view_output id=<useID>` path that retrieves any portion of the
// FULL cached output without re-running the underlying tool — critical for
// `run_task` / `run_command` where re-running is slow or non-idempotent.
// Tools with cheap, idempotent re-invocation paths (read_file with a different
// line range, web_read with offset/limit) also keep that alternate hint
// because slicing on the caller's terms is sometimes more useful than
// grepping. parseArgs is best-effort — when args don't contain a useful key
// the alternate-call suggestion is dropped and only view_output remains.
func truncationHint(useID, toolName, args string) string {
	a := parseArgs(args)
	viewOut := fmt.Sprintf("call view_output id=%q mode=grep pattern=<regex> (or mode=head / mode=tail with lines=<n>) to retrieve the cached full output without re-running this tool", useID)
	switch toolName {
	case "read_file":
		path := a["path"]
		if path == "" {
			return "To see more: " + viewOut + ", OR call read_file again with line=<n> limit=<m> to view a specific range."
		}
		return fmt.Sprintf("To see more: %s, OR call read_file path=%q with line=<n> limit=<m> for a specific range, or search_text path=%q pattern=<regex>.", viewOut, path, path)
	case "web_read", "web_read_raw":
		u := a["url"]
		if u == "" {
			return "To see more: " + viewOut + ", OR call this tool again with offset=<n> limit=<m> — the full body is cached server-side, so no re-fetch."
		}
		return fmt.Sprintf("To see more: %s, OR call %s again with url=%q offset=<n> limit=<m> — the full body is cached server-side, so no re-fetch.", viewOut, toolName, u)
	case "run_command", "run_task":
		return "To see more: " + viewOut + ". Do NOT re-run the command just to see more output — view_output reads the cached result."
	case "list_files":
		path := a["path"]
		if path == "" {
			return "To see more: " + viewOut + ", OR call list_files on a deeper subdirectory."
		}
		return fmt.Sprintf("To see more: %s, OR call list_files on a subdirectory of %q.", viewOut, path)
	case "search_text":
		return "To see more: " + viewOut + ", OR re-run search_text with a more specific pattern or narrower path."
	default:
		return "To see more: " + viewOut + "."
	}
}
