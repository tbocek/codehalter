package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ---------------------------------------------------------------------------
// Path security
// ---------------------------------------------------------------------------

func (a *agent) resolvePath(sid string, path string) (string, error) {
	sess := a.getSession(sid)
	if sess == nil {
		return "", fmt.Errorf("no session found")
	}
	inside := func(p string) bool {
		return p == sess.Cwd || strings.HasPrefix(p, sess.Cwd+string(filepath.Separator))
	}
	if filepath.IsAbs(path) {
		resolved := filepath.Clean(path)
		if !inside(resolved) {
			return "", fmt.Errorf("path %q is outside project directory", path)
		}
		return resolved, nil
	}
	rel := filepath.Clean(filepath.Join(sess.Cwd, path))
	if !inside(rel) {
		return "", fmt.Errorf("path %q is outside project directory", path)
	}
	// LLMs sometimes write absolute-looking paths with a missing leading "/"
	// (e.g. `workspaces/preveltekit/go.mod` when cwd is /workspaces/preveltekit).
	// If the cwd-joined interpretation doesn't exist on disk but the "/"-
	// prepended one does and stays inside cwd, prefer that.
	if _, err := os.Stat(rel); err != nil {
		abs := filepath.Clean("/" + path)
		if abs != rel && inside(abs) {
			if _, err := os.Stat(abs); err == nil {
				return abs, nil
			}
		}
	}
	return rel, nil
}

// ---------------------------------------------------------------------------
// Tool registry
// ---------------------------------------------------------------------------

type Tool struct {
	Def map[string]any
	// Execute returns the tool's output and a `failed` flag. `failed` is the
	// authoritative signal that the underlying operation reported a hard
	// failure (e.g. run_task observed a non-zero exit). It's surfaced as
	// ToolUse.Failed so the subtask orchestrator can override an LLM
	// "success=true" when codehalter itself saw the call fail. Most handlers
	// return (output, false); run_task and similar truth-bearing tools set
	// failed=true on a non-zero exit. edit_file/write_file also set it on a usage
	// error to feed the loop's fail cap — but runExecutePhase's verdict excludes
	// file-mutation tools, so a recovered edit doesn't condemn the subtask.
	Execute func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool)
	// Terminal marks a tool as the loop's exit point: when the model invokes
	// it, the agentic tool loop returns after this batch completes, with the
	// tool's Output (one Execute return value) becoming the assistant's final
	// text. Currently only `respond` is terminal — see tool_respond.go. The
	// loop guarantees the result is still recorded in history before exiting.
	Terminal bool
}

// terminalToolName returns the registered Terminal tool exposed in this phase
// (not in the exclude filter). There are two terminals: respond (execute /
// subagent) and submit_plan (plan). respond is the general-purpose one and
// always wins when both are available, so a phase that filters neither — an
// unfiltered loop — stays on respond rather than depending on registration
// order; submit_plan only takes over when respond is excluded (the plan phase
// excludes it). Returns "" when every terminal is filtered out (document
// excludes both), so the loop falls back to the legacy empty-tool-calls exit.
func terminalToolName(f toolFilter) string {
	registryMu.Lock()
	defer registryMu.Unlock()
	first := ""
	for _, t := range registeredTools {
		if !t.Terminal {
			continue
		}
		fn, _ := t.Def["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if name == "" || f.exclude[name] {
			continue
		}
		if name == respondToolName {
			return name
		}
		if first == "" {
			first = name
		}
	}
	return first
}

// registryMu guards registeredTools. Most writes happen at init() (single
// goroutine), but the MCP reconciler mutates the registry mid-session — both
// on startup (parallel server bring-up) and on every Prompt() (diff-and-apply
// after the user edits mcp.toml). Reads from runToolLoop / subagent dispatch
// happen after reconcile completes within a Prompt(), but ACP can deliver
// SessionUpdate to other sessions concurrently, so the lock is mandatory.
var (
	registryMu      sync.Mutex
	registeredTools []Tool
)

func RegisterTool(t Tool) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registeredTools = append(registeredTools, t)
}

// UnregisterToolsByPrefix removes every tool whose function name starts with
// the given prefix. Used by the MCP reconciler to drop a server's tools when
// it shuts down or its config changes. Returns the number removed.
func UnregisterToolsByPrefix(prefix string) int {
	registryMu.Lock()
	defer registryMu.Unlock()
	kept := registeredTools[:0]
	removed := 0
	for _, t := range registeredTools {
		fn, _ := t.Def["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if strings.HasPrefix(name, prefix) {
			removed++
			continue
		}
		kept = append(kept, t)
	}
	// Zero the tail so closed-over goroutines (e.g. an in-flight tools/call)
	// don't keep the old Execute closure alive through this slice.
	for i := len(kept); i < len(registeredTools); i++ {
		registeredTools[i] = Tool{}
	}
	registeredTools = kept
	return removed
}

// hasToolPrefix reports whether any registered tool starts with the prefix.
// Used by the MCP reconciler to detect leftover registrations.
func hasToolPrefix(prefix string) bool {
	registryMu.Lock()
	defer registryMu.Unlock()
	for _, t := range registeredTools {
		fn, _ := t.Def["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// toolFilter lets phases exclude specific tools from a turn (e.g. execute
// strips web_search/web_read so the model can't go fishing mid-edit). Every
// other tool is exposed in every phase — we run inside a devcontainer, so
// "this tool might mutate" is no longer a reason to hide it from the planner.
type toolFilter struct {
	exclude map[string]bool
}

func llmToolDefinitionsFiltered(f toolFilter) []map[string]any {
	registryMu.Lock()
	defer registryMu.Unlock()
	type named struct {
		name string
		def  map[string]any
	}
	var got []named
	for _, t := range registeredTools {
		fn, _ := t.Def["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if f.exclude[name] {
			continue
		}
		got = append(got, named{name, t.Def})
	}
	// Sort by tool name so the rendered `tools` block is byte-identical
	// turn-over-turn regardless of registration order. The MCP reconciler
	// re-adds server tools in a non-deterministic order; without this, that
	// reorders the tool list every reconcile, changing the prompt prefix and
	// busting the LLM's KV cache (full reprocess) — same rule as message bytes.
	sort.Slice(got, func(i, j int) bool { return got[i].name < got[j].name })
	defs := make([]map[string]any, len(got))
	for i, g := range got {
		defs[i] = g.def
	}
	return defs
}

// parseArgs extracts string arguments from raw JSON. For simple string params.
func parseArgs(rawArgs string) map[string]string {
	var args map[string]string
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		slog.Debug("parseArgs: tool arguments are not valid JSON", "err", err, "raw", truncate(rawArgs, 200))
	}
	if args == nil {
		args = make(map[string]string)
	}
	return args
}

func (a *agent) executeTool(ctx context.Context, sid string, tc toolCall) (string, bool) {
	slog.Info("executeTool", "tool", tc.Function.Name, "sid", sid, "args", tc.Function.Arguments)

	// Snapshot the registry under the lock so Execute can run without holding
	// it (Execute may take seconds — bash commands, LLM subagents — and we
	// must not block concurrent registration or other callers during that).
	registryMu.Lock()
	snap := make([]Tool, len(registeredTools))
	copy(snap, registeredTools)
	registryMu.Unlock()

	var names []string
	for _, t := range snap {
		fn, _ := t.Def["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if name == tc.Function.Name {
			return t.Execute(ctx, a, sid, tc.Function.Arguments)
		}
		if name != "" {
			names = append(names, name)
		}
	}
	// Listed names go back to the model as the tool result, so it can
	// self-correct on the next turn instead of looping on the same hallucination.
	return fmt.Sprintf("unknown tool %q. Use only the tools provided to you; available tools: %s",
		tc.Function.Name, strings.Join(names, ", ")), false
}

// ---------------------------------------------------------------------------
// Tool call UI
// ---------------------------------------------------------------------------

var toolCallCounter atomic.Uint64

func nextToolCallID() string {
	return fmt.Sprintf("tc_%d", toolCallCounter.Add(1))
}

// toolUseCounter assigns each recorded ToolUse a stable per-process handle so
// view_output can address it later without re-running the original tool.
// Process-global rather than session-scoped — search is already scoped to one
// session, and a single counter is simpler than tracking last-seen IDs per
// session across restarts (where the counter resets but historical IDs in
// session TOML survive — view_output handles "id not found" cleanly).
var toolUseCounter atomic.Uint64

func nextToolUseID() string {
	return fmt.Sprintf("tu_%d", toolUseCounter.Add(1))
}

// runToolCall executes one tool call and returns (a) the ToolUse recording its
// FULL output and (b) the model-visible content. The full output is cached here
// (in the session, saved to disk) so view_output can re-serve any portion
// without re-running the tool; the message stream only ever carries the
// model-visible copy. That copy is the output shrunk past truncateThreshold with
// a view_output hint — or, for view_image, the inline multimodal parts. The
// caller appends the returned ToolUse to its result set and the content to the
// message stream. Truncation lives here, not in individual tools, so every tool
// returns its complete output and this one place decides "small → whole, big →
// truncate + cache the rest".
func (a *agent) runToolCall(ctx context.Context, sid string, tc toolCall) (ToolUse, any) {
	started := time.Now()

	// view_image short-circuit: when the server supports images, deliver the
	// bytes as multimodal tool content in the SAME turn (so the next llmStream
	// call sees the image). The standard executeTool path returns text only.
	var result string
	var failed bool
	var multimodal any
	if tc.Function.Name == "view_image" && a.imagesSupported {
		text, parts, ferr := dispatchViewImage(a.getSession(sid), tc.Function.Arguments)
		result, failed = text, ferr
		if !ferr {
			multimodal = parts
		}
	} else {
		result, failed = a.executeTool(ctx, sid, tc)
	}

	useID := nextToolUseID()
	tu := ToolUse{
		ID:         useID,
		Name:       tc.Function.Name,
		Input:      tc.Function.Arguments,
		Output:     result,
		Failed:     failed,
		StartedAt:  started,
		DurationMs: time.Since(started).Milliseconds(),
	}
	// Cache the full output (saved incrementally so it survives a crash) for view_output.
	if sess := a.getSession(sid); sess != nil {
		sess.AppendToolUse(tu)
		sess.saveOrLog()
	}

	if multimodal != nil {
		return tu, multimodal
	}
	return tu, liveToolOutput(useID, tc.Function.Name, tc.Function.Arguments, result)
}

type toolCallUpdate struct {
	Kind       string             `json:"sessionUpdate"`
	ToolCallId string             `json:"toolCallId"`
	Title      string             `json:"title,omitempty"`
	ToolKind   string             `json:"kind,omitempty"`
	Status     string             `json:"status,omitempty"`
	Content    []ToolCallContent  `json:"content,omitempty"`
	Locations  []ToolCallLocation `json:"locations,omitempty"`
}

type ToolCallContent struct {
	Type    string        `json:"type"`
	Content *ContentBlock `json:"content,omitempty"`
	Path    string        `json:"path,omitempty"`
	OldText *string       `json:"oldText,omitempty"`
	NewText string        `json:"newText,omitempty"`
}

type ToolCallLocation struct {
	Path string `json:"path"`
	Line *int   `json:"line,omitempty"`
}

func TextContent(text string) ToolCallContent {
	b := ContentBlock{Type: "text", Text: text}
	return ToolCallContent{Type: "content", Content: &b}
}

func DiffContent(path string, oldText *string, newText string) ToolCallContent {
	return ToolCallContent{Type: "diff", Path: path, OldText: oldText, NewText: newText}
}

func (a *agent) StartToolCall(ctx context.Context, sid string, title, kind string, locations []ToolCallLocation) string {
	id := nextToolCallID()
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call",
		ToolCallId: id,
		Title:      title,
		ToolKind:   kind,
		Status:     "in_progress",
		Content:    []ToolCallContent{},
		Locations:  locations,
	})
	return id
}

func (a *agent) CompleteToolCall(ctx context.Context, sid string, id string, content []ToolCallContent) {
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: id,
		Status:     "completed",
		Content:    content,
	})
}

// CompleteToolCallTitled is like CompleteToolCall but also overwrites the
// tool-call's title. Use this to surface a result preview in the panel
// without requiring the user to expand the disclosure (e.g. change
// "go_symbols: Foo" → "go_symbols: Foo → router.go:27 (+1)").
func (a *agent) CompleteToolCallTitled(ctx context.Context, sid string, id, title string, content []ToolCallContent) {
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: id,
		Title:      title,
		Status:     "completed",
		Content:    content,
	})
}

func (a *agent) FailToolCall(ctx context.Context, sid string, id, errMsg string) {
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: id,
		Status:     "failed",
		Content:    []ToolCallContent{TextContent("❌ " + errMsg)},
	})
}

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
	// liveExemptCap bounds the LIVE output of the truncation-exempt tools
	// (read_file / continue_read / search_text). They self-bound by COUNT
	// (150-line chunk / 100 matches) but NOT by bytes — a minified file or
	// long-line matches can still be enormous and blow n_ctx in a single call.
	// ~32 KB ≈ 8k tokens: 20× the history clip, big enough to act on, but no
	// single call can balloon the context. Clipped at a line boundary so we
	// never slice mid-line / mid-match (the reason these are exempt at all).
	liveExemptCap = 32 * 1024
)

// liveToolOutput is the model-visible content for a tool call as it executes.
// The content-retrieval tools — read_file / continue_read / search_text and the
// web fetchers web_search / web_read / web_read_raw — manage their OWN size
// (line chunks, bounded match/result lists, offset/limit page slices) and the
// model must act on the whole result in the current turn, so their live output
// passes through whole rather than the 1.5 KB head/tail cap every other tool
// gets — byte-clipping them cut mid-line / mid-result / mid-page. They still get
// a generous line-aware ceiling (liveExemptCap) so one pathological call (a
// minified file, long-line matches) can't blow n_ctx. This is LIVE only:
// history.go re-renders stored outputs through truncateForLLM directly, which
// truncates EVERYTHING to 1.5 KB — an old read or search must NOT sit at full
// size in every later request (that ballooned the context past n_ctx; the model
// can re-read / view_output if it needs it again).
func liveToolOutput(useID, toolName, args, content string) string {
	switch toolName {
	case "read_file", "continue_read", "search_text", "web_search", "web_read", "web_read_raw":
		if len(content) <= liveExemptCap {
			return content
		}
		// Line-aware clip: keep whole lines up to the cap, then a pointer to
		// the cached full output. Falls back to a hard byte cut only if there's
		// no newline in the kept span (a single mega-line).
		cut := strings.LastIndexByte(content[:liveExemptCap], '\n')
		if cut <= 0 {
			cut = liveExemptCap
		}
		return fmt.Sprintf("%s\n\n[... %d of %d chars omitted (oversized output capped at %d KB). %s]",
			content[:cut], len(content)-cut, len(content), liveExemptCap/1024, truncationHint(useID, toolName, args))
	}
	return truncateForLLM(useID, toolName, args, content)
}

// truncateForLLM returns content unchanged if short; otherwise emits a
// head/tail-shaped slice with a per-tool "to see more" hint in the middle.
// useID is the ToolUse handle the caller just assigned — embedded in the hint
// so the model can `view_output id=useID mode=grep pattern=…` to retrieve any
// portion of the original without re-running the underlying tool. toolName +
// args let the hint name the exact alternate follow-up call (e.g. read_file
// path+line, web_read url+offset) so the model doesn't have to reconstruct what
// it just asked about. Used by history.go when re-rendering stored tool outputs
// (always truncates) and by liveToolOutput for the non-exempt live tools.
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
	case "web_search":
		return "To see more: " + viewOut + ", OR refine the query (fewer, more specific terms) and search again — then web_read the most promising result for its full text."
	default:
		return "To see more: " + viewOut + "."
	}
}
