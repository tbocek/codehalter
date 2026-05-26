package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
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
	// return (output, false); only run_task and similar truth-bearing tools
	// set failed=true.
	Execute func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool)
	// Terminal marks a tool as the loop's exit point: when the model invokes
	// it, the agentic tool loop returns after this batch completes, with the
	// tool's Output (one Execute return value) becoming the assistant's final
	// text. Currently only `respond` is terminal — see tool_respond.go. The
	// loop guarantees the result is still recorded in history before exiting.
	Terminal bool
}

// terminalToolName returns the name of the first registered tool with
// Terminal=true that is NOT in the exclude filter. Used by runToolLoopOn to
// decide whether the synthetic-respond exit applies for this phase: plan,
// verify, and document exclude the respond tool because they parse JSON
// output, so for those filters this returns "" and the loop falls back to the
// legacy text exit. Returns "" when no terminal tool is enabled.
func terminalToolName(f toolFilter) string {
	registryMu.Lock()
	defer registryMu.Unlock()
	for _, t := range registeredTools {
		if !t.Terminal {
			continue
		}
		fn, _ := t.Def["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if name == "" || f.exclude[name] {
			continue
		}
		return name
	}
	return ""
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
	var defs []map[string]any
	for _, t := range registeredTools {
		fn, _ := t.Def["function"].(map[string]any)
		name, _ := fn["name"].(string)
		if f.exclude[name] {
			continue
		}
		defs = append(defs, t.Def)
	}
	return defs
}

// parseArgs extracts string arguments from raw JSON. For simple string params.
func parseArgs(rawArgs string) map[string]string {
	var args map[string]string
	_ = json.Unmarshal([]byte(rawArgs), &args)
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

// imageCounter assigns each captured ImageData a stable per-process handle so
// the LLM can call view_image to retrieve an older attachment after
// buildLLMContext has replaced its inline data: URL with a text placeholder.
var imageCounter atomic.Uint64

func nextImageID() string {
	return fmt.Sprintf("img_%d", imageCounter.Add(1))
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
