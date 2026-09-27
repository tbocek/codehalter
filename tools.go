package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

func (a *agent) resolvePath(sid string, path string) (string, error) {
	sess := a.getSession(sid)
	if sess == nil {
		return "", fmt.Errorf("no session found")
	}
	inside := func(p string) bool {
		return p == sess.Cwd || strings.HasPrefix(p, sess.Cwd+string(filepath.Separator))
	}
	var final string
	if filepath.IsAbs(path) {
		final = filepath.Clean(path)
	} else {
		final = filepath.Clean(filepath.Join(sess.Cwd, path))
		// Models sometimes drop the leading "/" of an absolute path; prefer that
		// reading when only it exists and stays inside cwd.
		if _, err := os.Stat(final); err != nil {
			abs := filepath.Clean("/" + path)
			if abs != final && inside(abs) {
				if _, err := os.Stat(abs); err == nil {
					final = abs
				}
			}
		}
	}
	if !inside(final) {
		return "", fmt.Errorf("path %q is outside project directory", path)
	}
	// inside() is a string-prefix test, which an in-tree symlink to an
	// out-of-tree target would defeat.
	if !realInside(final, sess.Cwd) {
		return "", fmt.Errorf("path %q resolves through a symlink outside the project directory", path)
	}
	return final, nil
}

// realInside resolves the deepest existing ancestor of p, since a write target
// may not exist yet.
func realInside(p, cwd string) bool {
	rcwd, err := filepath.EvalSymlinks(cwd)
	if err != nil {
		rcwd = filepath.Clean(cwd)
	}
	p = filepath.Clean(p)
	for dir := p; ; {
		if resolved, err := filepath.EvalSymlinks(dir); err == nil {
			full := filepath.Clean(resolved + strings.TrimPrefix(p, dir))
			return full == rcwd || strings.HasPrefix(full, rcwd+string(filepath.Separator))
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return false
		}
		dir = parent
	}
}

type Tool struct {
	Def map[string]any
	// Execute's failed flag becomes ToolUse.Failed and overrides the model's own
	// "success"; edit_file/write_file also set it on usage errors for the fail cap.
	Execute func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool)
}

// phasePolicy is enforced at dispatch, not by pruning the tools array, so the
// array stays byte-identical across phases and the KV-cache prefix survives.
type phasePolicy struct {
	deny      map[string]bool
	terminals map[string]bool
	// proseEnds: a reply without a tool call is a valid exit (the documenter).
	// Otherwise it gets one contract nudge (see contractNudge).
	proseEnds bool
}

// contractNudge is the one corrective for a reply that missed its phase's exit:
// servers that ignore tool_choice (Ollama) let a small model answer in prose.
func contractNudge(policy phasePolicy, text string) string {
	var names []string
	for n := range policy.terminals {
		names = append(names, "`"+n+"`")
	}
	sort.Strings(names)
	what := "Your reply ended without a tool call"
	if strings.TrimSpace(text) == "" {
		what = "Your reply had no tool call and no visible text (your reasoning is never shown)"
	}
	return fmt.Sprintf("%s, and nothing outside a tool call reaches the user. Call %s if the step is done, or the tool for the next step.", what, strings.Join(names, " or "))
}

func toolName(t Tool) string {
	fn, _ := t.Def["function"].(map[string]any)
	name, _ := fn["name"].(string)
	return name
}

// The zero value holds the built-ins (seedLocked); add replaces an existing name.
type toolRegistry struct {
	mu    sync.Mutex
	tools map[string]Tool
}

func (r *toolRegistry) seedLocked() {
	if r.tools != nil {
		return
	}
	r.tools = make(map[string]Tool)
	// run_command/run_background come from discovery and MCP tools from the
	// reconciler, both through add.
	builtin := slices.Concat(fileTools, webTools, []Tool{
		askUserTool, submitPlanTool, respondTool,
		screenshotTool, viewImageTool,
	})
	for _, t := range builtin {
		r.tools[toolName(t)] = t
	}
}

func (r *toolRegistry) add(ts ...Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seedLocked()
	for _, t := range ts {
		r.tools[toolName(t)] = t
	}
}

func (r *toolRegistry) removePrefix(prefix string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seedLocked()
	for name := range r.tools {
		if strings.HasPrefix(name, prefix) {
			delete(r.tools, name)
		}
	}
}

func (r *toolRegistry) lookup(name string) (t Tool, ok bool, names []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seedLocked()
	if t, ok = r.tools[name]; !ok {
		names = slices.Sorted(maps.Keys(r.tools))
	}
	return t, ok, names
}

// defs is sorted by name so the tools block stays byte-identical across phases
// and turns regardless of add order; any change would bust the KV cache.
func (r *toolRegistry) defs() []map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seedLocked()
	names := slices.Sorted(maps.Keys(r.tools))
	defs := make([]map[string]any, len(names))
	for i, n := range names {
		defs[i] = r.tools[n].Def
	}
	return defs
}

// toolArgs is map[string]any because the schemas declare ints and bools, and
// decoding those into map[string]string fails the whole object.
type toolArgs map[string]any

// parseArgs never returns nil; on malformed JSON each tool's own validation
// reports the missing parameter.
func parseArgs(rawArgs string) toolArgs {
	var args toolArgs
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		slog.Debug("parseArgs: tool arguments are not valid JSON", "err", err, "raw", truncate(rawArgs, 200))
	}
	if args == nil {
		args = make(toolArgs)
	}
	return args
}

// str does not stringify numbers or bools: for file-content keys a coerced value
// would clobber the file, so callers pair it with wrongType.
func (a toolArgs) str(key string) string {
	s, _ := a[key].(string)
	return s
}

func (a toolArgs) wrongType(key string) bool {
	v, ok := a[key]
	if !ok {
		return false
	}
	_, isStr := v.(string)
	return !isStr
}

// num also accepts a quoted digit string, since models send either.
func (a toolArgs) num(key string) (int, bool) {
	switch v := a[key].(type) {
	case float64:
		return int(v), true
	case string:
		n, err := strconv.Atoi(strings.TrimSpace(v))
		return n, err == nil
	}
	return 0, false
}

func (a toolArgs) flag(key string) bool {
	switch v := a[key].(type) {
	case bool:
		return v
	case string:
		return strings.EqualFold(strings.TrimSpace(v), "true")
	}
	return false
}

func (a toolArgs) has(key string) bool {
	_, ok := a[key]
	return ok
}

var toolCallCounter atomic.Uint64

// toolUseCounter ids are also the wire id when the model sends no tool_call id
// of its own (see ToolUse.CallID).
var toolUseCounter atomic.Uint64

func (a *agent) recordToolUse(sid string, tc toolCall, tu ToolUse) ToolUse {
	tu.ID = fmt.Sprintf("tu_%d", toolUseCounter.Add(1))
	tu.CallID = tc.ID // replayed verbatim from history
	tu.Name = tc.Function.Name
	tu.Input = tc.Function.Arguments
	if sess := a.getSession(sid); sess != nil {
		sess.AppendToolUse(tu)
		sess.saveOrLog()
	}
	return tu
}

// runToolCall records the FULL output in the ToolUse and returns the
// model-visible content: truncated text, or multimodal parts.
func (a *agent) runToolCall(ctx context.Context, sid string, tc toolCall) (ToolUse, any) {
	slog.Info("runToolCall", "tool", tc.Function.Name, "sid", sid, "args", tc.Function.Arguments)
	started := time.Now()

	// With image support, view_image and screenshot bypass the registry to
	// return multimodal parts in the same turn; the registered tools are text only.
	var result string
	var failed bool
	var multimodal any
	var imageID string
	switch {
	case tc.Function.Name == "view_image" && a.imagesSupported:
		text, parts, ferr := dispatchViewImage(a.getSession(sid), tc.Function.Arguments)
		result, failed = text, ferr
		if !ferr {
			multimodal = parts
		}
	case tc.Function.Name == "screenshot" && a.imagesSupported:
		text, parts, id, ferr := dispatchScreenshot(ctx, a, sid, tc.Function.Arguments)
		result, failed = text, ferr
		if !ferr {
			multimodal, imageID = parts, id
		}
	default:
		// Execute outside the registry lock: a slow tool must not block the MCP
		// reconciler.
		if t, ok, names := a.tools.lookup(tc.Function.Name); ok {
			result, failed = t.Execute(ctx, a, sid, tc.Function.Arguments)
		} else {
			result = fmt.Sprintf("unknown tool %q. Use only the tools provided to you; available tools: %s",
				tc.Function.Name, strings.Join(names, ", "))
		}
		// Attach the rendered screen: the executor rarely calls screenshot on
		// its own.
		if !failed && tc.Function.Name == "run_command" && a.imagesSupported {
			if text, parts, id := a.attachRenderedScreen(ctx, sid, tc.Function.Arguments, result, started); id != "" {
				result, multimodal, imageID = text, parts, id
			}
		}
	}

	if multimodal == nil {
		if sess := a.getSession(sid); sess != nil {
			note, told := sess.batchHint(tc.Function.Name, tc.Function.Arguments, failed)
			if told != "" {
				result += note
				a.say(ctx, sid, told+"\n")
			}
		}
	}

	tu := a.recordToolUse(sid, tc, ToolUse{
		Output:     result,
		Failed:     failed,
		StartedAt:  started,
		DurationMs: time.Since(started).Milliseconds(),
		ImageID:    imageID,
	})
	if multimodal != nil {
		return tu, multimodal
	}
	return tu, liveToolOutput(tc.Function.Name, tc.Function.Arguments, result)
}

// denyToolCall rejects without executing. Failed is for the record only; the
// caller does not feed it to the fail cap.
func (a *agent) denyToolCall(ctx context.Context, sid, phase string, tc toolCall) (ToolUse, string) {
	// Say what to do instead: a bare refusal gets the same call again.
	hint := "it isn't allowed here; continue without it."
	switch phase {
	case "plan":
		hint = "planning is read-only; describe this change as a subtask in submit_plan and the executor will make it."
	case "document":
		hint = "the documentation phase only wraps up — write the note and stop, don't re-plan."
	}
	msg := fmt.Sprintf("error: %s is not available during the %s phase — %s", tc.Function.Name, phase, hint)
	tcId := a.StartToolCall(ctx, sid, tc.Function.Name+" (not allowed this phase)", "tool", nil)
	a.FailToolCall(ctx, sid, tcId, msg)
	return a.recordToolUse(sid, tc, ToolUse{Output: msg, Failed: true, StartedAt: time.Now()}), msg
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
	Type       string        `json:"type"`
	Content    *ContentBlock `json:"content,omitempty"`
	Path       string        `json:"path,omitempty"`
	OldText    *string       `json:"oldText,omitempty"`
	NewText    string        `json:"newText,omitempty"`
	TerminalId string        `json:"terminalId,omitempty"`
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
	id := fmt.Sprintf("tc_%d", toolCallCounter.Add(1))
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

func (a *agent) CompleteToolCallTitled(ctx context.Context, sid string, id, title string, content []ToolCallContent) {
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: id,
		Title:      title,
		Status:     "completed",
		Content:    content,
	})
}

// retitleToolCall sends no content: content would replace a card's live
// terminal view with a static copy.
func (a *agent) retitleToolCall(ctx context.Context, sid, id, title, status string) {
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: id,
		Title:      title,
		Status:     status,
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

// Tool output going back to the LLM is clipped to head + hint + tail; the full
// output stays in ToolUse.Output.
const (
	truncateThreshold = 1500
	truncateHeadChars = 600
	truncateTailChars = 600
	// liveExemptCap: the exempt tools bound by count, not bytes, so a minified
	// file could still blow n_ctx.
	liveExemptCap = 32 * 1024
)

// liveToolOutput is also how history.go re-renders stored outputs, so a replay
// is byte-identical to the live wire. continue_read stays listed for old sessions.
func liveToolOutput(toolName, args, content string) string {
	switch toolName {
	case "read_file", "continue_read", "web_search", "web_read":
		if len(content) <= liveExemptCap {
			return content
		}
		cut := strings.LastIndexByte(content[:liveExemptCap], '\n')
		if cut <= 0 {
			cut = liveExemptCap
		}
		return fmt.Sprintf("%s\n\n[... %d of %d chars omitted (oversized output capped at %d KB). %s]",
			content[:cut], len(content)-cut, len(content), liveExemptCap/1024, truncationHint(toolName, args))
	}
	return truncateForLLM(toolName, args, content)
}

func truncateForLLM(toolName, args, content string) string {
	if len(content) <= truncateThreshold {
		return content
	}
	omitted := len(content) - truncateHeadChars - truncateTailChars
	head := clipUTF8(content, truncateHeadChars)
	tail := tailUTF8(content, truncateTailChars)
	hint := truncationHint(toolName, args)
	return fmt.Sprintf("%s\n\n[... %d of %d chars omitted. %s]\n\n%s", head, omitted, len(content), hint, tail)
}

// truncationHint: there is deliberately no tool that re-serves the cached full
// output; advertising it cost more history than it saved.
func truncationHint(toolName, args string) string {
	a := parseArgs(args)
	switch toolName {
	case "web_read":
		if u := a["url"]; u != "" {
			return fmt.Sprintf("To see more: call %s again with url=%q offset=<n> limit=<m>. The full body is cached, so nothing is re-fetched.", toolName, u)
		}
		return "To see more: call this tool again with offset=<n> limit=<m>. The full body is cached, so nothing is re-fetched."
	case "run_command":
		return "To see more: re-run it with the output narrowed (`| grep <pattern>`, `| tail -n <n>`, `| head -n <n>`), or redirect it to a file and read_file that. If re-running is slow or has side effects, redirect to a file the FIRST time."
	case "web_search":
		return "To see more: refine the query (fewer, more specific terms) and search again, then web_read the most promising result."
	default:
		return "To see more: call the tool again with narrower arguments."
	}
}
