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
	var final string
	if filepath.IsAbs(path) {
		final = filepath.Clean(path)
	} else {
		final = filepath.Clean(filepath.Join(sess.Cwd, path))
		// LLMs sometimes write absolute-looking paths with a missing leading "/"
		// (e.g. `workspaces/preveltekit/go.mod` when cwd is /workspaces/preveltekit).
		// If the cwd-joined interpretation doesn't exist on disk but the "/"-
		// prepended one does and stays inside cwd, prefer that.
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
	// inside() above is a string-prefix test, which a symlink LIVING under cwd but
	// TARGETING outside it would defeat (e.g. `ln -s /etc cwd/x` then read x/passwd).
	// Re-check after resolving symlinks on the path's deepest existing ancestor.
	if !realInside(final, sess.Cwd) {
		return "", fmt.Errorf("path %q resolves through a symlink outside the project directory", path)
	}
	return final, nil
}

// realInside reports whether p stays within cwd after symlink resolution. It
// resolves the deepest EXISTING ancestor of p (a write target may not exist yet,
// and a not-yet-created tail can't itself be a symlink) and re-checks the prefix
// against the resolved cwd, so an in-tree symlink to an out-of-tree target is
// rejected while a symlink to another in-tree path still works.
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
			return false // walked to the root with nothing resolvable
		}
		dir = parent
	}
}

// ---------------------------------------------------------------------------
// Tool registry
// ---------------------------------------------------------------------------

type Tool struct {
	Def map[string]any
	// Execute returns the tool's output and a `failed` flag: the authoritative
	// signal that the operation itself failed (run_command saw a non-zero exit).
	// It becomes ToolUse.Failed, which lets the orchestrator override a model's
	// "success". edit_file/write_file set it on a usage error too, to feed the
	// fail cap, but the subtask verdict excludes file-mutation tools, so a
	// recovered edit does not condemn the subtask.
	Execute func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool)
}

// phasePolicy is a phase's per-call rules, enforced at DISPATCH instead of by
// pruning the tools array — so the array stays a byte-identical superset across
// phases (only MCP reconcile changes it) and the KV-cache prefix survives
// plan↔execute↔document. deny rejects a call with a teaching message; terminals
// are the tools whose call ends the loop (a phase may have several — execute
// exits on respond OR submit_plan).
type phasePolicy struct {
	deny      map[string]bool
	terminals map[string]bool
}

// terminalList renders the phase's terminal tools for a model-facing nudge,
// e.g. "`respond` or `submit_plan`". Sorted for a stable message.
func terminalList(p phasePolicy) string {
	names := make([]string, 0, len(p.terminals))
	for n := range p.terminals {
		names = append(names, "`"+n+"`")
	}
	sort.Strings(names)
	return strings.Join(names, " or ")
}

// toolName is the function name a tool is offered and called under.
func toolName(t Tool) string {
	fn, _ := t.Def["function"].(map[string]any)
	name, _ := fn["name"].(string)
	return name
}

// toolRegistry is the agent's tool set, keyed by name. Adding a name that is
// already there replaces it, so discovery running again for the next session
// cannot offer a tool twice. The zero value holds the built-ins (seedLocked),
// which is what a test fixture's zero agent starts with too.
//
// Most adds happen before any turn, but the MCP reconciler adds and removes
// tools on its own goroutine while ACP may run other sessions' turns, so the
// lock is needed.
type toolRegistry struct {
	mu    sync.Mutex
	tools map[string]Tool
}

// seedLocked gives a registry that was never used the built-ins. Caller holds mu.
func (r *toolRegistry) seedLocked() {
	if r.tools != nil {
		return
	}
	r.tools = make(map[string]Tool)
	// Every tool codehalter provides in any project. Discovery adds the
	// project's own (run_command and run_background inside a container) and
	// MCP adds its servers', both at runtime through add.
	builtin := slices.Concat(fileTools, webTools, []Tool{
		searchTextTool, askUserTool, submitPlanTool, respondTool,
		insightsTool, screenshotTool, viewImageTool,
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

// removePrefix drops every tool whose name starts with prefix; the MCP
// reconciler uses it when a server stops or its config changes.
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

// lookup returns the named tool, or ok=false and every tool's name, sorted.
func (r *toolRegistry) lookup(name string) (t Tool, ok bool, names []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seedLocked()
	if t, ok = r.tools[name]; !ok {
		names = slices.Sorted(maps.Keys(r.tools))
	}
	return t, ok, names
}

// defs returns EVERY tool's definition, sorted by name. Phases do not prune
// this (phasePolicy restricts calls at dispatch instead), and the order does not
// depend on when a tool was added (the MCP reconciler re-adds a server's tools
// in no particular order), so the rendered `tools` block is byte-identical
// across phases and turns. Only a change to the set itself alters it; anything
// else would change the prompt prefix and bust the server's KV cache.
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

// toolArgs is one decoded tool-call argument object. It is map[string]any, NOT
// map[string]string, because the schemas declare integers and booleans: a
// model that obeys them sends `{"line": 42}`, and decoding that into strings
// fails the whole object while leaving the key "". The tool then read from
// line 1 and reported success. Values are coerced per key at the point of use.
type toolArgs map[string]any

// parseArgs decodes a tool call's raw JSON arguments. A malformed payload
// yields an empty (non-nil) map; each tool's own validation reports the missing
// parameter, which is a better message than a JSON parse error.
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

// str returns a string-typed argument, or "" when it's absent or some other
// JSON type. Deliberately does NOT stringify a number or bool: for the keys that
// carry file content (write_file's `content`, edit_file's `old_text`/`new_text`)
// a coerced value would clobber a file, so callers pair this with wrongType.
func (a toolArgs) str(key string) string {
	s, _ := a[key].(string)
	return s
}

// wrongType reports whether `key` is present with a JSON type other than string.
// write_file / edit_file reject the call on this rather than writing a
// zero-byte file and reporting success.
func (a toolArgs) wrongType(key string) bool {
	v, ok := a[key]
	if !ok {
		return false
	}
	_, isStr := v.(string)
	return !isStr
}

// num returns an integer argument. Accepts both the schema-correct JSON number
// and a quoted digit string, since models emit either; ok is false when the key
// is absent or holds neither, so callers can tell "omitted" from "passed 0".
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

// flag returns a boolean argument, accepting the JSON literal `true` and the
// string "true" alike. Anything else, including absence, is false.
func (a toolArgs) flag(key string) bool {
	switch v := a[key].(type) {
	case bool:
		return v
	case string:
		return strings.EqualFold(strings.TrimSpace(v), "true")
	}
	return false
}

// has reports whether the key was supplied at all, whatever its type. Used
// where the presence of an argument changes behaviour independently of its
// value (web_read treats any `limit` as a range request).
func (a toolArgs) has(key string) bool {
	_, ok := a[key]
	return ok
}

func (a *agent) executeTool(ctx context.Context, sid string, tc toolCall) (string, bool) {
	slog.Info("executeTool", "tool", tc.Function.Name, "sid", sid, "args", tc.Function.Arguments)

	// Run outside the registry lock: a tool may take seconds (a build, a web
	// read) and must not block the MCP reconciler.
	t, ok, names := a.tools.lookup(tc.Function.Name)
	if ok {
		return t.Execute(ctx, a, sid, tc.Function.Arguments)
	}
	// The names go back to the model as the tool result, so it can self-correct
	// on the next turn instead of looping on the same hallucination.
	return fmt.Sprintf("unknown tool %q. Use only the tools provided to you; available tools: %s",
		tc.Function.Name, strings.Join(names, ", ")), false
}

// ---------------------------------------------------------------------------
// Tool call UI
// ---------------------------------------------------------------------------

var toolCallCounter atomic.Uint64

// toolUseCounter assigns each recorded ToolUse a per-process handle. It names
// the call in session.toml and is the wire id of last resort for a model that
// sends no tool_call id of its own (see ToolUse.CallID).
var toolUseCounter atomic.Uint64

func nextToolUseID() string {
	return fmt.Sprintf("tu_%d", toolUseCounter.Add(1))
}

// runToolCall executes one tool call and returns the ToolUse recording its FULL
// output, plus the model-visible content: the output shrunk past
// truncateThreshold with a "to see more" hint, or view_image's multimodal
// parts. The full output goes to the session; the wire carries only the
// visible copy. Truncation lives here, so every tool returns everything and
// one place decides what the model sees.
func (a *agent) runToolCall(ctx context.Context, sid string, tc toolCall) (ToolUse, any) {
	started := time.Now()

	// Image short-circuit: when the server supports images, deliver the bytes
	// as multimodal tool content in the SAME turn (so the next llmStream call
	// sees the image). The standard executeTool path returns text only. When
	// it does NOT support images these fall through to executeTool, whose
	// fallback says so instead of pretending bytes were delivered.
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
		result, failed = a.executeTool(ctx, sid, tc)
	}

	useID := nextToolUseID()
	tu := ToolUse{
		ID:         useID,
		CallID:     tc.ID, // the model's own tool_call id — replayed verbatim from history
		Name:       tc.Function.Name,
		Input:      tc.Function.Arguments,
		Output:     result,
		Failed:     failed,
		StartedAt:  started,
		DurationMs: time.Since(started).Milliseconds(),
		ImageID:    imageID,
	}
	// Record the full output, saved incrementally so it survives a crash.
	if sess := a.getSession(sid); sess != nil {
		sess.AppendToolUse(tu)
		sess.saveOrLog()
	}

	if multimodal != nil {
		return tu, multimodal
	}
	return tu, liveToolOutput(tc.Function.Name, tc.Function.Arguments, result)
}

// denyToolCall rejects a policy-forbidden tool WITHOUT executing it: a failed
// tool card + a recorded rejection so the model corrects. Failed is for the
// record only — the caller doesn't feed it to the fail cap (the repetition
// ladder catches genuine spamming).
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
	tu := ToolUse{
		ID:        nextToolUseID(),
		CallID:    tc.ID,
		Name:      tc.Function.Name,
		Input:     tc.Function.Arguments,
		Output:    msg,
		Failed:    true,
		StartedAt: time.Now(),
	}
	if sess := a.getSession(sid); sess != nil {
		sess.AppendToolUse(tu)
		sess.saveOrLog()
	}
	return tu, msg
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

// TerminalContent embeds a terminal created with terminal/create into a tool
// call, so the client renders its output live instead of us relaying it as
// message chunks. Must be sent before terminal/release; the client keeps
// showing the output afterwards.
func TerminalContent(terminalId string) ToolCallContent {
	return ToolCallContent{Type: "terminal", TerminalId: terminalId}
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
	// liveExemptCap bounds the LIVE output of the truncation-exempt tools. They
	// self-bound by COUNT (150-line chunk / 100 matches) but not by bytes — a
	// minified file or long-line matches could blow n_ctx. ~32 KB (20× the history
	// clip) is plenty to act on; clipped at a line boundary, never mid-line.
	liveExemptCap = 32 * 1024
)

// liveToolOutput is the model-visible content for a tool call. The content-
// retrieval tools (read_file/continue_read/search_text + web_search/web_read/
// web_read) pass through whole (up to the line-aware liveExemptCap); every
// other tool gets the 1.5 KB head/tail cap. history.go re-renders stored outputs
// through THIS same function, so a replay is byte-identical to the live wire
// (cache-warm) — re-sending a cached full read is free, whereas clipping it would
// change the bytes and force a reprocess. n_ctx is bounded by compaction instead.
func liveToolOutput(toolName, args, content string) string {
	switch toolName {
	case "read_file", "continue_read", "search_text", "web_search", "web_read":
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
			content[:cut], len(content)-cut, len(content), liveExemptCap/1024, truncationHint(toolName, args))
	}
	return truncateForLLM(toolName, args, content)
}

// truncateForLLM returns content unchanged if short; otherwise emits a
// head/tail-shaped slice with a per-tool "to see more" hint in the middle.
// toolName + args let the hint name the exact alternate follow-up call (e.g. read_file
// path+line, web_read url+offset) so the model doesn't have to reconstruct what
// it just asked about. Applied by liveToolOutput to the non-exempt tools (live
// and on history re-render alike).
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

// truncationHint returns the per-tool "to see more" pointer: how to get at
// the part that was cut. There is deliberately no tool that re-serves a cached
// full output: it was called after about 7% of truncations, and every hint paid
// ~300 bytes of history to advertise it. Narrowing the call is what the model
// does anyway, so the hint says how. parseArgs is best-effort; without a useful
// key the hint falls back to the generic wording.
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
	case "list_files":
		if path := a["path"]; path != "" {
			return fmt.Sprintf("To see more: call list_files on a subdirectory of %q.", path)
		}
		return "To see more: call list_files on a deeper subdirectory."
	case "search_text":
		return "To see more: re-run search_text with a more specific pattern or a narrower path."
	case "web_search":
		return "To see more: refine the query (fewer, more specific terms) and search again, then web_read the most promising result."
	default:
		return "To see more: call the tool again with narrower arguments."
	}
}
