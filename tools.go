package main

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"sync/atomic"
)

// ---------------------------------------------------------------------------
// Path security
// ---------------------------------------------------------------------------

func (a *agent) resolvePath(sid SessionId, path string) (string, error) {
	sess := a.getSession(sid)
	if sess == nil {
		return "", fmt.Errorf("no session found")
	}
	var resolved string
	if filepath.IsAbs(path) {
		resolved = filepath.Clean(path)
	} else {
		resolved = filepath.Clean(filepath.Join(sess.Cwd, path))
	}
	if !strings.HasPrefix(resolved, sess.Cwd+string(filepath.Separator)) && resolved != sess.Cwd {
		return "", fmt.Errorf("path %q is outside project directory", path)
	}
	return resolved, nil
}

// ---------------------------------------------------------------------------
// Tool registry
// ---------------------------------------------------------------------------

type Tool struct {
	Def      map[string]any
	ReadOnly bool // safe to use in discussion mode
	Execute  func(ctx context.Context, a *agent, sid SessionId, args map[string]string) string
}

var registeredTools []Tool

func RegisterTool(t Tool) {
	registeredTools = append(registeredTools, t)
}

func llmToolDefinitions(readOnly bool) []map[string]any {
	var defs []map[string]any
	for _, t := range registeredTools {
		if readOnly && !t.ReadOnly {
			continue
		}
		defs = append(defs, t.Def)
	}
	return defs
}

func (a *agent) executeTool(ctx context.Context, sid SessionId, tc toolCall) string {
	var args map[string]string
	_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)

	a.mu.Lock()
	mode := a.mode
	a.mu.Unlock()

	for _, t := range registeredTools {
		fn, _ := t.Def["function"].(map[string]any)
		if fn["name"] == tc.Function.Name {
			if mode == "discussion" && !t.ReadOnly {
				return fmt.Sprintf("tool %s is not available in discussion mode", tc.Function.Name)
			}
			return t.Execute(ctx, a, sid, args)
		}
	}
	return fmt.Sprintf("unknown tool: %s", tc.Function.Name)
}

// ---------------------------------------------------------------------------
// Tool call UI
// ---------------------------------------------------------------------------

var toolCallCounter atomic.Uint64

func nextToolCallID() string {
	return fmt.Sprintf("tc_%d", toolCallCounter.Add(1))
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
	b := TextBlock(text)
	return ToolCallContent{Type: "content", Content: &b}
}

func DiffContent(path string, oldText *string, newText string) ToolCallContent {
	return ToolCallContent{Type: "diff", Path: path, OldText: oldText, NewText: newText}
}

func (a *agent) StartToolCall(ctx context.Context, sid SessionId, title, kind string, locations []ToolCallLocation) string {
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

func (a *agent) CompleteToolCall(ctx context.Context, sid SessionId, id string, content []ToolCallContent) {
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: id,
		Status:     "completed",
		Content:    content,
	})
}

func (a *agent) FailToolCall(ctx context.Context, sid SessionId, id, errMsg string) {
	a.sendUpdate(ctx, sid, toolCallUpdate{
		Kind:       "tool_call_update",
		ToolCallId: id,
		Status:     "failed",
		Content:    []ToolCallContent{TextContent("❌ " + errMsg)},
	})
}
