package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

const maxSubagentDepth = 2

type subagentTask struct {
	Instructions string `json:"instructions"`
	Context      string `json:"context,omitempty"`
}

type subagentResult struct {
	Index   int    `json:"index"`
	Success bool   `json:"success"`
	Result  string `json:"result"`
	Error   string `json:"error,omitempty"`
}

// registerSubagentTool registers the launch_subagent tool if not already registered.
func (a *agent) registerSubagentTool() {
	for _, t := range registeredTools {
		fn, _ := t.Def["function"].(map[string]any)
		if fn["name"] == "launch_subagent" {
			return
		}
	}

	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "launch_subagent",
			"description": "Launch one or more subagents to work on independent subtasks in parallel. Each subagent gets its own plan/execute/verify cycle with access to all project tools. You SHOULD use this when the plan has 2+ independent steps that don't depend on each other (e.g. searching for two different things, editing unrelated files, researching while coding).",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"tasks"},
				"properties": map[string]any{
					"tasks": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"instructions": map[string]any{"type": "string", "description": "What this subagent should do"},
								"context":      map[string]any{"type": "string", "description": "Relevant context from the parent task (file contents, previous results, etc.)"},
							},
							"required": []string{"instructions"},
						},
					},
				},
			},
		},
	}, Execute: func(ctx context.Context, ag *agent, sid SessionId, rawArgs string) string {
		// Parse tasks from raw JSON arguments.
		var parsed struct {
			Tasks []subagentTask `json:"tasks"`
		}
		if err := json.Unmarshal([]byte(rawArgs), &parsed); err != nil {
			return "error: invalid JSON: " + err.Error()
		}
		tasks := parsed.Tasks
		if len(tasks) == 0 {
			return "error: no tasks provided"
		}

		// Check depth.
		sess := ag.getSession(sid)
		if sess == nil {
			return "error: no session"
		}
		if sess.Depth >= maxSubagentDepth {
			return "error: maximum subagent nesting depth reached"
		}

		tcId := ag.StartToolCall(ctx, sid, fmt.Sprintf("Launching %d subagent(s)", len(tasks)), "execute", nil)

		// Launch subagents in parallel.
		results := make([]subagentResult, len(tasks))
		var mu sync.Mutex

		parallel(len(tasks), func(i int) {
			task := tasks[i]
			subSess := newSubagentSession(sess.Cwd, sid, i, sess.Depth+1)
			ag.putSession(subSess)

			ag.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("[subagent %d] Starting: %s\n", i+1, truncate(task.Instructions, 100)))))

			// Use parent session ID for UI so Zed can show updates/permissions.
			result, err := ag.runSubagent(ctx, sid, subSess, task, sid)

			mu.Lock()
			if err != nil {
				results[i] = subagentResult{Index: i, Success: false, Error: err.Error()}
			} else {
				results[i] = subagentResult{Index: i, Success: true, Result: result}
			}
			mu.Unlock()

			ag.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("[subagent %d] Done\n", i+1))))
		})

		// Format results.
		var out strings.Builder
		for _, r := range results {
			if r.Success {
				fmt.Fprintf(&out, "=== Subagent %d ===\n%s\n\n", r.Index+1, r.Result)
			} else {
				fmt.Fprintf(&out, "=== Subagent %d (FAILED) ===\n%s\n\n", r.Index+1, r.Error)
			}
		}

		ag.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("%d subagent(s) completed", len(tasks)))})
		return out.String()
	}})
}

// runSubagent executes a full plan→execute→verify cycle for a subagent task.
func (a *agent) runSubagent(ctx context.Context, parentSid SessionId, subSess *Session, task subagentTask, uiSid SessionId) (string, error) {
	// Build the instruction with optional context.
	instructions := task.Instructions
	if task.Context != "" {
		instructions = "Context:\n" + task.Context + "\n\nTask:\n" + task.Instructions
	}

	subSess.AddUser(instructions)
	_ = subSess.Save()

	// Plan (auto-approve).
	conn, planSteps, _, err := a.planForSubagent(ctx, uiSid, instructions)
	if err != nil {
		return "", err
	}

	// Build execution content.
	content := instructions
	if len(planSteps) > 0 {
		var planCtx strings.Builder
		planCtx.WriteString("Follow these steps exactly:\n")
		for i, step := range planSteps {
			fmt.Fprintf(&planCtx, "%d. %s\n", i+1, step)
		}
		planCtx.WriteString("\nTask: ")
		planCtx.WriteString(instructions)
		content = planCtx.String()
	}

	// Add system prompt.
	sysPrompt, err := a.systemPrompt(subSess.ID)
	if err == nil && sysPrompt != "" {
		content = sysPrompt + "\n---\n" + content
	}

	messages := []llmMessage{{Role: "user", Content: content}}

	// Exclude launch_subagent if at max depth. Web tools are excluded by execute().
	var extraExclude []string
	if subSess.Depth >= maxSubagentDepth {
		extraExclude = append(extraExclude, "launch_subagent")
	}

	result, err := a.execute(ctx, uiSid, messages, extraExclude...)
	if err != nil {
		return result.Text, err
	}

	// Verify.
	result, _, err = a.verify(ctx, uiSid, conn, messages, result, instructions, planSteps)
	if err != nil {
		return result.Text, err
	}

	// Save subagent session.
	if result.Text != "" {
		subSess.AddAssistantWithTools(result.Text, result.ToolUses)
		_ = subSess.Save()
	}

	return result.Text, nil
}
