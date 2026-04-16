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
			"description": "Launch one or more subagents to work on independent subtasks in parallel. Each subagent gets its own plan/execute/verify cycle with access to the project's code tools. Use this when the plan has 2+ independent steps that don't depend on each other (e.g. searching for two different things, editing unrelated files, researching while coding). IMPORTANT: subagents cannot talk to the user — they cannot ask clarifying questions or request confirmation. Only launch a subagent when the task is unambiguous and self-contained. If you would need to ask the user anything to complete the task, do it yourself instead.",
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
			defer ag.deleteSession(subSess.ID)

			ag.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("[subagent %d] Starting: %s\n\n", i+1, truncate(task.Instructions, 100)))))

			// All internal plan/execute/verify calls use the subagent's own
			// session id. That keeps plan steps, tool uses, and assistant
			// output in subSess.toml — and because Zed does not know about
			// that session id, every session/update notification the tool
			// loop emits is silently dropped. The subagent therefore runs
			// without polluting the parent conversation; we surface only the
			// final result (via the tool return value below) to the parent.
			result, err := ag.runSubagent(ctx, subSess, task)

			mu.Lock()
			if err != nil {
				results[i] = subagentResult{Index: i, Success: false, Error: err.Error()}
			} else {
				results[i] = subagentResult{Index: i, Success: true, Result: result}
			}
			mu.Unlock()

			ag.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("[subagent %d] Done\n\n", i+1))))
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

// runSubagent executes a full plan→execute→verify cycle against subSess. All
// internal calls use subSess.ID so tool uses, plan steps, and assistant text
// are saved there (not in the parent). UI updates emitted by the tool loop
// target subSess.ID too and are dropped by the client since it has never been
// announced — only the returned result string flows back to the parent.
func (a *agent) runSubagent(ctx context.Context, subSess *Session, task subagentTask) (string, error) {
	sid := subSess.ID

	instructions := task.Instructions
	if task.Context != "" {
		instructions = "Context:\n" + task.Context + "\n\nTask:\n" + task.Instructions
	}

	subSess.AddUser(instructions)
	_ = subSess.Save()

	// Plan (auto-approve).
	conn, planSteps, _, err := a.planForSubagent(ctx, sid, instructions)
	if err != nil {
		return "", err
	}

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

	sysPrompt, err := a.systemPrompt(sid)
	if err == nil && sysPrompt != "" {
		content = sysPrompt + "\n---\n" + content
	}

	messages := []llmMessage{{Role: "user", Content: content}}

	// Subagents can't interact with the user (UI is dropped), so ask_user
	// would hang. launch_subagent is excluded at max depth to bound recursion.
	extraExclude := []string{"ask_user"}
	if subSess.Depth >= maxSubagentDepth {
		extraExclude = append(extraExclude, "launch_subagent")
	}

	result, err := a.execute(ctx, sid, messages, extraExclude...)
	if err != nil {
		return result.Text, err
	}

	result, _, err = a.verify(ctx, sid, conn, messages, result, instructions, planSteps)
	if err != nil {
		return result.Text, err
	}

	// The tool loop incrementally appends tool uses to the last assistant
	// message via AppendToolUse; here we just set that message's Content to
	// the final text (or add a new assistant message if none exists).
	if result.Text != "" {
		if n := len(subSess.Messages); n > 0 && subSess.Messages[n-1].Role == "assistant" {
			subSess.Messages[n-1].Content = result.Text
		} else {
			subSess.AddAssistantWithTools(result.Text, result.ToolUses)
		}
		_ = subSess.Save()
	}

	return result.Text, nil
}
