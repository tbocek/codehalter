package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type planResult struct {
	Clear      bool     `json:"clear"`
	Choices    []string `json:"choices"`
	Question   string   `json:"question"`
	Complexity string   `json:"complexity"`
	Steps      []string `json:"steps"`
	// Subtasks lists top-level pieces the planner wants to decompose the
	// request into. When non-empty, Prompt treats each entry as its own
	// plan→execute→verify cycle (see runSubtasks). Steps should be empty
	// in this case — each subtask's own planner pass produces its steps.
	Subtasks []string `json:"subtasks"`
}

func (a *agent) loadPlanPrompt(sid SessionId) string {
	return a.loadPromptFile(sid, "PLAN.md")
}

// planAndRoute analyzes the user's request, asks for clarification if needed,
// shows the plan, and asks the user to confirm before execution.
//
// Returns the LLM connection, the parsed plan (may be nil if parsing failed
// or there is no PLAN.md), the tool uses performed during planning, and an
// error. When the plan has Subtasks, the "Execute this plan?" confirmation
// is skipped — the caller (Prompt → runSubtasks) handles a one-shot
// Interactive/Automatic/Cancel prompt for the whole batch instead.
func (a *agent) planAndRoute(ctx context.Context, sid SessionId, userText string) (*LLMConnection, *planResult, []ToolUse, error) {
	thinking := a.settings.LLM("thinking")
	if thinking == nil {
		return nil, nil, nil, fmt.Errorf("no 'thinking' connection in .codehalter/settings.toml")
	}

	planPrompt := a.loadPlanPrompt(sid)
	if planPrompt == "" {
		return thinking, nil, nil, nil
	}

	// Build planner prompt with AGENT.md rules and project context.
	sess := a.getSession(sid)
	var prompt strings.Builder
	if sess != nil {
		if agentMD, err := os.ReadFile(filepath.Join(sess.Cwd, ".codehalter", "AGENT.md")); err == nil {
			prompt.Write(agentMD)
			prompt.WriteString("\n\n")
		}
	}
	prompt.WriteString(planPrompt)
	if sess != nil {
		projCtx := buildProjectContext(sess.Cwd, a.fileCache)
		if projCtx != "" {
			prompt.WriteString("\n\n")
			prompt.WriteString(projCtx)
		}
	}
	prompt.WriteString("\n\nUser request: ")
	prompt.WriteString(userText)

	// Ask thinking to plan with read-only tools available.
	messages := []llmMessage{{Role: "user", Content: prompt.String()}}
	planRes, err := a.runToolLoop(ctx, sid, thinking, messages, true)
	if err != nil {
		return thinking, nil, planRes.ToolUses, nil
	}
	response := planRes.Text

	response = trimJSON(response)

	var plan planResult
	if err := json.Unmarshal([]byte(response), &plan); err != nil {
		return thinking, nil, planRes.ToolUses, nil
	}

	// If the request is unclear, ask the user.
	if !plan.Clear && len(plan.Choices) > 0 {
		question := plan.Question
		if question == "" {
			question = "I'm not sure what you mean. Which of these?"
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(question)))

		tcId := a.StartToolCall(ctx, sid, "Clarification needed", "think", nil)
		choice, err := a.askChoiceAuto(ctx, sid, tcId, plan.Choices, 0)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + choice)})

		if err != nil || choice == "abort" {
			if sess != nil {
				sess.AddAssistant(question + "\n(choices: " + strings.Join(plan.Choices, ", ") + ")\nUser aborted.")
				_ = sess.Save()
			}
			return nil, nil, planRes.ToolUses, fmt.Errorf("user aborted")
		}

		if sess != nil {
			sess.AddAssistant(question + "\n(choices: " + strings.Join(plan.Choices, ", ") + ")\nUser chose: " + choice)
			_ = sess.Save()
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Understood: "+choice+"\n")))
	}

	// If the planner decomposed the request into subtasks, skip the per-plan
	// "Execute this plan?" prompt — the caller will show the subtask list and
	// ask Interactive/Automatic/Cancel for the whole batch instead.
	if len(plan.Subtasks) > 0 {
		return thinking, &plan, planRes.ToolUses, nil
	}

	// Show the plan and ask for confirmation.
	if len(plan.Steps) > 0 {
		var planText strings.Builder
		planText.WriteString("Plan:\n")
		for i, step := range plan.Steps {
			fmt.Fprintf(&planText, "%d. %s\n", i+1, step)
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(planText.String())))

		tcId := a.StartToolCall(ctx, sid, "Execute this plan?", "think", nil)
		ok, err := a.askYesNoAuto(ctx, sid, tcId, "Execute", "Cancel", true)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("User chose: %v", ok))})

		if err != nil || !ok {
			if sess != nil {
				sess.AddAssistant(planText.String() + "\nUser declined execution.")
				_ = sess.Save()
			}
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Cancelled.\n")))
			return nil, nil, planRes.ToolUses, fmt.Errorf("user declined execution")
		}
	}

	return thinking, &plan, planRes.ToolUses, nil
}

// planForSubagent is like planAndRoute but auto-approves without user interaction.
func (a *agent) planForSubagent(ctx context.Context, sid SessionId, instructions string) (*LLMConnection, []string, []ToolUse, error) {
	thinking := a.settings.LLM("thinking")
	if thinking == nil {
		return nil, nil, nil, fmt.Errorf("no 'thinking' connection")
	}

	planPrompt := a.loadPlanPrompt(sid)
	if planPrompt == "" {
		return thinking, nil, nil, nil
	}

	sess := a.getSession(sid)
	var prompt strings.Builder
	if sess != nil {
		if agentMD, err := os.ReadFile(filepath.Join(sess.Cwd, ".codehalter", "AGENT.md")); err == nil {
			prompt.Write(agentMD)
			prompt.WriteString("\n\n")
		}
	}
	prompt.WriteString(planPrompt)
	if sess != nil {
		projCtx := buildProjectContext(sess.Cwd, a.fileCache)
		if projCtx != "" {
			prompt.WriteString("\n\n")
			prompt.WriteString(projCtx)
		}
	}
	prompt.WriteString("\n\nUser request: ")
	prompt.WriteString(instructions)

	messages := []llmMessage{{Role: "user", Content: prompt.String()}}
	planRes, err := a.runToolLoop(ctx, sid, thinking, messages, true)
	if err != nil {
		return thinking, nil, planRes.ToolUses, nil
	}

	var plan planResult
	if err := json.Unmarshal([]byte(trimJSON(planRes.Text)), &plan); err != nil {
		return thinking, nil, planRes.ToolUses, nil
	}

	// Auto-approve — show the plan but don't ask.
	if len(plan.Steps) > 0 {
		var planText strings.Builder
		planText.WriteString("Plan:\n")
		for i, step := range plan.Steps {
			fmt.Fprintf(&planText, "%d. %s\n", i+1, step)
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(planText.String())))
	}

	return thinking, plan.Steps, planRes.ToolUses, nil
}
