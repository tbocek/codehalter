package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
)

// This file owns the four phases of a prompt cycle: plan → execute → verify →
// document. Each phase is a thin orchestration layer over runToolLoop; the
// orchestrator in prompt.go calls them in order and re-plans on verify failure.

// ---------------------------------------------------------------------------
// Plan phase
// ---------------------------------------------------------------------------

type planResult struct {
	Clear    bool     `json:"clear"`
	Choices  []string `json:"choices"`
	Question string   `json:"question"`
	Steps    []string `json:"steps"`
	// Subtasks lists top-level pieces the planner wants to decompose the
	// request into. When non-empty, Prompt treats each entry as its own
	// plan→execute→verify cycle (see runSubtasks). Steps should be empty
	// in this case — each subtask's own planner pass produces its steps.
	Subtasks []string `json:"subtasks"`
}

// runPlanLLM runs the planning LLM and parses the JSON response. Returns
// (nil, nil, nil, err) when no LLM is configured. Returns (thinking, nil, ...)
// when there's no PLAN.md, the tool loop fails, or the response can't be
// parsed — callers treat any of those as "no plan, proceed without one".
func (a *agent) runPlanLLM(ctx context.Context, sid SessionId, userText string) (*LLMConnection, *planResult, []ToolUse, error) {
	thinking := a.settings.LLMFor("thinking", a.llmTier(sid))
	if thinking == nil {
		return nil, nil, nil, fmt.Errorf("no [[llmconnections]] in .codehalter/settings.toml")
	}
	planPrompt := a.loadPromptFile(sid, "PLAN.md")
	if planPrompt == "" {
		return thinking, nil, nil, nil
	}
	messages := []llmMessage{{Role: "user", Content: planPrompt + "\n\nUser request: " + userText}}
	planRes, err := a.runToolLoop(ctx, sid, thinking, messages, toolFilter{readOnly: true})
	if err != nil {
		return thinking, nil, planRes.ToolUses, nil
	}
	var plan planResult
	if err := json.Unmarshal([]byte(trimJSON(planRes.Text)), &plan); err != nil {
		return thinking, nil, planRes.ToolUses, nil
	}
	return thinking, &plan, planRes.ToolUses, nil
}

// renderSteps shows the plan's steps to the user and returns the rendered
// text so the caller can fold it into the session transcript on cancel.
func (a *agent) renderSteps(ctx context.Context, sid SessionId, steps []string) string {
	if len(steps) == 0 {
		return ""
	}
	var planText strings.Builder
	planText.WriteString("Plan:\n")
	for i, step := range steps {
		fmt.Fprintf(&planText, "%d. %s\n", i+1, step)
	}
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(planText.String())))
	return planText.String()
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
	thinking, plan, toolUses, err := a.runPlanLLM(ctx, sid, userText)
	if err != nil || plan == nil {
		return thinking, plan, toolUses, err
	}
	sess := a.getSession(sid)

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
			return nil, nil, toolUses, fmt.Errorf("user aborted")
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
		return thinking, plan, toolUses, nil
	}

	// Show the plan and ask for confirmation.
	if len(plan.Steps) > 0 {
		planText := a.renderSteps(ctx, sid, plan.Steps)

		tcId := a.StartToolCall(ctx, sid, "Execute this plan?", "think", nil)
		ok, err := a.askYesNoAuto(ctx, sid, tcId, "Execute", "Cancel", true)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("User chose: %v", ok))})

		if err != nil || !ok {
			if sess != nil {
				sess.AddAssistant(planText + "\nUser declined execution.")
				_ = sess.Save()
			}
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Cancelled.\n")))
			return nil, nil, toolUses, fmt.Errorf("user declined execution")
		}
	}

	return thinking, plan, toolUses, nil
}

// planForSubagent is like planAndRoute but auto-approves without user interaction.
func (a *agent) planForSubagent(ctx context.Context, sid SessionId, instructions string) (*LLMConnection, []string, []ToolUse, error) {
	thinking, plan, toolUses, err := a.runPlanLLM(ctx, sid, instructions)
	if err != nil || plan == nil {
		return thinking, nil, toolUses, err
	}
	a.renderSteps(ctx, sid, plan.Steps)
	return thinking, plan.Steps, toolUses, nil
}

// ---------------------------------------------------------------------------
// Execute phase
// ---------------------------------------------------------------------------

// execute runs the execution phase. It prepends EXECUTE.md (if present) to the
// last user message, excludes web tools (information retrieval belongs to the
// planning phase), and runs the agentic tool loop with write-enabled tools.
// extraExclude lists additional tool names to exclude (e.g. launch_subagent
// when a subagent has reached its max nesting depth).
func (a *agent) execute(ctx context.Context, sid SessionId, messages []llmMessage, extraExclude ...string) (toolLoopResult, error) {
	if executeMD := a.loadPromptFile(sid, "EXECUTE.md"); executeMD != "" && len(messages) > 0 {
		last := len(messages) - 1
		if messages[last].Role == "user" {
			if content, ok := messages[last].Content.(string); ok {
				messages[last].Content = executeMD + "\n\n---\n\n" + content
			}
		}
	}
	exclude := map[string]bool{"web_search": true, "web_read": true, "web_read_raw": true}
	for _, name := range extraExclude {
		exclude[name] = true
	}
	return a.runToolLoop(ctx, sid, a.settings.LLMFor("execute", a.llmTier(sid)), messages, toolFilter{
		readOnly: false,
		exclude:  exclude,
	})
}

// ---------------------------------------------------------------------------
// Verify phase
// ---------------------------------------------------------------------------

type verifyResult struct {
	Success  bool     `json:"success"`
	Issues   []string `json:"issues"`
	FixSteps []string `json:"fix_steps,omitempty"`
}

const maxVerifyAttempts = 2

// preVerify performs a read-only check of the project state using VERIFY.md
// before planning begins. Returns the verification result if issues were found.
func (a *agent) preVerify(ctx context.Context, sid SessionId, conn *LLMConnection, userText string) (*verifyResult, error) {
	verifyPrompt := a.loadPromptFile(sid, "VERIFY.md")
	if verifyPrompt == "" {
		return &verifyResult{Success: true}, nil
	}

	var prompt strings.Builder
	prompt.WriteString(verifyPrompt)
	prompt.WriteString("\n\nUser request: ")
	prompt.WriteString(userText)
	prompt.WriteString("\n\nPerform a pre-planning check to see if the current project state is broken or has issues related to this request. Return JSON.")

	messages := []llmMessage{{Role: "user", Content: prompt.String()}}
	verifyRes, err := a.runToolLoop(ctx, sid, conn, messages, toolFilter{
		readOnly: true,
		include:  map[string]bool{"run_task": true},
	})
	if err != nil {
		return nil, err
	}

	trimmed := trimJSON(verifyRes.Text)
	var result verifyResult
	if err := json.Unmarshal([]byte(trimmed), &result); err != nil {
		slog.Error("preVerify: non-JSON response", "err", err, "snippet", truncate(trimmed, 200))
		return nil, fmt.Errorf("preVerify non-JSON: %w", err)
	}

	return &result, nil
}

// verify runs a self-check after execution. If the LLM finds issues, it gets
// another chance to fix them. Returns the final result and the verify outcome.
func (a *agent) verify(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, res toolLoopResult, userText string, planSteps []string) (toolLoopResult, *verifyResult, error) {
	verifyPrompt := a.loadPromptFile(sid, "VERIFY.md")
	if verifyPrompt == "" {
		return res, &verifyResult{Success: true}, nil
	}

	for attempt := 0; attempt < maxVerifyAttempts; attempt++ {
		// Build verification context.
		var prompt strings.Builder
		prompt.WriteString(verifyPrompt)
		prompt.WriteString("\n\nUser request: ")
		prompt.WriteString(userText)
		if len(planSteps) > 0 {
			prompt.WriteString("\n\nApproved plan:\n")
			for i, step := range planSteps {
				fmt.Fprintf(&prompt, "%d. %s\n", i+1, step)
			}
		}
		prompt.WriteString("\n\nYour response was:\n")
		prompt.WriteString(res.Text)

		// Run verify with read-only tools + run_task.
		verifyMessages := append(messages, llmMessage{Role: "user", Content: prompt.String()})
		verifyRes, err := a.runToolLoop(ctx, sid, conn, verifyMessages, toolFilter{
			readOnly: true,
			include:  map[string]bool{"run_task": true},
		})
		if err != nil {
			slog.Error("verify: LLM call failed; treating as success", "err", err)
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ Verification skipped: "+err.Error()+"\n")))
			return res, &verifyResult{Success: true}, nil
		}

		trimmed := trimJSON(verifyRes.Text)

		var result verifyResult
		if err := json.Unmarshal([]byte(trimmed), &result); err != nil {
			slog.Error("verify: LLM returned non-JSON response; treating as success",
				"err", err, "snippet", truncate(trimmed, 200))
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ Verification returned non-JSON; skipping self-check.\n")))
			return res, &verifyResult{Success: true}, nil
		}

		if result.Success {
			res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
			return res, &result, nil
		}

		// If there are fix_steps, return to the caller for a full retry cycle.
		if len(result.FixSteps) > 0 {
			res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
			return res, &result, nil
		}

		// No fix_steps — try an inline fix within this attempt.
		var issueText strings.Builder
		issueText.WriteString("⚠ Self-check found issues:\n")
		for _, issue := range result.Issues {
			fmt.Fprintf(&issueText, "- %s\n", issue)
		}
		issueText.WriteString("\nFixing...\n")
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(issueText.String())))

		fixPrompt := "Your previous response had these issues:\n"
		for _, issue := range result.Issues {
			fixPrompt += "- " + issue + "\n"
		}
		fixPrompt += "\nPlease fix these issues now."

		messages = append(messages,
			llmMessage{Role: "assistant", Content: res.Text},
			llmMessage{Role: "user", Content: fixPrompt},
		)

		fixRes, err := a.runToolLoop(ctx, sid, conn, messages, toolFilter{})
		if err != nil {
			slog.Error("verify: inline fix pass failed", "err", err)
			return res, &result, nil
		}
		res.Text = res.Text + "\n" + fixRes.Text
		res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
		res.ToolUses = append(res.ToolUses, fixRes.ToolUses...)
	}

	return res, &verifyResult{Success: true}, nil
}

// ---------------------------------------------------------------------------
// Document phase
// ---------------------------------------------------------------------------

// hasFileWrites reports whether any of the tool uses wrote to the filesystem.
// The document phase is gated on this so read-only turns (questions, lookups)
// don't trigger a doc-update LLM call.
func hasFileWrites(uses []ToolUse) bool {
	for _, u := range uses {
		if u.Name == "write_file" || u.Name == "edit_file" {
			return true
		}
	}
	return false
}

// document runs after a successful verify when the turn actually wrote files.
// It hands the executor's response to the thinking LLM with DOCUMENT.md and a
// read-only tool set augmented with write_file/edit_file so the LLM can update
// (or create) the project README when the change is user-visible. readOnly:true
// suppresses chat streaming so the user doesn't see "no doc change needed"
// noise on routine turns.
func (a *agent) document(ctx context.Context, sid SessionId, conn *LLMConnection, userText string, planSteps []string, exec toolLoopResult) (toolLoopResult, error) {
	docPrompt := a.loadPromptFile(sid, "DOCUMENT.md")
	if docPrompt == "" {
		return exec, nil
	}

	var prompt strings.Builder
	prompt.WriteString(docPrompt)
	prompt.WriteString("\n\nUser request: ")
	prompt.WriteString(userText)
	if len(planSteps) > 0 {
		prompt.WriteString("\n\nApproved plan:\n")
		for i, step := range planSteps {
			fmt.Fprintf(&prompt, "%d. %s\n", i+1, step)
		}
	}
	prompt.WriteString("\n\nExecutor response:\n")
	prompt.WriteString(exec.Text)

	messages := []llmMessage{{Role: "user", Content: prompt.String()}}
	docRes, err := a.runToolLoop(ctx, sid, conn, messages, toolFilter{
		readOnly: true,
		include: map[string]bool{
			"write_file": true,
			"edit_file":  true,
		},
	})
	if err != nil {
		slog.Warn("document phase failed", "err", err)
		return exec, nil
	}
	exec.ToolUses = append(exec.ToolUses, docRes.ToolUses...)
	return exec, nil
}
