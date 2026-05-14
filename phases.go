package main

import (
	"context"
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
	// ReportOnly is true when the planner already has the answer in hand
	// (pure-lookup tasks: "where is X?", "what version of Y?") and the
	// executor only needs to relay findings — no file edits, no commands.
	// When true, planAndRoute skips the "Execute this plan?" confirmation
	// because there is nothing destructive to gate. Default false → current
	// gating behavior preserved when the planner omits the field.
	ReportOnly bool `json:"report_only"`
}

// runPlanLLM runs the planning LLM and parses the JSON response. Returns
// (nil, nil, nil, err) when no LLM is configured. Returns (thinking, nil, ...)
// when there's no PLAN.md, the tool loop fails, or the response can't be
// parsed even after one corrective retry — callers treat any of those as
// "no plan, proceed without one".
func (a *agent) runPlanLLM(ctx context.Context, sid SessionId, userText string) (*LLMConnection, *planResult, []ToolUse, error) {
	thinking := a.pickAvailable(ctx, "thinking", a.llmTier(sid))
	if thinking == nil {
		return nil, nil, nil, fmt.Errorf("no [[llmconnections]] in .codehalter/settings.toml")
	}
	planPrompt := a.loadPromptFile(sid, "PLAN.md")
	if planPrompt == "" {
		return thinking, nil, nil, nil
	}
	messages := []llmMessage{{Role: "user", Content: planPrompt + "\n\nUser request: " + userText}}
	var plan planResult
	planRes, err := a.runToolLoopJSON(ctx, sid, thinking, messages, toolFilter{readOnly: true}, &plan)
	if err != nil {
		return thinking, nil, planRes.ToolUses, nil
	}
	return thinking, &plan, planRes.ToolUses, nil
}

// renderSteps shows the plan's steps to the user and returns the rendered
// text so the caller can fold it into the session transcript on cancel.
// header is the section heading ("Plan:" for actionable plans, "Findings:"
// for report-only plans where the executor will just relay the answer).
func (a *agent) renderSteps(ctx context.Context, sid SessionId, steps []string, header string) string {
	if len(steps) == 0 {
		return ""
	}
	var planText strings.Builder
	planText.WriteString(header + "\n")
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
			return nil, nil, toolUses, errUserCancelled
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
		header := "Plan:"
		if plan.ReportOnly {
			header = "Findings:"
		}
		planText := a.renderSteps(ctx, sid, plan.Steps, header)

		// Pure-lookup plans need no execution gate — the executor just
		// relays what the planner already gathered. PLAN.md tells the
		// planner to set report_only=true for these so we skip the prompt.
		if plan.ReportOnly {
			return thinking, plan, toolUses, nil
		}

		tcId := a.StartToolCall(ctx, sid, "Execute this plan?", "think", nil)
		ok, err := a.askYesNoAuto(ctx, sid, tcId, "Execute", "Cancel", true)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("User chose: %v", ok))})

		if err != nil || !ok {
			if sess != nil {
				sess.AddAssistant(planText + "\nUser declined execution.")
				_ = sess.Save()
			}
			return nil, nil, toolUses, errUserCancelled
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
	header := "Plan:"
	if plan.ReportOnly {
		header = "Findings:"
	}
	a.renderSteps(ctx, sid, plan.Steps, header)
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
	return a.runToolLoop(ctx, sid, a.pickAvailable(ctx, "execute", a.llmTier(sid)), messages, toolFilter{
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

// verify runs a self-check after execution. Result.FixSteps (when present)
// signals the caller to re-plan with the failure context — the inline-fix
// path was removed because small models tend to compound errors when handed
// a growing context of fix attempts. Returns the final result and the verify
// outcome.
func (a *agent) verify(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, res toolLoopResult, userText string, planSteps []string) (toolLoopResult, *verifyResult, error) {
	verifyPrompt := a.loadPromptFile(sid, "VERIFY.md")
	if verifyPrompt == "" {
		return res, &verifyResult{Success: true}, nil
	}

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

	verifyMessages := append(messages, llmMessage{Role: "user", Content: prompt.String()})
	var result verifyResult
	verifyRes, err := a.runToolLoopJSON(ctx, sid, conn, verifyMessages, toolFilter{
		readOnly: true,
		include:  map[string]bool{"run_task": true},
	}, &result)
	res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
	if err != nil {
		slog.Error("verify: skipped, treating as success", "err", err)
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ Verification skipped: "+err.Error()+"\n")))
		return res, &verifyResult{Success: true}, nil
	}
	return res, &result, nil
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
