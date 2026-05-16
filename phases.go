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
	thinking := a.pickAvailable(ctx, sid, "thinking")
	if thinking == nil {
		return nil, nil, nil, fmt.Errorf("no [llm] in .codehalter/settings.toml")
	}
	planPrompt := a.loadPromptFile(sid, "PLAN.md")
	if planPrompt == "" {
		return thinking, nil, nil, nil
	}
	// Tell the planner about the project's verify-class target so the final
	// plan step names `just:verify` (or equivalent) instead of `just:test`.
	// Without this, the small planner picks whatever target name comes to
	// mind first and skips the chained tidy/vet/build steps that `verify`
	// wraps. Same hint goes into execute/verify too, but those see the plan
	// already locked in — the planner is the load-bearing place.
	if hint := a.verifyTargetHint(); hint != "" {
		planPrompt = planPrompt + "\n\n" + hint
	}
	// Carry the full session history so plan shares a prefix with the other
	// phases (execute/verify/document). Without this, each turn's plan call
	// sends only `PLAN.md + userText` and the cache from prior turns is
	// useless.
	var messages []llmMessage
	if sess := a.getSession(sid); sess != nil {
		messages = a.buildLLMHistory(sess, -1)
	}
	messages = append(messages, llmMessage{Role: "user", Content: planPrompt + "\n\nUser request: " + userText})
	var plan planResult
	planRes, err := a.runToolLoopJSON(ctx, sid, thinking, messages, toolFilter{}, "plan", &plan)
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
		choice, err := a.askChoiceAuto(ctx, sid, tcId, plan.Choices)
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
		ok, err := a.askYesNoAuto(ctx, sid, tcId, "Execute", "Cancel")
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

// ---------------------------------------------------------------------------
// Execute phase
// ---------------------------------------------------------------------------

// pickVerifyTarget returns runner:target for the most "comprehensive" task
// the project advertises (priority: verify > check > ci > all) or "" when
// none exists. Used by execute() and verify() to inject a concrete hint that
// the model should use the project's full verification contract via run_task
// rather than running its constituents directly via run_command. Generic
// across runners: any justfile/Makefile/npm-script target with one of these
// names gets picked up automatically.
func (a *agent) pickVerifyTarget() string {
	priority := []string{"verify", "check", "ci", "all"}
	a.mu.Lock()
	runners := a.runners
	a.mu.Unlock()
	for _, want := range priority {
		for _, r := range runners {
			for _, t := range r.Tasks {
				if strings.EqualFold(t, want) {
					return r.Name + ":" + t
				}
			}
		}
	}
	return ""
}

// verifyTargetHint formats the per-project hint pointing at the comprehensive
// verify-class target (see pickVerifyTarget). Empty string when there is no
// such target — caller treats that as "no hint, use the generic rule".
func (a *agent) verifyTargetHint() string {
	vt := a.pickVerifyTarget()
	if vt == "" {
		return ""
	}
	return fmt.Sprintf(
		"This project declares `%s` as its verify-class target. Run it via run_task — its body is the project's contract for \"verified\" and may chain tidy/build steps that piecemeal commands miss. Do NOT substitute `go test ./...` / `npm test` / equivalent constituents.",
		vt,
	)
}

// execute runs the execution phase. It prepends EXECUTE.md (if present) to the
// last user message, excludes web tools (information retrieval belongs to the
// planning phase), and runs the agentic tool loop with write-enabled tools.
// extraExclude lists additional tool names to exclude (e.g. launch_subagent
// when a subagent has reached its max nesting depth).
//
// Routed via llmTier(sid): the main session always uses [llm] for cache
// consistency — every phase (plan/execute/verify/document) hits the same slot
// so the prefix cache stays warm across the whole turn. Subagent sessions
// route to [[subllm]] so they don't evict the main slot's cache.
func (a *agent) execute(ctx context.Context, sid SessionId, messages []llmMessage, extraExclude ...string) (toolLoopResult, error) {
	executeMD := a.loadPromptFile(sid, "EXECUTE.md")
	if hint := a.verifyTargetHint(); hint != "" {
		if executeMD != "" {
			executeMD = executeMD + "\n\n" + hint
		} else {
			executeMD = hint
		}
	}
	if executeMD != "" && len(messages) > 0 {
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
	return a.runToolLoop(ctx, sid, a.pickAvailable(ctx, sid, "execute"), messages, toolFilter{
		exclude: exclude,
	}, "execute")
}

// ---------------------------------------------------------------------------
// Verify phase
// ---------------------------------------------------------------------------

type verifyResult struct {
	Success  bool     `json:"success"`
	Issues   []string `json:"issues"`
	FixSteps []string `json:"fix_steps,omitempty"`
}

// verify runs a self-check after execution. It routes to the "execute" role
// (lower temperature, less thinking) rather than reusing the planner's
// connection — small thinking models tend to rationalize failures into
// success=true, while a colder execute model sticks to the rule. Result.FixSteps
// (when present) signals the caller to re-plan with the failure context — the
// inline-fix path was removed because small models tend to compound errors when
// handed a growing context of fix attempts. Returns the final result and the
// verify outcome.
//
// When fallbackConn is non-nil, it's used only if no LLM is configured for the
// execute role (test path with synthetic agent and no settings).
func (a *agent) verify(ctx context.Context, sid SessionId, fallbackConn *LLMConnection, messages []llmMessage, res toolLoopResult, userText string, planSteps []string) (toolLoopResult, *verifyResult, error) {
	verifyPrompt := a.loadPromptFile(sid, "VERIFY.md")
	if verifyPrompt == "" {
		return res, &verifyResult{Success: true}, nil
	}

	// Same routing as execute(): main session uses [llm], subagents use [[subllm]].
	conn := a.pickAvailable(ctx, sid, "execute")
	if conn == nil {
		conn = fallbackConn
	}

	var prompt strings.Builder
	prompt.WriteString(verifyPrompt)
	if hint := a.verifyTargetHint(); hint != "" {
		prompt.WriteString("\n\n")
		prompt.WriteString(hint)
	}
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

	// Capture executor's tool-use count before we append verify's own —
	// the Failed override only inspects executor tools.
	execToolCount := len(res.ToolUses)

	verifyMessages := append(messages, llmMessage{Role: "user", Content: prompt.String()})
	var result verifyResult
	verifyRes, err := a.runToolLoopJSON(ctx, sid, conn, verifyMessages, toolFilter{}, "verify", &result)
	res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
	if err != nil {
		slog.Error("verify: skipped, treating as success", "err", err)
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ Verification skipped: "+err.Error()+"\n")))
		return res, &verifyResult{Success: true}, nil
	}

	// Authoritative override: a typed-failed tool result during execution
	// trumps the LLM's verdict. Small models routinely return success=true
	// even after seeing a tool's "❌ TASK FAILED" banner in their context,
	// so we don't trust the JSON when the raw exit-code stream disagrees.
	if result.Success {
		for i := 0; i < execToolCount; i++ {
			u := res.ToolUses[i]
			if u.Failed {
				result.Success = false
				result.Issues = append(result.Issues,
					fmt.Sprintf("%s failed (exit-code override; verifier had reported success=true)", u.Name))
			}
		}
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
// It hands the executor's response to the thinking LLM with DOCUMENT.md so the
// LLM can update (or create) the project README when the change is user-visible.
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

	// Carry the session history so document shares a prefix with the other
	// phases — the prior turns are identical bytes to what plan/execute/
	// verify just saw, so the slot's cache stays warm into the doc pass
	// instead of being thrown away for a fresh fill.
	var messages []llmMessage
	if sess := a.getSession(sid); sess != nil {
		messages = a.buildLLMHistory(sess, -1)
	}
	messages = append(messages, llmMessage{Role: "user", Content: prompt.String()})
	docRes, err := a.runToolLoop(ctx, sid, conn, messages, toolFilter{}, "document")
	if err != nil {
		slog.Warn("document phase failed", "err", err)
		return exec, nil
	}
	exec.ToolUses = append(exec.ToolUses, docRes.ToolUses...)
	return exec, nil
}
