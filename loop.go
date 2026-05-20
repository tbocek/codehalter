package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
)

// This file owns the agent's per-task loop. One iteration is plan → execute →
// verify, with an optional document phase after a successful verify. When
// verify fails with fix steps, the loop re-plans with the failure context and
// runs again (up to maxAttempts, with Jaccard duplicate-issue detection to
// bail early when the model is rephrasing the same problem). Each phase is a
// thin wrapper over runToolLoop; runTaskCycle (at the bottom) is the retry
// engine. The Prompt orchestrator (prompt.go) calls runTaskCycle once per
// task — either directly for simple requests, or once per subtask for big
// requests decomposed by the planner.

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
		return nil, nil, nil, fmt.Errorf("no [[llm]] in .codehalter/settings.toml")
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
	// Exclude respond: planning emits a JSON object as plain text, so we don't
	// want the model to escape into the synthetic terminal-tool grammar that
	// execute uses. Same reasoning in verify() and document() below.
	planRes, err := a.runToolLoopJSON(ctx, sid, thinking, messages, toolFilter{exclude: map[string]bool{respondToolName: true}}, "plan", &plan)
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
// Routed via pickAvailable(sid): the main session always uses LLM[0] for
// cache consistency — every phase (plan/execute/verify/document) hits the
// same slot so the prefix cache stays warm across the whole turn. Subagent
// sessions route to their pinned LLM[i] for the same reason.
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
	// SustainabilityConcerns lists fixes the executor performed that won't
	// survive a container rebuild / fresh clone — e.g. `apt-get install X`
	// run via run_command without a matching `.devcontainer/Dockerfile`
	// edit. Independent of pass/fail: even when exit codes are clean, a
	// non-empty list downgrades the verdict to failure so the re-plan can
	// persist the fix. See VERIFY.md.
	SustainabilityConcerns []string `json:"sustainability_concerns,omitempty"`
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

	// Same routing as execute(): main session uses LLM[0], subagents their pinned LLM[i].
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
	verifyRes, err := a.runToolLoopJSON(ctx, sid, conn, verifyMessages, toolFilter{exclude: map[string]bool{respondToolName: true}}, "verify", &result)
	res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
	if err != nil {
		slog.Error("verify: skipped, treating as success", "err", err)
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ Verification skipped: "+err.Error()+"\n")))
		return res, &verifyResult{Success: true}, nil
	}

	// Exit-code authority: the executor's typed Failed flags are ground
	// truth for pass/fail. The LLM's `success` field is advisory — small
	// models hallucinate in BOTH directions (inventing failures on clean
	// runs, rationalizing real failures as success=true), so we override
	// either way from the raw tool stream.
	anyToolFailed := false
	for i := 0; i < execToolCount; i++ {
		if res.ToolUses[i].Failed {
			anyToolFailed = true
			break
		}
	}
	if anyToolFailed {
		if result.Success {
			result.Success = false
			if len(result.Issues) == 0 {
				for i := 0; i < execToolCount; i++ {
					u := res.ToolUses[i]
					if u.Failed {
						result.Issues = append(result.Issues,
							fmt.Sprintf("%s failed (exit-code override; verifier had reported success=true)", u.Name))
					}
				}
			}
		}
	} else if !result.Success {
		// LLM invented issues with no failed tool to back them up — discard.
		result.Success = true
		result.Issues = nil
		result.FixSteps = nil
	}

	// Sustainability: a clean exit with non-sustainable fixes (e.g. an
	// `apt-get install X` run via run_command without a matching Dockerfile
	// edit) is still incomplete — the fix will vanish on container rebuild.
	// Treat as failure with concerns folded into fix_steps so re-plan can
	// persist them.
	if result.Success && len(result.SustainabilityConcerns) > 0 {
		result.Success = false
		for _, c := range result.SustainabilityConcerns {
			result.Issues = append(result.Issues, "not yet sustainable: "+c)
			result.FixSteps = append(result.FixSteps, c)
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
	docRes, err := a.runToolLoop(ctx, sid, conn, messages, toolFilter{exclude: map[string]bool{respondToolName: true}}, "document")
	if err != nil {
		slog.Warn("document phase failed", "err", err)
		return exec, nil
	}
	exec.ToolUses = append(exec.ToolUses, docRes.ToolUses...)
	return exec, nil
}

// ---------------------------------------------------------------------------
// Loop: plan → execute → verify (→ document), with re-plan on verify failure
// ---------------------------------------------------------------------------

// runTaskCycle runs plan → execute → verify with retries for a single task.
// On attempt 0, a pre-computed plan may be supplied (prePlan/preConn/preToolUses)
// to avoid re-planning after an upfront planAndRoute call; retries always
// re-plan with the failure context. userText is the original user-facing
// request, planInput is what the planner sees (may carry failure context).
func (a *agent) runTaskCycle(
	ctx context.Context, sid SessionId,
	userText, planInput string,
	currentUserIdx int, isFirstInSession bool,
	images []ImageData,
	prePlan *planResult, preConn *LLMConnection, preToolUses []ToolUse,
) (toolLoopResult, error) {
	const maxAttempts = 20
	var result toolLoopResult
	var lastVR *verifyResult
	var seenBags []map[string]bool
	// wroteFilesEver tracks whether ANY attempt in this retry loop performed a
	// file write. The document gate at the bottom must see the cumulative
	// answer: when verify failed on attempt N (causing a re-plan) and the
	// final successful attempt only re-read files to confirm, result.ToolUses
	// alone misses the earlier edits and the document phase silently skips.
	wroteFilesEver := false
	sess := a.getSession(sid)

	conn := preConn
	var planSteps []string
	if prePlan != nil {
		planSteps = prePlan.Steps
	}
	planToolUses := preToolUses

	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 || prePlan == nil {
			a.sendPhase(ctx, sid, 0, false)
			c, p, tu, err := a.planAndRoute(ctx, sid, planInput)
			if err != nil {
				if errors.Is(err, errUserCancelled) {
					return result, err
				}
				if sess != nil && len(tu) > 0 {
					sess.AddAssistantWithTools("❌ "+err.Error(), tu)
					_ = sess.Save()
				}
				return result, err
			}
			conn = c
			planSteps = nil
			if p != nil {
				planSteps = p.Steps
			}
			planToolUses = tu
		}

		// composeUserContent yields the exact wire bytes for this turn —
		// planCtx + planInput, plus sysPrompt on the very first message of
		// the session. Persisting that wire format (rather than a stripped
		// version) is what keeps the LLM-side prefix cache hot across turns:
		// turn N+1's buildLLMHistory rebuilds the same bytes turn N sent.
		stored, err := a.composeUserContent(sid, planInput, planSteps, planToolUses, isFirstInSession)
		if err != nil {
			return result, err
		}

		if sess != nil {
			sess.UpdateLastMessageContent(currentUserIdx, stored)
			_ = sess.Save()
		}

		// History is read 1:1 from TOML (including images on user turns).
		var messages []llmMessage
		if sess != nil {
			messages = a.buildLLMHistory(sess, currentUserIdx)
		}
		messages = append(messages, a.buildUserMessage(stored, images))

		a.sendPhase(ctx, sid, 1, false)
		result, err = a.execute(ctx, sid, messages)
		if err != nil {
			return result, err
		}
		if hasFileWrites(result.ToolUses) {
			wroteFilesEver = true
		}

		a.sendPhase(ctx, sid, 2, false)
		var vr *verifyResult
		preVerifyToolCount := len(result.ToolUses)
		result, vr, err = a.verify(ctx, sid, conn, messages, result, planInput, planSteps)
		if err != nil {
			return result, err
		}
		// verify() runs the LLM with an unfiltered tool set, so it can
		// invoke edit_file/write_file while re-checking the result. Capture
		// any new file writes it produced so the cumulative wroteFilesEver
		// flag (and the document gate downstream) sees them.
		if len(result.ToolUses) > preVerifyToolCount &&
			hasFileWrites(result.ToolUses[preVerifyToolCount:]) {
			wroteFilesEver = true
		}
		lastVR = vr
		if vr == nil || vr.Success || len(vr.FixSteps) == 0 {
			break
		}

		// Loop-detection: if verify reports a near-duplicate of an earlier
		// failure (Jaccard ≥ threshold over the bag of issue words), the LLM
		// is stuck rephrasing the same problem. Bail rather than burn the
		// remaining retry budget. Order, casing and punctuation are ignored
		// so "missing import" and "import is missing" collapse together.
		currentBag := issueBag(vr.Issues)
		looped := false
		for _, prev := range seenBags {
			if jaccard(currentBag, prev) >= failureSimilarityThreshold {
				looped = true
				break
			}
		}
		if looped {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
				fmt.Sprintf("⚠ Same verification failure as a prior attempt — retrying would loop. Giving up. Last issues:\n%s\n", strings.Join(vr.Issues, "\n")),
			)))
			break
		}
		seenBags = append(seenBags, currentBag)

		if attempt == maxAttempts-1 {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
				fmt.Sprintf("⚠ Verification still failing after %d attempts — giving up. Last issues:\n%s\n", maxAttempts, strings.Join(vr.Issues, "\n")),
			)))
			break
		}

		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
			fmt.Sprintf("⚠ Verification failed (attempt %d/%d). Re-planning with the failure context:\n%s\n", attempt+1, maxAttempts, strings.Join(vr.FixSteps, "\n")),
		)))
		planInput = fmt.Sprintf("The previous attempt failed verification.\nIssues: %s\nFix steps: %s\n\nOriginal request: %s\n\nRe-plan with this information. If the fix steps conflict with the original request or the task is infeasible, say so instead of attempting it.",
			strings.Join(vr.Issues, "; "),
			strings.Join(vr.FixSteps, "; "),
			userText)
	}

	// Document phase: only when the executor wrote files and verify did not
	// declare the result broken. Failures with no fix_steps still reach here
	// (we couldn't fix it but didn't bail) — skip the doc pass for those, the
	// change isn't trustworthy enough to advertise in the README.
	lastPhase := 2
	if (lastVR == nil || lastVR.Success) && wroteFilesEver && conn != nil {
		a.sendPhase(ctx, sid, 3, false)
		result, _ = a.document(ctx, sid, conn, userText, planSteps, result)
		lastPhase = 3
	}
	a.sendPhase(ctx, sid, lastPhase, true)
	return result, nil
}

// composeUserContent assembles the wire bytes for one user turn. The result
// is BOTH persisted to sess.Messages and sent to the LLM, so the next turn's
// buildLLMHistory replays the same bytes — that's the prefix-cache contract.
//
// On the first message of a session it folds the system prompt (skills +
// project dir) and the empty-project hint into the content. Subsequent turns
// don't re-add those: they're already in sess.Messages[0] from turn 1, and
// re-adding would shift them and break the cache.
func (a *agent) composeUserContent(sid SessionId, planInput string, planSteps []string, planToolUses []ToolUse, isFirstInSession bool) (string, error) {
	var b strings.Builder
	if len(planSteps) > 0 {
		b.WriteString("The user approved this plan. Follow these steps exactly:\n")
		for i, step := range planSteps {
			fmt.Fprintf(&b, "%d. %s\n", i+1, step)
		}
	}
	if len(planToolUses) > 0 {
		b.WriteString("\nDuring planning, these tools were already called (do NOT repeat them):\n")
		for _, tu := range planToolUses {
			fmt.Fprintf(&b, "- %s(%s) → %s\n", tu.Name, tu.Input, truncate(tu.Output, 500))
		}
	}
	content := planInput
	if b.Len() > 0 {
		b.WriteString("\nUser request: ")
		b.WriteString(planInput)
		content = b.String()
	}

	if isFirstInSession {
		sysPrompt, err := a.systemPrompt(sid)
		if err != nil {
			return "", err
		}
		a.mu.Lock()
		empty := a.emptyProject
		a.mu.Unlock()
		prefix := sysPrompt + "\n---\n"
		if empty {
			prefix = emptyProjectHint + "\n---\n" + prefix
		}
		content = prefix + content
	}
	return content, nil
}

func (a *agent) buildUserMessage(content string, images []ImageData) llmMessage {
	if len(images) == 0 || !a.imagesSupported {
		return llmMessage{Role: "user", Content: content}
	}
	// Build OpenAI-style content array with text and image blocks.
	var parts []any
	parts = append(parts, map[string]any{
		"type": "text",
		"text": content,
	})
	for _, img := range images {
		parts = append(parts, map[string]any{
			"type": "image_url",
			"image_url": map[string]string{
				"url": fmt.Sprintf("data:%s;base64,%s", img.MimeType, img.Data),
			},
		})
	}
	return llmMessage{Role: "user", Content: parts}
}

// failureSimilarityThreshold is the Jaccard ratio above which two
// verify-failure bags are considered "the same problem." Tuned empirically
// for short LLM-generated issue strings: 0.6 catches "missing import" /
// "import is missing" (Jaccard 0.67) without collapsing genuinely different
// files (e.g. foo.go vs bar.go syntax errors land around 0.67-0.83 and
// would false-positive; the cost of an over-eager bail is recoverable by
// re-prompting, so we accept that for the upside of catching small-model
// rewordings).
const failureSimilarityThreshold = 0.6

// issueBag tokenises a list of issue strings into a single set of distinct
// lowercase alphanumeric words. Punctuation, casing and ordering are all
// discarded so two attempts reporting the same root cause in different
// phrasing collapse to comparable bags.
func issueBag(issues []string) map[string]bool {
	bag := make(map[string]bool)
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			bag[cur.String()] = true
			cur.Reset()
		}
	}
	for _, iss := range issues {
		for _, r := range strings.ToLower(iss) {
			switch {
			case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
				cur.WriteRune(r)
			default:
				flush()
			}
		}
		flush()
	}
	return bag
}

// jaccard returns |A ∩ B| / |A ∪ B| for two word sets. 1.0 = identical,
// 0.0 = disjoint. Two empty bags are treated as identical.
func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	inter := 0
	for w := range a {
		if b[w] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}
