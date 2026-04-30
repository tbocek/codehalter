package main

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
)

// This file owns the prompt orchestrator: the per-turn loop that drives the
// plan → execute → verify pipeline (runTaskCycle), the big-task branch that
// decomposes a request into subtasks (runSubtasks), the wire-bytes assembly
// that keeps the prefix cache hot (composeUserContent), and the small phase
// UI helpers used along the way (sendPhase, finalizePlan).

// phaseNames are the pipeline stages; only the current one is shown to the
// client at a time (the plan UI updates in place as phases progress).
var phaseNames = []string{"Planning", "Executing", "Verifying"}

// sendPhase emits a plan update with a single entry for the given phase,
// replacing any previous entry. done=true renders the phase as completed
// (spinner stops); done=false renders it as in_progress. The session tracks
// in-progress state so finalizePlan can mark whatever phase was running as
// completed if Prompt exits early.
func (a *agent) sendPhase(ctx context.Context, sid SessionId, phase int, done bool) {
	if phase < 0 || phase >= len(phaseNames) {
		return
	}
	status := "in_progress"
	if done {
		status = "completed"
	}
	if sess := a.getSession(sid); sess != nil {
		sess.mu.Lock()
		sess.phaseCurrent = phase
		sess.phaseActive = !done
		sess.mu.Unlock()
	}
	entry := PlanEntry{Content: phaseNames[phase], Priority: "medium", Status: status}
	a.sendUpdate(ctx, sid, PlanUpdate([]PlanEntry{entry}))
}

// finalizePlan marks the currently-in-progress phase as completed so the UI
// stops spinning. Idempotent and safe to call when no phase is active. Used
// from a Prompt-level defer to cover every exit path: errors mid-phase
// (LLM 500, tool failure), user cancel, or panic.
func (a *agent) finalizePlan(sid SessionId) {
	sess := a.getSession(sid)
	if sess == nil {
		return
	}
	sess.mu.Lock()
	active := sess.phaseActive
	phase := sess.phaseCurrent
	sess.phaseActive = false
	sess.mu.Unlock()
	if !active || phase < 0 || phase >= len(phaseNames) {
		return
	}
	entry := PlanEntry{Content: phaseNames[phase], Priority: "medium", Status: "completed"}
	// Background ctx so the finalize fires even when the request ctx is cancelled.
	a.sendUpdate(context.Background(), sid, PlanUpdate([]PlanEntry{entry}))
}

func (a *agent) replyError(ctx context.Context, sid SessionId, msg string) PromptResponse {
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("❌ Error: "+msg+"\n")))
	if sess := a.getSession(sid); sess != nil {
		sess.AddAssistant("❌ " + msg)
		_ = sess.Save()
	}
	return PromptResponse{StopReason: stopReasonFor(ctx)}
}

// stopReasonFor reports `cancelled` if the prompt's context was cancelled
// (user hit Cancel in the client), else `end_turn`. ACP clients use the
// stop reason to distinguish a clean turn from a user-initiated abort.
func stopReasonFor(ctx context.Context) StopReason {
	if ctx.Err() != nil {
		return StopReasonCancelled
	}
	return StopReasonEndTurn
}

func (a *agent) Prompt(ctx context.Context, req PromptRequest) (PromptResponse, error) {
	ctx, cancel := context.WithCancel(ctx)
	a.mu.Lock()
	a.cancel = cancel
	a.mu.Unlock()
	defer cancel()
	defer a.finalizePlan(req.SessionId)

	a.waitForIndex()

	// Extract user text and images from prompt blocks.
	var userText string
	var images []ImageData
	for _, block := range req.Content {
		if block.Text != nil {
			userText += block.Text.Text
		}
		if block.Image != nil {
			images = append(images, ImageData{
				MimeType: block.Image.MimeType,
				Data:     block.Image.Data,
			})
		}
	}

	// Slash commands (/improve, /clean) — handled before any session
	// mutation or LLM dispatch so the command itself isn't stored in
	// history.
	if a.handleSlashCommand(ctx, req.SessionId, userText) {
		return PromptResponse{StopReason: stopReasonFor(ctx)}, nil
	}

	// Store user message. The stored message is the raw userText only —
	// project context and plan/retry prefixes are injected onto the latest
	// prompt at send time (see buildLLMHistory) and are NOT persisted, so
	// history stays cacheable and compact.
	sess := a.getSession(req.SessionId)
	isFirstMessage := sess != nil && len(sess.Messages) == 0 && len(sess.History) == 0
	currentUserIdx := -1
	if sess != nil {
		if len(images) > 0 {
			sess.AddUserWithImages(userText, images)
		} else {
			sess.AddUser(userText)
		}
		currentUserIdx = len(sess.Messages) - 1
		_ = sess.Save()
	}

	// Generate title for new sessions.
	if isFirstMessage {
		go a.generateTitle(context.Background(), sess, userText)
	}

	slog.Info("Prompt", "sid", req.SessionId, "sessions", len(a.sessions))

	planInput := userText

	// Pre-planning verification.
	if thinkingConn := a.settings.LLMFor("thinking", a.llmTier(req.SessionId)); thinkingConn != nil {
		a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock("Running pre-planning verification...\n\n")))
		vr, err := a.preVerify(ctx, req.SessionId, thinkingConn, userText)
		if err == nil && vr != nil && !vr.Success {
			a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock(fmt.Sprintf("⚠ Pre-planning check found issues:\n%s\n\n", strings.Join(vr.Issues, "\n")))))
			planInput = fmt.Sprintf("The project is currently in a broken state:\nIssues: %s\n\nOriginal request: %s", strings.Join(vr.Issues, "; "), userText)
		} else if err != nil {
			slog.Error("preVerify failed", "error", err)
		} else {
			a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock("Project state verified.\n\n")))
		}
	}

	// Initial plan. Also decides whether the task is big enough to break
	// into subtasks (plan.Subtasks). For simple tasks, this same plan is
	// reused for attempt 0 of runTaskCycle — no double planning.
	a.sendPhase(ctx, req.SessionId, 0, false)
	firstConn, firstPlan, firstToolUses, err := a.planAndRoute(ctx, req.SessionId, planInput)
	if err != nil {
		if sess != nil && len(firstToolUses) > 0 {
			sess.AddAssistantWithTools("❌ "+err.Error(), firstToolUses)
			_ = sess.Save()
		}
		a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock("❌ "+err.Error()+"\n")))
		return PromptResponse{StopReason: stopReasonFor(ctx)}, nil
	}

	var result toolLoopResult
	if firstPlan != nil && len(firstPlan.Subtasks) > 0 {
		// Big task — iterate each subtask through its own plan→execute→verify.
		result, err = a.runSubtasks(ctx, req.SessionId, firstPlan.Subtasks, firstToolUses, currentUserIdx, isFirstMessage, images)
		if err != nil {
			// User cancelled or fatal; bail out without wrapping as an error reply
			// (runSubtasks already surfaced the message).
			return PromptResponse{StopReason: stopReasonFor(ctx)}, nil
		}
	} else {
		// Single task — reuse the initial plan on attempt 0; retries re-plan.
		result, err = a.runTaskCycle(ctx, req.SessionId, userText, planInput, currentUserIdx, isFirstMessage, images, firstPlan, firstConn, firstToolUses)
		if err != nil {
			return a.replyError(ctx, req.SessionId, err.Error()), nil
		}
	}
	a.sendPhase(ctx, req.SessionId, 2, true)

	if sess != nil && result.Text != "" {
		sess.UpsertLastAssistant(result.Text)
		_ = sess.Save()
		a.compressHistory(ctx, sess)
	}

	return PromptResponse{StopReason: stopReasonFor(ctx)}, nil
}

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
	const maxAttempts = 5
	var result toolLoopResult
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
				if sess != nil && len(tu) > 0 {
					sess.AddAssistantWithTools("❌ "+err.Error(), tu)
					_ = sess.Save()
				}
				a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("❌ "+err.Error()+"\n")))
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

		a.sendPhase(ctx, sid, 2, false)
		var vr *verifyResult
		result, vr, err = a.verify(ctx, sid, conn, messages, result, planInput, planSteps)
		if err != nil {
			return result, err
		}
		if vr == nil || vr.Success || len(vr.FixSteps) == 0 {
			break
		}

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
	return result, nil
}

// runSubtasks shows the decomposed subtask list, asks the user how to run
// the batch (Interactive / Automatic / Cancel), and iterates each subtask
// through its own runTaskCycle. Automatic flips the session into autopilot
// mode for the duration so inner prompts auto-answer. Returns the final
// subtask's result.
func (a *agent) runSubtasks(
	ctx context.Context, sid SessionId,
	subtasks []string, initialToolUses []ToolUse,
	currentUserIdx int, isFirstMessage bool,
	images []ImageData,
) (toolLoopResult, error) {
	sess := a.getSession(sid)

	var header strings.Builder
	fmt.Fprintf(&header, "This looks like a big task. I'd break it into %d subtasks:\n", len(subtasks))
	for i, s := range subtasks {
		fmt.Fprintf(&header, "%d. %s\n", i+1, s)
	}
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(header.String())))

	tcId := a.StartToolCall(ctx, sid, "How should I run these?", "think", nil)
	choice, err := a.askChoiceAuto(ctx, sid, tcId, []string{"Interactive", "Automatic"}, 0)
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + choice)})
	if err != nil || choice == "abort" {
		if sess != nil {
			sess.AddAssistant(header.String() + "\nUser cancelled.")
			_ = sess.Save()
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Cancelled.\n")))
		return toolLoopResult{}, fmt.Errorf("user cancelled")
	}

	if choice == "Automatic" {
		a.mu.Lock()
		origMode := a.mode
		a.mode = modeAutopilot
		a.mu.Unlock()
		defer func() {
			a.mu.Lock()
			a.mode = origMode
			a.mu.Unlock()
		}()
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("[Automatic] Running all subtasks without interruption.\n\n")))
	}

	var finalResult toolLoopResult
	userIdx := currentUserIdx
	firstInSession := isFirstMessage

	for i, sub := range subtasks {
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
			fmt.Sprintf("\n=== Subtask %d/%d: %s ===\n\n", i+1, len(subtasks), sub),
		)))

		// For subtasks after the first, append a new user message so the prior
		// subtask's assistant reply has a clean user turn to follow.
		if i > 0 && sess != nil {
			sess.AddUser(sub)
			userIdx = len(sess.Messages) - 1
			_ = sess.Save()
			firstInSession = false
		}

		// Subtask 1 inherits the planning tool uses from the decomposition
		// pass so the inner planner doesn't repeat those lookups.
		planInput := sub
		if i == 0 && len(initialToolUses) > 0 {
			var buf strings.Builder
			buf.WriteString("Context from initial task decomposition (these tools were already called — do not repeat):\n")
			for _, tu := range initialToolUses {
				fmt.Fprintf(&buf, "- %s(%s) → %s\n", tu.Name, tu.Input, truncate(tu.Output, 500))
			}
			buf.WriteString("\nSubtask: ")
			buf.WriteString(sub)
			planInput = buf.String()
		}

		result, err := a.runTaskCycle(ctx, sid, sub, planInput, userIdx, firstInSession, images, nil, nil, nil)
		if err != nil {
			return finalResult, err
		}
		finalResult = result

		// Persist this subtask's result so subtask N+1 sees it in history.
		if sess != nil && result.Text != "" {
			sess.UpsertLastAssistant(result.Text)
			_ = sess.Save()
		}

		// Images only travel with the real user message (subtask 0).
		images = nil
	}

	return finalResult, nil
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
