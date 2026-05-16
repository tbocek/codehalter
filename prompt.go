package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
)

// errUserCancelled flags a deliberate stop initiated by the user (e.g. they
// chose Abort in a tool-choice prompt). It's NOT an error to surface as a
// red box — the prompt returns a clean PromptResponse with StopReasonCancelled
// when this sentinel reaches the top level.
var errUserCancelled = errors.New("user cancelled")

// This file owns the prompt orchestrator: the per-turn loop that drives the
// plan → execute → verify pipeline (runTaskCycle), the big-task branch that
// decomposes a request into subtasks (runSubtasks), the wire-bytes assembly
// that keeps the prefix cache hot (composeUserContent), and the small phase
// UI helpers used along the way (sendPhase, finalizePlan).

// phaseNames are the pipeline stages; only the current one is shown to the
// client at a time (the plan UI updates in place as phases progress).
var phaseNames = []string{"Planning", "Executing", "Verifying", "Documenting"}

// phaseEntries builds the multi-row plan entries to render in the client: all
// phases up to and including `phase`, with prior phases marked completed and
// phase `phase` marked in_progress or completed depending on `done`. Optional
// `suffix` is appended to whichever row is in_progress (used for transient
// markers like " (thinking…)"). Document (phase 3) only appears once it
// actually starts, so a no-doc run ends with three rows.
func phaseEntries(phase int, done bool, suffix string) []PlanEntry {
	if phase < 0 || phase >= len(phaseNames) {
		return nil
	}
	entries := make([]PlanEntry, 0, phase+1)
	for i := 0; i <= phase; i++ {
		status := "completed"
		content := phaseNames[i]
		if i == phase && !done {
			status = "in_progress"
			content += suffix
		}
		entries = append(entries, PlanEntry{Content: content, Priority: "medium", Status: status})
	}
	return entries
}

// sendPhase emits a plan update covering every phase started so far. The
// current phase is in_progress (or completed when done=true); earlier phases
// are completed. The session tracks in-progress state so finalizePlan can
// mark whatever phase was running as completed if Prompt exits early.
func (a *agent) sendPhase(ctx context.Context, sid SessionId, phase int, done bool) {
	entries := phaseEntries(phase, done, "")
	if entries == nil {
		return
	}
	if sess := a.getSession(sid); sess != nil {
		sess.phaseMu.Lock()
		sess.phaseCurrent = phase
		sess.phaseActive = !done
		sess.phaseMu.Unlock()
	}
	a.sendUpdate(ctx, sid, PlanUpdate(entries))
}

// notifyPhaseSuffix re-emits the full multi-row plan with `suffix` appended
// to whichever row is currently in_progress. Used as a transient marker
// (e.g. " (thinking…)") while a phase-bound LLM call is in flight; pass "" to
// revert. No-op when no phase is active so background calls (history
// compaction) don't clobber the UI.
func (a *agent) notifyPhaseSuffix(ctx context.Context, sid SessionId, suffix string) {
	sess := a.getSession(sid)
	if sess == nil {
		return
	}
	sess.phaseMu.Lock()
	active := sess.phaseActive
	phase := sess.phaseCurrent
	sess.phaseMu.Unlock()
	if !active {
		return
	}
	entries := phaseEntries(phase, false, suffix)
	if entries == nil {
		return
	}
	a.sendUpdate(ctx, sid, PlanUpdate(entries))
}

// finalizePlan marks every phase up to and including the currently-active one
// as completed so the UI stops spinning. Idempotent and safe to call when no
// phase is active. Used from a Prompt-level defer to cover every exit path:
// errors mid-phase (LLM 500, tool failure), user cancel, or panic.
func (a *agent) finalizePlan(sid SessionId) {
	sess := a.getSession(sid)
	if sess == nil {
		return
	}
	sess.phaseMu.Lock()
	active := sess.phaseActive
	phase := sess.phaseCurrent
	sess.phaseActive = false
	sess.phaseMu.Unlock()
	if !active {
		return
	}
	entries := phaseEntries(phase, true, "")
	if entries == nil {
		return
	}
	// Background ctx so the finalize fires even when the request ctx is cancelled.
	a.sendUpdate(context.Background(), sid, PlanUpdate(entries))
}

// failPrompt records a fatal error in the session and returns it so the ACP
// dispatcher emits a JSON-RPC error response — Zed renders that as a red
// error box in the chat. Use only for failures that abort the prompt
// (LLM auth / out-of-credits / planAndRoute crash). Pass any tool uses
// captured before the failure so they're preserved in history. Recoverable
// warnings should keep using sendUpdate with a "⚠ ..." chunk.
func (a *agent) failPrompt(sid SessionId, err error, toolUses []ToolUse) (PromptResponse, error) {
	if sess := a.getSession(sid); sess != nil {
		if len(toolUses) > 0 {
			sess.AddAssistantWithTools("❌ "+err.Error(), toolUses)
		} else {
			sess.AddAssistant("❌ " + err.Error())
		}
		_ = sess.Save()
	}
	return PromptResponse{}, err
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
	isFirstMessage := sess != nil && len(sess.Messages) == 0 && sess.Summary == ""
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

	// Initial plan. Also decides whether the task is big enough to break
	// into subtasks (plan.Subtasks). For simple tasks, this same plan is
	// reused for attempt 0 of runTaskCycle — no double planning.
	a.sendPhase(ctx, req.SessionId, 0, false)
	firstConn, firstPlan, firstToolUses, err := a.planAndRoute(ctx, req.SessionId, planInput)
	if err != nil {
		if errors.Is(err, errUserCancelled) {
			return PromptResponse{StopReason: StopReasonCancelled}, nil
		}
		return a.failPrompt(req.SessionId, err, firstToolUses)
	}

	var result toolLoopResult
	if firstPlan != nil && len(firstPlan.Subtasks) > 0 {
		// Big task — iterate each subtask through its own plan→execute→verify.
		result, err = a.runSubtasks(ctx, req.SessionId, firstPlan.Subtasks, firstToolUses, currentUserIdx, isFirstMessage, images)
		if err != nil {
			// User cancellation is a clean stop, not an error to render as red.
			if errors.Is(err, errUserCancelled) {
				return PromptResponse{StopReason: StopReasonCancelled}, nil
			}
			return a.failPrompt(req.SessionId, err, nil)
		}
	} else {
		// Single task — reuse the initial plan on attempt 0; retries re-plan.
		result, err = a.runTaskCycle(ctx, req.SessionId, userText, planInput, currentUserIdx, isFirstMessage, images, firstPlan, firstConn, firstToolUses)
		if err != nil {
			if errors.Is(err, errUserCancelled) {
				return PromptResponse{StopReason: StopReasonCancelled}, nil
			}
			return a.failPrompt(req.SessionId, err, nil)
		}
	}

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
	const maxAttempts = 10
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
	choice, err := a.askChoiceAuto(ctx, sid, tcId, []string{"Interactive", "Automatic"})
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + choice)})
	if err != nil || choice == "abort" {
		if sess != nil {
			sess.AddAssistant(header.String() + "\nUser cancelled.")
			_ = sess.Save()
		}
		return toolLoopResult{}, errUserCancelled
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
