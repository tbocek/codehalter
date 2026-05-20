package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
)

// This file owns the prompt orchestrator: Prompt is the ACP entry point, it
// dispatches each user turn to either runTaskCycle (simple request, in
// loop.go) or runSubtasks (big-task decomposition with one cycle per
// subtask). The phase UI helpers (sendPhase, setStatus, finalizePlan) and
// the request-level error shaper (failPrompt, stopReasonFor) also live here
// — they're orchestrator concerns, not part of the loop body.

// errUserCancelled flags a deliberate stop initiated by the user (e.g. they
// chose Abort in a tool-choice prompt). It's NOT an error to surface as a
// red box — the prompt returns a clean PromptResponse with StopReasonCancelled
// when this sentinel reaches the top level.
var errUserCancelled = errors.New("user cancelled")

// phaseNames are the pipeline stages; only the current one is shown to the
// client at a time (the plan UI updates in place as phases progress).
var phaseNames = []string{"Planning", "Executing", "Verifying", "Documenting"}

// phaseEntries builds the multi-row plan entries to render in the client: all
// phases up to and including `phase`, with prior phases marked completed and
// phase `phase` marked in_progress or completed depending on `done`. Optional
// `suffix` is appended to whichever row is in_progress (used by setStatus to
// surface transient lifecycle markers like " (thinking…)" or " (running
// read_file…)"). Document (phase 3) only appears once it actually starts, so
// a no-doc run ends with three rows.
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

// setStatus re-emits the full multi-row plan with `suffix` appended to
// whichever row is currently in_progress. Used as a transient marker for
// lifecycle states: " (sent to llm…)" between HTTP POST and first token,
// " (thinking…)" while tokens stream, " (running read_file…)" while a tool
// executes. Pass "" to revert to the bare phase name. No-op when no phase is
// active so background calls (history compaction, per-turn summariser) don't
// clobber the UI.
func (a *agent) setStatus(ctx context.Context, sid SessionId, suffix string) {
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

	// Pick up anything the user edited between turns (mcp.toml today;
	// settings.toml / capabilities later). checkSettings is silent when
	// nothing changed and only emits tool-call cards for real diffs, so it's
	// safe to run on every prompt without spamming the chat.
	if sess := a.getSession(req.SessionId); sess != nil {
		a.checkSettings(ctx, sess.Cwd, req.SessionId)
	}

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
	subtaskPath := firstPlan != nil && len(firstPlan.Subtasks) > 0
	if subtaskPath {
		// Big task — iterate each subtask through its own plan→execute→verify.
		// runSubtasks fires backgroundSummarise per subtask iteration so every
		// turn pair gets a structured note; the epilogue below skips the
		// summary call in that path to avoid double-summarising the last pair.
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
		// Fire-and-forget per-turn summarisation onto a free LLM[1+] slot.
		// Skipped on the subtasks path — runSubtasks already fired it for
		// every iteration including the last, and double-firing would
		// duplicate the final pair's structured note in the shadow buffer.
		if !subtaskPath {
			a.backgroundSummarise(sess)
		}
		// Refresh .codehalter/.git_commit on LLM[2+] (or LLM[1] after
		// shadowPending — see pickGitCommitConn). Overwrites by design, so
		// it's safe to fire once per Prompt regardless of subtaskPath.
		a.backgroundGitCommit(sess)
		a.compressHistory(ctx, sess)
	}

	return PromptResponse{StopReason: stopReasonFor(ctx)}, nil
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
			// Per-subtask background summary so the shadow buffer covers
			// every turn pair, not just the final subtask's reply.
			a.backgroundSummarise(sess)
		}

		// Images only travel with the real user message (subtask 0).
		images = nil
	}

	return finalResult, nil
}
