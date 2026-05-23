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
// red box — the prompt returns a clean PromptResponse with stopReason
// "cancelled" when this sentinel reaches the top level.
var errUserCancelled = errors.New("user cancelled")

// isCancelled returns true for both the deliberate-cancel sentinel and a
// raw context.Canceled / DeadlineExceeded. The latter is what the LLM
// stream / HTTP client surface when the user hits the red Cancel button
// mid-request — surfacing it as a JSON-RPC error makes Zed render the
// AUTH_REQUIRED red box (because ACP reserves -32000 for that). All three
// must collapse to a clean "cancelled" stopReason at the top of Prompt.
func isCancelled(err error) bool {
	return errors.Is(err, errUserCancelled) ||
		errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded)
}

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
func (a *agent) sendPhase(ctx context.Context, sid string, phase int, done bool) {
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
func (a *agent) setStatus(ctx context.Context, sid string, suffix string) {
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
func (a *agent) finalizePlan(sid string) {
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
func (a *agent) failPrompt(sid string, err error, toolUses []ToolUse) (PromptResponse, error) {
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
func stopReasonFor(ctx context.Context) string {
	if ctx.Err() != nil {
		return "cancelled"
	}
	return "end_turn"
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
		// Clear per-turn read-dedup. Dedup is scoped to a single Prompt()
		// turn so that across-turn re-reads (after a user reply, edits made
		// outside our process, etc.) still go through.
		sess.readDedupMu.Lock()
		sess.readDedup = nil
		sess.readDedupMu.Unlock()
	}

	// Extract user text and images from prompt blocks.
	var userText string
	var images []ImageData
	for _, raw := range req.Content {
		var block ContentBlock
		if err := unmarshalContentBlock(raw, &block); err != nil {
			continue
		}
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

	// Store user message. On the very first turn of the session we render the
	// system prompt (skills + project dir) into sess.SystemPrompt — emitted by
	// buildLLMHistory as a separate leading user message, so it survives the
	// summariser when compressHistory rotates the conversation. The empty-
	// project hint stays folded into the first user message because it's a
	// one-shot nudge — once the project has files, we want the summariser to
	// drop it naturally rather than re-injecting it forever.
	sess := a.getSession(req.SessionId)
	isFirstMessage := sess != nil && len(sess.Messages) == 0 && sess.Summary == ""
	stored := userText
	if isFirstMessage {
		sysPrompt, err := a.systemPrompt(req.SessionId)
		if err != nil {
			return a.failPrompt(req.SessionId, err, nil)
		}
		sess.SystemPrompt = sysPrompt
		a.mu.Lock()
		empty := a.emptyProject
		a.mu.Unlock()
		if empty {
			stored = emptyProjectHint + "\n---\n" + userText
		}
	}
	if sess != nil {
		if len(images) > 0 {
			sess.AddUserWithImages(stored, images)
		} else {
			sess.AddUser(stored)
		}
		_ = sess.Save()
	}

	// Generate title for new sessions.
	if isFirstMessage {
		go a.generateTitle(context.Background(), sess, userText)
	}

	slog.Info("Prompt", "sid", req.SessionId, "sessions", len(a.sessions))

	// Initial plan. Also decides whether the task is big enough to break
	// into subtasks (plan.Subtasks). For simple tasks, this same plan is
	// reused for attempt 0 of runTaskCycle — no double planning.
	a.sendPhase(ctx, req.SessionId, 0, false)
	firstConn, firstPlan, firstToolUses, err := a.planAndRoute(ctx, req.SessionId, "")
	if err != nil {
		if isCancelled(err) {
			return PromptResponse{StopReason: "cancelled"}, nil
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
		result, err = a.runSubtasks(ctx, req.SessionId, firstPlan.Subtasks)
		if err != nil {
			// User cancellation is a clean stop, not an error to render as red.
			if isCancelled(err) {
				return PromptResponse{StopReason: "cancelled"}, nil
			}
			return a.failPrompt(req.SessionId, err, nil)
		}
	} else {
		// Single task — reuse the initial plan on attempt 0; retries re-plan.
		result, err = a.runTaskCycle(ctx, req.SessionId, firstPlan, firstConn)
		if err != nil {
			if isCancelled(err) {
				return PromptResponse{StopReason: "cancelled"}, nil
			}
			return a.failPrompt(req.SessionId, err, nil)
		}
	}

	// Each phase (plan / execute / verify / document) already persisted its
	// own response via UpsertLastAssistant. The epilogue here only fires
	// background work that observes the now-complete transcript.
	if sess != nil && result.Text != "" {
		// Fire-and-forget per-turn summarisation onto a free LLM[1+] slot.
		// Skipped on the subtasks path — runSubtasks already fired it for
		// every iteration including the last, and double-firing would
		// duplicate the final pair's structured note in the shadow buffer.
		if !subtaskPath {
			a.backgroundSummarise(sess)
		}
		// Refresh .codehalter/.git_commit on any background slot picked by
		// pickBackgroundLLM. Overwrites by design, so it's safe to fire once
		// per Prompt regardless of subtaskPath.
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
//
// The decomposition planner already ran upstream — its plan response (with
// Subtasks listed) and its tool uses are already in sess.Messages, so each
// inner runTaskCycle sees that context naturally via buildLLMHistory.
func (a *agent) runSubtasks(ctx context.Context, sid string, subtasks []string) (toolLoopResult, error) {
	sess := a.getSession(sid)

	var header strings.Builder
	fmt.Fprintf(&header, "This looks like a big task. I'd break it into %d subtasks:\n", len(subtasks))
	for i, s := range subtasks {
		fmt.Fprintf(&header, "%d. %s\n", i+1, s)
	}
	a.sendUpdate(ctx, sid, MessageChunk(KindAgentMessage, TextBlock(header.String())))

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
		a.sendUpdate(ctx, sid, MessageChunk(KindAgentMessage, TextBlock("[Automatic] Running all subtasks without interruption.\n\n")))
	}

	var finalResult toolLoopResult

	for i, sub := range subtasks {
		a.sendUpdate(ctx, sid, MessageChunk(KindAgentMessage, TextBlock(
			fmt.Sprintf("\n=== Subtask %d/%d: %s ===\n\n", i+1, len(subtasks), sub),
		)))

		// Append the subtask scope as a fresh user message so the inner
		// planner has a clear, focused target. Following PLAN.md / EXECUTE.md
		// / VERIFY.md user messages will accumulate after it.
		if sess != nil {
			sess.AddUser("Subtask: " + sub)
			_ = sess.Save()
		}

		result, err := a.runTaskCycle(ctx, sid, nil, nil)
		if err != nil {
			return finalResult, err
		}
		finalResult = result

		// Per-subtask background summary so the shadow buffer covers
		// every turn pair, not just the final subtask's reply.
		if sess != nil && result.Text != "" {
			a.backgroundSummarise(sess)
		}
	}

	return finalResult, nil
}
