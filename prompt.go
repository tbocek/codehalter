package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
)

// This file owns the prompt orchestrator. Prompt() is the ACP entry point.
// The pipeline per turn is:
//
//   1. Plan once (read-only, decomposes into subtasks each with a verify
//      recipe). User confirms (skipped in autopilot).
//   2. For each subtask, run a single bounded tool-calling loop where the
//      executor self-verifies before calling respond.
//   3. If any subtask fails, re-plan with the failure context — up to
//      maxReplans times. User confirms each replan (skipped in autopilot).
//      Jaccard similarity over failure reasons escalates the replan note
//      ("same problem N times — try a structurally different approach").
//   4. Once every subtask in the current plan succeeds, fire the document
//      phase exactly once.
//
// The phase UI helpers (sendPhase, setStatus, finalizePlan) and the
// request-level error shaper (failPrompt, stopReasonFor) live here.

// maxReplans caps the number of planner retries per Prompt. The planner
// has the conversation history (showing what's been tried and what failed)
// plus the orchestrator's REPLAN note; 10 budget is generous enough that
// any feasible request resolves within it, while bounding the cost when
// a request is infeasible.
const maxReplans = 10

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
var phaseNames = []string{"Planning", "Working", "Documenting"}

// phaseEntries builds the multi-row plan entries to render in the client: all
// phases up to and including `phase`, with prior phases marked completed and
// phase `phase` marked in_progress or completed depending on `done`. Optional
// `suffix` is appended to whichever row is in_progress (used by setStatus to
// surface transient lifecycle markers like " (thinking…)" or " (running
// read_file…)"). Document (phase 2) only appears once it actually starts,
// so a no-doc run ends with two rows.
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
	a.sendUpdate(ctx, sid, planUpdate{Kind: "plan", Entries: entries})
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
	a.sendUpdate(ctx, sid, planUpdate{Kind: "plan", Entries: entries})
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
	a.sendUpdate(context.Background(), sid, planUpdate{Kind: "plan", Entries: entries})
}

// failPrompt records a fatal error in the session and returns it so the ACP
// dispatcher emits a JSON-RPC error response — Zed renders that as a red
// error box in the chat. Use only for failures that abort the prompt
// (LLM auth / out-of-credits / planAndAsk crash). Pass any tool uses
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
	slog.Debug("Prompt: enter", "sid", req.SessionId, "blocks", len(req.Content))
	ctx, cancel := context.WithCancel(ctx)
	a.mu.Lock()
	a.cancel = cancel
	a.mu.Unlock()
	defer cancel()
	defer a.finalizePlan(req.SessionId)

	// Abort wins over pending-question. Once ensureDevcontainer has decided
	// the session can't proceed (set abortReason), the pending UI prompt is
	// moot — the user should see the real reason, not "answer the question".
	// Each refused turn also appends the reason to chat: Zed locks an open
	// red box until the user dismisses it, so the error response alone is
	// invisible after the first turn — the chat append gives fresh feedback.
	a.mu.Lock()
	abort := a.abortReason
	a.mu.Unlock()
	slog.Debug("Prompt: abort gate", "sid", req.SessionId, "abortReason", abort)
	if abort != "" {
		a.sendUpdate(ctx, req.SessionId, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: abort + "\n"}})
		return a.failPrompt(req.SessionId, errors.New(abort), nil)
	}

	// Non-blocking: if the bootstrap goroutine is still running it's parked
	// on an interactive prompt (devcontainer OS choice, gitignore choice).
	// Block the typed prompt with a red box so the user is told to answer
	// the pending question instead of their message hanging silently.
	if a.indexDone != nil {
		select {
		case <-a.indexDone:
			slog.Debug("Prompt: indexDone gate passed", "sid", req.SessionId)
		default:
			slog.Debug("Prompt: indexDone gate refused (bootstrap still running)", "sid", req.SessionId)
			return a.failPrompt(req.SessionId, errors.New("Please answer the pending question above first."), nil)
		}
	} else {
		slog.Debug("Prompt: indexDone nil, no gate", "sid", req.SessionId)
	}

	// Capture isFirstMessage and seed sess.SystemPrompt BEFORE prepare runs.
	// prepare may dispatch proposeFix → AddUser → orchestrator to install a
	// missing dev tool, which both fills sess.Messages (flipping isFirstMessage)
	// and triggers an LLM call that needs sess.SystemPrompt to carry the
	// skills / cwd context. Seeding sess.SystemPrompt here also makes
	// generateTitle pick the user's actual request as the title — not the
	// synthetic install prompt.
	sess := a.getSession(req.SessionId)
	isFirstMessage := sess != nil && len(sess.Messages) == 0 && sess.Summary == ""
	if sess != nil && sess.SystemPrompt == "" {
		sysPrompt, err := a.systemPrompt(req.SessionId)
		if err != nil {
			return a.failPrompt(req.SessionId, err, nil)
		}
		sess.SystemPrompt = sysPrompt
	}

	// Clear per-turn read-dedup. Dedup is scoped to a single Prompt() turn
	// so that across-turn re-reads (after a user reply, edits made outside
	// our process, etc.) still go through.
	if sess != nil {
		sess.readDedupMu.Lock()
		sess.readDedup = nil
		sess.readDedupMu.Unlock()
	} else {
		slog.Debug("Prompt: pre-turn getSession NIL", "sid", req.SessionId, "knownSessions", len(a.sessions))
	}

	// Extract user text and images from prompt blocks.
	var userText string
	var images []ImageData
	for _, block := range req.Content {
		switch block.Type {
		case "text":
			userText += block.Text
		case "image":
			images = append(images, ImageData{MimeType: block.MimeType, Data: block.Data})
		}
	}

	// The empty-project hint stays folded into the first user message because
	// it's a one-shot nudge — once the project has files, we want the
	// summariser to drop it naturally rather than re-injecting it forever.
	stored := userText
	if isFirstMessage {
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

	result, err := a.orchestrate(ctx, req.SessionId)
	if err != nil {
		if isCancelled(err) {
			return PromptResponse{StopReason: "cancelled"}, nil
		}
		return a.failPrompt(req.SessionId, err, nil)
	}

	// Each phase already persisted its own response via UpsertLastAssistant.
	// The epilogue here only fires background work that observes the
	// now-complete transcript.
	if sess != nil && result.Text != "" {
		// Per-subtask backgroundSummarise already fires inside orchestrate
		// for every subtask iteration; the final pair is covered too, so we
		// don't double-summarise here.
		a.backgroundGitCommit(sess)
		a.compressHistory(ctx, sess)
	}

	// Prepare phase runs at the end of every turn so the capabilities banner /
	// missing-tool fix cards are ready BEFORE the next user prompt arrives.
	// Bootstrap's first prepare (startIndexing) covers turn #1; this covers
	// turns #2+. Re-verifies the LLM (loops on a Retry card if unreachable),
	// refreshes env snapshot + MCP reconcile, emits one consolidated banner
	// only when something changed. Silent in steady state.
	slog.Debug("Prompt: about to call prepare (post-turn)", "sid", req.SessionId, "sessNil", sess == nil)
	if sess != nil {
		a.prepare(ctx, sess, req.SessionId)
	}

	return PromptResponse{StopReason: stopReasonFor(ctx)}, nil
}

// orchestrate drives the plan → subtasks → replan → document pipeline for
// one user turn. Returns the final subtask's result (used for the
// background epilogue) and the first error that isn't recoverable via
// replan. User cancellation surfaces as errUserCancelled.
func (a *agent) orchestrate(ctx context.Context, sid string) (toolLoopResult, error) {
	sess := a.getSession(sid)

	a.sendPhase(ctx, sid, 0, false)
	firstConn, plan, firstToolUses, err := a.planAndAsk(ctx, sid, "")
	if err != nil {
		if isCancelled(err) {
			return toolLoopResult{}, err
		}
		if sess != nil && len(firstToolUses) > 0 {
			sess.AddAssistantWithTools("❌ "+err.Error(), firstToolUses)
			_ = sess.Save()
		}
		return toolLoopResult{}, err
	}
	if plan == nil {
		// No PLAN.md or unparseable response — pipeline cannot proceed.
		return toolLoopResult{}, fmt.Errorf("planner returned no usable plan")
	}
	if len(plan.Subtasks) == 0 {
		// Clarification path: planner asked a question via clear=false +
		// choices, planAndAsk resolved it; if subtasks still empty, the
		// user's pick implied "no work needed".
		return toolLoopResult{}, nil
	}

	if err := a.confirmPlan(ctx, sid, plan, false); err != nil {
		return toolLoopResult{}, err
	}

	var lastResult toolLoopResult
	var failureBags []map[string]bool
	replans := 0

	for {
		allOk := true
		var failedAt int
		var failedReason string

		for i, st := range plan.Subtasks {
			a.sendPhase(ctx, sid, 1, false)
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("\n=== Subtask %d/%d: %s ===\n\n", i+1, len(plan.Subtasks), st.Description)}})

			outcome := a.runSubtaskLoop(ctx, sid, st, i, len(plan.Subtasks))
			lastResult = outcome.Result

			if sess != nil && outcome.Result.Text != "" {
				a.backgroundSummarise(sess)
			}

			if outcome.Success {
				continue
			}
			allOk = false
			failedAt = i
			failedReason = outcome.Reason
			break
		}

		if allOk {
			break
		}

		// Jaccard escalation: same failure surfaced before?
		bag := issueBag([]string{failedReason})
		dupCount := 1
		for _, prev := range failureBags {
			if jaccard(bag, prev) >= failureSimilarityThreshold {
				dupCount++
			}
		}
		failureBags = append(failureBags, bag)

		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("⚠ Subtask %d/%d failed: %s\n", failedAt+1, len(plan.Subtasks), failedReason)}})

		replans++
		if replans >= maxReplans {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("⚠ Replan budget (%d) exhausted — giving up.\n", maxReplans)}})
			return lastResult, nil
		}

		var replanCtx string
		if dupCount >= 2 {
			replanCtx = fmt.Sprintf("REPLAN: prior subtask failed: %s. Same failure has surfaced %d times — the prior fix didn't work; propose a structurally different approach. See history for executor attempts. Follow the 'Replanning' section in PLAN.md.", failedReason, dupCount)
		} else {
			replanCtx = fmt.Sprintf("REPLAN: prior subtask failed: %s. See history for executor attempts. Follow the 'Replanning' section in PLAN.md.", failedReason)
		}

		a.sendPhase(ctx, sid, 0, false)
		_, newPlan, _, err := a.planAndAsk(ctx, sid, replanCtx)
		if err != nil {
			if isCancelled(err) {
				return lastResult, err
			}
			return lastResult, err
		}
		if newPlan == nil || len(newPlan.Subtasks) == 0 {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Replan produced no further subtasks — stopping.\n"}})
			return lastResult, nil
		}

		if err := a.confirmPlan(ctx, sid, newPlan, true); err != nil {
			return lastResult, err
		}
		plan = newPlan
	}

	// Document phase: fire once at the end of a successful prompt.
	if firstConn != nil {
		a.sendPhase(ctx, sid, 2, false)
		lastResult, _ = a.document(ctx, sid, firstConn, lastResult)
		a.sendPhase(ctx, sid, 2, true)
	} else {
		a.sendPhase(ctx, sid, 1, true)
	}

	return lastResult, nil
}

// confirmPlan renders the planned subtasks and asks the user how to run
// them. Three-way: Execute (one subtask at a time), Automatic (flip to
// autopilot for the rest of this Prompt), Cancel. Skipped entirely when
// already in autopilot mode. Skipped for report_only plans where no
// mutating work happens. Replan plans use the same gate so the user sees
// the new approach before it runs.
func (a *agent) confirmPlan(ctx context.Context, sid string, plan *planResult, isReplan bool) error {
	header := "Plan:"
	if isReplan {
		header = "Replan:"
	}
	if plan.ReportOnly {
		header = "Findings:"
	}
	a.renderSubtasks(ctx, sid, plan.Subtasks, header)

	if plan.ReportOnly {
		return nil
	}
	if a.isAutopilot() {
		return nil
	}

	tcId := a.StartToolCall(ctx, sid, "How should I run these?", "think", nil)
	choice, err := a.askChoiceAuto(ctx, sid, tcId, []string{"Execute", "Automatic"})
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + choice)})

	sess := a.getSession(sid)
	if err != nil || choice == "abort" {
		appendAssistantNote(sess, "User declined execution.")
		if sess != nil {
			_ = sess.Save()
		}
		return errUserCancelled
	}

	if choice == "Automatic" {
		a.mu.Lock()
		a.mode = modeAutopilot
		a.mu.Unlock()
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "[Automatic] Running without further interruption.\n\n"}})
	}
	return nil
}
