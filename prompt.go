package main

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
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
// maxShadowInputBytes caps the bytes any single payload contributes to a
// background LLM call (per-turn summaries via backgroundSummarise, git diffs
// via backgroundGitCommit). Without this, a turn that produced megabytes of
// tool output (huge run_command dumps) would blow through the background
// LLM's own context window.
const maxShadowInputBytes = 20 * 1024

// clipBytes truncates s to at most max bytes, leaving a marker in the middle
// when it had to cut. Used to bound any single payload's contribution to a
// background LLM call (turn summaries via backgroundSummarise, git diffs via
// backgroundGitCommit) so one giant tool output can't blow the summariser's
// own context window.
func clipBytes(s string, max int) string {
	if len(s) <= max {
		return s
	}
	half := max / 2
	return s[:half] + fmt.Sprintf("\n[... %d bytes truncated ...]\n", len(s)-max) + s[len(s)-half:]
}

// resourcePath returns the local filesystem path an ACP resource URI points
// at: a file:// URI collapses to its percent-decoded path with any fragment
// (e.g. an editor line range "#L801-836") stripped, so the model can pass it
// straight to read_file. Non-file or unparseable URIs are returned verbatim.
func resourcePath(uri string) string {
	if uri == "" {
		return ""
	}
	if u, err := url.Parse(uri); err == nil && (u.Scheme == "file" || u.Scheme == "") && u.Path != "" {
		return u.Path
	}
	return strings.TrimPrefix(uri, "file://")
}

// resourceLabel is resourcePath plus the URI fragment in parentheses when
// present — used as a human-readable header for an attached snippet so the
// model sees which file (and line range) the context came from.
func resourceLabel(uri string) string {
	if uri == "" {
		return "attachment"
	}
	if u, err := url.Parse(uri); err == nil && (u.Scheme == "file" || u.Scheme == "") && u.Path != "" {
		if u.Fragment != "" {
			return u.Path + " (" + u.Fragment + ")"
		}
		return u.Path
	}
	return strings.TrimPrefix(uri, "file://")
}

// maxLinkedResourceBytes bounds how much of a whole-file resource_link we inline
// when it carries no line range — enough for a normal source file, capped so a
// huge attachment can't blow the prompt.
const maxLinkedResourceBytes = 32 * 1024

// readLinkedResource reads the file a resource_link / embedded-resource URI
// points at and returns it as an inline snippet plus a display label, honouring
// a #L<start>-<end> line-range fragment. ok is false (caller falls back to just
// noting the path) when the file is outside cwd, missing, or the URI isn't a
// readable local file — so the model never read-loops hunting for content the
// editor already handed us. Only files inside cwd are inlined.
func readLinkedResource(cwd, uri string) (snippet, label string, ok bool) {
	if cwd == "" || uri == "" {
		return "", "", false
	}
	path, start, end := parseResourceURI(uri)
	if path == "" {
		return "", "", false
	}
	clean := filepath.Clean(path)
	if !filepath.IsAbs(clean) {
		clean = filepath.Join(cwd, clean)
	}
	root := filepath.Clean(cwd)
	if clean != root && !strings.HasPrefix(clean, root+string(filepath.Separator)) {
		return "", "", false
	}
	data, err := os.ReadFile(clean)
	if err != nil {
		return "", "", false
	}
	base := filepath.Base(clean)
	if start > 0 {
		lines := strings.Split(string(data), "\n")
		if start > len(lines) {
			start = len(lines)
		}
		if end < start || end > len(lines) {
			end = len(lines)
		}
		return strings.Join(lines[start-1:end], "\n"), fmt.Sprintf("%s:%d-%d", base, start, end), true
	}
	s := string(data)
	if len(s) > maxLinkedResourceBytes {
		s = s[:maxLinkedResourceBytes] + "\n[... truncated ...]"
	}
	return s, base, true
}

// parseResourceURI splits a resource URI into its local path and an optional
// line range from the fragment (file:///x/llm.go#L810-845 → "/x/llm.go", 810,
// 845). Non-file or unparseable URIs return an empty path.
func parseResourceURI(uri string) (path string, start, end int) {
	if uri == "" {
		return "", 0, 0
	}
	var frag string
	if u, err := url.Parse(uri); err == nil && (u.Scheme == "file" || u.Scheme == "") && u.Path != "" {
		path = u.Path
		frag = u.Fragment
	} else {
		path = strings.TrimPrefix(uri, "file://")
		if i := strings.IndexByte(path, '#'); i >= 0 {
			frag, path = path[i+1:], path[:i]
		}
	}
	start, end = parseLineRange(frag)
	return path, start, end
}

// parseLineRange pulls a 1- or 2-number line range out of a URI fragment,
// tolerating the common encodings: "L810-845", "810:845", "810-845", "L810".
// Returns 0,0 when the fragment carries no digits (whole-file reference).
func parseLineRange(frag string) (int, int) {
	var nums []int
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			if n, err := strconv.Atoi(cur.String()); err == nil {
				nums = append(nums, n)
			}
			cur.Reset()
		}
	}
	for _, r := range frag {
		if r >= '0' && r <= '9' {
			cur.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	switch len(nums) {
	case 0:
		return 0, 0
	case 1:
		return nums[0], nums[0]
	default:
		return nums[0], nums[1]
	}
}

func isCancelled(err error) bool {
	return errors.Is(err, errUserCancelled) ||
		errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded)
}

// cancelReason renders a short human explanation for a cancelled turn so the
// stop is never silent. errUserCancelled is the in-app Abort/clarification
// path; a bare context.Canceled means the editor aborted the request (Cancel
// button, session switch, or — the case that bit us — a client-side request
// timeout while we were waiting on a busy LLM); DeadlineExceeded is a context
// deadline (codehalter sets none on the foreground, so it's the client's).
func cancelReason(err error) string {
	switch {
	case errors.Is(err, errUserCancelled):
		return "you stopped it"
	case errors.Is(err, context.DeadlineExceeded):
		return "a deadline was exceeded (client-side timeout)"
	case errors.Is(err, context.Canceled):
		return "the editor aborted the request (Cancel button, or a client-side timeout while the LLM was busy)"
	default:
		return "cancelled"
	}
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

// phaseActive reports whether a foreground phase is currently in progress for
// this session — used to gate the waiting-meter's stall warning so a slow
// background call (summariser / git-commit) doesn't emit a "server busy" line.
func (a *agent) phaseActive(sid string) bool {
	sess := a.getSession(sid)
	if sess == nil {
		return false
	}
	sess.phaseMu.Lock()
	defer sess.phaseMu.Unlock()
	return sess.phaseActive
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
	// skills / cwd context.
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

	// Extract user text and images from prompt blocks. Image bytes are
	// content-addressed (sha256[:8]) and written under .codehalter/images so
	// the ACP wire-side base64 doesn't get persisted in session.toml; what
	// stays on Message.Images is just {id, mime}. Re-pasting the same
	// screenshot collides on id and skips the write.
	var userText string
	var images []ImageData
	for _, block := range req.Content {
		slog.Debug("prompt: content block", "type", block.Type, "uri", block.URI, "hasResource", block.Resource != nil)
		switch block.Type {
		case "text":
			userText += block.Text
		case "image":
			bytes, err := base64.StdEncoding.DecodeString(block.Data)
			if err != nil {
				slog.Warn("prompt: skipping image with undecodable base64", "err", err)
				continue
			}
			// Content-addressed id ("img_<sha256[:8] hex>") — same bytes →
			// same id → same file path, so a re-pasted screenshot doesn't
			// re-write the store.
			h := sha256.Sum256(bytes)
			id := "img_" + hex.EncodeToString(h[:8])
			if sess != nil {
				if err := writeImageFile(sess.Cwd, id, block.MimeType, bytes); err != nil {
					slog.Warn("prompt: writing image file failed", "id", id, "err", err)
					continue
				}
			}
			images = append(images, ImageData{ID: id, MimeType: block.MimeType})
		case "resource":
			// Embedded resource: an editor selection / file excerpt attached via
			// Zed's "@ include context". The snippet text lives inline; without
			// this case it was silently dropped and the model saw only the bare
			// prompt, so a reference like "why do we need this?" had no referent.
			if block.Resource == nil {
				slog.Debug("prompt: resource block with no embedded resource")
				continue
			}
			label := resourceLabel(block.Resource.URI)
			switch {
			case block.Resource.Text != "":
				userText += fmt.Sprintf("\n\n[Attached context from %s]\n```\n%s\n```\n", label, block.Resource.Text)
			case block.Resource.Blob != "":
				// Binary embedded resource (rare from editors — images arrive as
				// "image" blocks). Note it rather than inlining opaque bytes.
				userText += fmt.Sprintf("\n\n[Attached binary resource %s (%s) — not inlined]\n", label, block.Resource.MimeType)
			default:
				// No inline text/blob but a URI — same fallback as resource_link:
				// read the linked file so the reference still resolves.
				cwd := ""
				if sess != nil {
					cwd = sess.Cwd
				}
				if snippet, l, ok := readLinkedResource(cwd, block.Resource.URI); ok {
					userText += fmt.Sprintf("\n\n[Attached context from %s]\n```\n%s\n```\n", l, snippet)
				} else {
					slog.Debug("prompt: empty embedded resource", "uri", block.Resource.URI)
				}
			}
		case "resource_link":
			// A pointer to a file (no inline content). Pull the linked file in —
			// honouring a #L<start>-<end> line range — so a bare reference like
			// "why do we need this?" resolves immediately instead of forcing the
			// model to read_file and risk a read-loop hunting for the snippet.
			// Falls back to noting the path when the file is outside the
			// workspace or unreadable.
			cwd := ""
			if sess != nil {
				cwd = sess.Cwd
			}
			if snippet, label, ok := readLinkedResource(cwd, block.URI); ok {
				userText += fmt.Sprintf("\n\n[Attached context from %s]\n```\n%s\n```\n", label, snippet)
			} else {
				name := block.Name
				path := resourcePath(block.URI)
				if name == "" {
					name = path
				}
				userText += fmt.Sprintf("\n\n[Referenced file: %s (%s)]\n", name, path)
			}
		default:
			slog.Debug("prompt: ignoring unsupported content block", "type", block.Type)
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

	// Reject prompts whose text alone would breach the compaction trigger.
	// Even on a fresh session, system prompt + skills + the assistant
	// reply won't fit alongside a user message that big, and letting
	// llmStream error out mid-stream is worse UX than refusing up front.
	// bytes/4 is a rough chars-per-token estimate; precision doesn't matter
	// here since the trigger leaves ~20% headroom anyway.
	if a.mainSlotTokens > 0 {
		estTokens := len(userText) / 4
		triggerTokens := a.mainSlotTokens * compactTriggerPct / 100
		if estTokens > triggerTokens {
			return a.failPrompt(req.SessionId, fmt.Errorf("message too large: ~%d tokens estimated, exceeds the %d-token compaction trigger (per-slot n_ctx %d × %d%%). Trim the prompt or restart your server with a larger -c N / --max-model-len N", estTokens, triggerTokens, a.mainSlotTokens, compactTriggerPct), nil)
		}
	}

	slog.Info("Prompt", "sid", req.SessionId, "sessions", len(a.sessions))

	result, err := a.orchestrate(ctx, req.SessionId)
	if err != nil {
		if isCancelled(err) {
			// Never silent: log the cancellation and tell the user why. Covers
			// the in-app Abort/Cancel button AND the editor aborting the turn
			// (e.g. a client-side request-deadline timeout while the LLM was
			// busy) — the latter used to vanish with no trace at all. Background
			// ctx for the notice since the request ctx is already cancelled.
			reason := cancelReason(err)
			slog.Warn("Prompt: turn cancelled", "sid", req.SessionId, "reason", reason, "err", err)
			a.sendUpdate(context.Background(), req.SessionId, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⏹ Turn cancelled — " + reason + ".\n"}})
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
	plan, firstToolUses, err := a.planAndAsk(ctx, sid, "")
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
		newPlan, _, err := a.planAndAsk(ctx, sid, replanCtx)
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

	// Document phase: fire once at the end of a successful prompt. Routes
	// to a non-foreground LLM entry internally so llm[0]'s prefix cache
	// isn't evicted by a one-shot README update.
	a.sendPhase(ctx, sid, 2, false)
	// document logs its own failures at Warn and always returns (exec, nil) —
	// a failed README update must not fail the whole turn — so the discarded
	// error is intentional and never non-nil, not a silent swallow.
	lastResult, _ = a.document(ctx, sid, lastResult)
	a.sendPhase(ctx, sid, 2, true)

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

// gitCommitFile is the path (relative to cwd) where the background updater
// stores the current draft commit message. EXECUTE.md tells the agent to
// hand this path to the user via `git commit -F` rather than committing
// itself. .codehalter/ is gitignored on first bootstrap so the file never
// gets accidentally staged.
const gitCommitFile = ".codehalter/.git_commit"

const gitCommitPrompt = "You are drafting a git commit message for the user's CURRENT uncommitted changes. " +
	"Reply with ONLY the message text — no preamble, no code fences.\n\n" +
	"Format:\n" +
	"- Line 1: short imperative subject, ≤72 chars (e.g. \"add ...\", \"fix ...\", \"refactor ...\").\n" +
	"- Blank line.\n" +
	"- Body: 1-3 short bullets or sentences covering WHY the change is being made, not a restatement of the diff.\n"

// gitStatusPorcelain runs `git status --porcelain` in cwd. Empty output means
// the working tree (including the index) is clean. Returns ("", err) when
// cwd isn't a git checkout or git isn't on PATH — callers treat that as
// "no work to do" without surfacing the error to the user.
func gitStatusPorcelain(cwd string) (string, error) {
	out, err := exec.Command("git", "-C", cwd, "status", "--porcelain").Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// gitDiffHead returns `git diff HEAD` in cwd — staged + unstaged tracked
// changes against HEAD. Untracked files are NOT included here; the porcelain
// status fed alongside this output is what lists those.
func gitDiffHead(cwd string) (string, error) {
	out, err := exec.Command("git", "-C", cwd, "diff", "HEAD").Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// cleanupGitCommitIfClean is called from checkMCP on every user prompt.
// Waits for any in-flight backgroundGitCommit to finish (otherwise its late
// write would resurrect a file we are about to delete), then checks
// `git status --porcelain`. Empty status means everything was committed
// externally — delete .codehalter/.git_commit so the next round starts
// fresh from the next uncommitted change.
func (a *agent) cleanupGitCommitIfClean(cwd string, sid string) {
	if info, err := os.Stat(filepath.Join(cwd, ".git")); err != nil || !info.IsDir() {
		return
	}
	if sess := a.getSession(sid); sess != nil {
		sess.gitCommitJob.Wait()
	}
	status, err := gitStatusPorcelain(cwd)
	if err != nil {
		return
	}
	if strings.TrimSpace(status) != "" {
		return
	}
	if err := os.Remove(filepath.Join(cwd, gitCommitFile)); err != nil && !os.IsNotExist(err) {
		slog.Debug("cleanupGitCommitIfClean: remove failed", "err", err)
	}
	// Reset hash so the next non-empty status regenerates the file, even
	// if (rarely) the new status+diff hashes identical to the prior one.
	if sess := a.getSession(sid); sess != nil {
		sess.gitCommitMu.Lock()
		sess.gitCommitLastHash = [32]byte{}
		sess.gitCommitMu.Unlock()
	}
}

// backgroundGitCommit fires after every assistant turn. Snapshots the current
// `git diff HEAD` + `git status --porcelain` and asks the LLM to (re)write
// .codehalter/.git_commit so it always matches the latest uncommitted state.
// Self-skips when:
//   - cwd has no .git directory (not a checkout, or .git not mounted),
//   - the working tree is clean (nothing to summarise),
//   - no eligible background slot is available (see pickBackgroundLLM).
//
// Multiple in-flight calls are allowed — they race on the file with
// last-write-wins, which is fine because each LLM call's snapshot is point-in-
// time and the freshest wins. The pre-write status re-check guards against
// the narrow race where the user commits during the LLM call. When this and
// the shadow summariser land on the same entry, the per-conn parallel
// semaphore in llmStream serialises them naturally — no explicit join needed.
func (a *agent) backgroundGitCommit(sess *Session) {
	if sess == nil {
		return
	}
	if info, err := os.Stat(filepath.Join(sess.Cwd, ".git")); err != nil || !info.IsDir() {
		return
	}
	status, err := gitStatusPorcelain(sess.Cwd)
	if err != nil || strings.TrimSpace(status) == "" {
		return
	}
	diff, err := gitDiffHead(sess.Cwd)
	if err != nil {
		// Diff is best-effort enrichment for the commit-message prompt;
		// status (already validated above) is the primary signal. Proceed
		// with an empty diff but don't drop the error silently.
		slog.Debug("backgroundGitCommit: git diff failed", "sid", sess.ID, "err", err)
	}
	hash := sha256.Sum256([]byte(status + "\x00" + diff))
	sess.gitCommitMu.Lock()
	unchanged := hash == sess.gitCommitLastHash
	sess.gitCommitMu.Unlock()
	if unchanged {
		return
	}

	if !sess.gitCommitJob.TryStart() {
		return
	}
	conn := a.pickBackgroundLLM()
	if conn == nil {
		sess.gitCommitJob.Done()
		return
	}

	go func() {
		defer sess.gitCommitJob.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		var buf strings.Builder
		buf.WriteString(gitCommitPrompt)
		buf.WriteString("\n<git_status>\n")
		buf.WriteString(status)
		buf.WriteString("</git_status>\n")
		if strings.TrimSpace(diff) != "" {
			buf.WriteString("\n<git_diff>\n")
			buf.WriteString(clipBytes(diff, maxShadowInputBytes))
			buf.WriteString("\n</git_diff>\n")
		}

		out, err := a.llmSimple(ctx, sess.ID, conn, []llmMessage{{Role: "user", Content: buf.String()}})
		if err != nil {
			slog.Debug("backgroundGitCommit: llm call failed", "sid", sess.ID, "err", err)
			return
		}

		// Race guard: re-check status before write. If the user committed
		// during the LLM call, skip — otherwise we'd resurrect a stale file
		// that cleanupGitCommitIfClean has not yet had a chance to delete.
		if s2, err := gitStatusPorcelain(sess.Cwd); err == nil && strings.TrimSpace(s2) == "" {
			return
		}

		path := filepath.Join(sess.Cwd, gitCommitFile)
		_ = os.MkdirAll(filepath.Dir(path), 0o755)
		if err := os.WriteFile(path, []byte(strings.TrimSpace(out)+"\n"), 0o644); err != nil {
			slog.Debug("backgroundGitCommit: writing .git_commit failed", "path", path, "err", err)
			return
		}
		sess.gitCommitMu.Lock()
		sess.gitCommitLastHash = hash
		sess.gitCommitMu.Unlock()
	}()
}
