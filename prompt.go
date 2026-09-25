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
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// This file owns the prompt orchestrator; Prompt() is the ACP entry point. Per
// turn: plan once (read-only, subtasks each with a verify recipe), run each
// subtask as one bounded tool loop that self-verifies before `respond`, replan
// on failure up to maxReplans times (Jaccard similarity over failure reasons
// escalates the note when the same problem recurs), then document once. The
// phase UI helpers and the request-level error shaper live here too.

// maxReplans caps the number of planner retries per Prompt. The planner has
// the conversation history (showing what's been tried and what failed) plus the
// orchestrator's REPLAN note. The budget is deliberately generous: the execute
// loop now caps failed rounds low (executeFailCap) and bounces stuck subtasks
// here, and replans are the only place web_search/web_read run — so most hard problems
// resolve across several cheap, web-capable replan rounds rather than one long
// web-blind execute loop. Still bounds total cost when a request is infeasible.
const maxReplans = 20

// maxUpserts caps how many times the executor may revise the plan mid-run via
// submit_plan before the orchestrator stops, so a model that re-plans instead of
// executing can't spin. Separate from (and additive to) the replan budget.
const maxUpserts = 20

// errUserCancelled flags a deliberate stop initiated by the user (e.g. they
// chose Abort in a tool-choice prompt). It's NOT an error to surface as a
// red box — the prompt returns a clean PromptResponse with stopReason
// "cancelled" when this sentinel reaches the top level.
var errUserCancelled = errors.New("user cancelled")

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
	path, frag := parseResourceURI(uri)
	start, end := parseLineRange(frag)
	if path == "" {
		return "", "", false
	}
	clean := filepath.Clean(path)
	if !filepath.IsAbs(clean) {
		clean = filepath.Join(cwd, clean)
	}
	// realInside, not a prefix test: a symlink living in the project but
	// pointing out of it passes a prefix test and would be inlined (see
	// resolvePath, which refuses the same path for the file tools).
	if !realInside(clean, cwd) {
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
	if len(s) > maxLLMInputBytes {
		s = clipUTF8(s, maxLLMInputBytes) + "\n[... truncated ...]"
	}
	return s, base, true
}

// parseResourceURI splits an ACP resource URI into the local path it points at
// (percent-decoded) and its fragment, which editors use for a line range
// (file:///x/llm.go#L810-845 → "/x/llm.go", "L810-845"). A URI that is not a
// file URI comes back as its own path, so the caller can still name it.
func parseResourceURI(uri string) (path, frag string) {
	if u, err := url.Parse(uri); err == nil && (u.Scheme == "file" || u.Scheme == "") && u.Path != "" {
		return u.Path, u.Fragment
	}
	path = strings.TrimPrefix(uri, "file://")
	if i := strings.IndexByte(path, '#'); i >= 0 {
		path, frag = path[:i], path[i+1:]
	}
	return path, frag
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

// promptContent turns a prompt's content blocks into the user's text and the
// images attached to it. Attached files and selections are inlined into the
// text, so a reference like "why do we need this?" has its referent. Image
// bytes are content-addressed (sha256[:8]) and written under
// .codehalter/images, so the wire's base64 never lands in session.toml and a
// re-pasted screenshot skips the write; what stays on the message is {id, mime}.
func promptContent(cwd string, blocks []ContentBlock) (text string, images []ImageData) {
	for _, block := range blocks {
		slog.Debug("prompt: content block", "type", block.Type, "uri", block.URI, "hasResource", block.Resource != nil)
		switch block.Type {
		case "text":
			text += block.Text
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
			if cwd != "" {
				if err := writeImageFile(cwd, id, block.MimeType, bytes); err != nil {
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
			label, frag := parseResourceURI(block.Resource.URI)
			switch {
			case label == "":
				label = "attachment"
			case frag != "":
				label += " (" + frag + ")"
			}
			switch {
			case block.Resource.Text != "":
				text += fmt.Sprintf("\n\n[Attached context from %s]\n```\n%s\n```\n", label, block.Resource.Text)
			case block.Resource.Blob != "":
				// Binary embedded resource (rare from editors — images arrive as
				// "image" blocks). Note it rather than inlining opaque bytes.
				text += fmt.Sprintf("\n\n[Attached binary resource %s (%s) — not inlined]\n", label, block.Resource.MimeType)
			default:
				// No inline text/blob but a URI — same fallback as resource_link:
				// read the linked file so the reference still resolves.
				if snippet, l, ok := readLinkedResource(cwd, block.Resource.URI); ok {
					text += fmt.Sprintf("\n\n[Attached context from %s]\n```\n%s\n```\n", l, snippet)
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
			if snippet, label, ok := readLinkedResource(cwd, block.URI); ok {
				text += fmt.Sprintf("\n\n[Attached context from %s]\n```\n%s\n```\n", label, snippet)
			} else {
				name := block.Name
				path, _ := parseResourceURI(block.URI)
				if name == "" {
					name = path
				}
				text += fmt.Sprintf("\n\n[Referenced file: %s (%s)]\n", name, path)
			}
		default:
			slog.Debug("prompt: ignoring unsupported content block", "type", block.Type)
		}
	}
	return text, images
}

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

// startStatusMeter refreshes the active phase row once a second with whatever
// render() returns, so anything that takes a while (an LLM round trip, a slow
// tool) shows a climbing counter instead of a frozen row. setStatus is a no-op
// when no phase is active, so this is safe to run unconditionally.
//
// The returned stop() halts the ticker AND waits for the goroutine, so a late
// tick can never re-set the row after the caller has cleared it. Callers that
// also `defer a.setStatus(ctx, sid, "")` must register that defer FIRST, so LIFO
// runs the join before the clear.
func (a *agent) startStatusMeter(ctx context.Context, sid string, render func() string) (stop func()) {
	done, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				a.setStatus(ctx, sid, render())
			}
		}
	}()
	return func() { close(done); <-stopped }
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
// (LLM auth / out-of-credits / runPlanPhase crash). Pass any tool uses
// captured before the failure so they're preserved in history. Recoverable
// warnings should keep using sendUpdate with a "⚠ ..." chunk.
func (a *agent) failPrompt(sid string, err error, toolUses []ToolUse) (PromptResponse, error) {
	if sess := a.getSession(sid); sess != nil {
		if len(toolUses) > 0 {
			sess.AddAssistantWithTools("❌ "+err.Error(), toolUses)
		} else {
			sess.AddAssistant("❌ " + err.Error())
		}
		sess.saveOrLog()
	}
	return PromptResponse{}, err
}

// sessionTitleMax is how many runes of the opening message become the thread
// name. Long enough to keep a real sentence, short enough not to be elided by
// the client's own thread list.
const sessionTitleMax = 60

// deriveTitle turns a user message into a one-line thread name: the first
// non-blank line, whitespace collapsed, cut to sessionTitleMax on a word
// boundary. Returns "" for a message with no text at all (an image-only
// prompt), which leaves the thread unnamed rather than naming it "".
func deriveTitle(raw string) string {
	var line string
	for _, l := range strings.Split(raw, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			line = l
			break
		}
	}
	line = strings.Join(strings.Fields(line), " ")
	if line == "" || utf8.RuneCountInString(line) <= sessionTitleMax {
		return line
	}
	cut := string([]rune(line)[:sessionTitleMax])
	// Only back up to a word boundary if one is reasonably close to the limit;
	// a single 60-rune token would otherwise collapse to almost nothing.
	if i := strings.LastIndex(cut, " "); i > sessionTitleMax/2 {
		cut = cut[:i]
	}
	return strings.TrimRight(cut, " ,.;:-") + "…"
}

// setSessionTitle names the thread in the client. ACP has no request for this:
// the agent volunteers a session_info_update, and a client that doesn't
// implement it ignores the unknown update kind, so there is no capability to
// gate on. Title is also persisted, so LoadSession can re-announce it.
func (a *agent) setSessionTitle(ctx context.Context, sess *Session, raw string) {
	title := deriveTitle(raw)
	if title == "" || title == sess.Title {
		return
	}
	sess.Title = title
	a.sendUpdate(ctx, sess.ID, sessionInfoUpdate{
		Kind:      "session_info_update",
		Title:     title,
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	})
}

func (a *agent) Prompt(ctx context.Context, req PromptRequest) (PromptResponse, error) {
	slog.Debug("Prompt: enter", "sid", req.SessionId, "blocks", len(req.Content))

	// Typing while a turn runs STEERS it: the text is queued and the running
	// turn picks it up between tool rounds (runToolLoop), so "also update the
	// README" costs a round rather than the whole turn. Interrupting is the
	// editor's stop button, which cancels the turn's context.
	//
	// Attached files and selections travel with it, inlined as text. Images do
	// not: the queue hands the model a text message, and the reply says so
	// rather than dropping them silently.
	if sess := a.getSession(req.SessionId); sess != nil && sess.turnRunning() {
		text, images := promptContent(sess.Cwd, req.Content)
		// "/spec stop" is an instruction to the loop, not text for the model.
		// The round in flight finishes and commits; the loop then reports and
		// returns instead of picking the next item.
		if strings.TrimSpace(text) == "/spec stop" {
			if sess.specFence() == "" {
				a.say(ctx, req.SessionId, "No /spec loop is running in this session.\n")
			} else {
				sess.requestSpecStop()
				a.say(ctx, req.SessionId, "⏹ /spec stops after the round in flight; its commit lands first. `/spec` later resumes where the ledger says.\n")
			}
			return PromptResponse{StopReason: "end_turn"}, nil
		}
		if strings.TrimSpace(text) != "" {
			sess.addSteer(text)
			note := "↪ Queued for the turn in flight, it lands at its next step. Stop the turn to interrupt it instead.\n"
			if len(images) > 0 {
				note = fmt.Sprintf("↪ Queued your message for the turn in flight (%d image(s) left out, send them once it finishes). Stop the turn to interrupt it instead.\n", len(images))
			}
			a.say(ctx, req.SessionId, note)
			return PromptResponse{StopReason: "end_turn"}, nil
		}
	}

	// A typed prompt replaces the turn in flight and waits for it to unwind
	// (holdTurn, turn.go). release runs on every exit path below.
	if sess := a.getSession(req.SessionId); sess != nil {
		var release func()
		ctx, release, _ = a.holdTurn(ctx, sess, true)
		defer release()
	} else {
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(ctx)
		defer cancel()
	}

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
		a.say(ctx, req.SessionId, abort+"\n")
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

	// Capture isFirstMessage and seed sess.SystemPrompt BEFORE the pre-turn
	// checks run. prepareChecks rebuilds the system prompt (skills / cwd context)
	// and an LLM call needs it seeded; isFirstMessage is read later for the
	// empty-project hint.
	sess := a.getSession(req.SessionId)
	isFirstMessage := sess != nil && len(sess.Messages) == 0 && sess.Summary == ""
	if sess != nil && sess.SystemPrompt == "" {
		sysPrompt, err := a.systemPrompt(req.SessionId)
		if err != nil {
			return a.failPrompt(req.SessionId, err, nil)
		}
		sess.SystemPrompt = sysPrompt
	}

	// Pre-turn freshness pass: verify a reachable LLM (looping on a Retry card
	// if not), refresh the env snapshot, and reconcile mcp.toml so THIS turn
	// runs against current config — a server or tool added last turn is live
	// before its work starts. Cheap in steady state (every check short-circuits).
	// Detected problems are held and offered as fix cards post-turn via
	// drainFixes, so an accepted card can't jump ahead of the user's request.
	var pendingFixes []fixProblem
	if sess != nil {
		pendingFixes = a.prepareChecks(ctx, sess, req.SessionId)
	}

	cwd := ""
	if sess != nil {
		cwd = sess.Cwd
	}
	userText, images := promptContent(cwd, req.Content)

	// Name the thread from its opening request. Done here, BEFORE macro
	// expansion, so a slash command titles as the command rather than as the
	// first line of the rendered template. The field is only set on the session; the
	// saveOrLog that stores this same message persists it.
	if isFirstMessage && sess != nil {
		a.setSessionTitle(ctx, sess, userText)
	}

	// `/<name> <args>` matching a TEMPLATE-<name>.md (user copy in .codehalter,
	// else the embedded default) expands into a full prompt and runs as a normal
	// turn. A macro that requires an arg ({{}}) but got none stops here with a
	// user-facing note — no model call.
	macroCwd := ""
	if sess != nil {
		macroCwd = sess.Cwd
	}
	// /spec runs a loop of whole turns rather than one: it owns the rest of this
	// Prompt (spec_loop.go).
	if name, args := splitMacro(userText); name == "spec" && sess != nil {
		return a.runSpec(ctx, req.SessionId, sess, args, pendingFixes)
	}
	if rendered, stopMsg, handled := a.expandMacro(ctx, req.SessionId, macroCwd, userText); handled {
		if stopMsg != "" {
			a.say(ctx, req.SessionId, stopMsg+"\n")
			return PromptResponse{StopReason: "end_turn"}, nil
		}
		userText = rendered
	}

	// The empty-project hint stays folded into the first user message because
	// it's a one-shot nudge — once the project has files, we want the
	// summariser to drop it naturally rather than re-injecting it forever.
	stored := userText
	if isFirstMessage && a.projectIsEmpty() {
		stored = emptyProjectHint + "\n---\n" + userText
	}
	if sess != nil {
		if len(images) > 0 {
			sess.AddUserWithImages(stored, images)
		} else {
			sess.AddUser(stored)
		}
		sess.saveOrLog()
	}

	// Reject prompts whose text alone would breach the context-size guard.
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

	if err := a.runTurn(ctx, req.SessionId); err != nil {
		if isCancelled(err) {
			// Never silent: an aborted turn says why, whether the user stopped it
			// or the editor did (a client-side request timeout while the LLM was
			// busy). Background ctx: the request's own is already cancelled.
			reason := cancelReason(err)
			slog.Warn("Prompt: turn cancelled", "sid", req.SessionId, "reason", reason, "err", err)
			msg := "⏹ Turn cancelled — " + reason + ".\n"
			if errors.Is(err, errUserCancelled) {
				msg = "⏹ Stopped.\n"
			}
			a.say(context.Background(), req.SessionId, msg)
			return PromptResponse{StopReason: "cancelled"}, nil
		}
		return a.failPrompt(req.SessionId, err, nil)
	}

	// Anything typed in the last seconds of the turn, after its final round had
	// already asked the model, is nobody's yet: run it as its own turn.
	if sess := a.getSession(req.SessionId); sess != nil {
		a.drainSteer(ctx, sess)
	}

	// Offer any fix cards the pre-turn checks detected, now that the user's
	// actual request has run. The freshness checks themselves moved pre-turn
	// (prepareChecks above); only the user-facing "fix it for me?" cards run
	// here so an accepted one can't dispatch its orchestrate cycle ahead of the
	// request it interrupted.
	slog.Debug("Prompt: draining pre-turn fix cards (post-turn)", "sid", req.SessionId, "fixes", len(pendingFixes))
	a.drainFixes(ctx, req.SessionId, pendingFixes)

	// A turn whose ctx was cancelled along the way still returns normally here:
	// the stop reason is how the client tells a clean turn from an abort.
	if ctx.Err() != nil {
		return PromptResponse{StopReason: "cancelled"}, nil
	}
	return PromptResponse{StopReason: "end_turn"}, nil
}

// runTurn drives one full turn through the single shared path: reset the
// per-turn stats window, run the orchestrator, and on a clean result fire the
// epilogue — the "✅ Done" stats line and history compaction. Both entry
// points use it — a typed user Prompt and an
// accepted proposeFix "install fix?" card — so a fix-dispatched turn gets the
// same stats line and compaction as a typed one. The caller owns error
// presentation (Prompt surfaces it over ACP, proposeFix logs it).
func (a *agent) runTurn(ctx context.Context, sid string) error {
	// MCP config this turn writes is applied at the BOUNDARY, never inside the
	// turn: registering tools rewrites the `tools` array, which the chat template
	// renders ahead of the whole conversation, so an in-turn reconcile would move
	// the prompt under a loop already running. wait() holds a turn that starts
	// while the previous flush is still bringing a child up; the deferred
	// schedule() applies whatever changed once this turn is fully done, coalesced
	// so a run of turns can never stack reconciles.
	a.mcp.wait()
	defer a.mcp.schedule(func() { a.flushMCP(sid) })

	sess := a.getSession(sid)
	if sess != nil {
		// Each turn starts with fresh scratch state: read/search dedup and paging
		// cursors are per turn, so a re-read after a user reply or an outside edit
		// still goes through.
		sess.startTurn(time.Now())
		// Anchor the in-flight turn at the prompt just appended by the caller
		// (Prompt / proposeFix), so the 400-recovery's first fold (foldHistory at
		// turnStartIndex) keeps this whole turn — human prompt plus the synthetic
		// subtask/doc prompts orchestrate adds — verbatim while folding only the
		// completed turns before it.
		sess.markTurnStart()
	}
	result, err := a.orchestrate(ctx, sid)

	// Summarise the just-finished turn on EVERY exit path — success, empty output
	// (orchestrate produced no answer/plan), or a non-recoverable error that still
	// added messages — so its messages always carry a note before a later
	// compaction can rotate them out. fold-all has no way to summarise an un-noted
	// turn after the fact, so a skipped note here is silent loss. This only
	// ENQUEUES; whichever compaction runs next folds it via waitSummarise. The
	// summariser's own LLM call uses context.Background(), so a cancelled turn
	// still gets a note.
	if sess != nil {
		a.backgroundSummarise(sess)
	}

	if err != nil {
		return err
	}
	if sess == nil || result.Text == "" {
		return nil
	}
	// Epilogue — the turn boundary, where control returns to the user. The note
	// was enqueued above; here we report stats and commit the turn's edits. There
	// is no proactive compaction: it is reactive now, driven by a context-overflow
	// 400 in runToolLoopSeeded (see foldHistory). Stats first.
	if r := sess.turnStats(); r.activeMs > 0 {
		// Context-window ring in the client (ACP usage_update, an unstable
		// feature): used = the last call's prompt_tokens (current context size),
		// size = the per-slot n_ctx. Skipped when either is unknown (a backend
		// that omits usage leaves lastPrompt 0, or n_ctx isn't probed yet), so no
		// data means no update rather than a misleading empty ring.
		if size := a.getMainSlotTokens(); size > 0 && r.lastPrompt > 0 {
			a.sendUpdate(ctx, sid, usageUpdate{Kind: "usage_update", Used: r.lastPrompt, Size: size})
		}
		// \n\n keeps the stats on their own markdown line. With the server cache
		// split, headline the work done (evaluated + gen) plus sent/cached%;
		// without it, show the final context size — don't guess.
		var line string
		if r.haveServerCache {
			// Headline the real prompt work: tokens sent that were NOT served from
			// cache (Σ evaluated). We don't show the gross prompt total, which
			// re-counts the cached prefix every call and balloons with tool-step
			// count (a multi-tool turn looked bigger than a later, larger-context one).
			line = fmt.Sprintf("\n\n✅ Done in %s · %s uncached + %s gen",
				humanDuration(r.activeMs), humanCount(r.evaluatedPrompt), humanCount(r.completion))
			if r.promptMs > 0 && r.evaluatedPrompt > 0 {
				line += " · " + humanRate(r.evaluatedPrompt, r.promptMs) + " pp/s"
			}
		} else {
			line = fmt.Sprintf("\n\n✅ Done in %s · %s ctx + %s gen",
				humanDuration(r.activeMs), humanCount(r.lastPrompt), humanCount(r.completion))
		}
		if r.genMs > 0 && r.completion > 0 {
			line += " · " + humanRate(r.completion, r.genMs) + " tg/s"
		}
		// Decode that was generated and then thrown away is already inside the
		// gen figure above, where it reads as productive output. Split it out: on
		// this hardware it is the most expensive thing that can go wrong in a
		// turn. One measured <think> stall burned the full 8192-token cap, 3m36s
		// at that server's 37.7 tok/s, against 57s for the prefix loss the same
		// event caused. Priced in the turn's own decode rate rather than a
		// constant, so it stays honest across backends.
		if r.wastedCompletion >= wastedCompletionFloor {
			line += fmt.Sprintf(" · %s discarded", humanCount(r.wastedCompletion))
			if r.genMs > 0 && r.completion > 0 {
				line += " (" + humanDuration(r.genMs*int64(r.wastedCompletion)/int64(r.completion)) + ")"
			}
		}
		// Prefix cache didn't hold across the turn's calls (noteCacheLineage).
		// Worth a mark: it is invisible otherwise (the turn still succeeds, just
		// several times slower), and it is nearly always one line of settings.
		if r.cacheRewinds > 0 {
			line += fmt.Sprintf("\n\n⚠ Prefix cache rewound %d× (%s re-read). ",
				r.cacheRewinds, humanCount(r.cacheRewound))
			// Two different faults, two different fixes. Say which one this was
			// instead of listing both: the render-change case is one line of
			// settings.toml, the other is not in settings.toml at all.
			if r.cacheRewindsRender > 0 {
				line += fmt.Sprintf("%d of those switched rendering mid-turn: `params_thinking` and `params_execute` must agree on "+
					"everything that is not a sampler (`chat_template_kwargs` above all), or this server needs `parallel = 2` "+
					"so each rendering keeps its own KV slot. ", r.cacheRewindsRender)
			} else {
				// Deliberately not naming a magic setting here. On the one 11.6h
				// session measured this way every stable-rendering rewind was an
				// idle eviction (gaps of 2h and 14min), and probing the server
				// afterwards showed preserve_thinking, the flag this line used to
				// recommend, changes nothing: same prompt tokens, cached=11507 of
				// 11511 with the flag removed, even with <think> blocks left inline
				// in the history. It is a no-op wherever the template ignores it.
				line += "The rendering never changed, so it is not the role split, and the CACHE lines say which of the other " +
					"two it was: a gap of minutes means the server dropped an idle slot and no setting will fix it, seconds " +
					"apart means something rewrote the middle of the prompt (a tool result that replayed differently, or a " +
					"chat template that repositions content as the conversation grows). "
			}
			line += "See the session log (CACHE lines) for the calls."
		}
		a.say(ctx, sid, line+"\n")
	}
	return nil
}

// orchestrate drives the plan → subtasks → replan → document pipeline for
// one user turn. Returns the final subtask's result (used for the
// background epilogue) and the first error that isn't recoverable via
// replan. User cancellation surfaces as errUserCancelled.
func (a *agent) orchestrate(ctx context.Context, sid string) (toolLoopResult, error) {
	sess := a.getSession(sid)

	a.sendPhase(ctx, sid, 0, false)
	p, firstToolUses, err := a.runPlanPhase(ctx, sid, "")
	if err != nil {
		if isCancelled(err) {
			return toolLoopResult{}, err
		}
		if sess != nil && len(firstToolUses) > 0 {
			sess.AddAssistantWithTools("❌ "+err.Error(), firstToolUses)
			sess.saveOrLog()
		}
		return toolLoopResult{}, err
	}
	if p == nil {
		// No PLAN.md or unparseable response — pipeline cannot proceed.
		return toolLoopResult{}, fmt.Errorf("planner returned no usable plan")
	}
	if len(p.Subtasks) == 0 {
		// No subtasks: a report_only direct answer — surface it (returning it as
		// result.Text lets Prompt's epilogue run). If the planner left it empty
		// even after the plan-phase nudge, warn rather than ending silently —
		// never leave the user with nothing after a turn that ran.
		switch {
		case p.answer != "":
			// Surface the answer, then say WHY the turn ends here: a report_only
			// plan means the planner judged this a question/diagnosis, not a code
			// change, so no execute phase runs. Without this note a "Completed
			// Plan — Planning" card reads as "stopped early", not "answered".
			a.say(ctx, sid, p.answer+"\n\nℹ Answered directly — no code change to execute.\n")
			return toolLoopResult{Text: p.answer}, nil
		default:
			a.say(ctx, sid, "⚠ I couldn't produce a clear answer or a plan for that — try rephrasing, or ask for a specific change.\n")
			return toolLoopResult{}, nil
		}
	}
	header := "Plan:"
	if p.ReportOnly {
		header = "Findings:"
	}
	a.renderPlan(ctx, sid, header, p.Subtasks)
	plan := p

	var lastResult toolLoopResult
	var failureBags []map[string]bool
	replans := 0
	upserts := 0

	for {
		// Bail the instant the turn is cancelled, instead of starting another
		// plan/execute phase on a turn that has been told to stop.
		if err := ctx.Err(); err != nil {
			return lastResult, err
		}
		allOk := true
		upserted := false
		var failedAt int
		var failedReason string

		for i, st := range plan.Subtasks {
			if err := ctx.Err(); err != nil {
				return lastResult, err
			}
			a.sendPhase(ctx, sid, 1, false)
			a.say(ctx, sid, fmt.Sprintf("\n=== Task %d/%d: %s ===\n\n", i+1, len(plan.Subtasks), st.Description))

			outcome := a.runExecutePhase(ctx, sid, st, i, len(plan.Subtasks))
			lastResult = outcome.Result

			if outcome.Upsert != nil {
				// Plan-upsert: adopt the revised plan and restart the subtask loop —
				// completed work stays, nothing cancelled, no replan-budget cost.
				plan = outcome.Upsert
				upserted = true
				break
			}
			if outcome.Success {
				continue
			}
			allOk = false
			failedAt = i
			failedReason = outcome.Reason
			break
		}

		if upserted {
			upserts++
			if upserts > maxUpserts {
				a.say(ctx, sid, fmt.Sprintf("⚠ Plan revised %d times — stopping to avoid a re-plan loop.\n", upserts))
				return lastResult, nil
			}
			a.renderPlan(ctx, sid, "\n📝 Plan updated — remaining:", plan.Subtasks)
			continue
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

		a.say(ctx, sid, fmt.Sprintf("⚠ Task %d/%d failed: %s\n", failedAt+1, len(plan.Subtasks), failedReason))

		replans++
		if replans >= maxReplans {
			a.say(ctx, sid, fmt.Sprintf("⚠ Replan budget (%d) exhausted — giving up.\n", maxReplans))
			return lastResult, nil
		}

		var replanCtx string
		if dupCount >= 2 {
			replanCtx = fmt.Sprintf("REPLAN: prior subtask failed: %s. Same failure has surfaced %d times — the prior fix didn't work; propose a structurally different approach. See history for executor attempts. Follow the 'Replanning' section in PLAN.md.", failedReason, dupCount)
		} else {
			replanCtx = fmt.Sprintf("REPLAN: prior subtask failed: %s. See history for executor attempts. Follow the 'Replanning' section in PLAN.md.", failedReason)
		}

		a.sendPhase(ctx, sid, 0, false)
		newPlan, _, err := a.runPlanPhase(ctx, sid, replanCtx)
		if err != nil {
			return lastResult, err
		}
		if newPlan == nil || len(newPlan.Subtasks) == 0 {
			a.say(ctx, sid, "Replan produced no further subtasks — stopping.\n")
			return lastResult, nil
		}

		a.renderPlan(ctx, sid, "Replan:", newPlan.Subtasks)
		plan = newPlan
	}

	// Document phase: fire once at the end of a successful prompt. Routes
	// to a non-foreground LLM entry internally so llm[0]'s prefix cache
	// isn't evicted by a one-shot README update.
	a.sendPhase(ctx, sid, 2, false)
	// runDocumentPhase logs its own failures at Warn and always returns (exec, nil) —
	// a failed README update must not fail the whole turn — so the discarded
	// error is intentional and never non-nil, not a silent swallow.
	lastResult, _ = a.runDocumentPhase(ctx, sid, lastResult)
	a.sendPhase(ctx, sid, 2, true)

	return lastResult, nil
}

// renderPlan shows the planned subtasks inline. There is no "Execute?" gate:
// codehalter runs inside a devcontainer, so the container is the approval and
// every subtask is confined to it. What the user gets here is visibility, not a
// question. Building that container is still gated (see bootstrap.go), and so
// is anything reaching outside it. `header` names the occasion ("Plan:",
// "Replan:", the mid-run revision notice).
func (a *agent) renderPlan(ctx context.Context, sid, header string, subtasks []subtask) {
	if len(subtasks) == 0 {
		return
	}
	// Already on screen: submit_plan's arguments streamed in as a live table
	// (planTableSink) carrying the full text, so repeating the list here would
	// show every subtask twice. The heading still prints, because the table has
	// none and cannot have one (see planTable) and "Plan:" versus "Findings:" is
	// the difference between work that will run and work that won't. The flag is
	// consumed, not just read: a later mid-run revision never streamed and must
	// still render in full.
	if sess := a.getSession(sid); sess != nil {
		sess.phaseMu.Lock()
		shown := sess.planTableShown
		sess.planTableShown = false
		sess.phaseMu.Unlock()
		if shown {
			a.say(ctx, sid, header+"\n")
			return
		}
	}
	var b strings.Builder
	b.WriteString(header)
	b.WriteString(planTableHead)
	for _, st := range subtasks {
		b.WriteString(planRow(st))
	}
	a.say(ctx, sid, b.String())
}

// ---------------------------------------------------------------------------
// System prompt + prompt-file loading
// ---------------------------------------------------------------------------

func (a *agent) systemPrompt(sid string) (string, error) {
	sess := a.getSession(sid)
	if sess == nil {
		return "", fmt.Errorf("no session found")
	}

	var b strings.Builder
	if skills := loadSkills(sess.Cwd, skillSet(sess.Cwd, sess.knownStacks)); skills != "" {
		b.WriteString(skills)
	}
	fmt.Fprintf(&b, "Project directory: %s\n", sess.Cwd)
	// Project-shipped agent instructions (AGENTS.md and common casings, project
	// root only): fold them into the cached prefix at session start so they ride
	// every turn alongside the SKILL files. Stable across the session, so no
	// mid-session cache churn.
	if name, content := loadAgentsFile(sess.Cwd); content != "" {
		fmt.Fprintf(&b, "\n\n## Project instructions (%s)\n\nThis project ships the following instructions for agents working in it. Follow them as authoritative project conventions, unless they conflict with a direct request from the user in this conversation. They are meant to hold across sessions, so when your work makes one of them wrong (the stack, the layout, how the project is built, run or tested), update %s in the same task: a line left stale here is believed by every session after this one.\n\n%s\n", name, name, content)
	}

	// Phase guidance lives in the system prompt (the stable, cached prefix) rather
	// than being re-injected as a multi-KB user message on every plan/execute
	// entry — that stacked a fresh 7-8 KB copy in the history each (re)plan and
	// forced repeated compactions. The per-phase user message now carries only the
	// trigger + the specific subtask. Both phases see both blocks; the live
	// instruction + the dispatch gate decide which one is in force.
	if plan := a.loadPromptFile(sid, "PLAN.md"); plan != "" {
		b.WriteString("\n\n")
		b.WriteString(plan)
	}
	if exec := a.loadPromptFile(sid, "EXECUTE.md"); exec != "" {
		b.WriteString("\n\n")
		b.WriteString(exec)
	}

	return b.String(), nil
}

// shippedPrompts is every phase prompt codehalter carries, by filename.
var shippedPrompts = map[string]string{
	"PLAN.md":        defaultPlanMD,
	"EXECUTE.md":     defaultExecuteMD,
	"DOCUMENT.md":    defaultDocumentMD,
	"SUMMARISE.md":   defaultSummariseMD,
	"RESUMMARISE.md": defaultResummariseMD,
	"SPEC.md":        defaultSpecMD,
	"SPEC-SETUP.md":  defaultSpecSetupMD,
	"SPEC-REMOVE.md": defaultSpecRemoveMD,
	"SPEC-FINAL.md":  defaultSpecFinalMD,
}

// loadPromptFile returns a phase prompt: the project's own copy in .codehalter
// when it has one, else the copy in the binary. The file is read every time, so
// an edit lands on the next turn.
//
// A file that exists wins even when EMPTY, which is the documented way to turn
// a phase off: emptying PLAN.md disables planning (see runPlanPhase).
func (a *agent) loadPromptFile(sid string, filename string) string {
	if sess := a.getSession(sid); sess != nil {
		if data, err := os.ReadFile(filepath.Join(sess.Cwd, ".codehalter", filename)); err == nil {
			return string(data)
		}
	}
	return shippedPrompts[filename]
}

// agentsFileNames are the project-root agent-instruction files codehalter folds
// into the system prompt at session start, in priority order: the AGENTS.md
// convention plus the common casings. The first that exists and is non-empty
// wins; subdirectory/nested files are NOT scanned — project root only.
var agentsFileNames = []string{"AGENTS.md", "AGENT.md", "agents.md", "agent.md"}

// loadAgentsFile returns the filename and trimmed contents of the first
// project-root agent-instruction file present (see agentsFileNames), or "", ""
// when none exists. Read fresh on each systemPrompt build so an edit lands on
// the next (re)build.
func loadAgentsFile(cwd string) (string, string) {
	for _, name := range agentsFileNames {
		data, err := os.ReadFile(filepath.Join(cwd, name))
		if err != nil {
			continue
		}
		if s := strings.TrimSpace(string(data)); s != "" {
			return name, s
		}
	}
	return "", ""
}

// humanCount renders a token count compactly: 543490 → "543k", 1_500_000 → "1.5m".
func humanCount(n int) string {
	switch {
	case n < 1000:
		return strconv.Itoa(n)
	case n < 1_000_000:
		return trimUnit(float64(n)/1e3, "k")
	case n < 1_000_000_000:
		return trimUnit(float64(n)/1e6, "m")
	default:
		return trimUnit(float64(n)/1e9, "g")
	}
}

// humanBytes renders a wire-payload size compactly in binary units: 900 →
// "900b", 12288 → "12kb", 2508268 → "2.4mb". Same trimUnit idiom as humanCount
// so the two halves of the LLM meter read alike, and it steps up to mb — the
// old fixed "%.2fkb" printed a 2 MB request body as "2451.39kb", which is both
// unreadable and easy to mistake for the ↓ side's token count.
func humanBytes(n int) string {
	switch {
	case n < 1024:
		return strconv.Itoa(n) + "b"
	case n < 1024*1024:
		return trimUnit(float64(n)/1024, "kb")
	default:
		return trimUnit(float64(n)/(1024*1024), "mb")
	}
}

func trimUnit(v float64, u string) string {
	if v >= 10 || v == float64(int64(v)) { // no decimal when big or whole (2g, not 2.0g)
		return fmt.Sprintf("%.0f%s", v, u)
	}
	return fmt.Sprintf("%.1f%s", v, u)
}

// humanDuration renders elapsed ms compactly: 5500 → "5.5s", 61000 → "1m1s",
// 3661000 → "1h1m1s".
func humanDuration(ms int64) string {
	d := time.Duration(ms) * time.Millisecond
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	h := int(d / time.Hour)
	m := int(d % time.Hour / time.Minute)
	s := int(d % time.Minute / time.Second)
	if h > 0 {
		return fmt.Sprintf("%dh%dm%ds", h, m, s)
	}
	return fmt.Sprintf("%dm%ds", m, s)
}

// humanRate renders tokens-per-second over a span of ms (k-suffixed when large).
func humanRate(tokens int, ms int64) string {
	if ms <= 0 {
		return "0"
	}
	r := float64(tokens) * 1000 / float64(ms)
	switch {
	case r >= 1000:
		return humanCount(int(r))
	case r >= 10:
		return fmt.Sprintf("%.0f", r)
	default:
		return fmt.Sprintf("%.1f", r)
	}
}
