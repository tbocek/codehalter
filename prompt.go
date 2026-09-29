package main

import (
	"cmp"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

// Generous on purpose: executeFailCap bounces stuck subtasks here, and replans
// are the only place web_search/web_read run.
const maxReplans = 20

// maxUpserts caps mid-run submit_plan revisions, separately from maxReplans.
const maxUpserts = 20

// errUserCancelled ends the turn with stopReason "cancelled", not an error.
var errUserCancelled = errors.New("user cancelled")

// readLinkedResource honours a #L<start>-<end> fragment; ok is false for a file
// outside cwd, missing, or not local.
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
	// realInside, not a prefix test: a symlink in the project pointing out of it
	// passes a prefix test.
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

// A non-file URI comes back as its own path, so the caller can still name it.
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

// parseLineRange accepts "L810-845", "810:845", "L810"; 0,0 means the whole file.
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

// Images are stored by content id, so the wire's base64 never lands in session.toml.
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
			id, err := storeImage(cwd, block.MimeType, bytes)
			if err != nil {
				slog.Warn("prompt: writing image file failed", "id", id, "err", err)
				continue
			}
			images = append(images, ImageData{ID: id, MimeType: block.MimeType})
		case "resource":
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
				text += fmt.Sprintf("\n\n[Attached binary resource %s (%s) — not inlined]\n", label, block.Resource.MimeType)
			default:
				if snippet, l, ok := readLinkedResource(cwd, block.Resource.URI); ok {
					text += fmt.Sprintf("\n\n[Attached context from %s]\n```\n%s\n```\n", l, snippet)
				} else {
					slog.Debug("prompt: empty embedded resource", "uri", block.Resource.URI)
				}
			}
		case "resource_link":
			// Inlined so the model doesn't read-loop hunting for the snippet.
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

// All three must end as stopReason "cancelled", never as a JSON-RPC error.
func isCancelled(err error) bool {
	return errors.Is(err, errUserCancelled) ||
		errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded)
}

// codehalter sets no foreground deadline, so DeadlineExceeded is the client's.
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

var phaseNames = []string{"Planning", "Working", "Documenting"}

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

// The tracked phase lets holdTurn's release complete it if the turn exits early.
func (a *agent) sendPhase(ctx context.Context, sid string, phase int, done bool) {
	if sess := a.getSession(sid); sess != nil { // closed mid-turn
		sess.phaseMu.Lock()
		sess.phaseCurrent = phase
		sess.phaseActive = !done
		sess.phaseMu.Unlock()
	}
	a.sendUpdate(ctx, sid, planUpdate{Kind: "plan", Entries: phaseEntries(phase, done, "")})
}

// setStatus is a no-op when no phase is active, so background calls don't clobber the UI.
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

// stop() joins the goroutine so a late tick can't re-set a cleared row. A caller
// that also defers setStatus(ctx, sid, "") must register that defer first.
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

const sessionTitleMax = 60

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
	// Only a nearby word boundary: one long token would otherwise collapse to nothing.
	if i := strings.LastIndex(cut, " "); i > sessionTitleMax/2 {
		cut = cut[:i]
	}
	return strings.TrimRight(cut, " ,.;:-") + "…"
}

// No capability gate: a client without session_info_update ignores the unknown kind.
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
	sess := a.getSession(req.SessionId)
	if sess == nil {
		return PromptResponse{}, fmt.Errorf("no session found")
	}

	// Typing while a turn runs steers it: the text is queued for its next tool
	// round. Images can't be queued, and the reply says so.
	if sess.turnRunning() {
		text, images := promptContent(sess.Cwd, req.Content)
		if strings.TrimSpace(text) == "/spec stop" {
			if sess.specFence() == "" {
				a.say(ctx, req.SessionId, "No /spec loop is running in this session.\n")
			} else {
				sess.requestSpecStop()
				a.say(ctx, req.SessionId, "⏹ /spec stops after the round in flight; its commit lands first. `/spec` later resumes where the ledger says.\n")
			}
			return PromptResponse{StopReason: "end_turn"}, nil
		}
		if t := strings.TrimSpace(text); strings.HasPrefix(t, "/spec") && sess.specFence() != "" {
			a.say(ctx, req.SessionId, "A /spec loop is running. `/spec stop` ends it after the round in flight; then `"+t+"`.\n")
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

	var release func()
	ctx, release, _ = a.holdTurn(ctx, sess, true)
	defer release()

	// These gates refuse before the message is stored, so history gets no reply.
	// The abort is also said in chat: Zed keeps the first red box open, so a later
	// error alone is invisible.
	a.mu.Lock()
	abort := a.abortReason
	a.mu.Unlock()
	slog.Debug("Prompt: abort gate", "sid", req.SessionId, "abortReason", abort)
	if abort != "" {
		a.say(ctx, req.SessionId, abort+"\n")
		return PromptResponse{}, errors.New(abort)
	}

	// Startup still running: a prompt waits for it, unless startup is asking the
	// user something, which the prompt would answer past. Refusing outright made
	// `--cli -p`, which prompts at once, fail every time with a question nobody asked.
	if a.indexDone != nil {
		select {
		case <-a.indexDone:
			slog.Debug("Prompt: indexDone gate passed", "sid", req.SessionId)
		default:
			if a.asking.Load() > 0 {
				slog.Debug("Prompt: indexDone gate refused (startup is asking)", "sid", req.SessionId)
				return PromptResponse{}, errors.New("Please answer the pending question above first.")
			}
			slog.Debug("Prompt: waiting for startup", "sid", req.SessionId)
			a.say(ctx, req.SessionId, "⏳ Still setting up; your message runs as soon as that is done.\n")
			select {
			case <-a.indexDone:
			case <-ctx.Done():
				return PromptResponse{}, ctx.Err()
			}
			// Startup may have ended on a problem the abort gate above did not see yet.
			a.mu.Lock()
			abort := a.abortReason
			a.mu.Unlock()
			if abort != "" {
				a.say(ctx, req.SessionId, abort+"\n")
				return PromptResponse{}, errors.New(abort)
			}
		}
	} else {
		slog.Debug("Prompt: indexDone nil, no gate", "sid", req.SessionId)
	}

	// Seeded before prepareChecks, whose LLM calls need a system prompt.
	isFirstMessage := len(sess.Messages) == 0 && sess.Summary == ""
	if sess.SystemPrompt == "" {
		sysPrompt, err := a.systemPrompt(req.SessionId)
		if err != nil {
			return PromptResponse{}, err
		}
		sess.SystemPrompt = sysPrompt
	}

	// Fix cards are held until after the turn, so an accepted one can't jump
	// ahead of the user's request.
	pendingFixes := a.prepareChecks(ctx, sess, req.SessionId)

	userText, images := promptContent(sess.Cwd, req.Content)

	// Before macro expansion, so a slash command titles as the command.
	if isFirstMessage {
		a.setSessionTitle(ctx, sess, userText)
	}

	if name, args := splitMacro(userText); name == "spec" {
		return a.runSpec(ctx, req.SessionId, sess, args, pendingFixes)
	}
	if rendered, stopMsg, handled := a.expandMacro(ctx, req.SessionId, sess.Cwd, userText); handled {
		if stopMsg != "" {
			a.say(ctx, req.SessionId, stopMsg+"\n")
			return PromptResponse{StopReason: "end_turn"}, nil
		}
		userText = rendered
	}

	// Before the message is stored: a refused message must not stay in history.
	if mst := int(a.mainSlotTokens.Load()); mst > 0 {
		estTokens := len(userText) / 4
		triggerTokens := mst * compactTriggerPct / 100
		if estTokens > triggerTokens {
			return PromptResponse{}, fmt.Errorf("message too large: ~%d tokens estimated, exceeds the %d-token compaction trigger (per-slot n_ctx %d × %d%%). Trim the prompt or restart your server with a larger -c N / --max-model-len N", estTokens, triggerTokens, mst, compactTriggerPct)
		}
	}

	// In the first message, not the system prompt, so the summariser drops it later.
	stored := userText
	if isFirstMessage && a.projectIsEmpty() {
		stored = emptyProjectHint + "\n---\n" + userText
	}
	sess.AddUser(stored, images...)
	sess.saveOrLog()

	slog.Info("Prompt", "sid", req.SessionId, "sessions", len(a.sessions))

	if err := a.runTurn(ctx, req.SessionId); err != nil {
		if isCancelled(err) {
			// Background ctx: the request's own is already cancelled.
			reason := cancelReason(err)
			slog.Warn("Prompt: turn cancelled", "sid", req.SessionId, "reason", reason, "err", err)
			msg := "⏹ Turn cancelled — " + reason + ".\n"
			if errors.Is(err, errUserCancelled) {
				msg = "⏹ Stopped.\n"
			}
			a.say(context.Background(), req.SessionId, msg)
			return PromptResponse{StopReason: "cancelled"}, nil
		}
		sess.AddAssistant("❌ " + err.Error())
		sess.saveOrLog()
		return PromptResponse{}, err
	}

	// Steering or a job note that came after the final round asked the model is nobody's yet.
	a.drainSteer(ctx, sess)

	slog.Debug("Prompt: draining pre-turn fix cards (post-turn)", "sid", req.SessionId, "fixes", len(pendingFixes))
	a.drainFixes(ctx, req.SessionId, pendingFixes)

	if sess.specHandoffPending() {
		return a.runSpec(ctx, req.SessionId, sess, "", nil)
	}

	// The stop reason is how the client tells a clean turn from an abort.
	if ctx.Err() != nil {
		return PromptResponse{StopReason: "cancelled"}, nil
	}
	return PromptResponse{StopReason: "end_turn"}, nil
}

// runTurn is shared by typed prompts and accepted fix cards; the caller presents errors.
func (a *agent) runTurn(ctx context.Context, sid string) error {
	sess := a.getSession(sid)
	if sess != nil {
		sess.startTurn(time.Now())
		// The 400-recovery's first fold keeps this turn verbatim and folds only
		// the completed turns before it.
		sess.markTurnStart()
	}
	result, err := a.orchestrate(ctx, sid)

	// On every exit path: fold-all can't note a turn after the fact, so a skipped
	// note is silent loss. Only enqueues.
	if sess != nil {
		a.backgroundSummarise(sess)
	}

	if err != nil {
		return err
	}
	if sess == nil || result.Text == "" {
		return nil
	}
	if r := sess.turnStats(); r.activeMs > 0 {
		if size := int(a.mainSlotTokens.Load()); size > 0 && r.lastPrompt > 0 {
			a.sendUpdate(ctx, sid, usageUpdate{Kind: "usage_update", Used: r.lastPrompt, Size: size})
		}
		var line string
		if r.haveServerCache {
			// Not the gross prompt total: it re-counts the cached prefix every call.
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
		// Discarded decode is already inside gen, where it reads as productive.
		if r.wastedCompletion >= wastedCompletionFloor {
			line += fmt.Sprintf(" · %s discarded", humanCount(r.wastedCompletion))
			if r.genMs > 0 && r.completion > 0 {
				line += " (" + humanDuration(r.genMs*int64(r.wastedCompletion)/int64(r.completion)) + ")"
			}
		}
		// Invisible otherwise: the turn still succeeds, just several times slower.
		if r.cacheRewinds > 0 {
			line += fmt.Sprintf("\n\n⚠ Prefix cache rewound %d× (%s re-read). ",
				r.cacheRewinds, humanCount(r.cacheRewound))
			if r.cacheRewindsRender > 0 {
				line += fmt.Sprintf("%d of those switched rendering mid-turn: `params_thinking` and `params_execute` must agree on "+
					"everything that is not a sampler (`chat_template_kwargs` above all), or this server needs `parallel = 2` "+
					"so each rendering keeps its own KV slot. ", r.cacheRewindsRender)
			} else {
				// Deliberately no setting named: preserve_thinking does not help here.
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

func (a *agent) orchestrate(ctx context.Context, sid string) (toolLoopResult, error) {
	sess := a.getSession(sid)
	sess.rt.mu.Lock()
	sess.rt.stuckCalls, sess.rt.stuckOutputs = nil, nil // a new request may need any call again
	sess.rt.mu.Unlock()

	a.sendPhase(ctx, sid, 0, false)
	p, err := a.runPlanPhase(ctx, sid, "")
	if err != nil {
		return toolLoopResult{}, err
	}
	if len(p.Redo) > 0 || len(p.Spec) > 0 {
		return a.specFromPlan(ctx, sid, sess, p)
	}
	if len(p.Subtasks) == 0 {
		// Returning the answer as Text lets runTurn's epilogue run.
		switch {
		case p.answer != "":
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

		// What the failed subtask spent its calls on: "see history" alone let replans
		// repeat the same hunt.
		var reads, edits, runs int
		var edited []string
		// Grouped by what came back, as the repetition tracker does: the same answer
		// under a new echo label or log name is the same call, shown as its first form.
		count := map[string]int{}
		first := map[string]string{}
		for _, u := range lastResult.ToolUses {
			args := parseArgs(u.Input)
			cmd := args.str("command")
			switch {
			case u.Name == "read_file", u.Name == "web_search", u.Name == "web_read", u.Name == "run_command" && onlyReads(cmd):
				reads++
			case u.Name == "edit_file" || u.Name == "write_file":
				if !strings.HasPrefix(u.Output, "file written") { // refused or unmatched
					break
				}
				edits++
				if p := args.str("path"); p != "" && !slices.Contains(edited, p) {
					edited = append(edited, p)
				}
			case u.Name == "run_command" && u.Changed:
				edits++
				runs++
			case u.Name == "run_command", u.Name == "run_background":
				runs++
			}
			// A short answer ("exit 0") says which call only by the call itself.
			h := "call " + u.Name + " " + u.Input
			if out := repeatText(u.Input, u.Output); len(out) >= repeatMinOutput {
				h = "answer " + out
			}
			if count[h]++; first[h] == "" {
				first[h] = u.Name + " " + u.Input
				if cmd != "" {
					first[h] = cmd
				}
			}
		}
		var digest strings.Builder
		fmt.Fprintf(&digest, "What the failed subtask did, counted by codehalter: %d tool calls, %d of them reads or searches, %d edits, %d commands run.", len(lastResult.ToolUses), reads, edits, runs)
		switch {
		case edits == 0:
			digest.WriteString(" It changed no file.")
		case len(edited) > 0:
			fmt.Fprintf(&digest, " Files changed through edits: %s.", strings.Join(edited, ", "))
		}
		keys := slices.Collect(maps.Keys(count))
		slices.SortFunc(keys, func(x, y string) int { return cmp.Or(count[y]-count[x], strings.Compare(first[x], first[y])) })
		var repeated []string
		for _, k := range keys {
			if count[k] < 3 || len(repeated) == 3 {
				break
			}
			repeated = append(repeated, fmt.Sprintf("`%s` %d times", truncate(first[k], 100), count[k]))
		}
		if len(repeated) > 0 {
			digest.WriteString(" Repeated: " + strings.Join(repeated, "; ") + ".")
		}
		if reads >= 20 && reads > 2*(edits+runs) {
			digest.WriteString(" It kept looking things up: put what it was hunting for (the verified signatures, paths and line ranges) into the new subtasks, so the executor does not have to find it again.")
		}
		var replanCtx string
		if dupCount >= 2 {
			replanCtx = fmt.Sprintf("REPLAN: prior subtask failed: %s. %s Same failure has surfaced %d times — the prior fix didn't work; propose a structurally different approach. See history for executor attempts. Follow the 'Replanning' section in PLAN.md.", failedReason, digest.String(), dupCount)
		} else {
			replanCtx = fmt.Sprintf("REPLAN: prior subtask failed: %s. %s See history for executor attempts. Follow the 'Replanning' section in PLAN.md.", failedReason, digest.String())
		}

		a.sendPhase(ctx, sid, 0, false)
		newPlan, err := a.runPlanPhase(ctx, sid, replanCtx)
		if err != nil {
			return lastResult, err
		}
		if len(newPlan.Subtasks) == 0 {
			a.say(ctx, sid, "Replan produced no further subtasks — stopping.\n")
			return lastResult, nil
		}

		a.renderPlan(ctx, sid, "Replan:", newPlan.Subtasks)
		plan = newPlan
	}

	// A /spec round plans its own docs step; a documenter after it edits past the round's test run.
	if sess == nil || sess.specFence() == "" {
		a.sendPhase(ctx, sid, 2, false)
		lastResult = a.runDocumentPhase(ctx, sid, lastResult)
	}
	a.sendPhase(ctx, sid, 2, true)

	return lastResult, nil
}

// No "Execute?" gate: the devcontainer is the approval.
func (a *agent) renderPlan(ctx context.Context, sid, header string, subtasks []subtask) {
	if len(subtasks) == 0 {
		return
	}
	// Already streamed as a live table, which has no heading. Consumed: a mid-run
	// revision never streamed and must render in full.
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
	if name, content := loadAgentsFile(sess.Cwd); content != "" {
		over := ""
		if len(content) > agentsFileBudget {
			over = fmt.Sprintf(" %s is %d KB now, over its %d KB budget: the next time you edit it, make it SHORTER, delete what the code, the tests and git already say.", name, len(content)/1024, agentsFileBudget/1024)
		}
		fmt.Fprintf(&b, "\n\n## Project instructions (%s)\n\nThis project ships the following instructions for agents working in it. Follow them as authoritative project conventions, unless they conflict with a direct request from the user in this conversation. They are meant to hold across sessions, so when your work makes one of them wrong (the stack, the layout, how the project is built, run or tested), fix that line in %s in the same task: a line left stale here is believed by every session after this one. It is a brief for the next agent, not a log of your work: change the line that became wrong, in as few words as it takes; never record what a task did, which widgets or constants it added, or what you measured (the code, the tests and git hold that). If you add a line, remove or shorten another; keep it under about 150 lines.%s\n\n%s\n", name, name, over, content)
	}

	// Only the directories, never the moving item count: the prefix must stay stable.
	if cfg, err := loadSpecConfig(sess.Cwd); err == nil && cfg != nil {
		fmt.Fprintf(&b, "\n\n## This project is built with /spec\n\nThe specification in `%s/` is implemented into `%s/` item by item, each item done when a test names it and the suite passes (the item ids are in the spec files). A chat request that amounts to many rounds of work, or that says something built does not work, is not a plan here: name the spec items it concerns in submit_plan's `redo` and codehalter rebuilds them one per round. Inside a /spec round (the prompt says which item it is) the item is already chosen: plan and execute that item, and leave AGENT.md alone: it is read-only while /spec runs.\n", cfg.SpecDir, cfg.OutDir)
	}

	// In the cached prefix, not re-sent per phase: each copy would stack in history.
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

// An existing file wins even when empty: an empty DOCUMENT.md turns the document phase off.
func (a *agent) loadPromptFile(sid string, filename string) string {
	cwd := ""
	if sess := a.getSession(sid); sess != nil {
		cwd = sess.Cwd
	}
	body, _ := builtin(cwd, filename)
	return body
}

var agentsFileNames = []string{"AGENTS.md", "AGENT.md", "agents.md", "agent.md"}

const agentsFileBudget = 12 * 1024

// agentsFileRefusal: during /spec rounds the brief is read-only, since every
// round planned a bullet for its item and it went from 73 to 179 lines in a day;
// elsewhere, over budget, it may not grow.
func (a *agent) agentsFileRefusal(sid, path, oldContent, newContent string) string {
	sess := a.getSession(sid)
	if sess == nil || filepath.Dir(path) != filepath.Clean(sess.Cwd) || !slices.Contains(agentsFileNames, filepath.Base(path)) {
		return ""
	}
	if sess.specFence() != "" {
		return fmt.Sprintf("refused: %s is read-only while /spec runs. It is the project brief, not a log: this round's work is recorded in the code, its tests and git. Nothing was written; go on with the task without it.", filepath.Base(path))
	}
	if len(newContent) <= agentsFileBudget || len(newContent) <= len(oldContent) {
		return ""
	}
	return fmt.Sprintf("refused: %s would be %.1f KB, over its %d KB budget, and this change makes it longer. Nothing was written. It is a brief for the next agent, not a log of the work: in the same edit, shorten or delete lines that the code, the tests and git already say, so the file does not grow.",
		filepath.Base(path), float64(len(newContent))/1024, agentsFileBudget/1024)
}

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
	if v >= 10 || v == float64(int64(v)) {
		return fmt.Sprintf("%.0f%s", v, u)
	}
	return fmt.Sprintf("%.1f%s", v, u)
}

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
