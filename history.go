package main

import (
	"context"
	"fmt"
	"strings"
	"time"
)

const (
	// rawBufferTokens is the FALLBACK compaction trigger used only when the
	// startup probe couldn't discover the model's context window. When n_ctx
	// is known, compactTriggerTokens computes the trigger from it instead so
	// the threshold scales to the actual model. Keep this conservative — a
	// fallback firing too late costs more than firing slightly early.
	rawBufferTokens = 60000

	// compactOverheadTokens reserves space for everything that isn't the
	// Messages array: system prompt + tool schemas (~5-6k), pending user
	// turn (1-3k), output budget (4-8k). Subtracted from the model's n_ctx
	// before applying the estimator-bias factor.
	compactOverheadTokens = 15000

	// compactSafetyPct accounts for our chars/4 token estimator under-counting
	// real BPE tokens, especially on tool-heavy JSON content (lots of short
	// tokens like ",", ":", '"'). Apply as a percentage so the trigger fires
	// before the *real* token count overruns the server's window.
	compactSafetyPct = 80

	// minCompactTrigger keeps the trigger sane for pathologically small
	// contexts — below this we'd compact every other turn and the summary
	// budget alone would dominate the post-compaction prefix.
	minCompactTrigger = 8000

	summaryBudget     = 12000
	charsPerToken     = 4
	roleTokenOverhead = 4
	maxTitleLen       = 60
	titleFallbackLen  = 50

	// maxSummaryItemBytes caps any single message's content when it's fed
	// to the summarize LLM. Without this, one giant message (e.g. a tool
	// dump bigger than rawBufferTokens) would blow through the summarize
	// model's own context window.
	maxSummaryItemBytes = 20 * 1024

	summarizePrompt = "You are a context summarization assistant. Read the conversation in <conversation>…</conversation> and output a structured summary following the exact format below.\n\n" +
		"Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.\n\n" +
		"Preserve exact file paths, function names, identifiers, versions, and error messages. Do NOT include source code verbatim — reference the file path and describe what changed. Drop pleasantries and redundant detail.\n\n" +
		"Format (use these EXACT H2 sections; omit a section by leaving its body empty, but keep the heading):\n\n" +
		"## Goal\n<what the user wants overall>\n\n" +
		"## Constraints & Preferences\n<rules / preferences the user has stated>\n\n" +
		"## Progress\n### Done\n- [x] <completed item>\n### In Progress\n- [ ] <item being worked on>\n### Blocked\n- [ ] <item blocked, with reason>\n\n" +
		"## Key Decisions\n<decisions already made — what + why>\n\n" +
		"## Next Steps\n<what's queued or open>\n\n" +
		"## Critical Context\n<paths, identifiers, versions, error strings that must not be lost>"
	titlePrompt   = "Generate a very short title (max 6 words) for this conversation. Reply with only the title, nothing else."
	retitlePrompt = "Generate a very short title (max 6 words) for this conversation based on the summary below. Reply with only the title, nothing else.\n\n"
)

// estimateTokens gives a rough token count for a string.
func estimateTokens(s string) int {
	return len(s) / charsPerToken
}

// compactTriggerTokens returns the estimated-token threshold at which
// compressHistory should fire. Derived from llm[0]'s n_ctx (discovered by
// the startup probe): subtract fixed overhead, then apply compactSafetyPct
// to absorb estimator bias. Falls back to rawBufferTokens when n_ctx is
// unknown (offline tests, server didn't report it).
func (a *agent) compactTriggerTokens() int {
	if a.mainContextTokens <= 0 {
		return rawBufferTokens
	}
	avail := a.mainContextTokens - compactOverheadTokens
	if avail < minCompactTrigger {
		return minCompactTrigger
	}
	return (avail * compactSafetyPct) / 100
}

// estimateMessagesTokens gives a rough token count for a slice of messages.
// Tool uses are counted because historyMessage serialises their name, input,
// and output into the assistant turn — so a tool-heavy conversation reaches
// the budget much sooner than the content field alone suggests.
func estimateMessagesTokens(msgs []Message) int {
	total := 0
	for _, m := range msgs {
		total += estimateTokens(m.Content) + roleTokenOverhead
		for _, tu := range m.ToolUses {
			total += estimateTokens(tu.Name) + estimateTokens(tu.Input) + estimateTokens(tu.Output) + roleTokenOverhead
		}
	}
	return total
}

// compressHistory rotates the session when the raw message buffer exceeds
// the budget: the current full state is frozen as a "session_archive_*"
// TOML, and the live session is rewritten in place with a single Summary
// covering all prior context plus the most recent 20% of messages kept
// verbatim. The session keeps its ACP-facing ID and on-disk path so Zed's
// panel continues without interruption — the archive is browsable via
// session/load if the user wants to inspect what was rotated out.
//
// On summarize failure we DO NOT rotate: better to keep the raw messages
// (and try again next turn) than to corrupt the live session with a junk
// summary. The user sees a warning instead.
//
// sess.mu is held across the whole routine so the background generateTitle
// goroutine cannot interleave a Save() (or read stale Messages while they're
// being resliced). retitle reacquires the lock via SetTitle, so we release
// before calling it.
func (a *agent) compressHistory(ctx context.Context, sess *Session) {
	sess.mu.Lock()
	if estimateMessagesTokens(sess.Messages) <= a.compactTriggerTokens() {
		sess.mu.Unlock()
		return
	}
	sess.mu.Unlock()

	// Join any in-flight background summarisation so the shadow buffer is
	// complete before we decide whether to use it. This happens BEFORE we
	// re-acquire sess.mu — backgroundSummarise's goroutine acquires sess.mu
	// briefly when reading messages, and we'd deadlock holding it here.
	sess.summariseJob.Wait()
	shadow := sess.drainShadow()

	sess.mu.Lock()

	// Split 20/80: the older 80% is folded into a fresh summary, the
	// recent 20% stays raw. (len*4)/5 lands at 0 only when len < 2 — in
	// that pathological case the single huge message is summarized whole
	// (and clipBytes below keeps its contribution bounded).
	splitIdx := (len(sess.Messages) * 4) / 5
	if splitIdx == 0 {
		splitIdx = len(sess.Messages)
	}
	keepMessages := append([]Message(nil), sess.Messages[splitIdx:]...)

	// Fast path: shadow buffer has structured per-turn notes covering the
	// turns we're about to rotate out. Use them directly instead of running
	// the synchronous summarize pass — the work has already been spread
	// across each turn while the user wasn't waiting. The existing Summary
	// (covering pre-previous-compaction context) is prepended verbatim so
	// nothing earlier gets lost.
	var summary string
	if shadow != "" {
		var b strings.Builder
		if sess.Summary != "" {
			b.WriteString(sess.Summary)
			b.WriteString("\n\n")
		}
		b.WriteString(shadow)
		summary = b.String()
	} else {
		// Main session uses LLM[0], subagents use their pinned LLM[i] — same
		// cache-consistency rule as the rest of the pipeline: every call on a
		// given session hits the same slot so the prefix cache stays warm.
		conn := a.pickAvailable(ctx, sess.ID, "execute")
		var buf strings.Builder
		if sess.Summary != "" {
			fmt.Fprintf(&buf, "[Earlier summary]\n%s\n\n", sess.Summary)
		}
		for _, m := range sess.Messages[:splitIdx] {
			fmt.Fprintf(&buf, "%s: %s\n\n", m.Role, clipBytes(m.Content, maxSummaryItemBytes))
		}
		s, err := a.summarize(ctx, sess.ID, conn, buf.String())
		if err != nil {
			sess.mu.Unlock()
			a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock(
				"⚠ History compaction skipped (summarize failed: "+err.Error()+"). Will retry next turn.\n\n")))
			return
		}
		summary = s
	}

	archiveID, err := sess.rotate(keepMessages, summary)
	if err != nil {
		sess.mu.Unlock()
		a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock("⚠ Compaction failed: "+err.Error()+"\n\n")))
		return
	}
	// Refresh the rendered system prompt so the next turn still sees current
	// skills + project context. Without this, the summariser's compression of
	// the older 80% would strip the skills + lookup-instruction paragraph
	// from the wire, and the LLM would drift after compaction. systemPrompt()
	// only reads sess.Cwd (immutable) and on-disk SKILL files, so it's safe
	// to call while holding sess.mu.
	if sysPrompt, err := a.systemPrompt(sess.ID); err == nil {
		sess.SystemPrompt = sysPrompt
	}
	_ = sess.saveLocked()
	sess.mu.Unlock()

	a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock(
		fmt.Sprintf("🗜 History compacted — archived as %s\n\n", archiveID))))

	a.retitle(ctx, sess)
}

// clipBytes truncates s to at most max bytes, leaving a marker in the middle
// when it had to cut. Used to bound any single message's contribution to the
// summarize prompt.
func clipBytes(s string, max int) string {
	if len(s) <= max {
		return s
	}
	half := max / 2
	return s[:half] + fmt.Sprintf("\n[... %d bytes truncated ...]\n", len(s)-max) + s[len(s)-half:]
}

// summarize calls the LLM to compress text. Returns an error on LLM failure
// so the caller can decide whether to proceed — we never silently substitute
// truncated raw text for a real summary.
func (a *agent) summarize(ctx context.Context, sid SessionId, conn *LLMConnection, text string) (string, error) {
	prompt := fmt.Sprintf("%s\n\nTarget length: ~%d tokens.\n\n<conversation>\n%s\n</conversation>", summarizePrompt, summaryBudget, text)
	return a.llmSimple(ctx, sid, conn, []llmMessage{{Role: "user", Content: prompt}})
}

// structuredTurnPrompt is the per-turn template fed to the background
// summariser. The six sections are the load-bearing ones for resuming a long
// task: Goal anchors what the user wants, Constraints capture stated rules,
// Progress + Decisions show what's locked in, Next Steps + Critical Context
// keep the thread alive across compaction. Each section is constrained to one
// line so the cumulative shadow buffer stays compact — a long session with
// many turns can otherwise dwarf the model's context.
const structuredTurnPrompt = "Summarise this exchange into a structured turn note. Each section is one line — be terse, no fluff. Skip any section that doesn't apply. Reply with just the structured note, no preamble.\n\n" +
	"Goal: what the user wants overall (not just this turn)\n" +
	"Constraints: rules/requirements/preferences the user has stated\n" +
	"Progress: concrete progress this turn (files changed, commands run, info gathered)\n" +
	"Decisions: choices made or directions taken\n" +
	"Next Steps: what's queued or open\n" +
	"Critical Context: paths, identifiers, versions, or state that must not be lost\n"

// maxShadowInputBytes bounds the bytes sent to the background summariser for
// any single turn — without this, a turn that produced megabytes of tool
// output (huge run_command dumps) would blow through the summariser's own
// context window. Matches maxSummaryItemBytes for symmetry with the
// synchronous compaction path.
const maxShadowInputBytes = 20 * 1024

// pickBackgroundLLM returns an LLM entry hosting a virtual slot ≥ 1 — i.e.
// any non-foreground background slot, picking the first reachable hosting
// entry in entry-major order. Virtual slot 0 is reserved for the warm
// foreground prefix cache and never selected here. Returns nil when:
//   - sid is a subagent session (subagents pin via SubagentPinOrder instead),
//   - no [[llm]] entries are configured,
//   - the total slot count across all entries is < 2 (only foreground exists),
//   - no eligible entry is reachable.
//
// The returned conn is the entry hosting the chosen slot; concurrency is
// bounded by that conn's `parallel` semaphore at llmStream, so two background
// callers landing on the same entry serialise naturally on the semaphore —
// no explicit waitForShadow coordination needed.
//
// Virtual slots flatten (entry, slot-within-entry) entry-major: one [[llm]]
// with parallel=2 exposes slots [0, 1] both hosted by entry 0; adding a
// second [[llm]] with parallel=2 adds slots [2, 3] hosted by entry 1.
func (a *agent) pickBackgroundLLM(sid SessionId) *LLMConnection {
	sess := a.getSession(sid)
	if sess == nil || sess.Depth > 0 {
		return nil
	}
	if len(a.settings.LLM) == 0 {
		return nil
	}
	totalSlots := 0
	for i := range a.settings.LLM {
		totalSlots += a.settings.LLM[i].parallelCap()
	}
	if totalSlots < 2 {
		return nil
	}
	reachable := func(idx int) bool {
		c := &a.settings.LLM[idx]
		return len(a.connReachable) == 0 || a.connReachable[connKey(c)]
	}
	// entryForSlot maps a virtual slot index to the [[llm]] entry that hosts
	// it under the entry-major flattening. Returns -1 when virtSlot is beyond
	// the total slot count.
	entryForSlot := func(virtSlot int) int {
		cum := 0
		for i := range a.settings.LLM {
			cap := a.settings.LLM[i].parallelCap()
			if virtSlot < cum+cap {
				return i
			}
			cum += cap
		}
		return -1
	}
	// Walk virtual slots [1, totalSlots) and return the first reachable host
	// entry. Each entry is checked at most once — multiple virtual slots can
	// map to the same entry, but they all share reachability.
	tried := map[int]bool{}
	for v := 1; v < totalSlots; v++ {
		e := entryForSlot(v)
		if e < 0 || tried[e] {
			continue
		}
		tried[e] = true
		if reachable(e) {
			return a.settings.ConnAt(e, "execute")
		}
	}
	return nil
}

// backgroundSummarise spawns a goroutine that condenses the just-completed
// turn pair (last user message + last assistant response) into the structured
// note format and appends it to the session's shadow buffer. Fired after each
// llmStream response inside the tool loop, so a long-running phase still
// produces incremental progress notes instead of one giant summary at the end.
// The goroutine uses context.Background() so it isn't killed when the user
// cancels the next prompt, but it carries a generous timeout so a stuck call
// can't pin a slot. The bgJob coalesces back-to-back fires from the tool loop
// (TryStart drops when one is in-flight) and exposes Wait so compressHistory
// can join before draining the shadow buffer.
func (a *agent) backgroundSummarise(sess *Session) {
	if sess == nil {
		return
	}
	if !sess.summariseJob.TryStart() {
		return
	}
	conn := a.pickBackgroundLLM(sess.ID)
	if conn == nil {
		sess.summariseJob.Done()
		return // no separate slot; falls through to sync summarize at compaction
	}

	// Snapshot the last user + assistant pair under the session lock so we
	// don't race with the next Prompt mutating Messages while we're reading.
	sess.mu.Lock()
	msgs := sess.Messages
	if len(msgs) < 2 {
		sess.mu.Unlock()
		sess.summariseJob.Done()
		return
	}
	var userMsg, assistantMsg Message
	for i := len(msgs) - 1; i >= 0; i-- {
		if assistantMsg.Role == "" && msgs[i].Role == "assistant" {
			assistantMsg = msgs[i]
			continue
		}
		if assistantMsg.Role != "" && msgs[i].Role == "user" {
			userMsg = msgs[i]
			break
		}
	}
	sess.mu.Unlock()
	if userMsg.Role == "" || assistantMsg.Role == "" {
		sess.summariseJob.Done()
		return
	}

	go func() {
		defer sess.summariseJob.Done()
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		var buf strings.Builder
		buf.WriteString(structuredTurnPrompt)
		buf.WriteString("\n<user_turn>\n")
		buf.WriteString(clipBytes(userMsg.Content, maxShadowInputBytes))
		buf.WriteString("\n</user_turn>\n<assistant_turn>\n")
		buf.WriteString(clipBytes(assistantMsg.Content, maxShadowInputBytes))
		buf.WriteString("\n</assistant_turn>")
		if len(assistantMsg.ToolUses) > 0 {
			buf.WriteString("\n<tool_calls>\n")
			for _, tu := range assistantMsg.ToolUses {
				fmt.Fprintf(&buf, "- %s(%s) → %s\n", tu.Name, tu.Input, truncate(tu.Output, 200))
			}
			buf.WriteString("</tool_calls>")
		}

		out, err := a.llmSimple(ctx, sess.ID, conn, []llmMessage{{Role: "user", Content: buf.String()}})
		if err != nil {
			// Silent — the synchronous path at next compaction will catch
			// anything we miss. No need to surface a UI warning for every
			// transient summariser hiccup.
			return
		}
		sess.appendShadow(out)
	}()
}

// generateTitle asks the LLM to create a short title from the first user
// message. Routes via the session's pin — main session always uses LLM[0]
// for cache consistency, subagents use their pinned LLM[i] entry.
func (a *agent) generateTitle(ctx context.Context, sess *Session, userText string) {
	conn := a.pickAvailable(ctx, sess.ID, "thinking")
	if conn == nil {
		a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock("⚠ Cannot generate title: no LLM connections configured\n")))
		return
	}
	messages := []llmMessage{{Role: "user", Content: titlePrompt + "\n\n" + userText}}
	title, err := a.llmSimple(ctx, sess.ID, conn, messages)
	if err != nil {
		a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock("⚠ Title generation failed: "+err.Error()+"\n")))
		title = userText
		if len(title) > titleFallbackLen {
			title = title[:titleFallbackLen]
		}
	}
	if title == "" {
		title = userText
		if len(title) > titleFallbackLen {
			title = title[:titleFallbackLen]
		}
	}
	sess.SetTitle(trimTitle(title))
	_ = sess.Save()
}

// trimTitle normalizes whitespace, strips wrapping quotes, and caps length.
func trimTitle(title string) string {
	title = strings.TrimSpace(title)
	title = strings.Trim(title, "\"'")
	if len(title) > maxTitleLen {
		title = title[:maxTitleLen]
	}
	return title
}

// retitle updates the session title based on the current summary. Same
// routing as generateTitle — main → LLM[0], subagent → pinned LLM[i].
func (a *agent) retitle(ctx context.Context, sess *Session) {
	if sess.Summary == "" {
		return
	}
	conn := a.pickAvailable(ctx, sess.ID, "thinking")
	if conn == nil {
		return // already warned in compressHistory
	}
	messages := []llmMessage{{Role: "user", Content: retitlePrompt + sess.Summary}}
	title, err := a.llmSimple(ctx, sess.ID, conn, messages)
	if err != nil {
		a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock("⚠ Retitle failed: "+err.Error()+"\n")))
		return
	}
	if title == "" {
		return
	}
	sess.SetTitle(trimTitle(title))
	_ = sess.Save()
}


// buildLLMHistory constructs the LLM message array from the session's stored
// system prompt, summary, and messages. Order is: SystemPrompt (skills +
// project context), then Summary (earlier-conversation digest if any), then
// the stored Messages. SystemPrompt and Summary are each emitted as a
// dedicated leading user message — set once on the first turn (SystemPrompt)
// or after a compressHistory rotation (both), so the rules and the digest
// survive compaction independently. skipIdx is the index in sess.Messages of
// the current user message; the caller appends it separately with the
// injected context.
func (a *agent) buildLLMHistory(sess *Session, skipIdx int) []llmMessage {
	var messages []llmMessage

	if sess.SystemPrompt != "" {
		messages = append(messages, llmMessage{
			Role:    "user",
			Content: sess.SystemPrompt,
		})
	}

	if sess.Summary != "" {
		messages = append(messages, llmMessage{
			Role:    "user",
			Content: "[Earlier conversation summary — most recent messages below take priority]\n\n" + sess.Summary,
		})
	}

	for i, m := range sess.Messages {
		if i == skipIdx {
			continue
		}
		messages = append(messages, a.historyMessage(m))
	}

	return messages
}

// historyMessage converts a stored Message to an llmMessage. Images pass
// through as OpenAI-style content blocks when the current LLM supports them;
// otherwise they're mentioned as text so the assistant knows they existed.
// Tool uses are summarized inline so the assistant knows what was already done.
func (a *agent) historyMessage(m Message) llmMessage {
	text := m.Content
	if len(m.ToolUses) > 0 {
		var buf strings.Builder
		buf.WriteString(text)
		buf.WriteString("\n\n[Tool calls performed:]\n")
		for _, tu := range m.ToolUses {
			fmt.Fprintf(&buf, "- %s(%s) → %s\n", tu.Name, tu.Input, truncate(tu.Output, 200))
		}
		text = buf.String()
	}

	if len(m.Images) == 0 {
		return llmMessage{Role: m.Role, Content: text}
	}
	if !a.imagesSupported {
		text = fmt.Sprintf("%s\n\n[Images: %d attached]", text, len(m.Images))
		return llmMessage{Role: m.Role, Content: text}
	}

	parts := []any{map[string]any{"type": "text", "text": text}}
	for _, img := range m.Images {
		parts = append(parts, map[string]any{
			"type": "image_url",
			"image_url": map[string]string{
				"url": fmt.Sprintf("data:%s;base64,%s", img.MimeType, img.Data),
			},
		})
	}
	return llmMessage{Role: m.Role, Content: parts}
}
