package main

import (
	"context"
	"fmt"
	"strings"
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

	summarizePrompt = "Summarize this conversation concisely. Preserve: the user's overall goal; any open questions or in-flight work; decisions already made; file paths referenced. Do NOT include source code verbatim — reference the file path and describe what changed. Drop pleasantries and redundant detail."
	titlePrompt     = "Generate a very short title (max 6 words) for this conversation. Reply with only the title, nothing else."
	retitlePrompt   = "Generate a very short title (max 6 words) for this conversation based on the summary below. Reply with only the title, nothing else.\n\n"
)

// estimateTokens gives a rough token count for a string.
func estimateTokens(s string) int {
	return len(s) / charsPerToken
}

// compactTriggerTokens returns the estimated-token threshold at which
// compressHistory should fire. Derived from the [llm] connection's n_ctx
// (discovered by the startup probe): subtract fixed overhead, then apply
// compactSafetyPct to absorb estimator bias. Falls back to rawBufferTokens
// when n_ctx is unknown (offline tests, server didn't report it).
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

	// Main session uses [llm], subagents use [[subllm]] — same cache-consistency
	// rule as the rest of the pipeline: every call on a given session hits the
	// same slot so the prefix cache stays warm.
	conn := a.pickAvailable(ctx, sess.ID, "execute")

	// Split 20/80: the older 80% is folded into a fresh summary, the
	// recent 20% stays raw. (len*4)/5 lands at 0 only when len < 2 — in
	// that pathological case the single huge message is summarized whole
	// (and clipBytes below keeps its contribution bounded).
	splitIdx := (len(sess.Messages) * 4) / 5
	if splitIdx == 0 {
		splitIdx = len(sess.Messages)
	}
	oldMessages := sess.Messages[:splitIdx]
	keepMessages := append([]Message(nil), sess.Messages[splitIdx:]...)

	var buf strings.Builder
	if sess.Summary != "" {
		fmt.Fprintf(&buf, "[Earlier summary]\n%s\n\n", sess.Summary)
	}
	for _, m := range oldMessages {
		fmt.Fprintf(&buf, "%s: %s\n\n", m.Role, clipBytes(m.Content, maxSummaryItemBytes))
	}

	summary, err := a.summarize(ctx, sess.ID, conn, buf.String())
	if err != nil {
		sess.mu.Unlock()
		a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock(
			"⚠ History compaction skipped (summarize failed: "+err.Error()+"). Will retry next turn.\n\n")))
		return
	}

	archiveID, err := sess.rotate(keepMessages, summary)
	if err != nil {
		sess.mu.Unlock()
		a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock("⚠ Compaction failed: "+err.Error()+"\n\n")))
		return
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
	prompt := fmt.Sprintf("%s\n\nTarget length: ~%d tokens.\n\n---\n%s", summarizePrompt, summaryBudget, text)
	return a.llmSimple(ctx, sid, conn, []llmMessage{{Role: "user", Content: prompt}})
}

// generateTitle asks the LLM to create a short title from the first user
// message. Routes via the session's pin — main session always uses [llm]
// for cache consistency, subagents use their pinned [[subllm]] entry.
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
// routing as generateTitle — main → [llm], subagent → pinned [[subllm]].
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
// summary and messages. Messages are read 1:1 from TOML — project context is
// NOT stored and must be injected by the caller onto the current user message.
// skipIdx is the index in sess.Messages of the current user message; the
// caller appends it separately with the injected context.
func (a *agent) buildLLMHistory(sess *Session, skipIdx int) []llmMessage {
	var messages []llmMessage

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
