package main

import (
	"context"
	"fmt"
	"strings"
)

const (
	// rawBufferTokens is the threshold at which compressHistory fires.
	// Set close to the model's context limit so the prompt prefix stays
	// stable across many turns — that maximizes prefix-cache reuse on the
	// LLM side. With a 128k model this leaves ~28k for system prompt,
	// summary levels, and output reservation.
	rawBufferTokens   = 100000
	summaryBudget     = 12000
	charsPerToken     = 4
	roleTokenOverhead = 4
	summaryWordRatio  = 3.0 / 4.0
	maxTitleLen       = 60
	titleFallbackLen  = 50

	summarizePrompt = "Summarize this conversation concisely. Preserve key decisions and file paths. Do NOT include source code verbatim — instead reference the file path and describe what was changed. Drop pleasantries and redundant detail."
	titlePrompt     = "Generate a very short title (max 6 words) for this conversation. Reply with only the title, nothing else."
	retitlePrompt   = "Generate a very short title (max 6 words) for this conversation based on the summary below. Reply with only the title, nothing else.\n\n"
)

// HistoryLevel is a summary of older conversation messages.
type HistoryLevel struct {
	Level   int    `toml:"level"`
	Content string `toml:"content"`
}

// estimateTokens gives a rough token count for a string.
func estimateTokens(s string) int {
	return len(s) / charsPerToken
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
// TOML, and the live session is rewritten in place with a single Level 0
// summary covering all prior context plus the most recent 20% of messages
// kept verbatim. The session keeps its ACP-facing ID and on-disk path so
// Zed's panel continues without interruption — the archive is browsable via
// session/load if the user wants to inspect what was rotated out.
//
// sess.mu is held across the whole routine so the background generateTitle
// goroutine cannot interleave a Save() (or read stale Messages while they're
// being resliced). retitle reacquires the lock via SetTitle, so we release
// before calling it.
func (a *agent) compressHistory(ctx context.Context, sess *Session) {
	sess.mu.Lock()
	if estimateMessagesTokens(sess.Messages) <= rawBufferTokens {
		sess.mu.Unlock()
		return
	}

	conn := a.settings.LLMFor("summary", a.llmTier(sess.ID))

	// Split 20/80: the older 80% is folded into a fresh summary, the
	// recent 20% stays raw. (len*4)/5 lands at 0 only when len < 2 — in
	// that pathological case the single huge message is summarized whole.
	splitIdx := (len(sess.Messages) * 4) / 5
	if splitIdx == 0 {
		splitIdx = len(sess.Messages)
	}
	oldMessages := sess.Messages[:splitIdx]
	keepMessages := append([]Message(nil), sess.Messages[splitIdx:]...)

	// Fold every existing history level + the older 80% raw messages into
	// the input so the new Level 0 captures all prior context (the archive
	// preserves the pre-rotation state for forensic browsing).
	var buf strings.Builder
	for _, h := range sess.History {
		fmt.Fprintf(&buf, "[Earlier summary, level %d]\n%s\n\n", h.Level, h.Content)
	}
	for _, m := range oldMessages {
		fmt.Fprintf(&buf, "%s: %s\n\n", m.Role, m.Content)
	}
	summary := a.summarize(ctx, sess.ID, conn, buf.String(), summaryBudget)

	archiveID, err := sess.rotate(keepMessages, []HistoryLevel{{Level: 0, Content: summary}})
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

// summarize calls the LLM to compress text into roughly targetTokens. sid is
// used only to route the per-session debug log (empty disables logging).
func (a *agent) summarize(ctx context.Context, sid SessionId, conn *LLMConnection, text string, targetTokens int) string {
	prompt := fmt.Sprintf("%s\n\nTarget length: ~%d words.\n\n---\n%s", summarizePrompt, int(float64(targetTokens)*summaryWordRatio), text)
	messages := []llmMessage{{Role: "user", Content: prompt}}

	// Non-streaming request for summarization.
	response, err := a.llmSimple(ctx, sid, conn, messages)
	if err != nil {
		// If summarization fails, just truncate.
		chars := targetTokens * charsPerToken
		if len(text) > chars {
			return text[:chars] + "\n[truncated]"
		}
		return text
	}
	return response
}

// generateTitle asks the LLM to create a short title from the first user message.
func (a *agent) generateTitle(ctx context.Context, sess *Session, userText string) {
	conn := a.settings.LLMFor("thinking", a.llmTier(sess.ID))
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

// retitle updates the session title based on the latest summary.
func (a *agent) retitle(ctx context.Context, sess *Session) {
	if len(sess.History) == 0 {
		return
	}
	conn := a.settings.LLMFor("thinking", a.llmTier(sess.ID))
	if conn == nil {
		return // already warned in compressHistory
	}
	latest := sess.History[len(sess.History)-1]
	messages := []llmMessage{{Role: "user", Content: retitlePrompt + latest.Content}}
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
// history and messages. Messages are read 1:1 from TOML — project context is
// NOT stored and must be injected by the caller onto the current user message.
// skipIdx is the index in sess.Messages of the current user message; the
// caller appends it separately with the injected context.
func (a *agent) buildLLMHistory(sess *Session, skipIdx int) []llmMessage {
	var messages []llmMessage

	if len(sess.History) > 0 {
		var historyBuf strings.Builder
		historyBuf.WriteString("[Conversation history - most recent decisions take priority]\n\n")

		for i := len(sess.History) - 1; i >= 0; i-- {
			h := sess.History[i]
			fmt.Fprintf(&historyBuf, "[Summary level %d]:\n%s\n\n", h.Level, h.Content)
		}
		messages = append(messages, llmMessage{Role: "user", Content: historyBuf.String()})
		messages = append(messages, llmMessage{
			Role:    "assistant",
			Content: "I've reviewed the conversation history. I'll prioritize the most recent decisions and context.",
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
