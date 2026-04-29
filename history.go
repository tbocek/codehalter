package main

import (
	"context"
	"fmt"
	"strings"
)

const (
	rawBufferTokens   = 10000
	summaryBudget     = 5000
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

// compressHistory checks if the raw message buffer exceeds the budget,
// and if so, summarizes older messages into cascading summary levels.
// Uses the "summary" LLM connection for summarization.
//
// sess.mu is held across the whole routine so that the background
// generateTitle goroutine cannot interleave a Save() (or read stale Messages
// while they're being resliced). retitle reacquires the lock via SetTitle, so
// we release before calling it.
func (a *agent) compressHistory(ctx context.Context, sess *Session) {
	sess.mu.Lock()
	if estimateMessagesTokens(sess.Messages) <= rawBufferTokens {
		sess.mu.Unlock()
		return
	}

	conn := a.settings.SummaryLLM()

	// Split messages: keep the most recent ~half, summarize the older half.
	splitIdx := len(sess.Messages) / 2
	oldMessages := sess.Messages[:splitIdx]
	sess.Messages = sess.Messages[splitIdx:]

	// Format old messages for summarization.
	var buf strings.Builder
	for _, m := range oldMessages {
		fmt.Fprintf(&buf, "%s: %s\n\n", m.Role, m.Content)
	}
	oldText := buf.String()

	// If there's an existing level 0 summary, prepend it.
	if len(sess.History) > 0 && sess.History[len(sess.History)-1].Level == 0 {
		oldText = sess.History[len(sess.History)-1].Content + "\n\n" + oldText
		sess.History = sess.History[:len(sess.History)-1]
	}

	summary := a.summarize(ctx, conn, oldText, summaryBudget)

	sess.History = append(sess.History, HistoryLevel{Level: 0, Content: summary})

	// Cascade: if adjacent levels have the same level number, merge and promote.
	for {
		n := len(sess.History)
		if n < 2 {
			break
		}
		prev, cur := sess.History[n-2], sess.History[n-1]
		if prev.Level != cur.Level {
			break
		}
		merged := prev.Content + "\n\n" + cur.Content
		if estimateTokens(merged) > rawBufferTokens {
			promoted := a.summarize(ctx, conn, merged, summaryBudget)
			sess.History = append(sess.History[:n-2], HistoryLevel{
				Level:   prev.Level + 1,
				Content: promoted,
			})
		} else {
			break
		}
	}

	_ = sess.saveLocked()
	sess.mu.Unlock()

	a.retitle(ctx, sess)
}

// summarize calls the LLM to compress text into roughly targetTokens.
func (a *agent) summarize(ctx context.Context, conn *LLMConnection, text string, targetTokens int) string {
	prompt := fmt.Sprintf("%s\n\nTarget length: ~%d words.\n\n---\n%s", summarizePrompt, int(float64(targetTokens)*summaryWordRatio), text)
	messages := []llmMessage{{Role: "user", Content: prompt}}

	// Non-streaming request for summarization.
	response, err := a.llmSimple(ctx, conn, messages)
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
	conn := a.settings.LLM("thinking")
	if conn == nil {
		a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock("⚠ Cannot generate title: no 'thinking' LLM connection\n")))
		return
	}
	messages := []llmMessage{{Role: "user", Content: titlePrompt + "\n\n" + userText}}
	title, err := a.llmSimple(ctx, conn, messages)
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
	title = strings.TrimSpace(title)
	title = strings.Trim(title, "\"'")
	if len(title) > maxTitleLen {
		title = title[:maxTitleLen]
	}
	sess.SetTitle(title)
	_ = sess.Save()
}

// retitle updates the session title based on the latest summary.
func (a *agent) retitle(ctx context.Context, sess *Session) {
	if len(sess.History) == 0 {
		return
	}
	conn := a.settings.LLM("thinking")
	if conn == nil {
		return // already warned in compressHistory
	}
	latest := sess.History[len(sess.History)-1]
	messages := []llmMessage{{Role: "user", Content: retitlePrompt + latest.Content}}
	title, err := a.llmSimple(ctx, conn, messages)
	if err != nil {
		a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock("⚠ Retitle failed: "+err.Error()+"\n")))
		return
	}
	if title == "" {
		return
	}
	title = strings.TrimSpace(title)
	title = strings.Trim(title, "\"'")
	if len(title) > maxTitleLen {
		title = title[:maxTitleLen]
	}
	sess.SetTitle(title)
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
