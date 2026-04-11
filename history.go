package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
)

const (
	rawBufferTokens    = 10000
	summaryBudget      = 5000
	charsPerToken      = 4
	roleTokenOverhead  = 4
	summaryWordRatio   = 3.0 / 4.0
	maxTitleLen        = 60
	titleFallbackLen   = 50
	wholeFileEndLine   = 999999

	summarizePrompt = "Summarize this conversation concisely. Preserve key decisions and file paths. Do NOT include source code verbatim — instead reference the file path and describe what was changed. Drop pleasantries and redundant detail."
	titlePrompt     = "Generate a very short title (max 6 words) for this conversation. Reply with only the title, nothing else."
	retitlePrompt   = "Generate a very short title (max 6 words) for this conversation based on the summary below. Reply with only the title, nothing else.\n\n"
)

// CodeRef tracks a file region referenced in a history summary.
type CodeRef struct {
	Path      string `toml:"path"`
	StartLine int    `toml:"start_line"`
	EndLine   int    `toml:"end_line"`
	Hash      string `toml:"hash"`
}

// HistoryLevel is a summary of older conversation messages.
type HistoryLevel struct {
	Level   int       `toml:"level"`
	Content string    `toml:"content"`
	Refs    []CodeRef `toml:"refs,omitempty"`
}

// estimateTokens gives a rough token count for a string.
func estimateTokens(s string) int {
	return len(s) / charsPerToken
}

// estimateMessagesTokens gives a rough token count for a slice of messages.
func estimateMessagesTokens(msgs []Message) int {
	total := 0
	for _, m := range msgs {
		total += estimateTokens(m.Content) + roleTokenOverhead // role overhead
	}
	return total
}

// compressHistory checks if the raw message buffer exceeds the budget,
// and if so, summarizes older messages into cascading summary levels.
// Uses the "summary" LLM connection for summarization.
func (a *agent) compressHistory(ctx context.Context, sess *Session) {
	if estimateMessagesTokens(sess.Messages) <= rawBufferTokens {
		return
	}

	conn := a.settings.SummaryLLM()
	if conn == nil {
		a.sendUpdate(ctx, sess.ID, AgentMessageChunk(TextBlock("⚠ Cannot compress history: no 'summary' or 'thinking' LLM connection\n")))
		return
	}

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

	// Collect refs from the session's pending refs (added by write/edit tools).
	a.mu.Lock()
	refs := a.pendingRefs
	a.pendingRefs = nil
	a.mu.Unlock()

	sess.History = append(sess.History, HistoryLevel{Level: 0, Content: summary, Refs: refs})

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
			// Merge refs from both levels, keeping the most recent hash per path.
			mergedRefs := mergeRefs(prev.Refs, cur.Refs)
			sess.History = append(sess.History[:n-2], HistoryLevel{
				Level:   prev.Level + 1,
				Content: promoted,
				Refs:    mergedRefs,
			})
		} else {
			break
		}
	}

	a.retitle(ctx, sess)
	_ = sess.Save()
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
	sess.Title = title
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
	sess.Title = title
	_ = sess.Save()
}

// hashLines computes a hash of lines startLine..endLine (1-based) in a file.
func hashLines(path string, startLine, endLine int) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()

	h := sha256.New()
	scanner := bufio.NewScanner(f)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		if lineNum >= startLine && lineNum <= endLine {
			h.Write(scanner.Bytes())
			h.Write([]byte{'\n'})
		}
		if lineNum > endLine {
			break
		}
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// MakeCodeRef creates a CodeRef for a file region with the current hash.
func MakeCodeRef(path string, startLine, endLine int) CodeRef {
	return CodeRef{
		Path:      path,
		StartLine: startLine,
		EndLine:   endLine,
		Hash:      hashLines(path, startLine, endLine),
	}
}

// MakeFileRef creates a CodeRef for an entire file.
func MakeFileRef(path string) CodeRef {
	return CodeRef{
		Path:      path,
		StartLine: 1,
		EndLine:   wholeFileEndLine,
		Hash:      hashFile(path),
	}
}

func hashFile(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	h := sha256.Sum256(data)
	return fmt.Sprintf("%x", h[:])
}

// checkInvalidRefs checks all code refs across history levels and returns
// a list of human-readable invalidation notes for refs whose files changed.
func checkInvalidRefs(history []HistoryLevel) []string {
	var notes []string
	for _, level := range history {
		for _, ref := range level.Refs {
			var currentHash string
			if ref.EndLine >= wholeFileEndLine {
				currentHash = hashFile(ref.Path)
			} else {
				currentHash = hashLines(ref.Path, ref.StartLine, ref.EndLine)
			}
			if currentHash == "" {
				notes = append(notes, fmt.Sprintf("⚠ %s (referenced in history) no longer exists", ref.Path))
			} else if currentHash != ref.Hash {
				notes = append(notes, fmt.Sprintf("⚠ %s (referenced in history) has changed since it was last discussed — re-read before editing", ref.Path))
			}
		}
	}
	// Deduplicate.
	seen := map[string]bool{}
	var unique []string
	for _, n := range notes {
		if !seen[n] {
			unique = append(unique, n)
			seen[n] = true
		}
	}
	return unique
}

// mergeRefs combines refs from two levels, keeping the latest hash per file path.
func mergeRefs(a, b []CodeRef) []CodeRef {
	byPath := map[string]CodeRef{}
	for _, r := range a {
		byPath[r.Path] = r
	}
	for _, r := range b {
		byPath[r.Path] = r // newer overwrites older
	}
	var result []CodeRef
	for _, r := range byPath {
		result = append(result, r)
	}
	return result
}

// buildLLMHistory constructs the message array for the LLM, including
// summarized history from older levels and recent raw messages.
func (a *agent) buildLLMHistory(sess *Session, currentContent string) []llmMessage {
	var messages []llmMessage

	// Add summary levels (oldest first) with invalidation notes.
	if len(sess.History) > 0 {
		var historyBuf strings.Builder
		historyBuf.WriteString("[Conversation history - most recent decisions take priority]\n\n")

		// Check for stale code references.
		invalidations := checkInvalidRefs(sess.History)
		if len(invalidations) > 0 {
			historyBuf.WriteString("[Code changes since last discussed]\n")
			for _, note := range invalidations {
				historyBuf.WriteString(note + "\n")
			}
			historyBuf.WriteString("\n")
		}

		for i := len(sess.History) - 1; i >= 0; i-- {
			h := sess.History[i]
			fmt.Fprintf(&historyBuf, "[Summary level %d]:\n%s\n\n", h.Level, h.Content)
		}
		messages = append(messages, llmMessage{
			Role:    "user",
			Content: historyBuf.String(),
		})
		messages = append(messages, llmMessage{
			Role:    "assistant",
			Content: "I've reviewed the conversation history. I'll prioritize the most recent decisions and context.",
		})
	}

	// Add recent raw messages (excluding the current one which is last).
	for _, m := range sess.Messages {
		// Skip the current message — it'll be added separately with context.
		if m.Role == "user" && m.Content == currentContent {
			continue
		}
		content := m.Content
		// Append tool use summaries so the LLM knows what was done.
		if len(m.ToolUses) > 0 {
			var buf strings.Builder
			buf.WriteString(content)
			buf.WriteString("\n\n[Tool calls performed:]\n")
			for _, tu := range m.ToolUses {
				fmt.Fprintf(&buf, "- %s(%s) → %s\n", tu.Name, tu.Input, truncate(tu.Output, 200))
			}
			content = buf.String()
		}
		messages = append(messages, llmMessage{
			Role:    m.Role,
			Content: content,
		})
	}

	return messages
}
