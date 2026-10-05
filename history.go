package main

import (
	"context"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"time"
)

// An input guard, NOT a compaction trigger: a prompt whose text alone exceeds
// this share of the slot is refused. Compaction runs only on the server's 400.
const compactTriggerPct = 80

// The 400-recovery loop escalates keepFrom (keepWindowStart, then lastAssistantIndex).
// The in-flight slice has no Shadow note yet, so it is summarised here.
func (a *agent) foldHistory(ctx context.Context, sess *Session, keepFrom int) bool {
	// Join pending notes first, or the turns rotating out lose them.
	sess.waitSummarise()

	sess.mu.Lock()
	if keepFrom > len(sess.Messages) {
		keepFrom = len(sess.Messages)
	}
	if keepFrom <= 0 {
		sess.mu.Unlock()
		return false
	}
	inFlightCompleted := append([]Message(nil), sess.Messages[sess.turnStartIdx:keepFrom]...)
	keepMessages := append([]Message(nil), sess.Messages[keepFrom:]...)
	sess.mu.Unlock()

	var notes strings.Builder
	if shadow := sess.drainShadow(); shadow != "" {
		notes.WriteString(shadow)
	}
	if len(inFlightCompleted) > 0 {
		a.say(ctx, sess.ID, "🗜 Context limit reached, compacting: summarising completed small turns, keeping the unfinished one…\n")
		var note string
		if prompt := a.loadPromptFile(sess.ID, "SUMMARISE.md"); prompt != "" {
			// Paste mode, never prefix-extension: the context already overflowed.
			if conn, _ := a.connForBackgroundLLM(); conn != nil {
				note = a.summariseSlice(ctx, sess, conn, prompt, inFlightCompleted)
			}
		}
		if note == "" {
			note = fallbackTurnNote(inFlightCompleted)
		}
		if notes.Len() > 0 {
			notes.WriteString("\n\n")
		}
		notes.WriteString(note)
	}
	if notes.Len() == 0 {
		return false
	}

	// waitSummarise above joined the background fold, so FoldedSummary is settled.
	var b strings.Builder
	prev, folded := sess.Summary, sess.FoldedSummary
	base := prev
	if folded != "" {
		base = folded
	}
	if base != "" {
		b.WriteString(base)
		b.WriteString("\n\n")
	}
	b.WriteString(notes.String())
	summary := b.String()

	sess.FoldedSummary = "" // consumed: it described the Summary being replaced
	archiveID, err := sess.rotate(keepMessages, summary)
	if err != nil {
		a.say(ctx, sess.ID, "⚠ Compaction failed: "+err.Error()+"\n\n")
		return false
	}
	// No lock: rotate runs with no concurrent writer.
	sess.turnStartIdx = 0
	// The fold rewrote the front, so the next full re-read is not a cache fault.
	sess.resetCacheLineage()
	// Only compaction may change the prefix, so refresh the system prompt here.
	if sysPrompt, err := a.systemPrompt(sess.ID); err == nil {
		sess.SystemPrompt = sysPrompt
		sess.promptSkills = skillSet(sess.Cwd, sess.knownStacks)
	}
	if err := sess.Save(); err != nil {
		a.say(ctx, sess.ID, fmt.Sprintf("⚠ Compacted in-memory but persisting failed: %s. Archive %s is on disk; the live session file will diverge until the next Save.\n\n", err.Error(), archiveID))
	} else {
		note := ""
		if folded != "" {
			note = fmt.Sprintf(", prior summary folded %d→%d KB", len(prev)/1024, len(folded)/1024)
		}
		a.say(ctx, sess.ID, fmt.Sprintf("🗜 Compacted, archived as %s%s\n\n", archiveID, note))
	}
	// After the Save: the fold's worker saves again when it lands.
	a.scheduleSummaryFold(sess, summary)
	return true
}

// Crossing it queues a background rewrite of the append-only Summary for the next
// compaction; Summary sits in the KV cache for every token decoded.
const maxSummaryBytes = 16 * 1024

// Lands in FoldedSummary, never Summary: swapping the front mid-session costs a full
// re-prefill. On the summariser queue so waitSummarise joins it before a compaction.
func (a *agent) scheduleSummaryFold(sess *Session, summary string) {
	if len(summary) <= maxSummaryBytes {
		return
	}
	prompt := a.loadPromptFile(sess.ID, "RESUMMARISE.md")
	conn, _ := a.connForBackgroundLLM()
	if prompt == "" || conn == nil {
		return
	}
	sess.enqueueSummarise(func() {
		// Longer than a turn note's: 16 KB in, thousands of tokens out, nothing waiting.
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		out, _, _, err := a.llmStream(ctx, sess.ID, conn.withThinkingDisabled(), []llmMessage{{
			Role: "user", Content: prompt + "\n\n" + clipBytes(summary, maxLLMInputBytes),
		}}, nil, nil, nil, nil)
		out = strings.TrimSpace(out)
		// On failure the next compaction concatenates: growing beats losing the record.
		if err != nil || out == "" || len(out) >= len(summary) {
			slog.Debug("summary fold: keeping the concatenated summary",
				"sid", sess.ID, "err", err, "in", len(summary), "out", len(out))
			return
		}
		sess.mu.Lock()
		// Defensive: a fold of a stale Summary would silently drop a compaction's notes.
		if sess.Summary == summary {
			sess.FoldedSummary = keepImageRefs(summary, out)
		}
		sess.mu.Unlock()
		sess.saveOrLog()
	})
}

var (
	imageRefLine = regexp.MustCompile(`(?m)^- (img_[0-9a-f]+) \([^)]*\) — call view_image id=[0-9a-z_]+ to view$`)
	imageID      = regexp.MustCompile(`img_[0-9a-f]+`)
)

// An image ref the model paraphrases away is unrecoverable: nothing else names
// the id, so view_image can never be asked for it.
func keepImageRefs(old, folded string) string {
	have := map[string]bool{}
	for _, id := range imageID.FindAllString(folded, -1) {
		have[id] = true
	}
	var missing []string
	for _, m := range imageRefLine.FindAllStringSubmatch(old, -1) {
		if !have[m[1]] {
			have[m[1]] = true // a duplicated ref in old shouldn't append twice
			missing = append(missing, m[0])
		}
	}
	if len(missing) == 0 {
		return folded
	}
	return strings.TrimRight(folded, "\n") + "\n\nAttached images:\n" + strings.Join(missing, "\n")
}

// Once per completed turn, so Shadow holds one note per turn. Background ctx:
// cancelling the next prompt must not kill a note in flight.
func (a *agent) backgroundSummarise(sess *Session) {
	if sess == nil {
		return
	}
	prompt := a.loadPromptFile(sess.ID, "SUMMARISE.md")
	if prompt == "" {
		return
	}
	conn, onMain := a.connForBackgroundLLM()
	if conn == nil {
		return
	}

	sess.mu.Lock()
	turn := append([]Message(nil), sess.Messages[sess.turnStartIdx:]...)
	sess.mu.Unlock()
	if len(turn) == 0 {
		return
	}

	// Prefix-extension: resend the cached conversation plus an instruction, so the
	// server evaluates only the instruction. Built at enqueue so a queued job still
	// summarises its own turn; the anchor quote is needed because a turn can hold
	// synthetic inner user messages.
	var msgs []llmMessage
	if onMain {
		msgs = a.buildLLMContext(sess)
		instr := prompt + "\n\nThe exchange to summarise is the FINAL turn of the conversation above, nothing before it."
		for _, m := range turn {
			if m.Role == "user" && strings.TrimSpace(m.Content) != "" {
				instr += " That turn began with the user message: \"" + clipBytes(strings.TrimSpace(m.Content), 200) + "\""
				break
			}
		}
		msgs = append(msgs, llmMessage{Role: "user", Content: instr})
	}
	sess.enqueueSummarise(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		var note string
		if onMain {
			// Same tools as the foreground, or the prefix diverges (see summariseCall).
			note = a.summariseCall(ctx, sess, conn.withBody("tool_choice", "none"), msgs, a.tools.defs(), turn)
		} else {
			note = a.summariseSlice(ctx, sess, conn, prompt, turn)
		}
		sess.appendShadow(note)
		// Now, so the note survives a kill in the idle gap before the next save.
		sess.saveOrLog()
	})
}

// tools must be the foreground's full array in prefix-extension mode (the template
// renders them at the prompt head), nil in paste mode. Reasoning off: it overruns the deadline.
func (a *agent) summariseCall(ctx context.Context, sess *Session, conn *LLMConnection, msgs []llmMessage, tools []map[string]any, turn []Message) string {
	out, _, _, err := a.llmStream(ctx, sess.ID, conn.withThinkingDisabled(), msgs, tools, nil, nil, nil)
	// By endpoint, not Slot: connForBackgroundLLM stamps its llm[0] fallback with Slot 1.
	dedicated := false
	a.cfgMu.RLock()
	if len(a.settings.LLM) > 0 {
		dedicated = conn.Server != a.settings.LLM[0].Server || conn.Model != a.settings.LLM[0].Model
	}
	a.cfgMu.RUnlock()
	if err != nil || strings.TrimSpace(out) == "" {
		// Warn, not Debug: a fallback note silently degrades every later compaction.
		slog.Warn("summarise: llm call failed, using raw fallback note", "sid", sess.ID, "server", conn.Server, "err", err)
		a.logSession(sess.ID, "SUMMARISE", "failed on %s (%s): turn note fell back to a raw transcript: %v", conn.Server, conn.Model, err)
		if dedicated {
			// Strikes take the summariser out of rotation in connForBackgroundLLM.
			a.summaryStrikes.Add(1)
			a.summaryStruckAt.Store(time.Now().UnixNano())
		}
		out = fallbackTurnNote(turn)
	} else if dedicated {
		a.summaryStrikes.Store(0)
	}
	return attachImageRefs(out, turn)
}

func (a *agent) summariseSlice(ctx context.Context, sess *Session, conn *LLMConnection, prompt string, turn []Message) string {
	var turnBuf strings.Builder
	for _, m := range turn {
		switch m.Role {
		case "user":
			turnBuf.WriteString("\n<user_turn>\n")
			turnBuf.WriteString(clipBytes(m.Content, maxLLMInputBytes))
			turnBuf.WriteString("\n</user_turn>")
		case "assistant":
			if strings.TrimSpace(m.Content) != "" {
				turnBuf.WriteString("\n<assistant_turn>\n")
				turnBuf.WriteString(clipBytes(m.Content, maxLLMInputBytes))
				turnBuf.WriteString("\n</assistant_turn>")
			}
			if len(m.ToolUses) > 0 {
				turnBuf.WriteString("\n<tool_calls>\n")
				for _, tu := range m.ToolUses {
					fmt.Fprintf(&turnBuf, "- %s(%s) → %s\n", tu.Name, tu.Input, truncateForLLM(tu.Name, tu.Input, tu.Output))
				}
				turnBuf.WriteString("</tool_calls>")
			}
		}
	}
	// A paste lacks the Summary prefix mode sees, so add it read-only: foldHistory
	// appends the note after that Summary, and copied text would be stored twice.
	sess.mu.Lock()
	prior := strings.TrimSpace(sess.Summary)
	sess.mu.Unlock()
	if prior != "" {
		prompt += "\n\n<already_recorded>\n" + clipBytes(prior, maxSummaryBytes) + "\n</already_recorded>\n" +
			"The block above is what has ALREADY been recorded about this session. " +
			"Use it only to stay consistent with names, goals and open constraints. " +
			"Do NOT repeat any of it: summarise ONLY the exchange below."
	}
	full := prompt + "\n" + clipBytes(turnBuf.String(), maxLLMInputBytes)
	return a.summariseCall(ctx, sess, conn, []llmMessage{{Role: "user", Content: full}}, nil, turn)
}

// Deterministic, so image IDs survive the summariser's paraphrasing.
func attachImageRefs(note string, turn []Message) string {
	var images []ImageData
	for _, m := range turn {
		images = append(images, m.Images...)
	}
	if len(images) == 0 {
		return note
	}
	var refs strings.Builder
	refs.WriteString("Attached images:")
	for _, img := range images {
		fmt.Fprintf(&refs, "\n- %s (%s) — call view_image id=%s to view", img.ID, img.MimeType, img.ID)
	}
	return strings.TrimRight(note, "\n") + "\n\n" + refs.String()
}

// Keeps the one-note-per-turn invariant compaction relies on when the summariser fails.
func fallbackTurnNote(turn []Message) string {
	var b strings.Builder
	b.WriteString("Progress: [automatic summary unavailable, raw excerpt follows]")
	for _, m := range turn {
		if c := strings.TrimSpace(m.Content); c != "" {
			fmt.Fprintf(&b, "\n%s: %s", m.Role, clipBytes(c, 800))
		}
		for _, tu := range m.ToolUses {
			fmt.Fprintf(&b, "\n%s tool: %s(%s)", m.Role, tu.Name, clipBytes(tu.Input, 200))
		}
	}
	return clipBytes(b.String(), 2400)
}

// Must reproduce the live wire bytes so the prompt stays cached: never a tighter clip.
// view_image is pure and re-runs; a tool that produced an image replays from ImageID.
func (a *agent) replayToolOutput(sess *Session, tu ToolUse) any {
	text := liveToolOutput(tu.Name, tu.Input, tu.Output)
	if tu.Failed || !a.imagesSupported.Load() {
		return text
	}
	// Image file gone: no replay can be identical, so fall back rather than fail.
	if tu.ImageID != "" {
		data, mime, err := readImageFile(sess.Cwd, tu.ImageID)
		if err != nil {
			return text
		}
		// tu.Output, not text: the live call put the untruncated text in parts[0].
		return imageParts(tu.Output, mime, data)
	}
	if tu.Name != "view_image" {
		return text
	}
	_, parts, failed := dispatchViewImage(sess, tu.Input)
	if failed {
		return text
	}
	return parts
}

// The model's own id first, so a rebuild is byte-identical to the live request.
func wireCallID(tu ToolUse) string {
	if tu.CallID != "" {
		return tu.CallID
	}
	return tu.ID
}

// Images are inlined every turn so a message's wire bytes stay identical until compaction.
func (a *agent) buildLLMContext(sess *Session) []llmMessage {
	// Snapshot: foldHistory/rotate can reassign Messages while this ranges it.
	sess.mu.Lock()
	systemPrompt, summary := sess.SystemPrompt, sess.Summary
	msgs := append([]Message(nil), sess.Messages...)
	sess.mu.Unlock()

	var messages []llmMessage

	if systemPrompt != "" {
		messages = append(messages, llmMessage{
			Role:    "user",
			Content: systemPrompt,
		})
	}

	if summary != "" {
		messages = append(messages, llmMessage{
			Role:    "user",
			Content: "[Earlier conversation summary — most recent messages below take priority]\n\n" + summary,
		})
	}

	for _, m := range msgs {
		var content any = m.Content
		if len(m.Images) > 0 {
			if !a.imagesSupported.Load() {
				var buf strings.Builder
				buf.WriteString(m.Content)
				for _, img := range m.Images {
					fmt.Fprintf(&buf, "\n\n[Image %s (%s) — call view_image id=%s to view]", img.ID, img.MimeType, img.ID)
				}
				content = buf.String()
			} else {
				parts := []any{map[string]any{"type": "text", "text": m.Content}}
				for _, img := range m.Images {
					data, mime, err := readImageFile(sess.Cwd, img.ID)
					if err != nil {
						parts = append(parts, map[string]any{
							"type": "text",
							"text": fmt.Sprintf("[Image %s (%s) — file missing on disk; call view_image id=%s to retry]", img.ID, img.MimeType, img.ID),
						})
						continue
					}
					parts = append(parts, imageURLPart(mime, data))
				}
				content = parts
			}
		}

		var toolCalls []toolCall
		for _, tu := range m.ToolUses {
			tc := toolCall{ID: wireCallID(tu), Type: "function"}
			tc.Function.Name = tu.Name
			tc.Function.Arguments = tu.Input
			toolCalls = append(toolCalls, tc)
		}

		messages = append(messages, llmMessage{Role: m.Role, Content: content, ToolCalls: toolCalls})
		for _, tu := range m.ToolUses {
			messages = append(messages, llmMessage{
				Role:       "tool",
				Content:    a.replayToolOutput(sess, tu),
				ToolCallID: wireCallID(tu),
			})
		}
	}

	return messages
}
