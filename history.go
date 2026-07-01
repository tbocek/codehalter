package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// compactTriggerPct is the up-front input guard, NOT a compaction trigger.
// A user prompt whose text alone would exceed this fraction of the slot budget
// is refused (prompt.go), and the startup banner reports it (prepare.go).
// Compaction itself is purely reactive now: driven by the server's
// context-overflow 400, see foldHistory and the recovery loop in
// runToolLoopSeeded. No chars/4 estimate anywhere.
const compactTriggerPct = 80

// foldHistory folds Messages[:keepFrom] into the rolling Summary and keeps
// Messages[keepFrom:] verbatim, returning true when it folded anything. The
// recovery loop in runToolLoopSeeded calls this on a context-overflow 400 and
// escalates keepFrom: first keepWindowStart (keep the unfinished small turn plus
// the most recent ~keepSmallTurnTokens of completed small turns, fold everything
// older), then lastAssistantIndex (keep ONLY the unfinished small turn) if that
// still overflows. The server's 400 between the steps decides whether the cheaper
// step was enough.
//
// Messages[:turnStartIdx] are completed LARGE turns, already noted in Shadow.
// Messages[turnStartIdx:keepFrom] is the in-flight slice being folded; it has no
// note yet, so it is summarised synchronously here. Reuses rotate's epilogue.
func (a *agent) foldHistory(ctx context.Context, sess *Session, keepFrom int) bool {
	// Drain pending background notes for the completed LARGE turns first, so
	// rotating them out doesn't lose their summary.
	sess.waitSummarise()

	sess.mu.Lock()
	if keepFrom > len(sess.Messages) {
		keepFrom = len(sess.Messages)
	}
	if keepFrom <= 0 {
		sess.mu.Unlock()
		return false // the kept window is the whole live context; nothing to fold
	}
	start := sess.turnStartIdx
	if start < 0 || start > keepFrom {
		start = 0
	}
	inFlightCompleted := append([]Message(nil), sess.Messages[start:keepFrom]...)
	keepMessages := append([]Message(nil), sess.Messages[keepFrom:]...)
	sess.mu.Unlock()

	// Notes for everything rotating out: drained Shadow (completed large turns,
	// Messages[:start]) plus a synchronous summary of any in-flight slice being
	// folded (Messages[start:keepFrom]), which has no pre-computed note.
	var notes strings.Builder
	if shadow := sess.drainShadow(); shadow != "" {
		notes.WriteString(shadow)
	}
	if len(inFlightCompleted) > 0 {
		// Announce: hitting the limit mid-turn is routine management, NOT an error
		// (no ⚠/❌). This synchronous summarise can take a moment.
		a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "🗜 Context limit reached — compacting: summarising completed small turns, keeping the unfinished one…\n"}})
		var note string
		if prompt := a.loadPromptFile(sess.ID, "SUMMARISE.md"); prompt != "" {
			// Always paste-style here, never prefix-extension: this fold runs
			// BECAUSE the context overflowed, so context + instruction would
			// overflow too.
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
		return false // nothing to fold (no shadow notes and no in-flight slice)
	}

	var b strings.Builder
	if sess.Summary != "" {
		b.WriteString(sess.Summary)
		b.WriteString("\n\n")
	}
	b.WriteString(notes.String())

	archiveID, err := sess.rotate(keepMessages, b.String())
	if err != nil {
		a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⚠ Compaction failed: " + err.Error() + "\n\n"}})
		return false
	}
	sess.resetTurnStart()
	// Re-render the system prompt so skills + project context survive the fold.
	if sysPrompt, err := a.systemPrompt(sess.ID); err == nil {
		sess.SystemPrompt = sysPrompt
		sess.promptSkills = skillFiles(sess.Cwd)
	}
	if err := sess.Save(); err != nil {
		a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("⚠ Compacted in-memory but persisting failed: %s. Archive %s is on disk; the live session file will diverge until the next Save.\n\n", err.Error(), archiveID)}})
		return true
	}
	a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("🗜 Compacted — archived as %s\n\n", archiveID)}})
	return true
}

// backgroundSummarise enqueues exactly one structured-note task for the
// just-completed turn — every message from turnStart to the end, which spans
// the human prompt plus any synthetic subtask/doc prompts and all assistant
// tool calls. Fired once per turn at the boundary (runTurn epilogue), never
// mid-turn, so the Shadow buffer holds one note per completed turn. A single
// worker drains the queue sequentially. context.Background() so user
// cancellation of the next prompt doesn't kill an in-flight note. On LLM
// failure it stores a clipped raw fallback rather than nothing, so a completed
// turn always contributes one entry and compaction never rotates it out unnoted.
func (a *agent) backgroundSummarise(sess *Session) {
	if sess == nil {
		return
	}
	if sess.improving.Load() {
		// /improve turn: throwaway scratch context, no compaction note. Critically,
		// the summariser runs in a background goroutine that can outlive
		// endImproveScratch (which restores Messages/Summary but NOT Shadow), so a
		// note here would appendShadow into the RESTORED real conversation. Skip it.
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
	start := sess.turnStartIdx
	if start < 0 || start > len(sess.Messages) {
		start = 0
	}
	turn := append([]Message(nil), sess.Messages[start:]...)
	sess.mu.Unlock()
	if len(turn) == 0 {
		return
	}

	task := summariseTask{Turn: turn, Conn: conn, Prompt: prompt}
	// Prefix-extension mode: the note generates on the conn whose KV cache
	// already holds this conversation, so send the conversation itself plus a
	// small instruction instead of re-pasting the turn — the server reuses the
	// whole cached prefix and evaluates only the instruction. This is what
	// makes a single-slot (parallel = 1) server viable. Depth guard: a
	// subagent's history lives on its pinned conn, not necessarily here.
	if onMain && sess.Depth == 0 {
		task.Msgs = a.appendSummariseMsgs(sess, prompt, turn)
	}
	sess.enqueueSummarise(task, func(t summariseTask) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		var note string
		if len(t.Msgs) > 0 {
			note = a.summariseCall(ctx, sess, t.Conn, t.Msgs, t.Turn)
		} else {
			note = a.summariseSlice(ctx, sess, t.Conn, t.Prompt, t.Turn)
		}
		sess.appendShadow(note)
		// Persist immediately so the note survives a process kill in the idle
		// gap before the next turn's save — that gap is exactly where the old
		// in-memory-only buffer lost notes across a restart.
		sess.saveOrLog()
	})
}

// appendSummariseMsgs builds the prefix-extension summarise request: the
// session's full wire context (byte-identical to the turn that just ran, so
// the server reuses the cached prefix whole) plus one instruction message.
// The turn content is NOT pasted — it is already the tail of the context. The
// anchor quote pins WHICH span is "the final turn": a turn can contain
// synthetic inner user messages (subtask prompts, skill disclosures), so
// "everything after the last user message" would be wrong.
func (a *agent) appendSummariseMsgs(sess *Session, prompt string, turn []Message) []llmMessage {
	msgs := a.buildLLMContext(sess)
	instr := prompt + "\n\nThe exchange to summarise is the FINAL turn of the conversation above — nothing before it."
	for _, m := range turn {
		if m.Role == "user" && strings.TrimSpace(m.Content) != "" {
			instr += " That turn began with the user message: \"" + clipBytes(strings.TrimSpace(m.Content), 200) + "\""
			break
		}
	}
	return append(msgs, llmMessage{Role: "user", Content: instr})
}

// summariseCall is the shared execution core of both note modes: one LLM call
// over the prepared messages, the raw-transcript fallback when it fails or
// returns empty, and the image-reference block. Message construction is the
// only thing that differs between prefix-extension (appendSummariseMsgs) and
// transcript paste (summariseSlice), so it stays with the callers.
func (a *agent) summariseCall(ctx context.Context, sess *Session, conn *LLMConnection, msgs []llmMessage, turn []Message) string {
	out, _, _, err := a.llmStream(ctx, sess.ID, conn, msgs, nil, nil, nil)
	if err != nil || strings.TrimSpace(out) == "" {
		slog.Debug("summarise: llm call failed — using raw fallback note", "sid", sess.ID, "err", err)
		out = fallbackTurnNote(turn)
	}
	return attachImageRefs(out, turn)
}

// summariseSlice renders a contiguous slice of messages and condenses it into a
// structured note via the SUMMARISE.md summariser, falling back to a clipped raw
// transcript when the LLM call fails or returns empty. Image IDs are appended
// deterministically so they survive the summariser's paraphrasing and view_image
// keeps working after compaction. Both the per-turn background summariser (whole
// large turn) and the in-flight overflow recovery (the completed small turns of
// the in-flight large turn, see foldHistory) go through here.
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
					fmt.Fprintf(&turnBuf, "- %s(%s) → %s\n", tu.Name, tu.Input, truncateForLLM(tu.ID, tu.Name, tu.Input, tu.Output))
				}
				turnBuf.WriteString("</tool_calls>")
			}
		}
	}
	// Cap the whole rendered slice so a long (many-iteration) turn can't blow the
	// background slot's context. clipBytes keeps the slice's head (the request +
	// early work) and tail (recent work + outcome), which is what a terse note
	// needs; without it a 100-iteration turn would 400 and degrade to the raw
	// fallback note.
	full := prompt + "\n" + clipBytes(turnBuf.String(), maxLLMInputBytes)
	return a.summariseCall(ctx, sess, conn, []llmMessage{{Role: "user", Content: full}}, turn)
}

// attachImageRefs appends the deterministic image-reference block to a note so
// image IDs survive the summariser's paraphrasing and view_image keeps working
// after compaction. No-op for a turn without images.
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

// fallbackTurnNote builds a terse, clipped raw transcript of a turn for when the
// background summariser's LLM call fails. It is never the happy path: it exists
// so a completed turn still leaves a note in the Shadow buffer (preserving the
// one-note-per-turn invariant compaction relies on) instead of silently
// vanishing when it rotates out.
func fallbackTurnNote(turn []Message) string {
	var b strings.Builder
	b.WriteString("Progress: [automatic summary unavailable, raw excerpt follows]")
	for _, m := range turn {
		if c := strings.TrimSpace(m.Content); c != "" {
			fmt.Fprintf(&b, "\n%s: %s", m.Role, clipBytes(c, 800))
		}
		// Include tool names so a tool-only turn (no assistant text) still leaves
		// a trace of what it did, not just the user prompt.
		for _, tu := range m.ToolUses {
			fmt.Fprintf(&b, "\n%s tool: %s(%s)", m.Role, tu.Name, clipBytes(tu.Input, 200))
		}
	}
	return clipBytes(b.String(), 2400)
}

// buildLLMContext renders SystemPrompt + Summary + stored Messages as the
// wire-shape llmMessage slice. Tool calls use OpenAI protocol — assistant
// with ToolCalls, then one Role:"tool" per call with ToolCallID = tu.ID.
// Every stored image is inlined as image_url every turn so wire bytes for a
// given message stay byte-identical until compaction rotates it out; after
// compaction the reference lives in Summary and view_image fetches on demand.
// wireCallID is the tool_call id to put on the wire for a ToolUse: the model's
// own id (what the live request used) so a rebuild from history is byte-identical.
// Falls back to the internal useID for older sessions / models that sent no id.
func wireCallID(tu ToolUse) string {
	if tu.CallID != "" {
		return tu.CallID
	}
	return tu.ID
}

func (a *agent) buildLLMContext(sess *Session) []llmMessage {
	// Snapshot under the lock: the background summariser (foldHistory/rotate
	// reassigns s.Messages) and a parallel subagent's buildLLMContext(parent) can
	// mutate the session while this ranges it. Copy the slice header + the prompt
	// strings, then build from the copy unlocked.
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
			if !a.imagesSupported {
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
						// File gone (manually deleted, race) — collapse to a
						// retry-via-view_image text part so the turn doesn't
						// fail just because bytes vanished.
						parts = append(parts, map[string]any{
							"type": "text",
							"text": fmt.Sprintf("[Image %s (%s) — file missing on disk; call view_image id=%s to retry]", img.ID, img.MimeType, img.ID),
						})
						continue
					}
					if mime == "" {
						mime = img.MimeType
					}
					parts = append(parts, map[string]any{
						"type": "image_url",
						"image_url": map[string]string{
							"url": fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(data)),
						},
					})
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
				Role: "tool",
				// Re-render EXACTLY like the live call (liveToolOutput, not a tighter
				// history clip) so a replay is byte-identical to the wire — a cached
				// re-send is free, whereas clipping changes the bytes and forces a
				// reprocess. n_ctx size is bounded by compaction, not by clipping
				// here. tool_call_id is the model's id so it matches the wire too.
				Content:    liveToolOutput(tu.ID, tu.Name, tu.Input, tu.Output),
				ToolCallID: wireCallID(tu),
			})
		}
	}

	return messages
}
