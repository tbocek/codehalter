package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// compactTriggerPct is the turn-boundary trigger: after a turn completes, if the
// context is at/above this fraction of the slot budget, fold every completed
// turn into Summary and start the next turn fresh. midTurnTriggerPct is the
// higher mid-turn trigger: only when an in-flight turn pushes the context this
// high do we compact without waiting for the boundary — and then we keep the
// in-flight turn verbatim and fold only the completed turns before it.
const (
	compactTriggerPct = 80
	midTurnTriggerPct = 90
)

// estimateMessageTokens is a chars/4 estimate of a context — the compaction
// fallback when the backend reports no prompt_tokens (a proxy stripping
// include_usage), where LastCompletePromptTokens stays 0 and the context would
// otherwise grow unbounded. ~30% off on tool JSON, but the 80% trigger has room.
func estimateMessageTokens(messages []llmMessage) int {
	chars := 0
	for _, m := range messages {
		switch c := m.Content.(type) {
		case string:
			chars += len(c)
		case nil:
		default:
			chars += len(fmt.Sprint(c)) // image parts etc. — rough; text dominates
		}
		for _, tc := range m.ToolCalls {
			chars += len(tc.Function.Name) + len(tc.Function.Arguments)
		}
	}
	return chars / 4
}

// compressHistory rotates the session when the prompt size crosses the trigger
// fraction of the slot budget — max(server prompt_tokens, curTokens), where
// curTokens is the caller's chars/4 estimate (keeps compaction working on
// backends that report no usage). The pre-rotation state is frozen as a
// "session_archive_*" TOML; the live session keeps its ID + path so Zed's panel
// continues. On any failure we DO NOT rotate.
//
// midTurn picks the policy. At the turn boundary (midTurn=false) the trigger is
// compactTriggerPct and we fold EVERY completed turn into Summary, keeping
// nothing verbatim — the next turn starts fresh. Mid-turn (midTurn=true, from
// the tool-loop overflow check) the trigger is the higher midTurnTriggerPct and
// we keep the in-flight turn (Messages[turnStart:]) verbatim, folding only the
// completed turns before it. Either way the background summariser has already
// produced one note per completed turn (boundary path runs it just before this;
// the in-flight turn has no note yet and is exactly what we keep), so folding
// the whole Shadow buffer rotates out exactly the turns those notes cover.
// Returns true when it compacted, so a mid-loop caller rebuilds its context.
func (a *agent) compressHistory(ctx context.Context, sess *Session, midTurn bool, curTokens int) bool {
	if server := sess.LastCompletePromptTokens(); server > curTokens {
		curTokens = server
	}
	trigger := compactTriggerPct
	if midTurn {
		trigger = midTurnTriggerPct
	}
	if curTokens <= a.mainSlotTokens*trigger/100 {
		return false
	}

	// keepFrom is the first message we DON'T fold. Boundary: fold everything.
	// Mid-turn: keep the in-flight turn verbatim, fold the completed turns ahead
	// of it. keepFrom==0 means there's nothing to compact yet (no messages, or
	// the in-flight turn IS the whole context) — not a failure.
	keepFrom := len(sess.Messages)
	if midTurn {
		keepFrom = sess.turnStartIndex()
	}
	if keepFrom <= 0 {
		return false
	}

	// Every completed turn we're about to rotate must have its note in the
	// buffer first — no anchor is held back, so wait for all of them.
	a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⏳ Waiting for background summary…\n"}})
	sess.waitSummarise()
	keepMessages := append([]Message(nil), sess.Messages[keepFrom:]...)
	shadow := sess.drainShadow()

	// Empty buffer → the completed turns produced no notes (background summariser
	// unconfigured/unreachable; backgroundSummarise stores a raw fallback note
	// when the LLM call itself fails, so this is genuinely "no summariser").
	// Skip rather than rotate those turns into oblivion; the next turn retries.
	if shadow == "" {
		a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⚠ History compaction skipped — no turn summaries available to fold. Check that the background-LLM slot is reachable; the session may approach n_ctx if this persists.\n\n"}})
		return false
	}
	var b strings.Builder
	if sess.Summary != "" {
		b.WriteString(sess.Summary)
		b.WriteString("\n\n")
	}
	b.WriteString(shadow)
	summary := b.String()

	archiveID, err := sess.rotate(keepMessages, summary)
	if err != nil {
		a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⚠ Compaction failed: " + err.Error() + "\n\n"}})
		return false
	}
	// The kept window now begins at index 0, so the in-flight turn (if any) does
	// too — re-anchor turnStart before more messages append to it.
	sess.resetTurnStart()
	// Re-render SystemPrompt so skills + project context survive the
	// summariser folding the original rendering into Summary. Compaction is the
	// one place the prompt may change, so this is where mid-session-seeded skills
	// (injected as user messages until now) finally enter the prefix — reset the
	// promptSkills baseline to match.
	if sysPrompt, err := a.systemPrompt(sess.ID); err == nil {
		sess.SystemPrompt = sysPrompt
		sess.promptSkills = skillFiles(sess.Cwd)
	}
	if err := sess.Save(); err != nil {
		a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("⚠ History compacted in-memory but persisting %s failed: %s. The archive %s is on disk, but the live session file is stale and will diverge until the next successful Save.\n\n", sess.filePath, err.Error(), archiveID)}})
		return true // in-memory state IS compacted; a rebuild from sess is valid
	}

	a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("🗜 History compacted — archived as %s\n\n", archiveID)}})
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
	prompt := a.loadPromptFile(sess.ID, "SUMMARISE.md")
	if prompt == "" {
		return
	}
	conn := a.connForBackgroundLLM()
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

	sess.enqueueSummarise(summariseTask{Turn: turn, Conn: conn}, func(t summariseTask) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		var turnBuf strings.Builder
		var images []ImageData
		for _, m := range t.Turn {
			images = append(images, m.Images...)
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
		// Cap the whole rendered turn so a long (many-iteration) turn can't blow the
		// background slot's context. clipBytes keeps the turn's head (the request +
		// early work) and tail (recent work + outcome), which is what a terse note
		// needs; without it a 100-iteration turn would 400 and degrade to the raw
		// fallback note.
		full := prompt + "\n" + clipBytes(turnBuf.String(), maxLLMInputBytes)

		out, _, _, err := a.llmStream(ctx, sess.ID, t.Conn, []llmMessage{{Role: "user", Content: full}}, nil, nil, nil)
		if err != nil || strings.TrimSpace(out) == "" {
			slog.Debug("backgroundSummarise: llm call failed — storing raw fallback note", "sid", sess.ID, "err", err)
			out = fallbackTurnNote(t.Turn)
		}
		// Image IDs are structured data — append deterministically so handles
		// survive the summariser's paraphrasing, otherwise view_image breaks
		// after compaction.
		if len(images) > 0 {
			var refs strings.Builder
			refs.WriteString("Attached images:")
			for _, img := range images {
				fmt.Fprintf(&refs, "\n- %s (%s) — call view_image id=%s to view", img.ID, img.MimeType, img.ID)
			}
			out = strings.TrimRight(out, "\n") + "\n\n" + refs.String()
		}
		sess.appendShadow(out)
		// Persist immediately so the note survives a process kill in the idle
		// gap before the next turn's save — that gap is exactly where the old
		// in-memory-only buffer lost notes across a restart.
		sess.saveOrLog()
	})
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

	for _, m := range sess.Messages {
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
