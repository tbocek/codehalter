package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

const compactTriggerPct = 80

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

// compressHistory rotates the session when the prompt size crosses
// compactTriggerPct of the slot budget — max(server prompt_tokens, curTokens),
// where curTokens is the caller's chars/4 estimate (keeps compaction working on
// backends that report no usage). The pre-rotation state is frozen as a
// "session_archive_*" TOML; the live session keeps its ID + path so Zed's panel
// continues. On any failure we DO NOT rotate.
//
// Called at the turn boundary (summariseFirst=false) AND mid-turn from the 80%
// overflow check (summariseFirst=true) — mid-turn the optimistic summariser
// hasn't run, so summariseFirst captures the current turn before rotating.
// Returns true when it compacted, so a mid-loop caller rebuilds its context.
func (a *agent) compressHistory(ctx context.Context, sess *Session, summariseFirst bool, curTokens int) bool {
	if server := sess.LastCompletePromptTokens(); server > curTokens {
		curTokens = server
	}
	if curTokens <= a.mainSlotTokens*compactTriggerPct/100 {
		return false
	}

	// Mid-turn: summarise the current turn now (the optimistic summariser only
	// runs at the boundary), so the content we're about to rotate has a note.
	if summariseFirst {
		a.backgroundSummarise(sess)
	}

	// Drain queued/in-flight summariser tasks EXCEPT the trailing one — it
	// covers the verbatim last message and can ride the next compaction.
	a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⏳ Waiting for background summary…\n"}})
	sess.waitSummarise()
	shadow := sess.drainShadow()

	if len(sess.Messages) < 2 {
		return false
	}
	splitIdx := len(sess.Messages) - 1
	keepMessages := append([]Message(nil), sess.Messages[splitIdx:]...)

	// Empty shadow → summariser failed every turn since the last rotation.
	// Skip rather than corrupt Summary; next turn retries.
	if shadow == "" {
		a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⚠ History compaction skipped — background summariser produced no notes since the last rotation. The session may approach n_ctx if this persists; check the background-LLM slot reachability.\n\n"}})
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

// backgroundSummarise enqueues a structured-note task for the just-completed
// turn pair. Fired after every llmStream response in the tool loop; a single
// worker drains the queue sequentially. context.Background() so user
// cancellation of the next prompt doesn't kill in-flight notes. Errors are
// silent — compressHistory warns only when EVERY note since the last
// rotation failed (empty shadow).
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
	msgs := sess.Messages
	if len(msgs) < 2 {
		sess.mu.Unlock()
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
		return
	}

	sess.enqueueSummarise(summariseTask{User: userMsg, Assistant: assistantMsg, Conn: conn}, func(t summariseTask) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		var buf strings.Builder
		buf.WriteString(prompt)
		buf.WriteString("\n<user_turn>\n")
		buf.WriteString(clipBytes(t.User.Content, maxLLMInputBytes))
		buf.WriteString("\n</user_turn>\n<assistant_turn>\n")
		buf.WriteString(clipBytes(t.Assistant.Content, maxLLMInputBytes))
		buf.WriteString("\n</assistant_turn>")
		if len(t.Assistant.ToolUses) > 0 {
			buf.WriteString("\n<tool_calls>\n")
			for _, tu := range t.Assistant.ToolUses {
				fmt.Fprintf(&buf, "- %s(%s) → %s\n", tu.Name, tu.Input, truncateForLLM(tu.ID, tu.Name, tu.Input, tu.Output))
			}
			buf.WriteString("</tool_calls>")
		}

		out, _, err := a.llmStream(ctx, sess.ID, t.Conn, []llmMessage{{Role: "user", Content: buf.String()}}, nil, nil, nil)
		if err != nil {
			slog.Debug("backgroundSummarise: llm call failed — turn note lost for this pair", "sid", sess.ID, "err", err)
			return
		}
		// Image IDs are structured data — append deterministically so handles
		// survive the summariser's paraphrasing, otherwise view_image breaks
		// after compaction.
		if len(t.User.Images) > 0 {
			var refs strings.Builder
			refs.WriteString("Attached images:")
			for _, img := range t.User.Images {
				fmt.Fprintf(&refs, "\n- %s (%s) — call view_image id=%s to view", img.ID, img.MimeType, img.ID)
			}
			out = strings.TrimRight(out, "\n") + "\n\n" + refs.String()
		}
		sess.appendShadow(out)
	})
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
