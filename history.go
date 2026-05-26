package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"time"
)

const compactTriggerPct = 80

// compressHistory rotates the session when the server-reported prompt_tokens
// cross compactTriggerPct of the slot budget. The pre-rotation state is
// frozen as a "session_archive_*" TOML; the live session keeps its
// ACP-facing ID + on-disk path so Zed's panel continues without interruption.
// On any failure we DO NOT rotate — keeping raw messages and retrying next
// turn beats corrupting the live session.
//
// Called at end-of-turn so we can block on the background summariser
// draining — the assistant's reply is already on screen.
func (a *agent) compressHistory(ctx context.Context, sess *Session) {
	if sess.LastCompletePromptTokens() <= a.mainSlotTokens*compactTriggerPct/100 {
		return
	}

	// Drain queued/in-flight summariser tasks EXCEPT the trailing one — it
	// covers the verbatim last message and can ride the next compaction.
	a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⏳ Waiting for background summary…\n"}})
	sess.waitSummarise()
	shadow := sess.drainShadow()

	if len(sess.Messages) < 2 {
		return
	}
	splitIdx := len(sess.Messages) - 1
	keepMessages := append([]Message(nil), sess.Messages[splitIdx:]...)

	// Empty shadow → summariser failed every turn since the last rotation.
	// Skip rather than corrupt Summary; next turn retries.
	if shadow == "" {
		a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⚠ History compaction skipped — background summariser produced no notes since the last rotation. The session may approach n_ctx if this persists; check the background-LLM slot reachability.\n\n"}})
		return
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
		return
	}
	// Re-render SystemPrompt so skills + project context survive the
	// summariser folding the original rendering into Summary.
	if sysPrompt, err := a.systemPrompt(sess.ID); err == nil {
		sess.SystemPrompt = sysPrompt
	}
	if err := sess.Save(); err != nil {
		a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("⚠ History compacted in-memory but persisting %s failed: %s. The archive %s is on disk, but the live session file is stale and will diverge until the next successful Save.\n\n", sess.filePath, err.Error(), archiveID)}})
		return
	}

	a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("🗜 History compacted — archived as %s\n\n", archiveID)}})
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
	conn := a.pickBackgroundLLM()
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
		buf.WriteString(clipBytes(t.User.Content, maxShadowInputBytes))
		buf.WriteString("\n</user_turn>\n<assistant_turn>\n")
		buf.WriteString(clipBytes(t.Assistant.Content, maxShadowInputBytes))
		buf.WriteString("\n</assistant_turn>")
		if len(t.Assistant.ToolUses) > 0 {
			buf.WriteString("\n<tool_calls>\n")
			for _, tu := range t.Assistant.ToolUses {
				fmt.Fprintf(&buf, "- %s(%s) → %s\n", tu.Name, tu.Input, truncateForLLM(tu.ID, tu.Name, tu.Input, tu.Output))
			}
			buf.WriteString("</tool_calls>")
		}

		out, err := a.llmSimple(ctx, sess.ID, t.Conn, []llmMessage{{Role: "user", Content: buf.String()}})
		if err != nil {
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
			tc := toolCall{ID: tu.ID, Type: "function"}
			tc.Function.Name = tu.Name
			tc.Function.Arguments = tu.Input
			toolCalls = append(toolCalls, tc)
		}

		messages = append(messages, llmMessage{Role: m.Role, Content: content, ToolCalls: toolCalls})
		for _, tu := range m.ToolUses {
			messages = append(messages, llmMessage{
				Role:       "tool",
				Content:    truncateForLLM(tu.ID, tu.Name, tu.Input, tu.Output),
				ToolCallID: tu.ID,
			})
		}
	}

	return messages
}
