package main

import (
	"context"
	"fmt"
	"strings"
	"time"
)

const compactTriggerPct = 80

// compressHistory rotates the session when the server-reported prompt_tokens
// for the most recent llmStream call crosses compactTriggerPct of the slot
// budget. The session is rewritten in place: SystemPrompt + Summary + the
// single most-recent message kept verbatim (every prior turn folds into the
// summary — the background summariser fires after each llmStream response,
// so the shadow buffer already covers the just-completed turn by the time
// we get here). The pre-rotation state is frozen as a "session_archive_*"
// TOML for inspection. The session keeps its ACP-facing ID and on-disk path
// so Zed's panel continues without interruption.
//
// Called at end-of-turn (post-orchestrate epilogue in prompt.go), so we can
// afford to block on the background summariser draining — the assistant's
// reply is already on screen and the next user prompt picks up the rotated
// state. On summarize failure we DO NOT rotate: keeping raw messages and
// retrying next turn is safer than corrupting the live session.
//
// Runs lock-free: the post-orchestrate epilogue has no concurrent writer
// (prompt() runs serially per session and the older summariser tasks have
// been joined by waitSummarise above), and the final Save() acquires sess.mu
// internally to serialise the encoder write.
func (a *agent) compressHistory(ctx context.Context, sess *Session) {
	// Trigger on the server's reported prompt_tokens — ground truth for
	// what just landed in n_ctx, no estimator needed. 0 means no turn has
	// run yet (process restart or test fixture); nothing to compact.
	if sess.LastCompletePromptTokens() <= a.mainSlotTokens*compactTriggerPct/100 {
		return
	}

	// Wait for every queued/in-flight summariser task EXCEPT the most
	// recent one — the trailing task covers the verbatim "last message"
	// we're about to keep, so it can ride the next compaction without
	// stalling end-of-turn. The user has the assistant's reply on screen
	// already; the status row tells them what's happening while the older
	// tasks drain.
	a.sendUpdate(ctx, sess.ID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⏳ Waiting for background summary…\n"}})
	sess.waitSummarise()
	shadow := sess.drainShadow()

	// Keep the single most-recent message verbatim; everything before it
	// folds into the summary. Nothing to rotate if there are fewer than 2
	// messages on disk.
	if len(sess.Messages) < 2 {
		return
	}
	splitIdx := len(sess.Messages) - 1
	keepMessages := append([]Message(nil), sess.Messages[splitIdx:]...)

	// Shadow buffer holds structured per-turn notes covering the turns we're
	// about to rotate out — ensureLLM guarantees a background summariser slot
	// is configured, so the foreground turn never has to run the summarize
	// pass synchronously. Empty shadow means the summariser hit transient
	// errors on every turn since the last compaction (errors are silent in
	// backgroundSummarise). We surface a warning and skip the rotation rather
	// than corrupt Summary with empty content; the next turn's background
	// summariser may recover and the compaction will retry.
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
	// Refresh the rendered system prompt so the next turn still sees current
	// skills + project context. Without this, the summariser's compression of
	// the older 80% would strip the skills + lookup-instruction paragraph
	// from the wire, and the LLM would drift after compaction. systemPrompt()
	// only reads sess.Cwd (immutable) and on-disk SKILL files.
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
// turn pair (last user message + last assistant response). Fired after each
// llmStream response inside the tool loop; every call enqueues — none are
// dropped, so a 50-iteration loop produces 50 progress notes covering every
// intermediate assistant snapshot. A single worker drains the queue
// sequentially (see Session.enqueueSummarise); the per-conn slot semaphore
// would serialise concurrent runners anyway. The runner uses
// context.Background() so user cancellation of the next prompt doesn't kill
// in-flight notes, with a per-task timeout so a stuck call can't pin the
// queue.
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
		// Defensive: ensureLLM gates startup on at least one configured LLM,
		// so this only fires if settings were emptied mid-session.
		return
	}

	// Snapshot the last user + assistant pair under the session lock so we
	// don't race with the next Prompt mutating Messages while we're reading.
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
			// Silent — a single failed background note isn't worth a UI
			// warning. The shadow buffer still holds older notes; only when
			// EVERY note since the last rotation fails does compressHistory
			// see an empty shadow and surface the "skipped" warning to the
			// user.
			return
		}
		sess.appendShadow(out)
	})
}

// buildLLMContext constructs the LLM message array from the session's stored
// system prompt, summary, and messages. Order is: SystemPrompt (skills +
// project context), then Summary (earlier-conversation digest if any), then
// the stored Messages. SystemPrompt and Summary are each emitted as a
// dedicated leading user message — set once on the first turn (SystemPrompt)
// or after a compressHistory rotation (both), so the rules and the digest
// survive compaction independently.
//
// Tool calls are rebuilt in the OpenAI protocol shape — assistant message with
// a ToolCalls field, followed by one Role:"tool" message per call carrying
// the truncated output and a ToolCallID pointer back to the call. tu.ID
// ("tu_<n>") is reused as the tool_call_id; tool_call_ids only need to be
// unique within a single request, and the model already knows tu_<n> from
// the view_output hints embedded in the truncated content.
//
// Image bytes are inlined as OpenAI-style image_url blocks ONLY for the
// trailing message — older messages become a text placeholder referencing
// the image id so the model can call view_image to retrieve any earlier
// attachment without paying the byte cost on every turn.
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

	lastIdx := len(sess.Messages) - 1
	for i, m := range sess.Messages {
		// Build Content: plain text by default, or text + image_url parts
		// for the trailing message when images are supported. Older
		// messages (or unsupported LLMs) get a text placeholder per image.
		var content any = m.Content
		if len(m.Images) > 0 {
			if !a.imagesSupported || i != lastIdx {
				var buf strings.Builder
				buf.WriteString(m.Content)
				for _, img := range m.Images {
					if img.ID == "" {
						fmt.Fprintf(&buf, "\n\n[Image (%s, %dKB) — id missing, cannot re-fetch]", img.MimeType, len(img.Data)*3/4/1024)
						continue
					}
					fmt.Fprintf(&buf, "\n\n[Image %s (%s, %dKB) — call view_image id=%s to view]", img.ID, img.MimeType, len(img.Data)*3/4/1024, img.ID)
				}
				content = buf.String()
			} else {
				parts := []any{map[string]any{"type": "text", "text": m.Content}}
				for _, img := range m.Images {
					parts = append(parts, map[string]any{
						"type": "image_url",
						"image_url": map[string]string{
							"url": fmt.Sprintf("data:%s;base64,%s", img.MimeType, img.Data),
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
