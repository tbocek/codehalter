package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"regexp"
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
		a.say(ctx, sess.ID, "🗜 Context limit reached — compacting: summarising completed small turns, keeping the unfinished one…\n")
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

	// The base is the previous compaction's Summary, or the background rewrite
	// of it when one is ready. waitSummarise above has already joined that fold,
	// so "ready" here is a decided fact, not a race.
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
	// rotate() trimmed the message prefix: the kept window now begins at index
	// 0, so the in-flight turn does too. rotate runs with no concurrent writer
	// (see its doc), so this needs no lock.
	sess.turnStartIdx = 0
	// The fold rewrote the front of the context, so the next call legitimately
	// re-reads almost everything. Drop the comparison point rather than report
	// that as a cache fault.
	sess.resetCacheLineage()
	// Re-render the system prompt so skills + project context survive the fold.
	if sysPrompt, err := a.systemPrompt(sess.ID); err == nil {
		sess.SystemPrompt = sysPrompt
		sess.promptSkills = skillFiles(sess.Cwd)
	}
	if err := sess.Save(); err != nil {
		a.say(ctx, sess.ID, fmt.Sprintf("⚠ Compacted in-memory but persisting failed: %s. Archive %s is on disk; the live session file will diverge until the next Save.\n\n", err.Error(), archiveID))
	} else {
		note := ""
		if folded != "" {
			note = fmt.Sprintf(", prior summary folded %d→%d KB", len(prev)/1024, len(folded)/1024)
		}
		a.say(ctx, sess.ID, fmt.Sprintf("🗜 Compacted — archived as %s%s\n\n", archiveID, note))
	}
	// Queue the NEXT fold, last and deliberately after the Save: its worker
	// saves again when it lands, minutes from now.
	a.scheduleSummaryFold(sess, summary)
	return true
}

// maxSummaryBytes bounds the rolling Summary. Crossing it makes a compaction
// queue a background rewrite of the summary into a summary of itself, which the
// NEXT compaction uses as its base instead of the concatenation.
//
// A bound is needed because Summary is otherwise append-only: foldHistory
// concatenates the previous one verbatim with the new notes, so it only ever
// grows. Measured on one 11.6h session: 0 -> 3801 -> 6521 tokens over two
// compactions, with the very first note of the session still byte-identical at
// the front, next to three near-duplicate notes about the same task and two
// "[automatic summary unavailable]" raw excerpts that were 19% of the whole
// thing on their own.
//
// It is not just prefix size. Summary sits in front of every request for the
// rest of the session, so it is resident in the KV cache for every token the
// model generates, and decode slows as that cache grows (fitted on the same
// session: 59 tok/s at 7k context, 34 at 100k). 6521 resident tokens cost ~6.5
// minutes of pure decode across that session, and the cost compounds because
// nothing ever removes a note.
//
// 16 KiB is roughly 4k tokens: about the size of the system prompt, which is
// the other permanently-resident block and a reasonable ceiling for "everything
// that happened before what is still in context".
const maxSummaryBytes = 16 * 1024

// scheduleSummaryFold queues the summary-of-summaries rewrite to run AFTER the
// compaction that produced `summary` has returned. Nothing in that turn needs
// the shorter text: it is consumed as the BASE of the NEXT compaction, a whole
// context-fill away. Doing it inline would put a second LLM call, over 16 KB of
// prose, in front of a user already waiting out a compaction they did not ask
// for, to save bytes nothing reads until much later.
//
// The result lands in FoldedSummary and never in Summary. Summary is the front
// of every request, so swapping shorter text in mid-session re-renders the
// whole prompt behind it: it would buy a few thousand resident tokens and pay a
// full re-prefill for them. foldHistory has already thrown the prefix away by
// the time it reads FoldedSummary (resetCacheLineage, a few lines up), so
// consuming it exactly there is the one place it is free.
//
// On the summariser queue rather than a bare goroutine, for three things that
// queue already gives: foldHistory's waitSummarise joins it, so the next
// compaction can never read a half-written fold; the single worker keeps the
// fold from racing a per-turn note for the same background slot, where two
// concurrent calls on one KV slot make each other re-prefill; and it is the
// same kind of work, on the same connection, as the notes it queues behind.
func (a *agent) scheduleSummaryFold(sess *Session, summary string) {
	if len(summary) <= maxSummaryBytes {
		return
	}
	prompt := a.loadPromptFile(sess.ID, "RESUMMARISE.md")
	conn, _ := a.connForBackgroundLLM()
	if prompt == "" || conn == nil {
		return
	}
	// summariseTask carries a turn; this job has none. The queue only hands the
	// value back to this closure, which reads the captured summary instead.
	sess.enqueueSummarise(summariseTask{}, func(summariseTask) {
		// Longer than a per-turn note's two minutes: this is 16 KB in and several
		// thousand tokens out on whatever the background model is, and the only
		// thing that ever waits on it is a compaction that is nowhere near.
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		// Reasoning off, like the per-turn notes (see summariseCall).
		out, _, _, err := a.llmStream(ctx, sess.ID, conn.withThinkingDisabled(), []llmMessage{{
			Role: "user", Content: prompt + "\n\n" + clipBytes(summary, maxLLMInputBytes),
		}}, nil, nil, nil, nil)
		out = strings.TrimSpace(out)
		// Every failure leaves FoldedSummary empty, so the next compaction uses
		// the concatenation unchanged. Growing beats losing the record.
		if err != nil || out == "" || len(out) >= len(summary) {
			slog.Debug("summary fold: keeping the concatenated summary",
				"sid", sess.ID, "err", err, "in", len(summary), "out", len(out))
			return
		}
		sess.mu.Lock()
		// Only store it if it still describes the live Summary. waitSummarise
		// makes a compaction mid-fold impossible today; the check keeps it
		// impossible if that ordering ever changes, because the failure would be
		// silent and would drop a whole compaction's worth of notes.
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

// keepImageRefs re-attaches any image reference the fold dropped. The
// summariser is told to copy them through verbatim, but a reference the model
// paraphrases away is unrecoverable: the bytes stay on disk, and nothing else
// in the session ever names their id again, so view_image can't be asked for
// them. Restoring them deterministically means the prompt is allowed to be
// wrong about this without costing anything.
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
	// makes a single-slot (parallel = 1) server viable.
	if onMain {
		task.Msgs = a.appendSummariseMsgs(sess, prompt, turn)
	}
	sess.enqueueSummarise(task, func(t summariseTask) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		var note string
		if len(t.Msgs) > 0 {
			// Prefix-extension: same tools array as the foreground turn (see
			// summariseCall — required for the rendered prompt to share the
			// foreground's prefix), tool_choice=none so the answer is the note.
			note = a.summariseCall(ctx, sess, t.Conn.withToolChoiceNone(), t.Msgs, a.tools.defs(), t.Turn)
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
//
// tools MUST be the foreground's full tool array in prefix-extension mode and
// nil in paste mode. The chat template renders the tools array into the HEAD
// of the prompt, so a prefix-extension call without it produces a rendered
// prompt that diverges from the foreground's at the very first token — the
// call re-evaluates the whole context cold AND (on a single slot) evicts the
// foreground's KV cache, and the next user turn then re-evaluates cold too.
// The callers pair the tools with tool_choice="none" on the conn so the
// summariser can't answer with a tool call.
//
// Reasoning is off. A note restates what the turn already says, and a thinking
// model reasoned 5 to 20 KB before writing one: past the two-minute deadline on
// a 27B, which cost the note and struck the summariser out. The closed think
// block is a suffix, so the prefix-extension call still reuses the cache.
func (a *agent) summariseCall(ctx context.Context, sess *Session, conn *LLMConnection, msgs []llmMessage, tools []map[string]any, turn []Message) string {
	out, _, _, err := a.llmStream(ctx, sess.ID, conn.withThinkingDisabled(), msgs, tools, nil, nil, nil)
	// Whether this ran on a DEDICATED summariser, by endpoint rather than by
	// Slot: connForBackgroundLLM stamps its llm[0] fallback with Slot 1 for the
	// meter, so Slot alone would blame llm[0] for llm[0]'s own failures and take
	// a healthy summariser out of rotation.
	dedicated := false
	a.cfgMu.RLock()
	if len(a.settings.LLM) > 0 {
		dedicated = conn.Server != a.settings.LLM[0].Server || conn.Model != a.settings.LLM[0].Model
	}
	a.cfgMu.RUnlock()
	if err != nil || strings.TrimSpace(out) == "" {
		// Not Debug: every failure here silently swaps a structured note for a
		// clipped transcript, which degrades every later compaction. One measured
		// session lost 23 notes this way to an endpoint that was unreachable from
		// inside the container, and the only trace was a Debug line nobody reads.
		slog.Warn("summarise: llm call failed — using raw fallback note", "sid", sess.ID, "server", conn.Server, "err", err)
		a.logSession(sess.ID, "SUMMARISE", "failed on %s (%s) — turn note fell back to a raw transcript: %v", conn.Server, conn.Model, err)
		if dedicated {
			// A dedicated summariser that keeps failing gets taken out of rotation
			// by connForBackgroundLLM, which then generates notes on llm[0].
			a.summaryStrikes.Add(1)
			a.summaryStruckAt.Store(time.Now().UnixNano())
		}
		out = fallbackTurnNote(turn)
	} else if dedicated {
		a.summaryStrikes.Store(0)
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
					fmt.Fprintf(&turnBuf, "- %s(%s) → %s\n", tu.Name, tu.Input, truncateForLLM(tu.Name, tu.Input, tu.Output))
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
	// Hand the paste the rolling Summary as background. Prefix-extension mode
	// gets this for free (appendSummariseMsgs builds the real wire context, and
	// buildLLMContext renders Summary in front of the messages), but a paste sees
	// nothing except the turn slice — so the first note written after a
	// compaction comes from something with no idea what the session had already
	// established. It restates the goal and the constraints that are sitting
	// three inches above it in the very Summary it is about to be appended to.
	// The two paste callers are exactly the two where that bites: the synchronous
	// in-flight fold inside foldHistory (which runs AT a compaction) and a
	// per-turn note routed to a separate purpose = "summary" connection.
	//
	// Framed read-only on purpose. foldHistory concatenates the resulting note
	// AFTER this same Summary, so anything copied out of it is stored twice and
	// then twice as likely to survive the next boundSummary fold.
	//
	// Clipped to maxSummaryBytes: the fold caller gets here BECAUSE the
	// foreground context overflowed, and the background slot must not follow it.
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

// replayToolOutput re-renders one stored tool result exactly as the live call
// put it on the wire, which is what keeps a rebuilt prompt byte-identical to
// the one the server already has cached.
//
// For every tool but view_image that is liveToolOutput's text (NOT a tighter
// history clip: a cached re-send is free, whereas clipping changes the bytes
// and forces a reprocess; n_ctx is bounded by compaction, not by clipping
// here). view_image is the exception. runToolCall puts []any{text, image_url}
// on the wire and stores only the text half in tu.Output, so replaying
// tu.Output alone silently DROPS an image out of the MIDDLE of the prompt and
// every message behind it shifts. Measured once on a real session: 431 prompt
// tokens saved against 30035 re-evaluated, a 70:1 losing trade.
//
// The parts are reproducible because dispatchViewImage is pure given Cwd and
// the stored arguments and the image store is content-addressed, so re-running
// it yields the same bytes. Same rule the m.Images branch follows.
//
// A tool that PRODUCED an image (screenshot) is NOT pure that way: re-rendering
// the page a turn later can yield different pixels, and the id is the only
// thing that pins the bytes the model actually saw. So that case replays from
// ImageID and never re-runs the tool.
func (a *agent) replayToolOutput(sess *Session, tu ToolUse) any {
	text := liveToolOutput(tu.Name, tu.Input, tu.Output)
	if tu.Failed || !a.imagesSupported {
		return text
	}
	// Bytes gone (file deleted since the live call)? Nothing can make either
	// replay identical, so fall back to the stored text rather than fail the
	// turn: the same trade the file-missing branch of m.Images makes.
	if tu.ImageID != "" {
		data, mime, err := readImageFile(sess.Cwd, tu.ImageID)
		if err != nil {
			return text
		}
		// tu.Output, not `text`: the live call put the untruncated text in
		// parts[0], so replaying the truncation hint here would change the
		// wire bytes of a message the model already saw.
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
	// reassigns s.Messages) can mutate the session while this ranges it. Copy the slice header + the prompt
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
				Role:       "tool",
				Content:    a.replayToolOutput(sess, tu),
				ToolCallID: wireCallID(tu),
			})
		}
	}

	return messages
}
