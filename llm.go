package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/tbocek/codehalter/llm"
)

// LLM message types for the OpenAI API.

// warnChatTemplateKwargsIgnored says so, once per model, when the server sent
// back reasoning after being asked not to reason. An OpenAI-compatible server
// is free to accept chat_template_kwargs and drop it on the floor, and several
// do: Ollama substitutes its own template, llama.cpp without --jinja runs a
// fallback template that has no enable_thinking to read. The symptom is
// invisible — correct answers, arriving at half speed, with the whole reasoning
// budget silently spent. It took a log analysis across an 11.6h session to
// notice it the first time, which is the argument for saying it out loud at the
// moment it happens. Setting enable_thinking=false costs a separate rendering
// (see llm.Conn.ParamsFor), so paying that and getting nothing back is the worst case of
// the trade.
//
// Once per Server+Model, not per session or per call: it is a property of the
// deployment, and a per-call warning on a 400-call session is worse than
// silence.
func (a *agent) warnChatTemplateKwargsIgnored(ctx context.Context, sid string, conn *llm.Conn, reqBody map[string]any, reasoningBytes int) {
	if reasoningBytes == 0 || sid == "" || llm.ThinkingOn(reqBody) {
		return
	}
	if _, seen := a.ctkIgnored.LoadOrStore(conn.Server+"\x00"+conn.Model, true); seen {
		return
	}
	// Two different mechanisms land here, so the advice has to split. The
	// execute-role phases and the stall retry continue a prefilled
	// <think></think>; everything else got here because the user's own params
	// carry enable_thinking=false.
	if cont, _ := reqBody["continue_final_message"].(bool); cont {
		slog.Warn("server ignored the thinking-off prefill",
			"server", conn.Server, "model", conn.Model, "reasoning_bytes", reasoningBytes)
		a.say(ctx, sid, "⚠ "+conn.Model+" kept reasoning after being handed a closed <think></think> to continue.\n"+
			"  The server started a fresh assistant turn instead of continuing the prefilled one, so it does not honour\n"+
			"  continue_final_message / add_generation_prompt=false. Execute-role calls will keep reasoning here, which\n"+
			"  costs decode time but nothing else; the damage stays capped, since reasoning that follows a closed block is short.\n"+
			"  If this model does not delimit reasoning with <think>/</think>, that is the likelier cause.\n"+
			"  The fallback is params_execute chat_template_kwargs = { enable_thinking = false }, which costs a second\n"+
			"  prompt rendering — worth it only on a server with 2+ slots (see settings.toml).\n\n")
		return
	}
	slog.Warn("server ignored chat_template_kwargs.enable_thinking=false",
		"server", conn.Server, "model", conn.Model, "reasoning_bytes", reasoningBytes)
	a.say(ctx, sid, "⚠ "+conn.Model+" kept reasoning after being asked not to: this server accepts chat_template_kwargs and ignores it.\n"+
		"  llama.cpp: start llama-server with --jinja, or the built-in fallback template runs and has no enable_thinking to read.\n"+
		"  vLLM: serve with --reasoning-parser qwen3, or set the default with --default-chat-template-kwargs '{\"enable_thinking\": false}'.\n"+
		"  Ollama: it substitutes its own template, so per-request kwargs cannot work; use llama.cpp or vLLM.\n"+
		"  Otherwise drop enable_thinking from params_execute: it is buying a second prompt rendering and nothing else.\n\n")
}

// prewarm pays the prompt-processing cost of the session's prefix (system
// prompt + summary + history + tool schemas) before the user's first message,
// so turn one only pays for its own delta. One synchronous 1-token call built
// through the exact renderers a real turn uses (buildLLMContext +
// toolRegistry.defs), which makes the rendered prompt a byte-prefix of the
// next real request; llama.cpp's longest-prefix slot routing then reuses the
// KV cache. Callers run it in a goroutine after the first prepare (probe done,
// skills seeded, SystemPrompt final). sid is passed to llmStream as "" so the
// call skips session logging and turn stats. Errors are swallowed by design:
// an unreachable server or a cache-less backend just makes this a no-op, and
// the real turn will surface any genuine problem.
func (a *agent) prewarm(sess *Session) {
	if sess == nil || !a.prewarmEnabled() {
		return
	}
	conn := a.connForSession(context.Background(), sess.ID, "thinking")
	if conn == nil {
		return
	}
	messages := a.buildLLMContext(sess)
	if len(messages) == 0 {
		return
	}
	// Generous ceiling: a 10k-token prefix on a slow local model is ~30s of
	// prompt processing; 122B-class models take a few times that.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	start := time.Now()
	// Log under the real sid so the session log records the prewarm's exact
	// request bytes and its cached/evaluated split — the only way to diagnose
	// a turn-one cache miss (diff this request against turn one's).
	// llm.Conn.NoTurnStats: a turn that starts while the warm is still streaming resets
	// the counters BEFORE the warm's usage lands, which would inflate that
	// turn's "uncached" stat by the whole prefill.
	warmConn := conn.WithMaxTokens(1)
	warmConn.NoTurnStats = true
	_, _, _, err := a.llmStream(ctx, sess.ID, warmConn, messages, a.tools.defs(), nil, nil, nil)
	elapsed := time.Since(start).Round(time.Millisecond)
	a.logSession(sess.ID, "PREWARM", "done in %s err=%v", elapsed, err)
	slog.Debug("prewarm: done", "sid", sess.ID, "elapsed", elapsed, "err", err)
}

// recordStreamStats folds one call's server-reported usage and timings into the
// turn's running totals for the "✅ Done" line, and runs the prefix-cache rewind
// check on tool-loop calls. sid="" (probes/tests) has no session, so it is a
// no-op there; so is a stream that broke before the usage chunk, whose counts
// are all 0.
func (a *agent) recordStreamStats(sid, connLabel string, conn *llm.Conn, r *llm.StreamResult) {
	sess := a.getSession(sid)
	if sess == nil || conn.NoTurnStats {
		return
	}
	// Derive evaluated (sent-but-not-cached) from cached_tokens when timings are
	// absent; -1 means no cache info reported. This is the only number we keep —
	// the gross prompt_tokens (cached prefix re-counted each call) is not summed.
	if r.EvaluatedTokens < 0 && r.CachedTokens >= 0 && r.PromptTokens > 0 {
		r.EvaluatedTokens = r.PromptTokens - r.CachedTokens
	}
	sess.addTurnTokens(r.PromptTokens, r.CompletionTokens, r.EvaluatedTokens)
	// Prefix-cache rewind check, tool-loop calls only. Each is the previous
	// call's messages plus an append, so the server should hand back
	// everything the previous call sent (cached ≈ its prompt) and evaluate
	// only the tail. A big shortfall means the prompt was re-rendered behind
	// our backs; logged per call, and reported once on the Done line.
	if conn.CacheLineage && r.PromptTokens > 0 {
		render := llm.RenderKey(conn.ExtraBody)
		rw := sess.noteCacheLineage(r.PromptTokens, r.CachedTokens, render, time.Now())
		if rw.tokens > 0 {
			// "(none)" rather than an empty string: a params table with no
			// template fields at all is the good configuration, and it should
			// not read like missing data.
			orNone := func(s string) string {
				if s == "" {
					return "(none)"
				}
				return s
			}
			// One line, three causes. The header is the same measurement every
			// time (how much was re-read, and how long the slot sat idle first,
			// because token counts alone cannot separate "something re-rendered
			// the prompt" from "the server reclaimed a slot we left sitting").
			// Only the diagnosis differs, so only the diagnosis branches.
			var cause string
			switch {
			case rw.renderChanged:
				cause = fmt.Sprintf("We asked for a different rendering than last call: template params went %s -> %s. "+
					"The server keeps a prompt state per rendering, so this call could only reuse what THIS rendering held "+
					"last time and had to re-evaluate everything the other role appended since. Make params_thinking and "+
					"params_execute agree on everything that is not a sampler.", orNone(rw.prevRender), orNone(render))
			case rw.idle >= idleEvictionSuspect:
				cause = fmt.Sprintf("Both calls asked for the same rendering (%s) and the slot sat idle that whole time, "+
					"which is the likeliest cause: servers reclaim idle slots and no setting prevents it. Nothing to fix "+
					"unless the gap surprises you.", orNone(render))
			default:
				cause = fmt.Sprintf("Both calls asked for the same rendering (%s) and came back to back, so an idle "+
					"eviction is unlikely: something rewrote the middle of the prompt. A tool result that replayed "+
					"differently than it was sent, or a chat template that repositions earlier messages as the "+
					"conversation grows.", orNone(render))
			}
			a.logSession(sid, connLabel+" CACHE",
				"prefix cache rewound: %d tokens the previous call had already sent were re-read "+
					"(prompt=%d cached=%d, %s since that call). %s",
				rw.tokens, r.PromptTokens, r.CachedTokens, humanDuration(rw.idle.Milliseconds()), cause)
		}
	}
	// Prefer the server's measured times over our TTFT proxy (which includes
	// queue + cache-load overhead → understates pp/s).
	pMs, gMs := r.ServerPromptMs, r.ServerGenMs
	if pMs == 0 && gMs == 0 && !r.FirstTokenAt.IsZero() {
		pMs = float64(r.FirstTokenAt.Sub(r.ReadStart).Milliseconds())
		gMs = float64(time.Since(r.FirstTokenAt).Milliseconds())
	}
	if pMs > 0 || gMs > 0 {
		sess.addTurnTiming(int64(pMs), int64(gMs))
	}
}

// streamOutcomeError turns how the stream ended into the one error llmStream
// returns and the RESPONSE log records. Order matters: an explicit in-band
// server error is checked FIRST, because gateways commonly emit an {"error":…}
// chunk and THEN drop the socket. Checking the transport error first would
// shadow the server's verbatim cause (e.g. "prompt exceeds n_ctx") behind a
// generic "unexpected EOF", and send a fatal prompt down the useless
// transient-retry path instead of surfacing the real reason.
//   - fired: a stream rule matched and we abandoned the generation ourselves.
//   - streamErrMsg: server sent an {"error":…} chunk under HTTP 200; surface
//     its message verbatim (it names the real cause, e.g. prompt > n_ctx).
//   - scanErr: stream broke mid-flight (e.g. a router model swap force-kills
//     the connection) with no in-band error to explain it.
//   - finish_reason="length": truncated at a length limit. If completion hit
//     the requested max_tokens cap the model is genuinely verbose/looping and we
//     bail (the message guides tuning); if it stopped BELOW the cap it hit the
//     n_ctx ceiling (prompt fit but left no room), recoverable, signalled via
//     llm.ErrContextCeiling so the tool loop folds history and retries.
func (a *agent) streamOutcomeError(conn *llm.Conn, reqBody map[string]any, r *llm.StreamResult, fired *streamRule) error {
	switch {
	case fired != nil:
		// Checked first: we aborted this stream on purpose, so scanErr (the reader
		// stopping mid-body) and a missing finish_reason are consequences of that
		// decision, not independent failures, and either would otherwise shadow the
		// real cause. The token counts stay 0 — we never reached the usage chunk —
		// so an aborted generation is simply not attributed in the turn stats.
		return &streamRuleError{Rule: fired.Name, Reminder: fired.Reminder, Matched: r.Text.String()}
	case r.StreamErrMsg != "":
		return fmt.Errorf("LLM returned an error mid-stream (role=%s, model=%s): %s", conn.Tag, conn.Model, r.StreamErrMsg)
	case r.ScanErr != nil:
		return fmt.Errorf("reading SSE stream: %w", r.ScanErr)
	case r.FinishReason == "length":
		// Two very different causes. (1) completion reached the requested max_tokens
		// cap → the model is genuinely verbose/looping; bail (the message guides
		// tuning). (2) completion is BELOW the cap → it hit the n_ctx ceiling: the
		// prompt fit but left no room to generate. (2) is recoverable — the context
		// is too full, same as a 400 — so signal llm.IsContextFull and let the tool loop
		// fold history and retry.
		reqMax := 0
		switch v := reqBody["max_tokens"].(type) {
		case int:
			reqMax = v
		case int64:
			reqMax = int(v)
		case float64:
			reqMax = int(v)
		}
		// The length limit was the n_ctx ceiling, not the cap, when EITHER the
		// generation stopped below the cap (completion < reqMax), OR the prompt
		// left less than a full generation of room (prompt + reqMax > n_ctx). The
		// second form still catches the ceiling when the server omits
		// completion_tokens (completion == 0), so it can't be compared to the cap.
		belowCap := r.CompletionTokens > 0 && reqMax > 0 && r.CompletionTokens < reqMax
		mst := a.getMainSlotTokens()
		noRoom := r.PromptTokens > 0 && reqMax > 0 && mst > 0 && r.PromptTokens+reqMax > mst
		switch {
		case belowCap || noRoom:
			return fmt.Errorf("generation hit the context ceiling (prompt=%d gen=%d, n_ctx=%d, role=%s): %w",
				r.PromptTokens, r.CompletionTokens, mst, conn.Tag, llm.ErrContextCeiling)
		case r.Text.Len() == 0 && len(r.Calls) == 0 && r.Reasoning.Len() > 0 && llm.ThinkingOn(reqBody):
			// All budget went to reasoning with nothing to show — the model looped
			// in <think>. Recoverable: the tool loop retries once with thinking off
			// so it answers directly (see llm.ErrStuckThinking).
			return fmt.Errorf("model stuck in <think> (%d B reasoning, 0 content/calls, role=%s): %w",
				r.Reasoning.Len(), conn.Tag, llm.ErrStuckThinking)
		default:
			return &llm.CapHitError{Cap: reqMax, Msg: fmt.Sprintf("LLM hit max_tokens cap (role=%s, model=%s) — response truncated (%d B content, %d B reasoning, %d tool calls). Likely the model is looping or stuck in <think>; raise max_tokens in params_%s, or set chat_template_kwargs.enable_thinking=false for this role if reasoning is dominating the budget",
				conn.Tag, conn.Model, r.Text.Len(), r.Reasoning.Len(), len(r.Calls), conn.Tag)}
		}
	default:
		return nil
	}
}

// logStreamResponse writes the one RESPONSE block per call, on every exit path.
// The raw SSE wire is one event per token (~100 lines for a short reply,
// thousands for a tool-call argument blob) — useless for skimming; this
// collapses the deltas into the reconstructed transcript. The token counts are
// the server's exact ones from the trailing usage chunk (0 when the backend
// didn't report them).
func (a *agent) logStreamResponse(sid, connLabel string, r *llm.StreamResult, err error) {
	text, reasoning := r.Text.String(), r.Reasoning.String()
	var rb strings.Builder
	if r.PromptTokens > 0 || r.CompletionTokens > 0 || r.FinishReason != "" {
		// finish: "stop" = model ended; "length" = hit max_tokens (truncated);
		// "tool_calls" = ended on a tool call; "(none)" = stream broke with no
		// finish_reason (interrupted). Distinguishes a cap/interrupt from a clean end.
		fr := r.FinishReason
		if fr == "" {
			fr = "(none)"
		}
		fmt.Fprintf(&rb, "tokens: prompt=%d completion=%d finish=%s", r.PromptTokens, r.CompletionTokens, fr)
		// Cache split when the server reported it: cached = prompt tokens served
		// from the KV cache, evaluated = prompt tokens actually processed. This
		// is THE line for diagnosing prefix-cache misses ("N k uncached" in the
		// Done stats without this split is unattributable).
		if r.CachedTokens >= 0 || r.EvaluatedTokens >= 0 {
			fmt.Fprintf(&rb, " cached=%d evaluated=%d", r.CachedTokens, r.EvaluatedTokens)
		}
		rb.WriteString("\n")
	}
	if reasoning != "" {
		fmt.Fprintf(&rb, "reasoning_content (%d B):\n%s\n", len(reasoning), reasoning)
	}
	if text != "" {
		fmt.Fprintf(&rb, "content:\n%s\n", text)
	}
	for i, c := range r.Calls {
		fmt.Fprintf(&rb, "tool_call[%d] %s id=%s args=%s\n", i, c.Function.Name, c.ID, c.Function.Arguments)
	}
	if text == "" && reasoning == "" && len(r.Calls) == 0 {
		rb.WriteString("(empty response)\n")
	}
	if err != nil {
		fmt.Fprintf(&rb, "[stream error] %v\n", err)
	}
	a.logSession(sid, connLabel+" RESPONSE", "%s", rb.String())
}

// llmStream is the core LLM call. Streams SSE, collects text and tool calls.
// sid scopes the debug log: req body and raw SSE response are appended to
// .codehalter/session_<sid>.log so a single file captures everything that
// went over the wire for a session. Pass "" to disable logging (used by tests
// and pre-session probes). think (nil to discard) receives reasoning_content
// tokens — kept separate from `on` so callers can surface chain-of-thought to
// the UI as agent_thought_chunk without polluting agent_message_chunk. onArgs
// (nil to discard) receives each tool-call argument delta tagged with its call
// index and tool name, which is the only way to see a structured terminal tool
// being written: its payload never reaches `on`.
//
// The body it sends, the stream it reads, and the three passes over the result
// (turn stats, outcome classification, the RESPONSE log) each live in their own
// function above; what is left here is the round trip itself — the concurrency
// gate, the status meter, the HTTP call and its non-200 handling.
func (a *agent) llmStream(ctx context.Context, sid string, conn *llm.Conn, messages []llm.Message, tools []map[string]any, on, think func(string), onArgs func(idx int, name, delta string)) (string, []llm.ToolCall, string, error) {
	reqBody := llm.BuildChatRequest(conn, messages, tools)
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", nil, "", fmt.Errorf("marshalling LLM request body: %w", err)
	}

	// Per-conn concurrency gate: cap in-flight calls to this conn at its
	// configured `parallel`. The token is held only for this call (released on
	// return), so between calls the conn frees up and a background call (the
	// summariser) can take its turn on a pool of size 1; the wait
	// shows as "(queued…)". Find the conn's semaphore index by matching
	// server+model; -1 (test mocks / probes not in settings.LLM) means "no gate,
	// dispatch directly".
	// Find the conn's semaphore index and bind its channel under cfgMu (RLock): a
	// foreground prepare phase can reassign a.settings.LLM / a.connSems while a
	// background LLM call sits in this gate. Read the pair, capture the channel into
	// a local, release the lock, THEN do the blocking acquire on that local. Binding
	// once also survives a rebuild: probeAllLLMs swaps in a fresh a.connSems per
	// prompt, so re-reading a.connSems[slot] at release time could hit a NEW empty
	// channel and block forever (the permit lives in the OLD one).
	a.cfgMu.RLock()
	slot := -1
	for i := range a.settings.LLM {
		if a.settings.LLM[i].Server == conn.Server && a.settings.LLM[i].Model == conn.Model {
			slot = i
			break
		}
	}
	var sem chan struct{}
	if slot >= 0 && slot < len(a.connSems) {
		sem = a.connSems[slot]
	}
	// Stream rules ride the same lock (they're reassigned wholesale with the rest
	// of the config). Armed only where the caller asked for it (llm.Conn.ForToolLoop).
	// See llm.Conn.StreamRulesArmed for why this is opt-in and not simply
	// "any call that passes tools".
	var matcher *ruleMatcher
	if conn.StreamRulesArmed && len(a.streamRules) > 0 {
		matcher = &ruleMatcher{rules: a.streamRules}
	}
	a.cfgMu.RUnlock()
	if sem != nil {
		// Try non-blocking first; only emit the queued suffix when we're
		// actually about to wait. Avoids flashing the wrong status on the
		// common hot path where the slot is free.
		select {
		case sem <- struct{}{}:
		default:
			a.setStatus(ctx, sid, " (queued…)")
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				a.setStatus(ctx, sid, "")
				return "", nil, "", ctx.Err()
			}
		}
		defer func() { <-sem }()
	}

	// slotLabel is the *display* index the user reads in the meter (llm[0]
	// foreground, llm[1] background) — distinct from `slot` above, the semaphore
	// index into settings.LLM. slot=-1 (test mocks, probes) → "?".
	slotLabel := "?"
	if slot >= 0 {
		slotLabel = fmt.Sprintf("%d", conn.Slot)
	}

	// Drive the phase-row suffix across the round-trip: "(sent…)" until the first
	// token, then the live ↑/↓ meter (setStatus is a no-op when no phase active).
	// ↑ is THIS call's request-body size on the wire, ↓ (below) is
	// generated tokens. Two different quantities in one row, so each carries its
	// own unit: a bare "↑2451.39kb ↓8.5k" reads as if ↓ were kb too. Display-only.
	upLabel := humanBytes(len(body))
	a.setStatus(ctx, sid, fmt.Sprintf(" (llm[%s] ↑%s sent…)", slotLabel, upLabel))
	defer a.setStatus(ctx, sid, "")

	// Until the first generated byte the row shows "(sent… Ns)", so a busy or
	// queuing server reads as "(sent… 25s)" rather than a frozen "(sent…)"; once
	// bytes arrive it switches to the live "↑<body bytes> ↓<gen tokens>" estimate.
	// The two halves are different quantities, hence the explicit "tok" suffix.
	// genChars is written by llm.ReadSSEStream while this reads it, hence atomic. No
	// warning is ever emitted from here: while the call is alive the climbing
	// counter is the signal, and if it dies llmStream surfaces the transport
	// error directly.
	//
	// Registered after the status-clear defer above so LIFO joins the meter first.
	var genChars int64
	meterStart := time.Now()
	defer a.startStatusMeter(ctx, sid, func() string {
		if g := atomic.LoadInt64(&genChars); g > 0 {
			// approx tokens (chars/4), compact via humanCount. Display-only.
			return fmt.Sprintf(" (llm[%s] ↑%s ↓%s tok…)", slotLabel, upLabel, humanCount(int(g)/4))
		}
		return fmt.Sprintf(" (llm[%s] ↑%s sent… %ds)", slotLabel, upLabel, int(time.Since(meterStart).Seconds()))
	})()

	// Per-session log: a REQUEST block now, one aggregated RESPONSE block at the
	// end (the per-token SSE wire is too noisy to skim). connLabel carries
	// llm[<slot>] + role + model so grepping "llm[1]" finds every request routed
	// to a given entry.
	connLabel := fmt.Sprintf("llm[%s] %s model=%s", slotLabel, conn.Tag, conn.Model)
	a.logSession(sid, connLabel+" REQUEST", "%s", string(body))

	httpReq, err := http.NewRequestWithContext(ctx, "POST", conn.Endpoint("/v1/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return "", nil, "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if conn.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
	}

	// llm.HTTPClient caps the wait for the server's first response byte at 90s
	// (ResponseHeaderTimeout). This bounds the hang when a network switch breaks
	// an in-flight TCP connection: without it, TCP retransmission keeps the
	// request alive for up to ~15 minutes before the OS gives up. 90s is enough
	// for a busy or queued LLM server to start streaming; cancellation via the
	// request ctx still applies for the rest of the stream.
	resp, err := llm.HTTPClient.Do(httpReq)
	if err != nil {
		a.logSession(sid, connLabel, "[transport error] %v", err)
		return "", nil, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", nil, "", fmt.Errorf("HTTP %d, failed to read body: %w", resp.StatusCode, err)
		}
		a.logSession(sid, connLabel, "[HTTP %d] %s", resp.StatusCode, string(bodyBytes))
		// Prefer the OpenAI-style {"error":{"message":…}} text; fall back to raw body.
		msg := string(bodyBytes)
		var apiErr struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(bodyBytes, &apiErr) == nil && apiErr.Error.Message != "" {
			msg = apiErr.Error.Message
		}
		// chat_template_kwargs is a llama.cpp / vLLM extension, not part of the
		// OpenAI API, and it only ever reaches the wire from a params_* table. Say
		// where it came from: the OpenAI API's "unrecognized request argument" is
		// otherwise a puzzle about a field the user set months ago.
		if strings.Contains(msg, "chat_template_kwargs") {
			msg += "\n\nThis backend does not accept chat_template_kwargs (it is a llama.cpp / vLLM extension). Remove it from params_thinking / params_execute for this [[llm]] entry."
		}
		return "", nil, "", &llm.HTTPError{Status: resp.StatusCode, Body: msg, URL: resp.Request.URL.String()}
	}

	var fired *streamRule
	stop := func(delta string) bool { fired = matcher.feed(delta); return fired != nil }
	res := llm.ReadSSEStream(resp.Body, conn, stop, on, think, onArgs, &genChars)
	a.recordStreamStats(sid, connLabel, conn, res)

	// A server that quietly ignores chat_template_kwargs looks exactly like a
	// server that honours it, right up until the reasoning tokens arrive.
	a.warnChatTemplateKwargsIgnored(ctx, sid, conn, reqBody, res.Reasoning.Len())

	err = a.streamOutcomeError(conn, reqBody, res, fired)
	// A generation the caller cannot use is decode time spent for nothing, and
	// it is already inside the turn's completion total. Name it, or the Done
	// line reports the worst turns as the most productive ones.
	if err != nil && res.CompletionTokens > 0 {
		if sess := a.getSession(sid); sess != nil && !conn.NoTurnStats {
			sess.addWastedCompletion(res.CompletionTokens)
		}
	}
	a.logStreamResponse(sid, connLabel, res, err)
	return res.Text.String(), res.Calls, res.Reasoning.String(), err
}

// connForSession resolves the connection for the role. Every session runs on
// LLM[0], whose KV cache owns the conversation prefix. Concurrency is enforced
// by per-conn semaphores in llmStream; MainLLM returns a value copy, safe to use
// after the lock is released.
func (a *agent) connForSession(_ context.Context, _ string, role string) *llm.Conn {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	return a.settings.MainLLM(role)
}

// summaryMaxStrikes is how many consecutive failures the dedicated summariser
// connection gets before background notes move to llm[0] for summaryCooldown.
// Each failure costs that turn's note (summariseCall falls back to a clipped
// raw transcript), so the threshold is low: two, enough to ride out a single
// timeout, not enough to spend a session degrading every note.
const summaryMaxStrikes = 2

// summaryCooldown is how long a struck-out summariser stays out of rotation
// before it gets another call. It used to be the rest of the run, which retired
// a summariser for 16 hours after two timeouts while the server was up the
// whole time. A call that fails again renews the cooldown, so a server that
// really is down costs one note per cooldown.
const summaryCooldown = 10 * time.Minute

// connForBackgroundLLM returns the connection to host background work (the
// per-turn summariser). It walks the entries marked `purpose = "summary"` and
// returns the first with free semaphore capacity, so marking several spreads
// load across them instead of stacking on one. If all are busy (or none are
// marked) it falls back to LLM[0], labelled llm[1] when that conn has >=2 slots
// so the meter shows the work routed off the foreground turn. The capacity peek
// is racy by design — llmStream's semaphore just queues if the slot was taken
// meanwhile; falling back rather than queueing keeps a busy summariser conn from
// stalling the turn's note behind somebody else's call.
//
// Index 0 is skipped in the walk because the fallback below already lands there
// with the right display slot, so `purpose = "summary"` on LLM[0] means the same
// thing as not marking anything.
//
// The second return reports that fallback: true means the call will land on
// the server whose KV cache holds the foreground conversation. Callers use it
// to switch to prefix-extension prompts (conversation context + instruction
// tail) so the call reuses that cache instead of evicting it — what makes a
// single-slot (parallel = 1) server viable.
func (a *agent) connForBackgroundLLM() (*llm.Conn, bool) {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	for i := 1; i < len(a.settings.LLM); i++ {
		if !strings.EqualFold(a.settings.LLM[i].Purpose, purposeSummary) {
			continue
		}
		// A summariser the probe couldn't reach, or one that has failed its way
		// through summaryMaxStrikes, is not a summariser. Using it anyway costs
		// the note outright — summariseCall's only fallback is a clipped raw
		// transcript — and llm[0] is right there, one line down, where the note
		// generates as a cheap prefix extension. Read of connProbe is unlocked,
		// same as hasReachableLLM: it is written by the probe in the pre-turn
		// prepare, never concurrently with a turn.
		if p, ok := a.connProbe[a.settings.LLM[i].Server+"\x00"+a.settings.LLM[i].Model]; ok && !p.Reachable {
			slog.Debug("background llm: summariser unreachable, using llm[0]", "server", a.settings.LLM[i].Server)
			break
		}
		if a.summaryStrikes.Load() >= summaryMaxStrikes &&
			time.Since(time.Unix(0, a.summaryStruckAt.Load())) < summaryCooldown {
			break
		}
		if i < len(a.connSems) && a.connSems[i] != nil &&
			len(a.connSems[i]) < cap(a.connSems[i]) {
			return a.settings.ConnAt(i, "execute"), false
		}
	}
	// No entry designated for the summariser (or all busy): fall back to llm[0].
	// When it has >= 2 parallel slots, label this as llm[1] — same connection and
	// semaphore, but a distinct display slot so the meter shows background
	// routed off the foreground turn (the server picks the real KV slot).
	c := a.settings.ConnAt(0, "execute")
	if c != nil && a.settings.LLM[0].ParallelCap() >= 2 {
		c.Slot = 1
	}
	return c, true
}

// buildConnSems sizes one buffered channel per LLM entry to its llm.Conn.ParallelCap.
// Called on startup AND on every settings reload (per prompt). It's a no-op when
// the shape is unchanged so we don't needlessly swap channels out from under
// in-flight llmStream calls (each binds its slot's channel at acquire and
// releases on it — see llm.go's capture). Only an actual cap change rebuilds.
// Caller MUST hold a.cfgMu (write lock): it reads a.settings.LLM and reassigns
// a.connSems, which the background-reachable readers touch under cfgMu.RLock.
func (a *agent) buildConnSems() {
	if len(a.connSems) == len(a.settings.LLM) {
		unchanged := true
		for i := range a.settings.LLM {
			if cap(a.connSems[i]) != a.settings.LLM[i].ParallelCap() {
				unchanged = false
				break
			}
		}
		if unchanged {
			return
		}
	}
	sems := make([]chan struct{}, len(a.settings.LLM))
	for i := range a.settings.LLM {
		sems[i] = make(chan struct{}, a.settings.LLM[i].ParallelCap())
	}
	a.connSems = sems
}

// isTransientStreamError reports whether err is a mid-flight connection drop:
// the server or router closed the stream (EOF, reset, broken pipe) or a network
// error hit the request — as opposed to a clean LLM error or a deliberate
// cancel. These are usually momentary (a router model swap, a brief blip), so
// the tool loop retries a couple of times before surfacing a clear message. A
// cancel is excluded: it's intentional and already has its own user message.
func isTransientStreamError(err error) bool {
	if err == nil || isCancelled(err) {
		return false
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	var ne net.Error
	if errors.As(err, &ne) {
		return true
	}
	s := err.Error()
	return strings.Contains(s, "connection reset") ||
		strings.Contains(s, "broken pipe") ||
		strings.Contains(s, "unexpected EOF") ||
		strings.Contains(s, "EOF")
}
