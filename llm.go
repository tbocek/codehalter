package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// llmHTTPClient is used for all LLM API calls. ResponseHeaderTimeout caps the
// wait for the first response byte so a broken TCP connection (e.g. after a
// network switch) doesn't hang for the full OS retransmission window (~15 min).
// The dialer's KeepAlive matches http.DefaultTransport so idle connections are
// probed every 30s. No Client.Timeout: streaming generations run unbounded and
// are cancelled only via the request context.
var llmHTTPClient = &http.Client{
	Transport: &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ResponseHeaderTimeout: 90 * time.Second,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	},
}

// LLM message types for the OpenAI API.

type llmMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type toolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type sseChunk struct {
	Choices []struct {
		Delta struct {
			Content   string     `json:"content"`
			ToolCalls []toolCall `json:"tool_calls"`
			// ReasoningContent is the chain-of-thought channel thinking models
			// emit when the server splits <think>…</think> off the content
			// stream. Kept OUT of Content (merging would break prefix-cache
			// keys) but accumulated + logged, so an all-thinking turn reports
			// "N B reasoning, 0 visible" instead of "(empty response)".
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	// Usage is the OpenAI-compatible token count block. With
	// stream_options.include_usage=true the server emits one final chunk
	// (choices empty) carrying this — the prompt_tokens count is ground
	// truth for what the server actually packed into n_ctx, and feeds the
	// per-turn "✅ Done" usage stats (see addTurnTokens). Not all backends
	// send it (a proxy may strip include_usage); when absent the stats line
	// just shows less detail.
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		// CachedTokens (OpenAI-standard) = prompt tokens served from cache;
		// evaluated = prompt_tokens - cached_tokens.
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	// Timings is llama.cpp's per-request block (timings_per_token=true): prompt_n =
	// evaluated, cache_n = reused, _ms = measured times. A proxy that strips
	// non-standard fields won't send it (we then fall back to usage cached_tokens).
	Timings *struct {
		PromptN     int     `json:"prompt_n"`
		CacheN      int     `json:"cache_n"`
		PromptMs    float64 `json:"prompt_ms"`    // server-measured prompt-eval time
		PredictedMs float64 `json:"predicted_ms"` // server-measured generation time
	} `json:"timings"`
	// Error is an error delivered INSIDE the SSE stream under an HTTP 200 — how
	// llama.cpp / llama-swap and some gateways signal a mid-stream failure
	// (e.g. a prompt that exceeds the model's real context length) rather than a
	// non-200 status. Such a chunk has empty Choices, so without this field it
	// would be skipped and the whole call would surface as a baffling
	// "(empty response)" → "plan not valid JSON" three layers up. Captured and
	// raised as the call error so the server's own message reaches the user.
	// Both shapes seen in the wild: nested {"error":{"message":…}} (OpenAI
	// style) and a bare top-level {"message":…}; Message catches the latter.
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
	Message string `json:"message"`
}

// chunkErrorMessage extracts an in-stream error message from an SSE chunk, or
// "" when the chunk is a normal content/usage frame. Handles the nested OpenAI
// shape {"error":{"message":…}} and a bare {"message":…} — the latter only when
// the frame carries nothing else (no choices/usage/timings), so a stray field
// can never make a normal chunk look like a failure.
func chunkErrorMessage(c *sseChunk) string {
	if c.Error != nil && c.Error.Message != "" {
		return c.Error.Message
	}
	if c.Message != "" && len(c.Choices) == 0 && c.Usage == nil && c.Timings == nil {
		return c.Message
	}
	return ""
}

// llmHTTPError is a non-200 response from the LLM endpoint. It carries the
// status code so callers can distinguish a context-overflow 400 (the tool loop
// recovers from it by summarising completed small turns, see foldHistory)
// from other failures. Error() reproduces the prior bare-string message so logs
// and UI surfaces are unchanged.
type llmHTTPError struct {
	Status int
	Body   string
	URL    string
}

func (e *llmHTTPError) Error() string {
	return fmt.Sprintf("LLM returned %d: %s [URL: %s]", e.Status, e.Body, e.URL)
}

// errContextCeiling marks a generation that truncated at the n_ctx ceiling: the
// prompt fit (no 400) but left so little room that the model hit finish=length
// below its max_tokens cap. Same cause as a 400 (context too full), so the tool
// loop recovers it the same way — fold history and retry. Wrapped with %w so
// isContextFull can detect it.
var errContextCeiling = errors.New("generation hit the context ceiling")

// isContextFull reports whether err means the prompt filled the context: the
// server rejected it outright (HTTP 400) or a generation truncated at the n_ctx
// ceiling (errContextCeiling). Both are recovered by folding history and retrying.
func isContextFull(err error) bool {
	var he *llmHTTPError
	if errors.As(err, &he) && he.Status == 400 {
		return true
	}
	return errors.Is(err, errContextCeiling)
}

// errStuckThinking marks a generation that spent its whole max_tokens budget on
// reasoning_content with no message text and no tool calls — the model looped in
// <think> and produced nothing usable. Recoverable: the tool loop retries once on
// a thinking-disabled copy of the connection so the model must answer directly.
// Only raised when thinking was ON, so the retry can't re-trigger it.
var errStuckThinking = errors.New("model stuck in reasoning")

// isStuckThinking reports whether err is the <think>-loop stall recovered by a
// thinking-off retry (see errStuckThinking).
func isStuckThinking(err error) bool { return errors.Is(err, errStuckThinking) }

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

// thinkingOn reports whether the request had reasoning enabled. Absent
// chat_template_kwargs (or an absent enable_thinking) counts as on, since the
// stall is only classified when reasoning_content was actually produced; an
// explicit enable_thinking=false (a retry) counts as off so it can't loop.
func thinkingOn(reqBody map[string]any) bool {
	ctk, ok := reqBody["chat_template_kwargs"].(map[string]any)
	if !ok {
		return true
	}
	et, ok := ctk["enable_thinking"].(bool)
	return !ok || et
}

// withThinkingDisabled returns a shallow copy of the connection with
// chat_template_kwargs.enable_thinking forced false in a deep-copied ExtraBody,
// so the stuck-in-<think> retry answers directly. Slot/Server/Model are
// unchanged, so it routes to the same connSem. The original conn is untouched.
func (c *LLMConnection) withThinkingDisabled() *LLMConnection {
	cp := *c
	eb := make(map[string]any, len(c.ExtraBody)+1)
	maps.Copy(eb, c.ExtraBody)
	ctk := map[string]any{}
	if existing, ok := eb["chat_template_kwargs"].(map[string]any); ok {
		maps.Copy(ctk, existing)
	}
	ctk["enable_thinking"] = false
	eb["chat_template_kwargs"] = ctk
	cp.ExtraBody = eb
	return &cp
}

// llmStream is the core LLM call. Streams SSE, collects text and tool calls.
// sid scopes the debug log: req body and raw SSE response are appended to
// .codehalter/session_<sid>.log so a single file captures everything that
// went over the wire for a session. Pass "" to disable logging (used by tests
// and pre-session probes). think (nil to discard) receives reasoning_content
// tokens — kept separate from `on` so callers can surface chain-of-thought to
// the UI as agent_thought_chunk without polluting agent_message_chunk.
func (a *agent) llmStream(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, tools []map[string]any, on, think func(string)) (string, []toolCall, string, error) {
	// Seed with extra_body (per-role sampler/reasoning overrides), then write
	// core fields last so model/messages/stream/tools can't be hijacked from
	// settings.toml.
	reqBody := map[string]any{}
	maps.Copy(reqBody, conn.ExtraBody)
	if _, ok := reqBody["max_tokens"]; !ok {
		reqBody["max_tokens"] = defaultMaxTokens
	}
	reqBody["model"] = conn.Model
	reqBody["stream"] = true
	// Ask llama.cpp for per-request timings (prompt_n/cache_n/_ms) so the stats use
	// server ground truth for the cache split. Harmless if ignored.
	reqBody["timings_per_token"] = true
	// stream_options.include_usage asks the server to emit a final SSE chunk
	// carrying prompt_tokens / completion_tokens, the server's own count for
	// the per-turn "✅ Done" usage stats. A backend that ignores it (or a proxy
	// that strips it) leaves the counts at 0, so the stats line just shows less
	// detail.
	reqBody["stream_options"] = map[string]any{"include_usage": true}
	reqBody["messages"] = messages
	if tools != nil {
		reqBody["tools"] = tools
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", nil, "", fmt.Errorf("marshalling LLM request body: %w", err)
	}

	// Per-conn concurrency gate: cap in-flight calls to this conn at its
	// configured `parallel`. The token is held only for this call (released on
	// return), not a subagent's lifetime — so between calls the conn frees up,
	// and pool size 1 serialises without deadlocking nested subagents; the wait
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
	// ↑ is THIS call's request-body size as two-decimal KiB — display-only.
	upLabel := fmt.Sprintf("%.2fkb", float64(len(body))/1024)
	a.setStatus(ctx, sid, fmt.Sprintf(" (llm[%s] ↑%s sent…)", slotLabel, upLabel))
	defer a.setStatus(ctx, sid, "")

	// streamWaitMeter (see its doc) refreshes the phase row each second for the
	// whole call. genChars is the running generated-byte count it reads per tick,
	// written concurrently by the scanner loop below — hence atomic.
	var genChars int64
	waitDone := make(chan struct{})
	go a.streamWaitMeter(ctx, sid, slotLabel, upLabel, time.Now(), &genChars, waitDone)
	defer close(waitDone)

	// Per-session log: a REQUEST block now, one aggregated RESPONSE block at the
	// end (the per-token SSE wire is too noisy to skim). connLabel carries
	// llm[<slot>] + role + model so grepping "llm[1]" finds every request routed
	// to a given entry.
	connLabel := fmt.Sprintf("llm[%s] %s model=%s", slotLabel, conn.Tag, conn.Model)
	a.logSession(sid, connLabel+" REQUEST", "%s", string(body))

	httpReq, err := http.NewRequestWithContext(ctx, "POST", conn.endpoint("/v1/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return "", nil, "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if conn.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
	}

	// llmHTTPClient caps the wait for the server's first response byte at 90s
	// (ResponseHeaderTimeout). This bounds the hang when a network switch breaks
	// an in-flight TCP connection: without it, TCP retransmission keeps the
	// request alive for up to ~15 minutes before the OS gives up. 90s is enough
	// for a busy or queued LLM server to start streaming; cancellation via the
	// request ctx still applies for the rest of the stream.
	resp, err := llmHTTPClient.Do(httpReq)
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
		return "", nil, "", &llmHTTPError{Status: resp.StatusCode, Body: msg, URL: resp.Request.URL.String()}
	}

	var fullText strings.Builder
	var reasoningText strings.Builder
	var calls []toolCall
	var finishReason string
	// streamErrMsg holds an error the server delivered in-band (HTTP 200, an
	// {"error":…} SSE chunk). Surfaced as the call error so it isn't swallowed
	// as an empty response.
	var streamErrMsg string
	var promptTokens, completionTokens int
	// Server-reported cache split (see sseChunk.Timings / PromptTokensDetails).
	// evaluatedTokens = prompt tokens actually run through the model this call;
	// cachedTokens = reused from KV cache. -1 = the server didn't report it.
	evaluatedTokens, cachedTokens := -1, -1
	// Server-measured eval/gen times (ms) — exact, vs our TTFT proxy.
	var serverPromptMs, serverGenMs float64

	scanner := bufio.NewScanner(resp.Body)
	// SSE chunks can carry large tool-call argument blobs; the default 64 KB
	// line limit silently truncates. 4 MB matches common reverse-proxy caps.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	// TTFT (readStart→firstToken) and gen (firstToken→end) — the rate timing
	// fallback when the server sends no _ms.
	readStart := time.Now()
	var firstTokenAt time.Time
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk sseChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		// In-band error: the gateway/llama.cpp can return HTTP 200 and put the
		// failure in an {"error":…} chunk (empty Choices). Capture and stop —
		// checked before the empty-Choices skip below, which would drop it and
		// leave the call looking like a silent "(empty response)".
		if msg := chunkErrorMessage(&chunk); msg != "" {
			streamErrMsg = msg
			break
		}
		// Usage arrives in its own trailing chunk (choices empty) when
		// stream_options.include_usage=true. Capture and keep going — there
		// may still be a [DONE] line after it.
		if chunk.Usage != nil {
			if chunk.Usage.PromptTokens > 0 {
				promptTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				completionTokens = chunk.Usage.CompletionTokens
			}
			if d := chunk.Usage.PromptTokensDetails; d != nil {
				cachedTokens = d.CachedTokens
			}
		}
		// llama.cpp timings (prefer over usage cached_tokens — it carries both
		// sides directly): prompt_n = evaluated, cache_n = reused.
		if chunk.Timings != nil {
			evaluatedTokens = chunk.Timings.PromptN
			cachedTokens = chunk.Timings.CacheN
			serverPromptMs = chunk.Timings.PromptMs
			serverGenMs = chunk.Timings.PredictedMs
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		if r := chunk.Choices[0].FinishReason; r != "" {
			finishReason = r
		}

		delta := chunk.Choices[0].Delta

		if firstTokenAt.IsZero() && (delta.Content != "" || delta.ReasoningContent != "" || len(delta.ToolCalls) > 0) {
			firstTokenAt = time.Now()
		}

		if delta.ReasoningContent != "" {
			reasoningText.WriteString(delta.ReasoningContent)
			atomic.AddInt64(&genChars, int64(len(delta.ReasoningContent)))
			if think != nil {
				think(delta.ReasoningContent)
			}
		}

		if delta.Content != "" {
			fullText.WriteString(delta.Content)
			atomic.AddInt64(&genChars, int64(len(delta.Content)))
			if on != nil {
				on(delta.Content)
			}
		}

		for _, tc := range delta.ToolCalls {
			atomic.AddInt64(&genChars, int64(len(tc.Function.Name)+len(tc.Function.Arguments)))
			if tc.ID != "" {
				calls = append(calls, tc)
			} else if len(calls) > 0 {
				last := &calls[len(calls)-1]
				last.Function.Arguments += tc.Function.Arguments
			}
		}
	}
	scanErr := scanner.Err()

	// Sum the server's reported usage into the turn for the "✅ Done" stats line.
	// sid="" (probes/tests) has no session, skip. A scan failure means the stream
	// broke before the usage chunk, so the counts are 0 and this is a no-op.
	if sess := a.getSession(sid); sess != nil {
		// Derive evaluated from cached_tokens when timings are absent; -1 means no
		// cache info reported.
		if evaluatedTokens < 0 && cachedTokens >= 0 && promptTokens > 0 {
			evaluatedTokens = promptTokens - cachedTokens
		}
		sess.addTurnTokens(promptTokens, completionTokens, evaluatedTokens, cachedTokens)
		// Prefer the server's measured times over our TTFT proxy (which includes
		// queue + cache-load overhead → understates pp/s).
		pMs, gMs := serverPromptMs, serverGenMs
		if pMs == 0 && gMs == 0 && !firstTokenAt.IsZero() {
			pMs = float64(firstTokenAt.Sub(readStart).Milliseconds())
			gMs = float64(time.Since(firstTokenAt).Milliseconds())
		}
		if pMs > 0 || gMs > 0 {
			sess.addTurnTiming(int64(pMs), int64(gMs))
		}
	}

	// One outcome error, shared by the RESPONSE log block and the return. Order
	// matters: an explicit in-band server error is checked FIRST, because gateways
	// commonly emit an {"error":…} chunk and THEN drop the socket. Checking the
	// transport error first would shadow the server's verbatim cause (e.g. "prompt
	// exceeds n_ctx") behind a generic "unexpected EOF", and send a fatal prompt
	// down the useless transient-retry path instead of surfacing the real reason.
	//   - streamErrMsg: server sent an {"error":…} chunk under HTTP 200; surface
	//     its message verbatim (it names the real cause, e.g. prompt > n_ctx).
	//   - scanErr: stream broke mid-flight (e.g. a router model swap force-kills
	//     the connection) with no in-band error to explain it.
	//   - finish_reason="length": truncated at a length limit. If completion hit
	//     the requested max_tokens cap the model is genuinely verbose/looping and we
	//     bail (the message guides tuning); if it stopped BELOW the cap it hit the
	//     n_ctx ceiling (prompt fit but left no room), recoverable, signalled via
	//     errContextCeiling so the tool loop folds history and retries.
	switch {
	case streamErrMsg != "":
		err = fmt.Errorf("LLM returned an error mid-stream (role=%s, model=%s): %s", conn.Tag, conn.Model, streamErrMsg)
	case scanErr != nil:
		err = fmt.Errorf("reading SSE stream: %w", scanErr)
	case finishReason == "length":
		// Two very different causes. (1) completion reached the requested max_tokens
		// cap → the model is genuinely verbose/looping; bail (the message guides
		// tuning). (2) completion is BELOW the cap → it hit the n_ctx ceiling: the
		// prompt fit but left no room to generate. (2) is recoverable — the context
		// is too full, same as a 400 — so signal isContextFull and let the tool loop
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
		belowCap := completionTokens > 0 && reqMax > 0 && completionTokens < reqMax
		noRoom := promptTokens > 0 && reqMax > 0 && a.mainSlotTokens > 0 && promptTokens+reqMax > a.mainSlotTokens
		if belowCap || noRoom {
			err = fmt.Errorf("generation hit the context ceiling (prompt=%d gen=%d, n_ctx=%d, role=%s): %w",
				promptTokens, completionTokens, a.mainSlotTokens, conn.Tag, errContextCeiling)
		} else if fullText.Len() == 0 && len(calls) == 0 && reasoningText.Len() > 0 && thinkingOn(reqBody) {
			// All budget went to reasoning with nothing to show — the model looped
			// in <think>. Recoverable: the tool loop retries once with thinking off
			// so it answers directly (see errStuckThinking).
			err = fmt.Errorf("model stuck in <think> (%d B reasoning, 0 content/calls, role=%s): %w",
				reasoningText.Len(), conn.Tag, errStuckThinking)
		} else {
			err = fmt.Errorf("LLM hit max_tokens cap (role=%s, model=%s) — response truncated (%d B content, %d B reasoning, %d tool calls). Likely the model is looping or stuck in <think>; raise max_tokens in params_%s, or set chat_template_kwargs.enable_thinking=false for this role if reasoning is dominating the budget",
				conn.Tag, conn.Model, fullText.Len(), reasoningText.Len(), len(calls), conn.Tag)
		}
	default:
		err = nil
	}

	// One RESPONSE block per call, on every exit path. The raw SSE wire is one
	// event per token (~100 lines for a short reply, thousands for a tool-call
	// argument blob) — useless for skimming; this collapses the deltas into the
	// reconstructed transcript. promptTokens/completionTokens are the server's
	// exact counts from the trailing usage chunk (0 when the backend didn't
	// report them).
	text, reasoning := fullText.String(), reasoningText.String()
	var rb strings.Builder
	if promptTokens > 0 || completionTokens > 0 || finishReason != "" {
		// finish: "stop" = model ended; "length" = hit max_tokens (truncated);
		// "tool_calls" = ended on a tool call; "(none)" = stream broke with no
		// finish_reason (interrupted). Distinguishes a cap/interrupt from a clean end.
		fr := finishReason
		if fr == "" {
			fr = "(none)"
		}
		fmt.Fprintf(&rb, "tokens: prompt=%d completion=%d finish=%s\n", promptTokens, completionTokens, fr)
	}
	if reasoning != "" {
		fmt.Fprintf(&rb, "reasoning_content (%d B):\n%s\n", len(reasoning), reasoning)
	}
	if text != "" {
		fmt.Fprintf(&rb, "content:\n%s\n", text)
	}
	for i, c := range calls {
		fmt.Fprintf(&rb, "tool_call[%d] %s id=%s args=%s\n", i, c.Function.Name, c.ID, c.Function.Arguments)
	}
	if text == "" && reasoning == "" && len(calls) == 0 {
		rb.WriteString("(empty response)\n")
	}
	if err != nil {
		fmt.Fprintf(&rb, "[stream error] %v\n", err)
	}
	a.logSession(sid, connLabel+" RESPONSE", "%s", rb.String())
	return fullText.String(), calls, reasoningText.String(), err
}

// streamWaitMeter refreshes the active phase row once per second for the whole
// LLM call. Until the first generated byte (genChars==0) it shows "(sent… Ns)"
// so a busy/queuing server reads as "(sent… 25s)" instead of a frozen
// "(sent…)"; once bytes arrive it shows the live "↑up ↓tokens…" estimate.
// genChars is read atomically — the scanner loop writes it concurrently.
// Returns when the call ends (done closed) or ctx is cancelled; it never aborts
// the request. No warning is emitted: while the call is alive the climbing
// counter is the signal, and if it dies (e.g. a router model swap force-kills
// the connection) llmStream surfaces the transport error directly.
func (a *agent) streamWaitMeter(ctx context.Context, sid, slotLabel, upLabel string, start time.Time, genChars *int64, done <-chan struct{}) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ctx.Done():
			return
		case <-ticker.C:
			if g := atomic.LoadInt64(genChars); g > 0 {
				// approx tokens (chars/4), compact via humanCount. Display-only.
				tok := humanCount(int(g) / 4)
				a.setStatus(ctx, sid, fmt.Sprintf(" (llm[%s] ↑%s ↓%s…)", slotLabel, upLabel, tok))
			} else {
				elapsed := int(time.Since(start).Seconds())
				a.setStatus(ctx, sid, fmt.Sprintf(" (llm[%s] ↑%s sent… %ds)", slotLabel, upLabel, elapsed))
			}
		}
	}
}

// probeResult is what a single LLM probe call yields: server reachability,
// model presence (when the endpoint enumerates models), image support, and
// (when discoverable) the model's context window in tokens. Empty value
// means "we could not reach the server at all."
type probeResult struct {
	Reachable    bool // got 200 from any probe endpoint
	ModelKnown   bool // /v1/models enumerated models — ModelLoaded is meaningful
	ModelLoaded  bool // the configured model was in the enumeration
	ImageSupport bool
	// AvailableModels is every model id /v1/models enumerated, in server
	// order. renderLLMStatus shows it when the configured model isn't among
	// them so the user can read off the correct name. Empty when the server
	// didn't enumerate (ModelKnown false) or its list was empty.
	AvailableModels []string
	// ContextSize is the TOTAL n_ctx (prompt+output) the server was launched
	// with — from /v1/models' -c / --ctx-size launch arg, or older llama.cpp
	// /props top-level n_ctx. 0 = unknown. probeAllLLMs divides it by the slot
	// count to get the per-slot window.
	ContextSize int
	// SlotCtx is the PER-SLOT n_ctx reported directly by modern llama.cpp /props
	// (default_generation_settings.n_ctx — already total ÷ -np). Preferred over
	// ContextSize when set: no division, robust to how the server splits. 0 =
	// not reported.
	SlotCtx int
	// TotalSlots is llama.cpp's -np concurrent slot count from /props
	// total_slots. probeAllLLMs back-fills it into any [[llm]] that left
	// `parallel` unset. 0 = not reported (non-llama backends).
	TotalSlots int
}

// probeLLM checks reachability, model presence, and (best-effort) image
// support + context size via cheap metadata endpoints. /v1/models is the
// universal reachability endpoint — every OpenAI-compatible backend exposes
// it. /props is a llama.cpp-specific enrichment step that fills in image /
// ctx metadata when /v1/models returns the bare OpenAI shape. Backends
// without /props (OpenAI, Ollama, vLLM, OpenWebUI, LiteLLM, …) leave those
// fields unset and the user supplies them via settings.toml (probeAllLLMs
// applies that precedence).
func (a *agent) probeLLM(ctx context.Context, conn *LLMConnection) probeResult {
	if conn == nil {
		return probeResult{}
	}
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	r, _ := a.probeViaModels(probeCtx, conn)
	// Enrich via /props when reachable: it carries image + ctx metadata AND
	// llama.cpp's per-slot n_ctx / total_slots, none of which /v1/models exposes.
	// Non-llama backends 404 here (ok=false) and we keep the /v1/models result.
	p, ok := a.probeViaProps(probeCtx, conn, "/props")
	// llama.cpp router mode (role:"router", models_autoload): bare /props reports
	// n_ctx=0 — the loaded model's real n_ctx is only returned when the request is
	// routed to it via the ?model=<id> query param (and that also autoloads the
	// model). url.QueryEscape is required: ids routinely carry spaces and
	// semicolons ("Qwen3.5 (122B-A10B; …)") and a raw ';' is a query separator, so
	// the name would be truncated → "model not found". Own longer timeout covers a
	// cold load. Scoped to reachable-but-zero /props so direct servers (real n_ctx)
	// and OpenAI/vLLM (404 /props) are untouched.
	if ok && p.ContextSize == 0 && p.SlotCtx == 0 {
		upCtx, upCancel := context.WithTimeout(ctx, 180*time.Second)
		if up, upOK := a.probeViaProps(upCtx, conn, "/props?model="+url.QueryEscape(conn.Model)); upOK {
			p = up
		}
		upCancel()
	}
	if ok {
		r.Reachable = true
		if !r.ImageSupport {
			r.ImageSupport = p.ImageSupport
		}
		if r.ContextSize == 0 {
			r.ContextSize = p.ContextSize
		}
		if r.SlotCtx == 0 {
			r.SlotCtx = p.SlotCtx
		}
		if r.TotalSlots == 0 {
			r.TotalSlots = p.TotalSlots
		}
	}

	if r.Reachable {
		slog.Info("probeLLM", "model", conn.Model, "loaded", r.ModelLoaded, "image", r.ImageSupport, "ctx", r.ContextSize)
	} else {
		slog.Info("probeLLM: unreachable", "server", conn.Server, "model", conn.Model)
	}
	return r
}

// probeViaModels asks /v1/models for the configured model. Always confirms
// reachability + model presence; image_support / context_size only land
// when the response carries llama-swap-style `status.args` (--mmproj /
// --ctx-size). OpenAI/Ollama/vLLM/LiteLLM all 200 here but return the bare
// OpenAI shape, so the caller's /props enrichment + settings.toml fallback
// fills the gap. ok=false only on network / non-200 — a bare response still
// returns ok=true so the caller knows the server is up. Records every
// enumerated id in AvailableModels so renderLLMStatus can show the real names
// when the configured model isn't found.
func (a *agent) probeViaModels(ctx context.Context, conn *LLMConnection) (probeResult, bool) {
	modelsURL := conn.endpoint("/v1/models")
	req, err := http.NewRequestWithContext(ctx, "GET", modelsURL, nil)
	if err == nil && conn.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+conn.APIKey)
	}
	var resp *http.Response
	if err == nil {
		resp, err = http.DefaultClient.Do(req)
	}
	if err != nil {
		slog.Info("probeViaModels: request failed", "url", modelsURL, "err", err)
		return probeResult{}, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return probeResult{}, false
	}
	var models struct {
		Data []struct {
			ID     string `json:"id"`
			Status struct {
				Args []string `json:"args"`
			} `json:"status"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		return probeResult{}, false
	}
	r := probeResult{Reachable: true, ModelKnown: true}
	for _, m := range models.Data {
		r.AvailableModels = append(r.AvailableModels, m.ID)
		if m.ID != conn.Model {
			continue
		}
		r.ModelLoaded = true
		// Scan launch args for two facts: --mmproj (vision) and the
		// context-size flag (--ctx-size / -c, either spaced or =-joined).
		// Don't `break` on the first hit — we want both signals.
		for i, arg := range m.Status.Args {
			switch arg {
			case "--mmproj":
				r.ImageSupport = true
			case "--ctx-size", "-c":
				if i+1 < len(m.Status.Args) {
					if n, err := strconv.Atoi(m.Status.Args[i+1]); err == nil {
						r.ContextSize = n
					}
				}
			default:
				if v, ok := strings.CutPrefix(arg, "--ctx-size="); ok {
					if n, err := strconv.Atoi(v); err == nil {
						r.ContextSize = n
					}
				} else if v, ok := strings.CutPrefix(arg, "-c="); ok {
					if n, err := strconv.Atoi(v); err == nil {
						r.ContextSize = n
					}
				}
			}
		}
	}
	return r, true
}

// probeViaProps reads a llama-server /props endpoint. Tells us the server is
// reachable and (via modalities.vision) whether the loaded model supports image
// input, but cannot tell us *which* model is loaded — so ModelKnown stays false.
// path is "/props" for a direct llama-server, or "/props?model=<url-encoded id>"
// to reach a specific model in llama.cpp router mode (whose bare /props reports
// n_ctx=0).
func (a *agent) probeViaProps(ctx context.Context, conn *LLMConnection, path string) (probeResult, bool) {
	propsURL := conn.endpoint(path)
	req, err := http.NewRequestWithContext(ctx, "GET", propsURL, nil)
	if err == nil && conn.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+conn.APIKey)
	}
	var resp *http.Response
	if err == nil {
		resp, err = http.DefaultClient.Do(req)
	}
	if err != nil {
		slog.Info("probeViaProps: request failed", "url", propsURL, "err", err)
		return probeResult{}, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		slog.Info("probeViaProps: non-OK", "url", propsURL, "status", resp.StatusCode, "body", string(body))
		return probeResult{}, false
	}
	var props struct {
		Modalities *struct {
			Vision bool `json:"vision"`
		} `json:"modalities"`
		// llama.cpp's /props exposes n_ctx in two shapes across releases, and
		// they mean different things: modern builds nest a PER-SLOT n_ctx under
		// default_generation_settings (already total ÷ -np), older ones surface
		// the TOTAL at the top level. total_slots is the -np slot count.
		DefaultGenerationSettings *struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
		NCtx       int `json:"n_ctx"`
		TotalSlots int `json:"total_slots"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&props); err != nil {
		return probeResult{}, false
	}
	r := probeResult{Reachable: true, ContextSize: props.NCtx, TotalSlots: props.TotalSlots}
	if props.Modalities != nil {
		r.ImageSupport = props.Modalities.Vision
	}
	if props.DefaultGenerationSettings != nil {
		r.SlotCtx = props.DefaultGenerationSettings.NCtx
	}
	return r, true
}

// connForSession resolves the connection for the session and role. The main
// session always uses LLM[0] — its KV cache owns the parent's conversation
// prefix. Subagent sessions carry a PinnedLLMIdx assigned at launch by
// launch_subagent; every call from that session routes back to the same
// entry so the conn's prefix cache stays warm across plan/execute switches.
// Concurrency is enforced by per-conn semaphores in llmStream; this picker
// just resolves the routing target.
func (a *agent) connForSession(_ context.Context, sid string, role string) *LLMConnection {
	// Resolve the session under a.mu (getSession) BEFORE taking cfgMu, so cfgMu
	// stays a strict leaf. Then read settings under RLock; ConnAt/MainLLM return
	// value copies, safe to use after the lock is released.
	sess := a.getSession(sid)
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	if sess != nil && sess.Depth > 0 && sess.PinnedLLMIdx >= 0 {
		if c := a.settings.ConnAt(sess.PinnedLLMIdx, role); c != nil {
			return c
		}
	}
	return a.settings.MainLLM(role)
}

// connForBackgroundLLM returns the connection to host background work (per-turn
// summariser, git-commit, document). It walks the background tier LLM[1..x] and
// returns the first with free semaphore capacity, spreading load across servers
// instead of stacking on LLM[1]. If all are busy (or none exist) it falls back
// to LLM[0], labelled llm[1] when that conn has >=2 slots so the meter shows the
// work routed off the foreground turn. The capacity peek is racy by design —
// llmStream's semaphore just queues if the slot was taken meanwhile.
func (a *agent) connForBackgroundLLM() *LLMConnection {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	for i := 1; i < len(a.settings.LLM); i++ {
		if i < len(a.connSems) && a.connSems[i] != nil &&
			len(a.connSems[i]) < cap(a.connSems[i]) {
			return a.settings.ConnAt(i, "execute")
		}
	}
	// No separate background entry (or all busy): fall back to llm[0]. When it
	// has >= 2 parallel slots, label this as llm[1] — same connection and
	// semaphore, but a distinct display slot so the meter shows background
	// routed off the foreground turn (the server picks the real KV slot).
	c := a.settings.ConnAt(0, "execute")
	if c != nil && a.settings.LLM[0].parallelCap() >= 2 {
		c.Slot = 1
	}
	return c
}

// buildConnSems sizes one buffered channel per LLM entry to its parallelCap.
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
			if cap(a.connSems[i]) != a.settings.LLM[i].parallelCap() {
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
		sems[i] = make(chan struct{}, a.settings.LLM[i].parallelCap())
	}
	a.connSems = sems
}
