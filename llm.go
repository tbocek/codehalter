package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

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
			// ReasoningContent is the chain-of-thought channel emitted by
			// thinking models (Qwen3, DeepSeek-R1, GPT-OSS, …) when the
			// upstream server is configured to split <think>…</think> off
			// the main content stream (llama.cpp `--reasoning-format
			// deepseek`, vLLM `--reasoning-parser deepseek_r1`, etc.). We
			// don't merge it into Content because then prefix-cache lookups
			// would key on a different string than the model actually
			// generated — but we DO accumulate and log it so a turn that
			// burns its whole budget thinking reports as "N bytes reasoning,
			// 0 visible" instead of "(empty response)".
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	// Usage is the OpenAI-compatible token count block. With
	// stream_options.include_usage=true the server emits one final chunk
	// (choices empty) carrying this — the prompt_tokens count is ground
	// truth for what the server actually packed into n_ctx, and replaces
	// the chars/4 estimator on the compaction trigger.
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// llmHTTPClient is the dedicated client for LLM completion calls. It owns its
// timeouts instead of inheriting http.DefaultClient/DefaultTransport's, whose
// defaults bite an LLM workload in surprising ways:
//   - DefaultTransport.TLSHandshakeTimeout = 10s — a hammered HTTPS endpoint can
//     be too busy to complete the handshake inside 10s; we lift it to 60s so a
//     slow-but-alive server isn't killed at exactly 10s.
//   - No Client.Timeout and ResponseHeaderTimeout left at 0: a long generation,
//     or a server queuing the request before it answers, is legitimate and must
//     NOT be capped — cancellation is driven by the request ctx only.
//
// The 30s dial timeout (TCP connect) is kept: a connection that can't even be
// established should fail fast and visibly rather than hang.
var llmHTTPClient = func() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.TLSHandshakeTimeout = 60 * time.Second
	t.ResponseHeaderTimeout = 0
	return &http.Client{Transport: t}
}()

// llmStream is the core LLM call. Streams SSE, collects text and tool calls.
// sid scopes the debug log: req body and raw SSE response are appended to
// .codehalter/session_<sid>.log so a single file captures everything that
// went over the wire for a session. Pass "" to disable logging (used by tests
// and pre-session probes). think (nil to discard) receives reasoning_content
// tokens — kept separate from `on` so callers can surface chain-of-thought to
// the UI as agent_thought_chunk without polluting agent_message_chunk.
func (a *agent) llmStream(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, tools []map[string]any, on, think func(string)) (string, []toolCall, error) {
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
	// stream_options.include_usage asks the server to emit a final SSE chunk
	// carrying prompt_tokens / completion_tokens. The chars/4 estimator can be
	// 30% wrong on tool-heavy JSON; this gives us the server's own count so
	// compressHistory triggers on ground truth instead of a guess.
	reqBody["stream_options"] = map[string]any{"include_usage": true}
	reqBody["messages"] = messages
	if tools != nil {
		reqBody["tools"] = tools
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", nil, fmt.Errorf("marshalling LLM request body: %w", err)
	}

	// Per-conn slot semaphore: cap concurrent in-flight calls to this conn's
	// configured `parallel`. The token is released on llmStream return, NOT for
	// the lifetime of a subagent — so during the tool-dispatch gap between two
	// llmStream calls the conn is free for another caller. With pool size 1
	// this naturally serialises everything (no deadlock even when a subagent
	// nests another), and with N>1 fan-out fills the pool until full and queues
	// the rest. Surface the wait as a status suffix so the UI shows "(queued…)"
	// instead of looking frozen while a slot is busy.
	slot := a.connSlot(conn)
	if slot >= 0 && slot < len(a.slotSems) {
		// Try non-blocking first; only emit the queued suffix when we're
		// actually about to wait. Avoids flashing the wrong status on the
		// common hot path where the slot is free.
		select {
		case a.slotSems[slot] <- struct{}{}:
		default:
			a.setStatus(ctx, sid, " (queued…)")
			select {
			case a.slotSems[slot] <- struct{}{}:
			case <-ctx.Done():
				a.setStatus(ctx, sid, "")
				return "", nil, ctx.Err()
			}
		}
		defer func() { <-a.slotSems[slot] }()
	}

	// slotLabel is the flat display slot for this call: llm[0] for the
	// foreground turn, llm[1] for background work (summariser/git-commit) even
	// when it's the same connection — see LLMConnection.Slot / pickBackgroundLLM.
	// `slot` above is the *semaphore* index (the configured [[llm]] entry);
	// conn.Slot is what the user reads. slot=-1 (test mocks, probes) → "?".
	slotLabel := "?"
	if slot >= 0 {
		slotLabel = fmt.Sprintf("%d", conn.Slot)
	}

	// Drive the lifecycle suffix on the active phase entry across the LLM
	// round-trip: "(sent…)" while we wait for the first SSE token (cache
	// misses, queueing, slow TTFT), then the live ↑/↓ meter once streaming
	// starts. setStatus is a no-op when no phase is active so summarisation
	// calls between phases don't flash anything.
	//
	// Upload (↑) is just THIS call's request-body size in kB — what we sent to
	// the LLM right now, not a running total or peak. Paired with llm[<slot>]
	// so a background summarise visibly routes off the foreground slot.
	upLabel := fmtKB(int64(len(body)))
	a.setStatus(ctx, sid, fmt.Sprintf(" (llm[%s] ↑%s sent…)", slotLabel, upLabel))
	defer a.setStatus(ctx, sid, "")

	// Waiting meter: a busy/queuing server holds the request open without
	// responding, and there's no client timeout (by design — a slow generation
	// is legitimate), so without this the phase row would sit frozen at
	// "(sent…)" with no sign of life. Tick elapsed seconds onto it until the
	// first token arrives, and warn once after a long silence — the request
	// keeps waiting, the user just learns the server is the bottleneck.
	firstByte := make(chan struct{})
	var firstByteOnce sync.Once
	waitDone := make(chan struct{})
	go a.streamWaitMeter(ctx, sid, slotLabel, upLabel, time.Now(), firstByte, waitDone)
	defer close(waitDone)

	// Per-session log: request header + body. The raw SSE wire is too noisy
	// to read (one event per token, each repeating the full envelope), so we
	// parse the stream and write a single aggregated RESPONSE block at the
	// end. Mid-stream events (HTTP errors, partial state on transport
	// failure) are appended inline below. Closed on return.
	// Header records llm[<slot>] alongside the role+model so a grep for
	// "llm[1]" surfaces every request routed to a given entry — useful for
	// confirming that document / summarise / git_commit landed off the
	// foreground prefix cache.
	connLabel := fmt.Sprintf("llm[%s] %s model=%s", slotLabel, conn.Tag, conn.Model)
	logF := a.sessionLog(sid)
	if logF != nil {
		defer logF.Close()
		fmt.Fprintf(logF, "\n=== %s [%s] REQUEST ===\n%s\n",
			time.Now().Format(time.RFC3339), connLabel, string(body))
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", conn.URL, bytes.NewReader(body))
	if err != nil {
		return "", nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if conn.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
	}

	resp, err := llmHTTPClient.Do(httpReq)
	if err != nil {
		if logF != nil {
			fmt.Fprintf(logF, "[transport error] %v\n", err)
		}
		return "", nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// 1. Read the body
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", nil, fmt.Errorf("HTTP %d, failed to read body: %w", resp.StatusCode, err)
		}
		if logF != nil {
			fmt.Fprintf(logF, "[HTTP %d] %s\n", resp.StatusCode, string(bodyBytes))
		}

		// 2. Try to parse the JSON to find a specific error message
		var apiErr map[string]any
		if jsonErr := json.Unmarshal(bodyBytes, &apiErr); jsonErr == nil {
			if errMsg, ok := apiErr["error"]; ok {
				if errMap, ok := errMsg.(map[string]any); ok {
					if msg, ok := errMap["message"].(string); ok {
						// Include the URL in the error
						return "", nil, fmt.Errorf("LLM returned %d: %s [URL: %s]", resp.StatusCode, msg, resp.Request.URL.String())
					}
				}
			}
		}

		// 3. Fallback: If JSON parsing failed, show raw body and URL
		return "", nil, fmt.Errorf("LLM returned %d: %s [URL: %s]", resp.StatusCode, string(bodyBytes), resp.Request.URL.String())
	}

	var fullText strings.Builder
	var reasoningText strings.Builder
	var calls []toolCall
	var finishReason string
	var promptTokens, completionTokens int
	// Live token meter: genChars accumulates every generated byte (content +
	// reasoning + tool-call args); lastMeterChars is the genChars value at the
	// last status re-emit so we refresh roughly once per statusMeterChars
	// bytes (~20 tokens) instead of on every SSE event.
	genChars, lastMeterChars := 0, 0

	scanner := bufio.NewScanner(resp.Body)
	// SSE chunks can carry large tool-call argument blobs; the default 64 KB
	// line limit silently truncates. 4 MB matches common reverse-proxy caps.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
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
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		if r := chunk.Choices[0].FinishReason; r != "" {
			finishReason = r
		}

		delta := chunk.Choices[0].Delta

		if delta.ReasoningContent != "" {
			reasoningText.WriteString(delta.ReasoningContent)
			genChars += len(delta.ReasoningContent)
			if think != nil {
				think(delta.ReasoningContent)
			}
		}

		if delta.Content != "" {
			fullText.WriteString(delta.Content)
			genChars += len(delta.Content)
			if on != nil {
				on(delta.Content)
			}
		}

		for _, tc := range delta.ToolCalls {
			genChars += len(tc.Function.Name) + len(tc.Function.Arguments)
			if tc.ID != "" {
				calls = append(calls, tc)
			} else if len(calls) > 0 {
				last := &calls[len(calls)-1]
				last.Function.Arguments += tc.Function.Arguments
			}
		}

		// Refresh the live meter on the first generated byte and then once per
		// statusMeterChars thereafter. Shows sent (estimated up front) and the
		// running received estimate so the user sees "↑3.4k ↓120…" climb during
		// long planner / reasoning stretches instead of a frozen "sent…".
		if genChars > 0 && (lastMeterChars == 0 || genChars-lastMeterChars >= statusMeterChars) {
			firstByteOnce.Do(func() { close(firstByte) }) // stop the waiting meter
			lastMeterChars = genChars
			a.setStatus(ctx, sid, fmt.Sprintf(" (llm[%s] ↑%s ↓%s…)", slotLabel, upLabel, fmtTokens(genChars/4)))
		}
	}
	if err := scanner.Err(); err != nil {
		writeResponseLog(logF, connLabel, fullText.String(), reasoningText.String(), calls, promptTokens, completionTokens, err)
		return fullText.String(), calls, fmt.Errorf("reading SSE stream: %w", err)
	}

	// Record the server's reported prompt_tokens on the session so
	// compressHistory can trigger on ground truth. Only the main session
	// owns the foreground prefix cache, but storing on every session is
	// harmless and keeps subagent telemetry honest. sid="" (probes/tests)
	// has no session, skip.
	if promptTokens > 0 {
		if sess := a.getSession(sid); sess != nil {
			sess.SetLastCompletePromptTokens(promptTokens)
		}
	}

	// finish_reason="length" means the server stopped because we hit
	// max_tokens, not because the model emitted a stop token. The output is
	// truncated mid-thought (or mid-tool-call JSON) and almost always means
	// the model was looping or rambling. Bail out cleanly rather than
	// retrying — the planner's JSON corrective retry is for malformed
	// prose, not for runaways, and prompt.go's retry loop would just hit
	// the same wall on the next attempt. The reasoning-byte count is
	// surfaced so an "empty response" cap-hit is recognisable as "the model
	// spent its whole budget thinking" vs an actual runaway in the visible
	// stream.
	if finishReason == "length" {
		err := fmt.Errorf("LLM hit max_tokens cap (role=%s, model=%s) — response truncated (%d B content, %d B reasoning, %d tool calls). Likely the model is looping or stuck in <think>; raise max_tokens in params_%s, or set chat_template_kwargs.enable_thinking=false for this role if reasoning is dominating the budget",
			conn.Tag, conn.Model, fullText.Len(), reasoningText.Len(), len(calls), conn.Tag)
		writeResponseLog(logF, connLabel, fullText.String(), reasoningText.String(), calls, promptTokens, completionTokens, err)
		return fullText.String(), calls, err
	}

	writeResponseLog(logF, connLabel, fullText.String(), reasoningText.String(), calls, promptTokens, completionTokens, nil)
	return fullText.String(), calls, nil
}

// statusMeterChars is how many generated bytes accumulate between live
// token-meter refreshes (~20 tokens at chars/4). Small enough that the count
// visibly climbs, large enough that we re-emit the plan a few dozen times for
// a long response instead of once per SSE event.
const statusMeterChars = 80

// llmStallWarnSeconds is how long the LLM may stay silent (no first token)
// before the waiting meter emits a one-line "server may be busy" warning. The
// request is NOT aborted — a large prompt on a slow model can legitimately take
// this long; the warning just tells the user the wait is on the server, not a
// freeze.
const llmStallWarnSeconds = 60

// streamWaitMeter ticks an elapsed-seconds suffix onto the active phase row
// while llmStream waits for the first token, so a busy/queuing server reads as
// "(sent… 25s)" instead of a frozen "(sent…)". After llmStallWarnSeconds of
// total silence it emits one soft warning (foreground turns only). Returns when
// the first byte arrives (firstByte closed), the call ends (done closed), or
// ctx is cancelled — it never aborts the request.
func (a *agent) streamWaitMeter(ctx context.Context, sid, slotLabel, upLabel string, start time.Time, firstByte, done <-chan struct{}) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	warned := false
	for {
		select {
		case <-firstByte:
			return
		case <-done:
			return
		case <-ctx.Done():
			return
		case <-ticker.C:
			elapsed := int(time.Since(start).Seconds())
			a.setStatus(ctx, sid, fmt.Sprintf(" (llm[%s] ↑%s sent… %ds)", slotLabel, upLabel, elapsed))
			if !warned && elapsed >= llmStallWarnSeconds && a.phaseActive(sid) {
				warned = true
				a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("⚠ No response from llm[%s] after %ds — the server may be busy or queuing the request. Still waiting (no timeout)…\n", slotLabel, elapsed)}})
			}
		}
	}
}

// fmtKB renders a byte count as kilobytes with two decimals for the upload
// meter, e.g. 12_636 → "12.34kb". Uses 1024-byte KiB; always "kb" (no MB
// rollover) so the cumulative upload total reads consistently as it climbs.
// Display-only.
func fmtKB(n int64) string {
	return fmt.Sprintf("%.2fkb", float64(n)/1024)
}

// fmtTokens renders an approximate token count compactly for the live status
// meter: <1000 prints as-is, ≥1000 collapses to a "k" suffix (1234 → "1.2k",
// 12000 → "12k"). Display-only — never used for budget or compaction math.
func fmtTokens(n int) string {
	if n < 1000 {
		return strconv.Itoa(n)
	}
	v := float64(n) / 1000
	if v >= 10 {
		return fmt.Sprintf("%.0fk", v)
	}
	return fmt.Sprintf("%.1fk", v)
}

// writeResponseLog emits one compact RESPONSE block per LLM call. The raw SSE
// wire is one event per token (~100 lines for a short reply, thousands for a
// tool-call argument blob) — useless for skimming. This collapses all the
// deltas into the reconstructed text and tool calls so the log reads like a
// transcript instead of a packet capture. promptTokens/completionTokens are
// the server's exact counts from the trailing usage chunk (0 when the backend
// didn't report them).
func writeResponseLog(logF *os.File, connLabel, text, reasoning string, calls []toolCall, promptTokens, completionTokens int, streamErr error) {
	if logF == nil {
		return
	}
	fmt.Fprintf(logF, "=== [%s] RESPONSE ===\n", connLabel)
	if promptTokens > 0 || completionTokens > 0 {
		fmt.Fprintf(logF, "tokens: prompt=%d completion=%d\n", promptTokens, completionTokens)
	}
	if reasoning != "" {
		fmt.Fprintf(logF, "reasoning_content (%d B):\n%s\n", len(reasoning), reasoning)
	}
	if text != "" {
		fmt.Fprintf(logF, "content:\n%s\n", text)
	}
	for i, c := range calls {
		fmt.Fprintf(logF, "tool_call[%d] %s id=%s args=%s\n", i, c.Function.Name, c.ID, c.Function.Arguments)
	}
	if text == "" && len(calls) == 0 && reasoning == "" {
		fmt.Fprintln(logF, "(empty response)")
	}
	if streamErr != nil {
		fmt.Fprintf(logF, "[stream error] %v\n", streamErr)
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
	// ContextSize is the model's max prompt+output tokens as reported by
	// /props (n_ctx) or /v1/models launch args (--ctx-size / -c). 0 means
	// unknown — ensureLLM blocks the session on a Retry card until the
	// server starts reporting this.
	ContextSize int
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
	// Enrich missing metadata from /props when /v1/models was bare. Skip
	// when /v1/models was unreachable too and try /props on its own — some
	// llama-server builds don't expose /v1/models.
	if r.Reachable && (!r.ImageSupport || r.ContextSize == 0) {
		if p, ok := a.probeViaProps(probeCtx, conn); ok {
			if !r.ImageSupport {
				r.ImageSupport = p.ImageSupport
			}
			if r.ContextSize == 0 {
				r.ContextSize = p.ContextSize
			}
		}
	} else if !r.Reachable {
		if p, ok := a.probeViaProps(probeCtx, conn); ok {
			r = p
		}
	}

	if r.Reachable {
		slog.Info("probeLLM", "model", conn.Model, "loaded", r.ModelLoaded, "image", r.ImageSupport, "ctx", r.ContextSize)
	} else {
		slog.Info("probeLLM: unreachable", "url", conn.URL, "model", conn.Model)
	}
	return r
}

// probeViaModels asks /v1/models for the configured model. Always confirms
// reachability + model presence; image_support / context_size only land
// when the response carries llama-swap-style `status.args` (--mmproj /
// --ctx-size). OpenAI/Ollama/vLLM/LiteLLM all 200 here but return the bare
// OpenAI shape, so the caller's /props enrichment + settings.toml fallback
// fills the gap. ok=false only on network / non-200 — a bare response still
// returns ok=true so the caller knows the server is up.
func (a *agent) probeViaModels(ctx context.Context, conn *LLMConnection) (probeResult, bool) {
	modelsURL, ok := deriveServerURL(conn.URL, "/v1/models")
	if !ok {
		return probeResult{}, false
	}
	resp, err := getWithAuth(ctx, modelsURL, conn.APIKey)
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
		break
	}
	return r, true
}

// probeViaProps reads llama-server's /props. Tells us the server is reachable
// and (via modalities.vision) whether the loaded model supports image input,
// but cannot tell us *which* model is loaded — so ModelKnown stays false.
func (a *agent) probeViaProps(ctx context.Context, conn *LLMConnection) (probeResult, bool) {
	propsURL, ok := deriveServerURL(conn.URL, "/props")
	if !ok {
		return probeResult{}, false
	}
	resp, err := getWithAuth(ctx, propsURL, conn.APIKey)
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
		// llama.cpp's /props exposes n_ctx in two shapes across releases:
		// recent builds nest it under default_generation_settings, older
		// ones surface it at the top level. We accept either.
		DefaultGenerationSettings *struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
		NCtx int `json:"n_ctx"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&props); err != nil {
		return probeResult{}, false
	}
	r := probeResult{Reachable: true}
	if props.Modalities != nil {
		r.ImageSupport = props.Modalities.Vision
	}
	if props.DefaultGenerationSettings != nil {
		r.ContextSize = props.DefaultGenerationSettings.NCtx
	}
	if r.ContextSize == 0 {
		r.ContextSize = props.NCtx
	}
	return r, true
}

// pickAvailable resolves the connection for the session and role. The main
// session always uses LLM[0] — its KV cache owns the parent's conversation
// prefix. Subagent sessions carry a PinnedLLMIdx assigned at launch by
// launch_subagent; every call from that session routes back to the same
// entry so the conn's prefix cache stays warm across plan/execute switches.
// Concurrency is enforced by per-conn semaphores in llmStream; this picker
// just resolves the routing target.
func (a *agent) pickAvailable(_ context.Context, sid string, role string) *LLMConnection {
	sess := a.getSession(sid)
	if sess != nil && sess.Depth > 0 && sess.PinnedLLMIdx >= 0 {
		if c := a.settings.ConnAt(sess.PinnedLLMIdx, role); c != nil {
			return c
		}
	}
	return a.settings.MainLLM(role)
}

// pickBackgroundLLM returns the connection that should host background work
// (per-turn summariser, git-commit, document). Walks LLM[1..x] — the
// background tier — and returns the first entry whose semaphore reports
// free capacity, so a 3-LLM setup spreads concurrent calls across servers
// instead of stacking them on LLM[1]. When every background entry is busy
// we fall back to LLM[0]: ensureLLM has gated startup on parallelCap >= 2
// so its server's slot allocator gives background a different KV cache
// slot, and one cache-pollution turn is preferable to stalling the
// background queue waiting for a [1..x] slot. The free-capacity peek is
// racy by design — by the time llmStream re-acquires the same slot,
// another caller may have grabbed it; llmStream's per-conn semaphore
// queues in that case, which is still no worse than the simple "always
// LLM[1]" fallback. Session-independent: every caller is on the foreground
// path (call sites filter on Depth==0), and the picking is the same
// regardless of which session triggered it.
func (a *agent) pickBackgroundLLM() *LLMConnection {
	for i := 1; i < len(a.settings.LLM); i++ {
		if i < len(a.slotSems) && a.slotSems[i] != nil &&
			len(a.slotSems[i]) < cap(a.slotSems[i]) {
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

// connSlot returns the index into settings.LLM that this connection
// corresponds to, used by llmStream to find the matching semaphore. Returns
// -1 when the conn is not in the configured list (test mocks, probe-only
// stubs); the caller treats -1 as "no semaphore" and dispatches directly.
func (a *agent) connSlot(conn *LLMConnection) int {
	for i := range a.settings.LLM {
		if a.settings.LLM[i].URL == conn.URL && a.settings.LLM[i].Model == conn.Model {
			return i
		}
	}
	return -1
}

// buildSlotSems sizes one buffered channel per LLM entry to its parallelCap.
// Called once on agent startup after settings load. Re-init on settings
// reload is handled by the caller (ensureLLM invokes loadSettings →
// re-init slotSems).
func (a *agent) buildSlotSems() {
	sems := make([]chan struct{}, len(a.settings.LLM))
	for i := range a.settings.LLM {
		sems[i] = make(chan struct{}, a.settings.LLM[i].parallelCap())
	}
	a.slotSems = sems
}

// getWithAuth issues a GET with an optional bearer token and follows redirects
// (http.DefaultClient follows up to 10 by default, needed for http→https).
func getWithAuth(ctx context.Context, rawURL, token string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", rawURL, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return http.DefaultClient.Do(req)
}

// deriveServerURL strips a chat-completions-style path from rawURL and appends
// suffix. Handles "…/v1/chat/completions" and bare "…/chat/completions".
func deriveServerURL(rawURL, suffix string) (string, bool) {
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" {
		return "", false
	}
	path := u.Path
	if i := strings.Index(path, "/v1/"); i >= 0 {
		path = path[:i]
	} else if j := strings.LastIndex(path, "/chat/completions"); j >= 0 {
		path = path[:j]
	}
	u.Path = strings.TrimRight(path, "/") + suffix
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), true
}

// llmSimple sends a no-tools LLM call, logs to stderr. sid scopes per-session
// debug logging (see llmStream); pass "" for unscoped calls.
func (a *agent) llmSimple(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage) (string, error) {
	text, _, err := a.llmStream(ctx, sid, conn, messages, nil, func(token string) {
		fmt.Fprint(os.Stderr, token)
	}, nil)
	fmt.Fprintln(os.Stderr)
	return text, err
}

// parallel runs fn for each index [0, n) with up to `cap` concurrent
// goroutines. Callers pass an explicit upper bound matched to the work-list
// (e.g. launch_subagent's SubagentPinOrder length, probeAllLLMs's len(conns))
// so excess work queues instead of contending for slots.
func parallel(n, cap int, fn func(i int)) {
	if cap > n {
		cap = n
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, cap)
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			fn(i)
		}(i)
	}
	wg.Wait()
}
