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
	"strconv"
	"strings"
	"sync/atomic"
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

	// Per-conn concurrency gate: cap in-flight calls to this conn at its
	// configured `parallel`. The token is held only for this call (released on
	// return), not a subagent's lifetime — so between calls the conn frees up,
	// and pool size 1 serialises without deadlocking nested subagents; the wait
	// shows as "(queued…)". Find the conn's semaphore index by matching
	// server+model; -1 (test mocks / probes not in settings.LLM) means "no gate,
	// dispatch directly".
	slot := -1
	for i := range a.settings.LLM {
		if a.settings.LLM[i].Server == conn.Server && a.settings.LLM[i].Model == conn.Model {
			slot = i
			break
		}
	}
	if slot >= 0 && slot < len(a.connSems) {
		// Try non-blocking first; only emit the queued suffix when we're
		// actually about to wait. Avoids flashing the wrong status on the
		// common hot path where the slot is free.
		select {
		case a.connSems[slot] <- struct{}{}:
		default:
			a.setStatus(ctx, sid, " (queued…)")
			select {
			case a.connSems[slot] <- struct{}{}:
			case <-ctx.Done():
				a.setStatus(ctx, sid, "")
				return "", nil, ctx.Err()
			}
		}
		defer func() { <-a.connSems[slot] }()
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

	// One meter, one cadence: streamWaitMeter refreshes the phase row once per
	// second for the whole call. Before the first token it shows "(sent… Ns)" so
	// a busy/queuing server isn't a frozen "(sent…)"; once bytes arrive it shows
	// the live "↑up ↓tokens…" estimate. genChars is the running generated-byte
	// count it reads each tick — written by the scanner loop below, read by the
	// meter goroutine, so it's atomic. There's no client timeout (a slow
	// generation is legitimate); if the server drops the connection the scanner
	// surfaces the transport error directly.
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
		return "", nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if conn.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
	}

	// http.DefaultClient already has no Client.Timeout and DefaultTransport
	// leaves ResponseHeaderTimeout at 0, so a long generation or a server that
	// queues the request before answering is never capped — cancellation is
	// driven by the request ctx only. The 30s dial timeout (TCP connect) is the
	// only ceiling, which is what we want: an unreachable server fails fast.
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		a.logSession(sid, connLabel, "[transport error] %v", err)
		return "", nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", nil, fmt.Errorf("HTTP %d, failed to read body: %w", resp.StatusCode, err)
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
		return "", nil, fmt.Errorf("LLM returned %d: %s [URL: %s]", resp.StatusCode, msg, resp.Request.URL.String())
	}

	var fullText strings.Builder
	var reasoningText strings.Builder
	var calls []toolCall
	var finishReason string
	var promptTokens, completionTokens int

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

	// Record the server's reported prompt_tokens on the session so
	// compressHistory can trigger on ground truth. Only the main session
	// owns the foreground prefix cache, but storing on every session is
	// harmless and keeps subagent telemetry honest. sid="" (probes/tests)
	// has no session, skip. On a scan failure the stream broke before the
	// usage chunk, so promptTokens is 0 and this is a no-op anyway.
	if scanErr == nil && promptTokens > 0 {
		if sess := a.getSession(sid); sess != nil {
			sess.SetLastCompletePromptTokens(promptTokens)
		}
	}

	// One outcome error, shared by the RESPONSE log block and the return:
	//   - scanErr: stream broke mid-flight (e.g. a router model swap force-kills
	//     the connection).
	//   - finish_reason="length": hit max_tokens, not a stop token — output is
	//     truncated and usually means the model looped. We bail rather than retry
	//     (prompt.go's retry would hit the same wall); the message guides tuning.
	switch {
	case scanErr != nil:
		err = fmt.Errorf("reading SSE stream: %w", scanErr)
	case finishReason == "length":
		err = fmt.Errorf("LLM hit max_tokens cap (role=%s, model=%s) — response truncated (%d B content, %d B reasoning, %d tool calls). Likely the model is looping or stuck in <think>; raise max_tokens in params_%s, or set chat_template_kwargs.enable_thinking=false for this role if reasoning is dominating the budget",
			conn.Tag, conn.Model, fullText.Len(), reasoningText.Len(), len(calls), conn.Tag)
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
	if promptTokens > 0 || completionTokens > 0 {
		fmt.Fprintf(&rb, "tokens: prompt=%d completion=%d\n", promptTokens, completionTokens)
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
	return fullText.String(), calls, err
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
				// approx tokens, compact: <1000 as-is, else "k" suffix
				// (1234 → "1.2k", 12000 → "12k"). Display-only.
				n := int(g) / 4
				tok := strconv.Itoa(n)
				if n >= 1000 {
					if v := float64(n) / 1000; v >= 10 {
						tok = fmt.Sprintf("%.0fk", v)
					} else {
						tok = fmt.Sprintf("%.1fk", v)
					}
				}
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
	// Always enrich via /props when reachable: it carries image + ctx metadata
	// AND llama.cpp's per-slot n_ctx / total_slots, none of which /v1/models
	// exposes. Non-llama backends 404 here (ok=false) and we keep the
	// /v1/models result; /props also serves as a reachability fallback for
	// llama-server builds that don't expose /v1/models.
	if p, ok := a.probeViaProps(probeCtx, conn); ok {
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
// returns ok=true so the caller knows the server is up.
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
	propsURL := conn.endpoint("/props")
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
	if sess := a.getSession(sid); sess != nil && sess.Depth > 0 && sess.PinnedLLMIdx >= 0 {
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
// Called once on agent startup after settings load. Re-init on settings
// reload is handled by the caller (ensureLLM invokes loadSettings →
// re-init connSems).
func (a *agent) buildConnSems() {
	sems := make([]chan struct{}, len(a.settings.LLM))
	for i := range a.settings.LLM {
		sems[i] = make(chan struct{}, a.settings.LLM[i].parallelCap())
	}
	a.connSems = sems
}
