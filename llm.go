package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
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
	for k, v := range conn.ExtraBody {
		reqBody[k] = v
	}
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

	// Drive the lifecycle suffix on the active phase entry across the LLM
	// round-trip: "(sent to llm…)" while we wait for the first SSE token (cache
	// misses, queueing on a busy llama-server, slow TTFT), then flip to
	// "(thinking…)" the moment streaming starts. setStatus is a no-op when no
	// phase is active so summarisation calls between phases don't flash
	// anything.
	// sentEst is a rough chars/4 estimate of the prompt size for the live
	// token meter — the server's exact prompt_tokens only arrives in the
	// trailing usage chunk, too late to show while we wait for the first
	// token. len(body) includes the tool-schema JSON, so this slightly
	// overcounts vs the tokenised prompt; it's display-only, never used for
	// budget math.
	sentEst := len(body) / 4
	a.setStatus(ctx, sid, fmt.Sprintf(" (↑%s sent…)", fmtTokens(sentEst)))
	defer a.setStatus(ctx, sid, "")

	// Per-session log: request header + body. The raw SSE wire is too noisy
	// to read (one event per token, each repeating the full envelope), so we
	// parse the stream and write a single aggregated RESPONSE block at the
	// end. Mid-stream events (HTTP errors, partial state on transport
	// failure) are appended inline below. Closed on return.
	// Header records llm[<slot>] alongside the role+model so a grep for
	// "llm[1]" surfaces every request routed to a given entry — useful for
	// confirming that document / summarise / git_commit landed off the
	// foreground prefix cache. slot=-1 (test mocks, probes) renders as "?".
	slotLabel := "?"
	if slot >= 0 {
		slotLabel = fmt.Sprintf("%d", slot)
	}
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

	resp, err := http.DefaultClient.Do(httpReq)
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
		var apiErr map[string]interface{}
		if jsonErr := json.Unmarshal(bodyBytes, &apiErr); jsonErr == nil {
			if errMsg, ok := apiErr["error"]; ok {
				if errMap, ok := errMsg.(map[string]interface{}); ok {
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
			lastMeterChars = genChars
			a.setStatus(ctx, sid, fmt.Sprintf(" (↑%s ↓%s…)", fmtTokens(sentEst), fmtTokens(genChars/4)))
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
			switch {
			case arg == "--mmproj":
				r.ImageSupport = true
			case arg == "--ctx-size" || arg == "-c":
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
	return a.settings.ConnAt(0, "execute")
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
	for i := 0; i < n; i++ {
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

// trimJSON extracts a JSON object from an LLM response. Small models often
// wrap the JSON in prose ("Sure, here's the JSON: { … } Let me know!") or
// markdown fences; we just locate the first `{` and the matching `}` and
// keep that slice. Brace counting respects strings + escapes so braces inside
// string values don't confuse the scan. Returns the trimmed input unchanged
// if no balanced object is found — caller surfaces the parse error.
func trimJSON(s string) string {
	s = strings.TrimSpace(s)
	start := strings.IndexByte(s, '{')
	if start < 0 {
		return s
	}
	depth := 0
	inStr := false
	esc := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}
	return s
}

type toolLoopResult struct {
	Text     string
	ToolUses []ToolUse
	// RespondCalled is true when the loop exited because the model invoked
	// the registered terminal tool (typically `respond`). False when the
	// loop exited via the legacy empty-tool-calls path, hit the soft cap,
	// or returned an error. Subtask runners use this as the primary
	// success signal — a loop that ran out of turns without calling
	// `respond` is a failed subtask regardless of what's in res.Text.
	RespondCalled bool
	// StartedAt is when the first llmStream call of this loop began.
	// DurationMs is the cumulative wall-clock time spent in llmStream calls
	// across all iterations (excludes tool execution). Phase is the pipeline
	// stage tag passed in by the caller ("plan", "execute", "document",
	// "subagent"). Callers that own the final assistant message (prompt.go,
	// runSubagent) use these to stamp the message they create after the
	// loop returns.
	StartedAt  time.Time
	DurationMs int64
	Phase      string
}

// maxToolLoopIterations bounds runToolLoop so a model that keeps producing
// "different enough" tool calls (e.g. path variations to dodge perceived
// repetition) can't spin forever. One iteration is one LLM round-trip; a
// complex execute pass is usually 10-20, so 100 leaves comfortable headroom
// for unusually long but legitimate runs while still bailing on genuine
// runaways. The signature nudge and per-name escalation above catch the
// common stuck patterns earlier, so this cap is the last-resort backstop.
const maxToolLoopIterations = 100

// toolNameEscalateThreshold is the number of *redundant* calls to a single
// tool name in one loop after which the connection switches from the
// "execute" role to "thinking". Same server (so the KV prefix cache stays
// warm — sampler params never enter the cache key), warmer sampler. A call
// is redundant when its (name, arguments) pair has already been seen this
// loop; legitimate fan-out across distinct files/queries (e.g. surveying
// every go.mod in a multi-module repo) does NOT count, so broad read
// passes no longer trip the escalation. Complementary to the signature-
// based nudge above: that one catches byte-for-byte consecutive repeats,
// this one catches interleaved revisits to the same args. One-shot per
// loop.
const toolNameEscalateThreshold = 5

// toolCallSig produces a stable signature for the tool calls emitted in one
// iteration: byte-for-byte name + arguments, joined when there's more than
// one. Used to spot tight repetition where the model keeps re-running the
// same call (same `grep`, same `read_file`) without doing anything with the
// result — the most common pathology when small models get stuck.
func toolCallSig(calls []toolCall) string {
	if len(calls) == 0 {
		return ""
	}
	var b strings.Builder
	for i, c := range calls {
		if i > 0 {
			b.WriteByte('|')
		}
		b.WriteString(c.Function.Name)
		b.WriteByte('(')
		b.WriteString(c.Function.Arguments)
		b.WriteByte(')')
	}
	return b.String()
}

// failureSkillHint returns a short pointer to SKILL-*.md files relevant to the
// failed tool, or "" when no skill docs exist or the tool isn't one of the
// shell-style ones where skills typically apply. Used by runToolLoopOn to nudge
// small models toward reading the skill docs after a hard failure instead of
// blindly retrying. Returns paths relative to cwd so the model's read_file
// call works without further escaping.
func (a *agent) failureSkillHint(sid string, toolName string) string {
	// Only fire on tools where a SKILL doc plausibly helps. Edit-failures
	// usually mean the model passed wrong content, not that it needs a skill.
	switch toolName {
	case "run_command", "run_task":
	default:
		return ""
	}
	sess := a.getSession(sid)
	if sess == nil {
		return ""
	}
	entries, err := os.ReadDir(filepath.Join(sess.Cwd, ".codehalter"))
	if err != nil {
		return ""
	}
	var skills []string
	for _, e := range entries {
		n := e.Name()
		if strings.HasPrefix(n, "SKILL-") && strings.HasSuffix(n, ".md") {
			skills = append(skills, ".codehalter/"+n)
		}
	}
	if len(skills) == 0 {
		return ""
	}
	sort.Strings(skills)
	return "[Hint: this command failed. The project ships skill docs that cover this kind of failure — read one with read_file before retrying: " +
		strings.Join(skills, ", ") + "]"
}

// runToolLoop runs the agentic tool loop with the default token-streaming
// callback so the user sees execute / document phases live. The internal
// JSON phases (planner) call runToolLoopOn directly with a no-op. softCap
// is the per-call soft iteration cap (0 = use the maxToolLoopIterations
// hard backstop only); when the soft cap is hit, the loop exits gracefully
// with RespondCalled=false so the caller can treat it as a failed turn.
func (a *agent) runToolLoop(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, filter toolFilter, phase string, softCap int) (toolLoopResult, error) {
	on := func(token string) {
		if sid != "" {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: token}})
		}
	}
	think := func(token string) {
		if sid != "" {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentThought, Content: ContentBlock{Type: "text", Text: token}})
		}
	}
	return a.runToolLoopOn(ctx, sid, conn, messages, filter, phase, on, think, softCap)
}

// runToolLoopOn is the core agentic tool loop: send to LLM, execute tool
// calls, repeat. Callers supply the per-token callback (default streaming
// for runToolLoop, no-op for the planner's JSON pass in runPlanLLM). The phase string ("plan",
// "execute", "document", "subagent") flows onto the trailing assistant
// message via MarkLastAssistantTiming together with the cumulative
// llmStream wall-clock — so session.toml records who ran the turn and how
// much of its time was model generation vs tool execution.
//
// softCap is an optional per-call soft iteration ceiling. When > 0 and the
// loop reaches it without the terminal tool firing, the function returns
// (res, nil) with RespondCalled=false instead of erroring. 0 means "no
// soft cap — only the maxToolLoopIterations runaway backstop applies".
func (a *agent) runToolLoopOn(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, filter toolFilter, phase string, on, think func(string), softCap int) (toolLoopResult, error) {
	tools := llmToolDefinitionsFiltered(filter)

	// termName is the registered Terminal tool exposed in this phase ("" when
	// none — e.g. plan/verify/document filter respond out). When non-empty,
	// the empty-tool-calls branch below stops meaning "model finished" and
	// starts meaning "model dropped out of tool-calling grammar" — see the
	// nudge + fallback there.
	termName := terminalToolName(filter)
	hasTerminal := termName != ""

	var res toolLoopResult
	res.Phase = phase
	var allText strings.Builder
	var genElapsed time.Duration
	// stampTiming applies the accumulated start/duration/phase to the
	// trailing assistant message in session. Called on every exit path so
	// even error returns leave a recorded turn for postmortem analysis.
	stampTiming := func() {
		if res.StartedAt.IsZero() {
			return
		}
		res.DurationMs = genElapsed.Milliseconds()
		if sess := a.getSession(sid); sess != nil {
			sess.MarkLastAssistantTiming(res.StartedAt, res.DurationMs, phase)
		}
	}
	// Repetition state: track the signature of the previous iteration's tool
	// calls. Two consecutive identical signatures inject a corrective and
	// give the model one chance to break out; a third (post-nudge) bails.
	// A different signature resets the streak — natural variation isn't
	// punished, only tight loops.
	var lastSig string
	var sameSigCount int
	var nudged bool
	// Per-name cumulative counts across the whole loop. When any single tool
	// name crosses toolNameEscalateThreshold we swap conn to the "thinking"
	// role on the same server (sampler-only change; KV prefix cache key is
	// derived from prompt tokens, not sampler params, so the cache stays
	// warm). One-shot per loop — if the warmer sampler doesn't help, the
	// signature nudge / 50-iter cap will still bail us out.
	toolNameCounts := map[string]int{}
	// toolArgSeen[name] is the set of (already-seen) argument strings for a
	// given tool name. The per-name escalation only counts a call as
	// redundant when its args have been seen before — fan-out across
	// distinct files (e.g. read_file on go.mod, examples/go.mod, …) doesn't
	// trip the escalation; only genuine revisits do.
	toolArgSeen := map[string]map[string]bool{}
	var escalated bool
	// respondNudged: when respondEnabled and the model returns an empty tool
	// call list, we nudge once to call respond. If the next turn still has no
	// tool calls we fall through to the legacy text-only exit so the loop
	// can't spin forever on a model that refuses the synthetic terminal.
	var respondNudged bool
	for iter := 0; ; iter++ {
		if iter >= maxToolLoopIterations {
			res.Text = allText.String()
			stampTiming()
			return res, fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations)
		}
		// Soft cap: exit gracefully with RespondCalled=false so the caller
		// (e.g. the subtask runner) can treat "ran out of turns without
		// calling respond" as a failure to replan against, rather than as
		// a hard error.
		if softCap > 0 && iter >= softCap {
			res.Text = allText.String()
			stampTiming()
			return res, nil
		}
		streamStart := time.Now()
		if res.StartedAt.IsZero() {
			res.StartedAt = streamStart
		}
		text, calls, err := a.llmStream(ctx, sid, conn, messages, tools, on, think)
		genElapsed += time.Since(streamStart)
		if err != nil {
			res.Text = allText.String()
			stampTiming()
			return res, err
		}
		allText.WriteString(text)

		if len(calls) == 0 {
			// Terminal-tool mode: empty tool_calls means the model dropped
			// out of tool-calling grammar instead of finishing. Nudge it to
			// either call the terminal tool or another tool, but only once —
			// a model that refuses twice gets the legacy text exit so we
			// don't loop indefinitely on the new constraint.
			if hasTerminal && !respondNudged {
				respondNudged = true
				messages = append(messages, llmMessage{Role: "assistant", Content: text})
				messages = append(messages, llmMessage{
					Role: "user",
					Content: fmt.Sprintf("Your last response was plain text with no tool call. "+
						"This turn ends only when you call `%s` with your final "+
						"user-facing message, or another tool if you still have work "+
						"to do. Do not reply in prose — call a tool.", termName),
				})
				continue
			}
			res.Text = allText.String()
			stampTiming()
			return res, nil
		}

		sig := toolCallSig(calls)
		if sig != "" && sig == lastSig {
			sameSigCount++
		} else {
			sameSigCount = 1
			nudged = false
		}
		lastSig = sig

		// Third identical call after a nudge → give up. The model had its
		// recovery chance and didn't take it; further iterations will burn
		// the same time. Surfaces as a clean error rather than waiting for
		// the 50-iter cap.
		if sameSigCount >= 3 && nudged {
			res.Text = allText.String()
			stampTiming()
			return res, fmt.Errorf("tool loop stuck on identical call after nudge: %s", truncate(sig, 200))
		}
		// Second identical call → nudge once. We still EXECUTE the duplicate
		// (legitimate read-after-write needs the fresh read to land, even
		// when its sig matches the prior read), but tack on a user-role
		// corrective at the end of this iteration so the next LLM round-trip
		// sees a break-out instruction alongside the (possibly unchanged)
		// tool result.
		nudgeThisIter := sameSigCount == 2 && !nudged
		if nudgeThisIter {
			nudged = true
			if sid != "" {
				a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⚠ Repeated tool call detected — nudging the model to try a different action.\n"}})
			}
		}

		messages = append(messages, llmMessage{
			Role:      "assistant",
			Content:   text,
			ToolCalls: calls,
		})

		// terminalCalled flips when this batch contains a Terminal tool call;
		// we finish processing the batch (so its tool result lands in history
		// for postmortem) and then exit with the message as res.Text.
		var terminalCalled bool
		var terminalMessage string

		for _, tc := range calls {
			// Subagent sessions don't surface their own UI (Zed doesn't know
			// their sid), so forward a one-liner to the parent before each
			// tool call. Gives the user a live feed of what each subagent is
			// up to instead of just "Starting…" → 5 minutes → "Done".
			if sess := a.getSession(sid); sess != nil && sess.ParentID != "" && sess.DisplayLabel != "" {
				a.sendUpdate(ctx, sess.ParentID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("[%s] %s %s\n\n", sess.DisplayLabel, tc.Function.Name, truncate(tc.Function.Arguments, 80))}})
			}
			a.setStatus(ctx, sid, " (running "+tc.Function.Name+"…)")
			started := time.Now()

			// view_image short-circuit: when the server supports images, we
			// read the bytes off the content-addressed store and deliver them
			// as multimodal tool content in the SAME turn (so the next
			// llmStream call sees the image). The standard executeTool path
			// only returns text, which would defeat the point.
			var result string
			var failed bool
			var multimodalContent any
			if tc.Function.Name == "view_image" && a.imagesSupported {
				sess := a.getSession(sid)
				text, parts, ferr := dispatchViewImage(sess, tc.Function.Arguments)
				result = text
				failed = ferr
				if !ferr {
					multimodalContent = parts
				}
			} else {
				result, failed = a.executeTool(ctx, sid, tc)
			}

			if hasTerminal && tc.Function.Name == termName && !terminalCalled {
				terminalCalled = true
				terminalMessage = result
			}
			useID := nextToolUseID()
			tu := ToolUse{
				ID:         useID,
				Name:       tc.Function.Name,
				Input:      tc.Function.Arguments,
				Output:     result,
				Failed:     failed,
				StartedAt:  started,
				DurationMs: time.Since(started).Milliseconds(),
			}
			res.ToolUses = append(res.ToolUses, tu)

			// Save incrementally so tool results survive crashes.
			if sess := a.getSession(sid); sess != nil {
				sess.AppendToolUse(tu)
				_ = sess.Save()
			}
			// Shrink the model-visible tool result before it enters the
			// message stream: anything past truncateThreshold is replaced
			// with head/tail + a per-tool "to see more" hint. session.toml
			// still records the full output via tu.Output above — only the
			// in-flight messages[] that get re-sent every turn shrink. The
			// hint embeds useID so the model can `view_output id=useID …`
			// to retrieve any portion of the original without re-running.
			content := truncateForLLM(useID, tc.Function.Name, tc.Function.Arguments, result)
			// Small models routinely keep retrying a failing run_command
			// without consulting the SKILL-*.md docs that were loaded into
			// the system prompt at session start. Re-surface them at the
			// moment the model would benefit most: right after the failure.
			// Hint is appended to the live tool result only — not to the
			// stored ToolUse.Output — so session.toml stays clean.
			if failed {
				if hint := a.failureSkillHint(sid, tc.Function.Name); hint != "" {
					content = content + "\n\n" + hint
				}
			}
			var toolContent any = content
			if multimodalContent != nil {
				toolContent = multimodalContent
			}
			messages = append(messages, llmMessage{
				Role:       "tool",
				Content:    toolContent,
				ToolCallID: tc.ID,
			})
		}

		// Per-LLM-call progress fan-out: fire summary + git_commit after each
		// iteration's tool batch. Without this, a planner that spends 17
		// minutes in one tool loop produces zero shadow notes and a stale
		// .codehalter/.git_commit — the existing Prompt() epilogue only runs
		// after the whole task finishes. Summariser enqueues every fire so a
		// 50-iteration loop produces 50 progress notes; git commit coalesces
		// via gitCommitJob (bgJob) since gitCommitLastHash already dedupes
		// identical snapshots. Subagents (Depth>0) skip — they already route
		// via their pinned slot and don't own the shadow buffer.
		if sess := a.getSession(sid); sess != nil && sess.Depth == 0 {
			a.backgroundSummarise(sess)
			a.backgroundGitCommit(sess)
		}

		// Terminal tool called: stream the message to the UI as one chunk
		// (the model emitted it as tool arguments, which never went through
		// the text-stream callback) and exit. The same-sig nudge and per-name
		// escalation below are skipped — this turn is over.
		if terminalCalled {
			if on != nil && terminalMessage != "" {
				on(terminalMessage)
			}
			res.Text = terminalMessage
			res.RespondCalled = true
			stampTiming()
			return res, nil
		}

		// If this iteration was flagged as a repeat, append a corrective
		// user message after the tool results so the model sees the nudge
		// alongside the (often unchanged) output it just got back.
		if nudgeThisIter {
			messages = append(messages, llmMessage{
				Role: "user",
				Content: "You just repeated the same tool call with the same arguments. Re-running rarely produces new information. Either:\n" +
					"1. Act on the output you already have (edit a file, run a different command, or summarise your finding), or\n" +
					"2. If you are stuck or the task is infeasible, say so and stop.\n\n" +
					"Do not call the same tool with the same arguments again unless you have first changed state that the call observes (e.g. a file you just edited).",
			})
		}

		// Per-name escalation: redundant calls to the same tool (same args
		// seen before) mean the sampler is too cold to abandon a stuck plan.
		// Distinct args don't count — surveying many files via read_file
		// doesn't trip this. Switch to the "thinking" role's params — same
		// URL/model so the prefix cache survives — and let the warmer
		// sampler pick a different action. One-shot per loop. Skip entirely
		// when conn.Tag is already "thinking" (plan phase): the swap would be
		// a no-op and the warning would mislead.
		if !escalated && conn != nil && conn.Tag != "thinking" {
			for _, c := range calls {
				seen := toolArgSeen[c.Function.Name]
				if seen == nil {
					seen = map[string]bool{}
					toolArgSeen[c.Function.Name] = seen
				}
				if seen[c.Function.Arguments] {
					toolNameCounts[c.Function.Name]++
				} else {
					seen[c.Function.Arguments] = true
				}
			}
			for name, n := range toolNameCounts {
				if n < toolNameEscalateThreshold {
					continue
				}
				thinkConn := a.pickAvailable(ctx, sid, "thinking")
				if thinkConn == nil {
					break
				}
				conn = thinkConn
				escalated = true
				if sid != "" {
					a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("⚠ Tool '%s' invoked %d× this loop — switching to thinking sampler to break out.\n", name, n)}})
				}
				break
			}
		}
	}
}
