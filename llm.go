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

// ResponseHeaderTimeout bounds the hang when a network switch breaks the TCP
// connection. No Client.Timeout: generations stream unbounded until ctx cancel.
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

// No ResponseHeaderTimeout: a llama.cpp router answers /props?model= only once
// the model has loaded (minutes); the probe's ctx bounds that wait.
var metaHTTPClient = &http.Client{
	Transport: &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout: 10 * time.Second,
		IdleConnTimeout:     90 * time.Second,
	},
}

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
			// Kept out of Content: merging would break prefix-cache keys.
			ReasoningContent string `json:"reasoning_content"`
			// vLLM's spelling of reasoning_content.
			Reasoning string `json:"reasoning"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
	// llama.cpp only (timings_per_token): prompt_n = evaluated, cache_n = reused.
	Timings *struct {
		PromptN     int     `json:"prompt_n"`
		CacheN      int     `json:"cache_n"`
		PromptMs    float64 `json:"prompt_ms"`
		PredictedMs float64 `json:"predicted_ms"`
	} `json:"timings"`
	// An error delivered in-stream under HTTP 200 (llama.cpp, llama-swap, gateways);
	// both this nested shape and a bare "message" occur.
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
	Message string `json:"message"`
}

type llmHTTPError struct {
	Status int
	Body   string
	Type   string // the OpenAI-style error.type, when the body carried one
	URL    string
}

func (e *llmHTTPError) Error() string {
	return fmt.Sprintf("LLM returned %d: %s [URL: %s]", e.Status, e.Body, e.URL)
}

// llmStreamError is a refusal the server sent in the stream after a 200.
type llmStreamError struct{ Role, Model, Msg string }

func (e *llmStreamError) Error() string {
	return fmt.Sprintf("LLM returned an error mid-stream (role=%s, model=%s): %s", e.Role, e.Model, e.Msg)
}

// llmCallError marks a turn that failed because the model call did, not because of
// what the model did: /spec does not count it against the item.
type llmCallError struct{ err error }

func (e *llmCallError) Error() string { return e.err.Error() }
func (e *llmCallError) Unwrap() error { return e.err }

// finish=length below the max_tokens cap: the prompt fit but left no room, so it
// is recovered like a context-overflow 400.
var errContextCeiling = errors.New("generation hit the context ceiling")

// A 400 also answers request shapes a server refuses (llama.cpp's "Cannot continue
// an assistant message that contains tool calls"), which no fold can fix, so the
// body must be about the context. Halogen counts prompt + max_tokens against it.
func isContextFull(err error) bool {
	var he *llmHTTPError
	if errors.As(err, &he) {
		if he.Status == 413 || he.Type == "exceed_context_size_error" {
			return true
		}
		if he.Status != 400 {
			return false
		}
		msg := strings.ToLower(he.Body)
		for _, p := range []string{"context", "too many tokens", "too long", "maximum prompt"} {
			if strings.Contains(msg, p) {
				return true
			}
		}
		return false
	}
	// Halogen refuses more than 64 pictures in the stream; a fold drops the older ones.
	var se *llmStreamError
	if errors.As(err, &se) && strings.Contains(strings.ToLower(se.Msg), "img count") {
		return true
	}
	return errors.Is(err, errContextCeiling)
}

// The partial output is unusable (truncated tool-call JSON cannot be resumed);
// the tool loop retries once with a nudge that names Cap, then once on 2*Cap.
type capHitError struct {
	Cap int
	msg string
}

func (e *capHitError) Error() string { return e.msg }

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
		strings.Contains(s, "EOF")
}

// Absent enable_thinking counts as on; a continued closed-think prefill counts as off.
func thinkingOn(reqBody map[string]any) bool {
	if cont, _ := reqBody["continue_final_message"].(bool); cont {
		return false
	}
	ctk, ok := reqBody["chat_template_kwargs"].(map[string]any)
	if !ok {
		return true
	}
	et, ok := ctk["enable_thinking"].(bool)
	return !ok || et
}

// A server may accept chat_template_kwargs and drop it (Ollama, llama.cpp without
// --jinja), which is otherwise invisible. Once per Server+Model.
func (a *agent) warnChatTemplateKwargsIgnored(ctx context.Context, sid string, conn *LLMConnection, reqBody map[string]any, reasoningBytes int) {
	if reasoningBytes == 0 || sid == "" || thinkingOn(reqBody) {
		return
	}
	if _, seen := a.ctkIgnored.LoadOrStore(conn.Server+"\x00"+conn.Model, true); seen {
		return
	}
	if cont, _ := reqBody["continue_final_message"].(bool); cont {
		slog.Warn("server ignored the thinking-off prefill",
			"server", conn.Server, "model", conn.Model, "reasoning_bytes", reasoningBytes)
		a.say(ctx, sid, "⚠ "+conn.Model+" kept reasoning after being handed a closed <think></think> to continue.\n"+
			"  The server started a fresh assistant turn instead of continuing the prefilled one, so it does not honour\n"+
			"  continue_final_message / add_generation_prompt=false. Execute-role calls will keep reasoning here, which\n"+
			"  costs decode time but nothing else; the damage stays capped, since reasoning that follows a closed block is short.\n"+
			"  If this model does not delimit reasoning with <think>/</think>, that is the likelier cause.\n"+
			"  The fallback is params_execute chat_template_kwargs = { enable_thinking = false }, which costs a second\n"+
			"  prompt rendering, worth it only on a server with 2+ slots (see settings.toml).\n\n")
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

// What Qwen3's template emits for enable_thinking=false; a model with other
// reasoning delimiters needs a different string.
const noThinkPrefillContent = "<think>\n\n</think>\n\n"

// Continues a closed think block rather than set enable_thinking=false, which
// re-renders the prompt and loses the prefix cache. nil in, nil out.
func (c *LLMConnection) withThinkingDisabled() *LLMConnection {
	if c == nil {
		return nil
	}
	cp := *c
	cp.noThinkPrefill = true
	return &cp
}

// The keys in use (max_tokens, tool_choice) are samplers, so the prefix cache is untouched.
// nil in, nil out.
func (c *LLMConnection) withBody(key string, v any) *LLMConnection {
	if c == nil {
		return nil
	}
	cp := *c
	eb := make(map[string]any, len(c.ExtraBody)+1)
	maps.Copy(eb, c.ExtraBody)
	eb[key] = v
	cp.ExtraBody = eb
	return &cp
}

// Same renderers as a real turn, so its prompt is a byte-prefix of the next request.
func (a *agent) warmCall(ctx context.Context, sess *Session, conn *LLMConnection, messages []llmMessage, tag string) error {
	// A long prefix on a slow local model takes minutes to process.
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	defer cancel()
	// A turn starting mid-warm would otherwise count the warm's prefill as uncached.
	warm := conn.withBody("max_tokens", 1)
	warm.noTurnStats = true
	start := time.Now()
	_, _, _, err := a.llmStream(ctx, sess.ID, warm, messages, a.tools.defs(), nil, nil, nil)
	a.logSession(sess.ID, tag, "done in %s err=%v", time.Since(start).Round(time.Millisecond), err)
	return err
}

// Keeps an idle slot from being reclaimed while the user is away or a long tool
// runs. next() runs per tick, so the current prefix is the one refreshed.
func (a *agent) keepWarm(sess *Session, conn *LLMConnection, next func() []llmMessage) (stop func()) {
	every := a.keepWarmInterval()
	if sess == nil || conn == nil || every <= 0 {
		return func() {}
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		tick := time.NewTicker(every)
		defer tick.Stop()
		giveUp := time.After(keepWarmFor)
		for {
			select {
			case <-ctx.Done():
				return
			case <-giveUp:
				return
			case <-tick.C:
				messages := next()
				if len(messages) == 0 {
					continue
				}
				if err := a.warmCall(ctx, sess, conn, messages, "WARM"); err != nil && ctx.Err() == nil {
					slog.Debug("keepWarm: refresh failed", "sid", sess.ID, "err", err)
				}
			}
		}
	}()
	return cancel
}

// Errors are only logged: the real turn reports an unreachable server.
func (a *agent) prewarm(sess *Session) {
	if sess == nil || !a.prewarmEnabled() {
		return
	}
	conn := a.connFor("thinking")
	if conn == nil {
		return
	}
	messages := a.buildLLMContext(sess)
	if len(messages) == 0 {
		return
	}
	err := a.warmCall(context.Background(), sess, conn, messages, "PREWARM")
	slog.Debug("prewarm: done", "sid", sess.ID, "err", err)
}

func buildChatRequest(conn *LLMConnection, messages []llmMessage, tools []map[string]any) map[string]any {
	// Core fields go last so settings.toml cannot override model/messages/stream/tools.
	reqBody := map[string]any{}
	maps.Copy(reqBody, conn.ExtraBody)
	if _, ok := reqBody["max_tokens"]; !ok {
		reqBody["max_tokens"] = defaultMaxTokens
	}
	reqBody["model"] = conn.Model
	reqBody["stream"] = true
	// llama.cpp only; other servers ignore it.
	reqBody["timings_per_token"] = true
	reqBody["stream_options"] = map[string]any{"include_usage": true}
	reqBody["messages"] = messages
	switch {
	case conn.noThinkPrefill && !conn.noPrefill:
		// On reqBody, not ExtraBody: renderKey reads ExtraBody, and the prefill must not
		// count as a different rendering. llama.cpp does not echo the continued prefix.
		withPrefill := make([]llmMessage, len(messages), len(messages)+1)
		copy(withPrefill, messages)
		reqBody["messages"] = append(withPrefill, llmMessage{Role: "assistant", Content: noThinkPrefillContent})
		reqBody["add_generation_prompt"] = false
		reqBody["continue_final_message"] = true
	case conn.noThinkPrefill && reqBody["tool_choice"] != "required":
		// No continuation on this server. A forced tool choice already renders without
		// reasoning, so the flag beside "required" would be a third rendering.
		kw, _ := reqBody["chat_template_kwargs"].(map[string]any)
		merged := make(map[string]any, len(kw)+1)
		maps.Copy(merged, kw)
		merged["enable_thinking"] = false
		reqBody["chat_template_kwargs"] = merged
	}
	if tools != nil {
		reqBody["tools"] = tools
	}
	return reqBody
}

type streamResult struct {
	text      strings.Builder
	reasoning strings.Builder
	calls     []toolCall

	finishReason string
	streamErrMsg string
	scanErr      error

	promptTokens, completionTokens int
	// -1: the server did not report it.
	evaluatedTokens, cachedTokens int
	serverPromptMs, serverGenMs   float64

	readStart, firstTokenAt time.Time
}

// Never closes body. genChars is read concurrently by the status meter.
func readSSEStream(body io.Reader, conn *LLMConnection, on, think func(string), onArgs func(idx int, name, delta string), genChars *int64) *streamResult {
	r := &streamResult{evaluatedTokens: -1, cachedTokens: -1}

	scanner := bufio.NewScanner(body)
	// Tool-call argument blobs overflow the default 64 KB line limit.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	r.readStart = time.Now()
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
			slog.Debug("llm: skipped unparseable SSE frame", "role", conn.Tag, "err", err, "frame", truncate(data, 200))
			continue
		}
		// Before the empty-Choices skip, which would drop it. A bare "message" counts
		// only on an otherwise empty frame, so a stray field cannot fake a failure.
		if chunk.Error != nil && chunk.Error.Message != "" {
			r.streamErrMsg = chunk.Error.Message
			break
		}
		if chunk.Message != "" && len(chunk.Choices) == 0 && chunk.Usage == nil && chunk.Timings == nil {
			r.streamErrMsg = chunk.Message
			break
		}
		if chunk.Usage != nil {
			if chunk.Usage.PromptTokens > 0 {
				r.promptTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				r.completionTokens = chunk.Usage.CompletionTokens
			}
			if d := chunk.Usage.PromptTokensDetails; d != nil {
				r.cachedTokens = d.CachedTokens
			}
		}
		// Timings override usage cached_tokens: they carry both sides directly.
		if chunk.Timings != nil {
			r.evaluatedTokens = chunk.Timings.PromptN
			r.cachedTokens = chunk.Timings.CacheN
			r.serverPromptMs = chunk.Timings.PromptMs
			r.serverGenMs = chunk.Timings.PredictedMs
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		if fr := chunk.Choices[0].FinishReason; fr != "" {
			r.finishReason = fr
		}

		delta := chunk.Choices[0].Delta
		if delta.ReasoningContent == "" { // fold vLLM's spelling into the standard one
			delta.ReasoningContent = delta.Reasoning
		}

		if r.firstTokenAt.IsZero() && (delta.Content != "" || delta.ReasoningContent != "" || len(delta.ToolCalls) > 0) {
			r.firstTokenAt = time.Now()
		}

		if delta.ReasoningContent != "" {
			r.reasoning.WriteString(delta.ReasoningContent)
			atomic.AddInt64(genChars, int64(len(delta.ReasoningContent)))
			if think != nil {
				think(delta.ReasoningContent)
			}
		}

		if delta.Content != "" {
			r.text.WriteString(delta.Content)
			atomic.AddInt64(genChars, int64(len(delta.Content)))
			if on != nil {
				on(delta.Content)
			}
		}

		for _, tc := range delta.ToolCalls {
			atomic.AddInt64(genChars, int64(len(tc.Function.Name)+len(tc.Function.Arguments)))
			if tc.ID != "" {
				r.calls = append(r.calls, tc)
			} else if len(r.calls) > 0 {
				last := &r.calls[len(r.calls)-1]
				last.Function.Arguments += tc.Function.Arguments
			}
			// The name rides only the first chunk, so read it off the accumulator.
			if onArgs != nil && tc.Function.Arguments != "" && len(r.calls) > 0 {
				onArgs(len(r.calls)-1, r.calls[len(r.calls)-1].Function.Name, tc.Function.Arguments)
			}
		}
	}
	r.scanErr = scanner.Err()
	return r
}

func (a *agent) recordStreamStats(sid, connLabel string, conn *LLMConnection, r *streamResult) {
	sess := a.getSession(sid)
	if sess == nil || conn.noTurnStats {
		return
	}
	if r.evaluatedTokens < 0 && r.cachedTokens >= 0 && r.promptTokens > 0 {
		r.evaluatedTokens = r.promptTokens - r.cachedTokens
	}
	sess.addTurnTokens(r.promptTokens, r.completionTokens, r.evaluatedTokens)
	// Each tool-loop call extends the previous one, so the server should reuse all
	// of it; a big shortfall means the prompt was re-rendered or the slot evicted.
	if conn.cacheLineage && r.promptTokens > 0 {
		render := renderKey(conn.ExtraBody)
		if conn.noPrefill {
			// A forced tool choice is a rendering of its own on this server.
			render += fmt.Sprintf(" tool_choice=%v", conn.ExtraBody["tool_choice"])
		}
		rw := sess.noteCacheLineage(r.promptTokens, r.cachedTokens, render, time.Now())
		if rw.tokens > 0 {
			var cause string
			switch {
			case rw.renderChanged && conn.noPrefill:
				cause = "Expected on this server: it keeps one prompt state per rendering and has no cache-preserving way to switch " +
					"thinking off, so a role switch re-reads what the other role appended since. The re-read is real; nothing is misconfigured."
			case rw.renderChanged:
				cause = fmt.Sprintf("We asked for a different rendering than last call: template params went %s -> %s. "+
					"The server keeps a prompt state per rendering, so this call could only reuse what THIS rendering held "+
					"last time and had to re-evaluate everything the other role appended since. Make params_thinking and "+
					"params_execute agree on everything that is not a sampler.", orElse(rw.prevRender, "(none)"), orElse(render, "(none)"))
			case rw.idle >= idleEvictionSuspect:
				cause = fmt.Sprintf("Both calls asked for the same rendering (%s) and the slot sat idle that whole time, "+
					"which is the likeliest cause: servers reclaim idle slots and no setting prevents it. Nothing to fix "+
					"unless the gap surprises you.", orElse(render, "(none)"))
			default:
				cause = fmt.Sprintf("Both calls asked for the same rendering (%s) and came back to back, so an idle "+
					"eviction is unlikely: something rewrote the middle of the prompt. A tool result that replayed "+
					"differently than it was sent, or a chat template that repositions earlier messages as the "+
					"conversation grows.", orElse(render, "(none)"))
			}
			a.logSession(sid, connLabel+" CACHE",
				"prefix cache rewound: %d tokens the previous call had already sent were re-read "+
					"(prompt=%d cached=%d, %s since that call). %s",
				rw.tokens, r.promptTokens, r.cachedTokens, humanDuration(rw.idle.Milliseconds()), cause)
		}
	}
	// The TTFT proxy includes queueing, so it is only the fallback.
	pMs, gMs := r.serverPromptMs, r.serverGenMs
	if pMs == 0 && gMs == 0 && !r.firstTokenAt.IsZero() {
		pMs = float64(r.firstTokenAt.Sub(r.readStart).Milliseconds())
		gMs = float64(time.Since(r.firstTokenAt).Milliseconds())
	}
	if pMs > 0 || gMs > 0 {
		sess.addTurnTiming(int64(pMs), int64(gMs))
	}
}

// The in-band error goes first: gateways send {"error":…} and THEN drop the
// socket, and the EOF would send a fatal prompt down the transient-retry path.
func (a *agent) streamOutcomeError(conn *LLMConnection, reqBody map[string]any, r *streamResult) error {
	switch {
	case r.streamErrMsg != "":
		return &llmStreamError{Role: conn.Tag, Model: conn.Model, Msg: r.streamErrMsg}
	case r.scanErr != nil:
		return fmt.Errorf("reading SSE stream: %w", r.scanErr)
	case r.finishReason == "length":
		// AT the cap the model is verbose or looping; BELOW it the n_ctx ceiling was hit,
		// which folding recovers like a 400.
		reqMax := 0
		switch v := reqBody["max_tokens"].(type) {
		case int:
			reqMax = v
		case int64:
			reqMax = int(v)
		case float64:
			reqMax = int(v)
		}
		// A one-token warm-up hitting its cap is the request doing its job.
		if reqMax == 1 {
			return nil
		}
		// noRoom still catches the ceiling when the server omits completion_tokens.
		belowCap := r.completionTokens > 0 && reqMax > 0 && r.completionTokens < reqMax
		mst := int(a.mainSlotTokens.Load())
		noRoom := r.promptTokens > 0 && reqMax > 0 && mst > 0 && r.promptTokens+reqMax > mst
		switch {
		case belowCap || noRoom:
			return fmt.Errorf("generation hit the context ceiling (prompt=%d gen=%d, n_ctx=%d, role=%s): %w",
				r.promptTokens, r.completionTokens, mst, conn.Tag, errContextCeiling)
		default:
			return &capHitError{Cap: reqMax, msg: fmt.Sprintf("LLM hit max_tokens cap (role=%s, model=%s): response truncated (%d B content, %d B reasoning, %d tool calls). Likely the model is looping or stuck in <think>; raise max_tokens in params_%s, or set chat_template_kwargs.enable_thinking=false for this role if reasoning is dominating the budget",
				conn.Tag, conn.Model, r.text.Len(), r.reasoning.Len(), len(r.calls), conn.Tag)}
		}
	default:
		return nil
	}
}

func (a *agent) logStreamResponse(sid, connLabel string, r *streamResult, err error) {
	text, reasoning := r.text.String(), r.reasoning.String()
	var rb strings.Builder
	if r.promptTokens > 0 || r.completionTokens > 0 || r.finishReason != "" {
		// "(none)": the stream broke before a finish_reason.
		fmt.Fprintf(&rb, "tokens: prompt=%d completion=%d finish=%s", r.promptTokens, r.completionTokens, orElse(r.finishReason, "(none)"))
		if r.cachedTokens >= 0 || r.evaluatedTokens >= 0 {
			fmt.Fprintf(&rb, " cached=%d evaluated=%d", r.cachedTokens, r.evaluatedTokens)
		}
		rb.WriteString("\n")
	}
	if reasoning != "" {
		fmt.Fprintf(&rb, "reasoning_content (%d B):\n%s\n", len(reasoning), reasoning)
	}
	if text != "" {
		fmt.Fprintf(&rb, "content:\n%s\n", text)
	}
	for i, c := range r.calls {
		fmt.Fprintf(&rb, "tool_call[%d] %s id=%s args=%s\n", i, c.Function.Name, c.ID, c.Function.Arguments)
	}
	if text == "" && reasoning == "" && len(r.calls) == 0 {
		rb.WriteString("(empty response)\n")
	}
	if err != nil {
		fmt.Fprintf(&rb, "[stream error] %v\n", err)
	}
	a.logSession(sid, connLabel+" RESPONSE", "%s", rb.String())
}

// Each request repeats the previous one's prefix, so a log entry keeps only the
// bytes from the first change on.
const requestLogSame = "[same as the previous request to this connection for the first "

// An early change is called out: it is the prefix cache being lost.
func requestLogDelta(prev, body []byte) string {
	// The wrapper before "messages" changes between renderings while the cache keys
	// on the messages, so the comparison starts at the messages.
	pm, bm := bytes.Index(prev, []byte(`"messages":`)), bytes.Index(body, []byte(`"messages":`))
	if len(prev) == 0 || pm < 0 || bm < 0 {
		return string(body)
	}
	wrapper, rest, prevRest := body[:bm], body[bm:], prev[pm:]
	n := 0
	for n < len(prevRest) && n < len(rest) && prevRest[n] == rest[n] {
		n++
	}
	if n == 0 {
		return string(body)
	}
	head := fmt.Sprintf("%s%s%d of %d bytes of messages; the rest:]\n", string(wrapper), requestLogSame, n, len(rest))
	if n*2 < len(prevRest) {
		head = fmt.Sprintf("%s%s%d of %d bytes of messages ONLY: a change this early is a prefix the server cannot reuse; the rest:]\n", string(wrapper), requestLogSame, n, len(rest))
	}
	return head + strings.ToValidUTF8(string(rest[n:]), "")
}

// think gets reasoning apart from on; onArgs gets tool-call argument deltas. A 400
// refusing the closed-think continuation marks the entry and retries once without it.
func (a *agent) llmStream(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, tools []map[string]any, on, think func(string), onArgs func(idx int, name, delta string)) (string, []toolCall, string, error) {
	// Connections are copied freely, so the mark lives on the settings entry.
	if conn != nil && conn.noThinkPrefill && !conn.noPrefill {
		a.cfgMu.RLock()
		i := a.entryIndex(conn)
		rejected := i >= 0 && a.settings.LLM[i].noPrefill
		a.cfgMu.RUnlock()
		if rejected {
			cp := *conn
			cp.noPrefill = true
			conn = &cp
		}
	}
	text, calls, reasoning, err := a.llmStreamOnce(ctx, sid, conn, messages, tools, on, think, onArgs)
	var he *llmHTTPError
	if conn != nil && conn.noThinkPrefill && !conn.noPrefill && errors.As(err, &he) && he.Status == 400 && strings.Contains(he.Body, "continue_final_message") {
		a.cfgMu.Lock()
		if i := a.entryIndex(conn); i >= 0 && !a.settings.LLM[i].noPrefill {
			a.settings.LLM[i].noPrefill = true
			slog.Info("server rejects the closed-think continuation; thinking off is its own tool_choice / enable_thinking from here on, one prompt rendering per role",
				"server", conn.Server, "model", conn.Model)
		}
		a.cfgMu.Unlock()
		cp := *conn
		cp.noPrefill = true
		return a.llmStreamOnce(ctx, sid, &cp, messages, tools, on, think, onArgs)
	}
	return text, calls, reasoning, err
}

// Caller holds cfgMu. -1 for test mocks and probes.
func (a *agent) entryIndex(conn *LLMConnection) int {
	for i := range a.settings.LLM {
		if a.settings.LLM[i].Server == conn.Server && a.settings.LLM[i].Model == conn.Model {
			return i
		}
	}
	return -1
}

// acquireConnSlot waits for a free call slot on conn's [[llm]] entry. slot is the
// entry index, -1 for test mocks and probes, which have no limit.
func (a *agent) acquireConnSlot(ctx context.Context, sid string, conn *LLMConnection) (slot int, release func(), err error) {
	// Release on this channel: a prepare phase can swap a.connSems meanwhile, and
	// releasing on the new channel would block forever.
	a.cfgMu.RLock()
	slot = a.entryIndex(conn)
	var sem chan struct{}
	if slot >= 0 && slot < len(a.connSems) {
		sem = a.connSems[slot]
	}
	a.cfgMu.RUnlock()
	if sem == nil {
		return slot, func() {}, nil
	}
	select {
	case sem <- struct{}{}:
	default:
		a.setStatus(ctx, sid, " (queued…)")
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			a.setStatus(ctx, sid, "")
			return slot, nil, ctx.Err()
		}
	}
	return slot, func() { <-sem }, nil
}

// llmHTTPErrorFrom prefers the OpenAI-style error.message over the raw body.
func (a *agent) llmHTTPErrorFrom(sid, connLabel string, resp *http.Response) error {
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("HTTP %d, failed to read body: %w", resp.StatusCode, err)
	}
	a.logSession(sid, connLabel, "[HTTP %d] %s", resp.StatusCode, string(bodyBytes))
	msg := string(bodyBytes)
	var apiErr struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if json.Unmarshal(bodyBytes, &apiErr) == nil && apiErr.Error.Message != "" {
		msg = apiErr.Error.Message
	}
	if strings.Contains(msg, "chat_template_kwargs") {
		msg += "\n\nThis backend does not accept chat_template_kwargs (it is a llama.cpp / vLLM extension). Remove it from params_thinking / params_execute for this [[llm]] entry."
	}
	return &llmHTTPError{Status: resp.StatusCode, Body: msg, Type: apiErr.Error.Type, URL: resp.Request.URL.String()}
}

func (a *agent) llmStreamOnce(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, tools []map[string]any, on, think func(string), onArgs func(idx int, name, delta string)) (string, []toolCall, string, error) {
	reqBody := buildChatRequest(conn, messages, tools)
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", nil, "", fmt.Errorf("marshalling LLM request body: %w", err)
	}

	slot, release, err := a.acquireConnSlot(ctx, sid, conn)
	if err != nil {
		return "", nil, "", err
	}
	defer release()

	// The display index (llm[1] for background work), not the semaphore index slot.
	slotLabel := "?"
	if slot >= 0 {
		slotLabel = fmt.Sprintf("%d", conn.Slot)
	}

	upLabel := humanBytes(len(body))
	a.setStatus(ctx, sid, fmt.Sprintf(" (llm[%s] ↑%s sent…)", slotLabel, upLabel))
	defer a.setStatus(ctx, sid, "")

	// Deferred after the status clear, so LIFO stops the meter first.
	var genChars int64
	meterStart := time.Now()
	defer a.startStatusMeter(ctx, sid, func() string {
		if g := atomic.LoadInt64(&genChars); g > 0 {
			// chars/4 approximates tokens.
			return fmt.Sprintf(" (llm[%s] ↑%s ↓%s tok…)", slotLabel, upLabel, humanCount(int(g)/4))
		}
		return fmt.Sprintf(" (llm[%s] ↑%s sent… %ds)", slotLabel, upLabel, int(time.Since(meterStart).Seconds()))
	})()

	connLabel := fmt.Sprintf("llm[%s] %s model=%s", slotLabel, conn.Tag, conn.Model)
	logKey := sid + "\x00" + connLabel
	a.logPrevMu.Lock()
	prevBody := a.logPrev[logKey]
	if a.logPrev == nil {
		a.logPrev = map[string][]byte{}
	}
	a.logPrev[logKey] = body
	a.logPrevMu.Unlock()
	a.logSession(sid, connLabel+" REQUEST", "%s", requestLogDelta(prevBody, body))

	httpReq, err := http.NewRequestWithContext(ctx, "POST", conn.endpoint("/v1/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return "", nil, "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if conn.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
	}

	resp, err := llmHTTPClient.Do(httpReq)
	if err != nil {
		a.logSession(sid, connLabel, "[transport error] %v", err)
		return "", nil, "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", nil, "", a.llmHTTPErrorFrom(sid, connLabel, resp)
	}

	res := readSSEStream(resp.Body, conn, on, think, onArgs, &genChars)
	a.recordStreamStats(sid, connLabel, conn, res)

	a.warnChatTemplateKwargsIgnored(ctx, sid, conn, reqBody, res.reasoning.Len())

	err = a.streamOutcomeError(conn, reqBody, res)
	// An unusable generation is already in the completion total; name it, or the
	// Done line ranks the worst turns as the most productive.
	if err != nil && res.completionTokens > 0 {
		if sess := a.getSession(sid); sess != nil && !conn.noTurnStats {
			sess.addWastedCompletion(res.completionTokens)
		}
	}
	a.logStreamResponse(sid, connLabel, res, err)
	return res.text.String(), res.calls, res.reasoning.String(), err
}

// The zero value means the server could not be reached.
type probeResult struct {
	Reachable       bool // got 200 from any probe endpoint
	ModelKnown      bool // /v1/models enumerated models, so ModelLoaded is meaningful
	ModelLoaded     bool // the configured model was in the enumeration
	ImageSupport    bool
	AvailableModels []string
	// TOTAL n_ctx across all slots; 0 = unknown.
	ContextSize int
	// PER-SLOT n_ctx from modern llama.cpp; preferred over ContextSize. 0 = unknown.
	SlotCtx int
	// llama.cpp's -np; 0 = unknown.
	TotalSlots int
}

// /v1/models works on every backend; /props is a llama.cpp-only enrichment.
func probeLLM(ctx context.Context, conn *LLMConnection) probeResult {
	if conn == nil {
		return probeResult{}
	}
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	r := probeViaModels(probeCtx, conn)
	p := probeViaProps(probeCtx, conn, "/props")
	// A llama.cpp router reports n_ctx=0 on bare /props; ?model= routes to the model
	// and autoloads it (hence the timeout). QueryEscape: ids carry spaces and ';'.
	if p.Reachable && p.ContextSize == 0 && p.SlotCtx == 0 {
		upCtx, upCancel := context.WithTimeout(ctx, 180*time.Second)
		if up := probeViaProps(upCtx, conn, "/props?model="+url.QueryEscape(conn.Model)); up.Reachable {
			p = up
		}
		upCancel()
	}
	if p.Reachable {
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

// false means unusable through this endpoint, and the reason is already logged.
func probeGetJSON(ctx context.Context, conn *LLMConnection, path, who string, v any) bool {
	url := conn.endpoint(path)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		slog.Info(who+": unusable request URL", "url", url, "err", err)
		return false
	}
	if conn.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+conn.APIKey)
	}
	resp, err := metaHTTPClient.Do(req)
	if err != nil {
		slog.Info(who+": request failed", "url", url, "err", err)
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		slog.Info(who+": non-OK", "url", url, "status", resp.StatusCode, "body", string(body))
		return false
	}
	if err := json.NewDecoder(resp.Body).Decode(v); err != nil {
		slog.Info(who+": undecodable body", "url", url, "err", err)
		return false
	}
	return true
}

// Image support and context size come only from llama-swap's status.args.
func probeViaModels(ctx context.Context, conn *LLMConnection) probeResult {
	var models struct {
		Data []struct {
			ID     string `json:"id"`
			Status struct {
				Args []string `json:"args"`
			} `json:"status"`
		} `json:"data"`
	}
	if !probeGetJSON(ctx, conn, "/v1/models", "probeViaModels", &models) {
		return probeResult{}
	}
	r := probeResult{Reachable: true, ModelKnown: true}
	for _, m := range models.Data {
		r.AvailableModels = append(r.AvailableModels, m.ID)
		if m.ID != conn.Model {
			continue
		}
		r.ModelLoaded = true
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
				for _, prefix := range []string{"--ctx-size=", "-c="} {
					if v, ok := strings.CutPrefix(arg, prefix); ok {
						if n, err := strconv.Atoi(v); err == nil {
							r.ContextSize = n
						}
					}
				}
			}
		}
	}
	return r
}

// /props cannot say which model is loaded, so ModelKnown stays false.
func probeViaProps(ctx context.Context, conn *LLMConnection, path string) probeResult {
	var props struct {
		Modalities *struct {
			Vision bool `json:"vision"`
		} `json:"modalities"`
		// Modern builds nest a PER-SLOT n_ctx here; older ones put the TOTAL at the top.
		DefaultGenerationSettings *struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
		NCtx       int `json:"n_ctx"`
		TotalSlots int `json:"total_slots"`
	}
	if !probeGetJSON(ctx, conn, path, "probeViaProps", &props) {
		return probeResult{}
	}
	r := probeResult{Reachable: true, ContextSize: props.NCtx, TotalSlots: props.TotalSlots}
	if props.Modalities != nil {
		r.ImageSupport = props.Modalities.Vision
	}
	if props.DefaultGenerationSettings != nil {
		r.SlotCtx = props.DefaultGenerationSettings.NCtx
	}
	return r
}

// Always LLM[0], whose KV cache owns the conversation prefix.
func (a *agent) connFor(role string) *LLMConnection {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	return a.settings.ConnAt(0, role)
}

// Each failure costs a turn's note, so this only rides out a single timeout.
const summaryMaxStrikes = 2

// Bounded so a healthy server is not retired for the whole run; a new failure renews it.
const summaryCooldown = 10 * time.Minute

// The capacity peek is racy by design: falling back beats queueing. true means the
// llm[0] fallback, the cue for a prefix-extension prompt that reuses its cache.
func (a *agent) connForBackgroundLLM() (*LLMConnection, bool) {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	for i := 1; i < len(a.settings.LLM); i++ {
		if !strings.EqualFold(a.settings.LLM[i].Purpose, purposeSummary) {
			continue
		}
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
	// Display-only: the meter shows background work as llm[1]; the server picks the slot.
	c := a.settings.ConnAt(0, "execute")
	if c != nil && a.settings.LLM[0].parallelCap() >= 2 {
		c.Slot = 1
	}
	return c, true
}

// setSettings installs loaded settings. A reload must not drop what was learned
// about the same server and model: the probed total_slots for an unset parallel,
// and a rejected closed-think continuation. Caller holds a.cfgMu for writing.
func (a *agent) setSettings(s Settings) {
	for i := range s.LLM {
		c := &s.LLM[i]
		if p := a.connProbe[c.Server+"\x00"+c.Model]; c.Parallel == nil && p.TotalSlots > 0 {
			n := p.TotalSlots
			c.Parallel = &n
		}
		if j := a.entryIndex(c); j >= 0 && a.settings.LLM[j].noPrefill {
			c.noPrefill = true
		}
	}
	a.settings = s
	a.buildConnSems()
}

// No-op when no cap changed, so in-flight calls keep their channels. Caller holds
// a.cfgMu for writing.
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
