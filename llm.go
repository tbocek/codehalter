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

// metaHTTPClient serves the short metadata requests: the LLM probes, the setup
// check, the update check and download. Same dial and handshake bounds as
// llmHTTPClient, so a dead route fails in seconds. No ResponseHeaderTimeout on
// purpose: a llama.cpp router answers /props?model= only once the model is
// loaded, which takes minutes, and the probe's own context bounds that wait.
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
			// Reasoning is the same channel under vLLM's spelling. Captured live
			// from the user's llmhub endpoint: 2053 frames, every one of them
			// carrying `reasoning`, none carrying `reasoning_content`, so reading
			// only the standard name dropped the whole chain of thought and made an
			// all-thinking turn look like a stalled stream to errStuckThinking.
			Reasoning string `json:"reasoning"`
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
	// Error is an error delivered INSIDE the stream under HTTP 200, which is how
	// llama.cpp / llama-swap and some gateways report a mid-stream failure (a
	// prompt over the real context length). Such a chunk has empty Choices, so
	// without this field it surfaced as "(empty response)" three layers up. Both
	// shapes occur: nested {"error":{"message":…}} and a bare {"message":…}.
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
	Type   string // the OpenAI-style error.type, when the body carried one
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
// server said so on a 400 (or a 413), or a generation truncated at the n_ctx
// ceiling (errContextCeiling). Both are recovered by folding history and
// retrying.
//
// The status alone is not enough. A 400 is also what a server answers to a
// request shape it will not take: llama.cpp's "Cannot continue an assistant
// message that contains tool calls", Halogen's refusal of continue_final_message
// beside a forced tool choice. Reading those as a full context started the
// compaction ladder over a request that no amount of folding could fix. So the
// body has to be about the context: llama.cpp's structured type, or a message
// that names it. Every server words that differently and a phrase list missed
// one within a day, so the test is the word itself:
//
//	llama.cpp: request (262314 tokens) exceeds the available context size (262144 tokens)
//	Halogen:   max_tokens 32768 does not fit: prompt is 255003 tokens and the context is 262144, leaving room for 7141
//	vLLM/OpenAI: This model's maximum context length is 8192 tokens
//
// plus the servers that say "too long" or "too many tokens" without the word.
// Halogen's check is prompt + max_tokens against the window, where llama.cpp
// checks the prompt alone, so on Halogen the fold comes max_tokens earlier.
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
	return errors.Is(err, errContextCeiling)
}

// errStuckThinking marks a generation that spent its whole max_tokens budget on
// reasoning_content with no message text and no tool calls — the model looped in
// <think> and produced nothing usable. Recoverable: the tool loop retries once on
// a thinking-disabled copy of the connection so the model must answer directly.
// Only raised when thinking was ON, so the retry can't re-trigger it.
var errStuckThinking = errors.New("model stuck in reasoning")

// capHitError marks a generation truncated AT the requested max_tokens cap with
// actual content or tool calls in flight (a reasoning-only burn classifies as
// errStuckThinking instead): the model needed more room than the cap allowed, or
// was looping. The partial output is unusable — truncated tool-call JSON can't be
// resumed through the chat API. Recoverable: the tool loop retries once with a
// be-concise nudge, then once more on a doubled cap, before surfacing the failure
// (see the cap ladder in runToolLoopSeeded). Cap carries the request's max_tokens
// so that retry can compute the doubled budget.
type capHitError struct {
	Cap int
	msg string
}

func (e *capHitError) Error() string { return e.msg }

// asCapHit returns the capHitError inside err, or nil when err isn't one.
func asCapHit(err error) *capHitError {
	var ce *capHitError
	if errors.As(err, &ce) {
		return ce
	}
	return nil
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

// thinkingOn reports whether the request had reasoning enabled. Absent
// chat_template_kwargs (or an absent enable_thinking) counts as on, since the
// stall is only classified when reasoning_content was actually produced. Two
// things count as off so a retry can't loop: the user's own
// enable_thinking=false, and our own prefilled-and-continued <think></think>.
func thinkingOn(reqBody map[string]any) bool {
	// A continued prefill is an already-closed <think></think> block, so the
	// model cannot open one no matter what the template says.
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

// warnChatTemplateKwargsIgnored says so when the server reasoned after being
// asked not to. A server may accept chat_template_kwargs and drop it: Ollama
// substitutes its own template, llama.cpp without --jinja has no
// enable_thinking to read. The symptom is invisible (correct answers at half
// speed), and the separate rendering it costs (see paramsFor) then buys
// nothing. Once per Server+Model: it is a property of the deployment.
func (a *agent) warnChatTemplateKwargsIgnored(ctx context.Context, sid string, conn *LLMConnection, reqBody map[string]any, reasoningBytes int) {
	if reasoningBytes == 0 || sid == "" || thinkingOn(reqBody) {
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

// noThinkPrefillContent is the assistant prefix a thinking-off call continues
// from: an already-closed reasoning block, which is what Qwen3's template emits
// for enable_thinking=false. It is the one model-specific literal on this path:
// a model with other reasoning delimiters needs a different string. That is
// the price of keeping the prefix cache across the switch, in both directions.
const noThinkPrefillContent = "<think>\n\n</think>\n\n"

// withThinkingDisabled returns a shallow copy whose next call answers without
// reasoning. It does not touch ExtraBody: llmStream appends
// noThinkPrefillContent as a trailing assistant message for the server to
// continue, so every earlier token, and the prefix cache, is untouched. Used by
// the executor, the documenter, the summariser and the <think>-stall retry.
// The alternatives cost more: /no_think is unreliable (237 of 388 responses
// reasoned anyway) and enable_thinking=false re-renders (see paramsFor).
// nil in, nil out, because callers chain it onto connFor.
func (c *LLMConnection) withThinkingDisabled() *LLMConnection {
	if c == nil {
		return nil
	}
	cp := *c
	cp.noThinkPrefill = true
	return &cp
}

// withMaxTokens returns a shallow copy of the connection with max_tokens
// forced to n in a copied ExtraBody (llmStream copies ExtraBody into the
// request first, so this overrides the role default). Slot/Server/Model are
// unchanged, so it routes to the same connSem. Used by prewarm to cap the
// warming call at a single generated token.
func (c *LLMConnection) withMaxTokens(n int) *LLMConnection {
	cp := *c
	eb := make(map[string]any, len(c.ExtraBody)+1)
	maps.Copy(eb, c.ExtraBody)
	eb["max_tokens"] = n
	cp.ExtraBody = eb
	return &cp
}

// forToolLoop returns a shallow copy of the connection marked as a tool-loop
// call: stream rules armed, and the call folded into the session's prefix-cache
// lineage. Both belong to the tool loop alone: it is the one caller with a retry
// ladder that can act on a rule abort, and the one caller whose successive calls
// are guaranteed to be appends to each other (see noteCacheLineage). Nothing
// about the request body changes, so the prefix cache is unaffected.
func (c *LLMConnection) forToolLoop() *LLMConnection {
	cp := *c
	cp.streamRulesArmed = true
	cp.cacheLineage = true
	return &cp
}

// withToolChoice returns a shallow copy of the connection that sends
// tool_choice=v. llama.cpp turns it into a grammar over the reply, so no prompt
// token changes and the prefix cache is untouched (see samplerParams).
//
//   - "required": the execute phase ends only on a terminal tool, so prose is
//     always a slip; requiring a call beats nudging for one afterwards. Not the
//     planner (it reasons before its call and has its own corrective retry) and
//     not the documenter, whose one-line answer is legitimate prose.
//   - "none": the prefix-extension summariser must still SEND the tools array,
//     because the template renders it into the head of the prompt, but has to
//     answer with the note rather than a call.
func (c *LLMConnection) withToolChoice(v string) *LLMConnection {
	cp := *c
	eb := make(map[string]any, len(c.ExtraBody)+1)
	maps.Copy(eb, c.ExtraBody)
	eb["tool_choice"] = v
	cp.ExtraBody = eb
	return &cp
}

// keepWarm refreshes the server's cached prefix for this conversation while
// nothing else is calling it, and returns the function that stops it. The call
// is prewarm's (one token, same renderers), so it is a byte-prefix of whatever
// comes next and costs the server a lookup. next() runs per tick, so the prefix
// refreshed is the current one. It covers both ways a prefix goes cold: a slot
// reclaimed while the user is away, and one reclaimed during a long tool run
// (measured: a 174k-token prompt re-read after a 2m59s gap during a test).
func (a *agent) keepWarm(sess *Session, conn *LLMConnection, next func() []llmMessage) (stop func()) {
	every := a.keepWarmInterval()
	if sess == nil || conn == nil || every <= 0 {
		return func() {}
	}
	ctx, cancel := context.WithCancel(context.Background())
	warm := conn.withMaxTokens(1)
	warm.noTurnStats = true // a refresh is not the turn's work; see noTurnStats
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
				start := time.Now()
				callCtx, callCancel := context.WithTimeout(ctx, 5*time.Minute)
				_, _, _, err := a.llmStream(callCtx, sess.ID, warm, messages, a.tools.defs(), nil, nil, nil)
				callCancel()
				a.logSession(sess.ID, "WARM", "kept the prefix warm in %s err=%v", time.Since(start).Round(time.Millisecond), err)
				if err != nil && ctx.Err() == nil {
					slog.Debug("keepWarm: refresh failed", "sid", sess.ID, "err", err)
				}
			}
		}
	}()
	return cancel
}

// prewarm pays the prompt-processing cost of the session's prefix before the
// user's first message. One synchronous 1-token call through the exact
// renderers a real turn uses, so the prompt is a byte-prefix of the next
// request and llama.cpp reuses the KV cache. Errors are swallowed by design: an
// unreachable server makes this a no-op and the real turn reports the problem.
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
	// Generous ceiling: a 10k-token prefix on a slow local model is ~30s of
	// prompt processing; 122B-class models take a few times that.
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	start := time.Now()
	// Log under the real sid so the session log records the prewarm's exact
	// request bytes and its cached/evaluated split — the only way to diagnose
	// a turn-one cache miss (diff this request against turn one's).
	// noTurnStats: a turn that starts while the warm is still streaming resets
	// the counters BEFORE the warm's usage lands, which would inflate that
	// turn's "uncached" stat by the whole prefill.
	warmConn := conn.withMaxTokens(1)
	warmConn.noTurnStats = true
	_, _, _, err := a.llmStream(ctx, sess.ID, warmConn, messages, a.tools.defs(), nil, nil, nil)
	elapsed := time.Since(start).Round(time.Millisecond)
	a.logSession(sess.ID, "PREWARM", "done in %s err=%v", elapsed, err)
	slog.Debug("prewarm: done", "sid", sess.ID, "elapsed", elapsed, "err", err)
}

// buildChatRequest assembles the OpenAI chat-completions body for one call.
// Split out of llmStream because it is the whole of what goes on the wire and
// nothing else: a pure function of the connection and the messages, so a test
// can assert the shape without standing a server up.
func buildChatRequest(conn *LLMConnection, messages []llmMessage, tools []map[string]any) map[string]any {
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
	switch {
	case conn.noThinkPrefill && !conn.noPrefill:
		// Append the closed think block and tell the server to continue that
		// message rather than open a fresh assistant turn. Written straight onto
		// reqBody and not into ExtraBody on purpose: renderKey reads ExtraBody to
		// decide whether two calls asked for different renderings, and this pair
		// must not read as one. Verified streaming against llama.cpp: the
		// continued prefix is not echoed back in the deltas, so the content
		// arrives clean and needs no stripping.
		withPrefill := make([]llmMessage, len(messages), len(messages)+1)
		copy(withPrefill, messages)
		reqBody["messages"] = append(withPrefill, llmMessage{Role: "assistant", Content: noThinkPrefillContent})
		reqBody["add_generation_prompt"] = false
		reqBody["continue_final_message"] = true
	case conn.noThinkPrefill && reqBody["tool_choice"] != "required":
		// No continuation on this server (see LLMConnection.noPrefill). A forced
		// tool choice already renders without reasoning there, so only a call
		// that forces nothing needs the flag; sending it beside "required" would
		// be a third rendering for no gain.
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

// streamResult is everything one SSE stream produced: the reconstructed
// response, the server's own accounting of it, and how it ended. It exists so
// the three passes that run after the scan — turn stats, outcome
// classification, the RESPONSE log — can each take one value instead of a
// dozen arguments.
type streamResult struct {
	text      strings.Builder
	reasoning strings.Builder
	calls     []toolCall

	// finishReason is the server's own word for how generation ended: "stop",
	// "length", "tool_calls", or "" when the stream broke before it said.
	finishReason string
	// streamErrMsg holds an error the server delivered in-band (HTTP 200, an
	// {"error":…} SSE chunk). Surfaced as the call error so it isn't swallowed
	// as an empty response.
	streamErrMsg string
	// scanErr is set when the stream broke mid-flight with no in-band error to
	// explain it.
	scanErr error
	// firedRule is set when a stream rule matched the content and the generation
	// was abandoned mid-flight. The partial is discarded, so nothing downstream
	// reads text in that case.
	firedRule *streamRule

	promptTokens, completionTokens int
	// Server-reported cache split (see sseChunk.Timings / PromptTokensDetails).
	// evaluatedTokens = prompt tokens actually run through the model this call;
	// cachedTokens = reused from KV cache. -1 = the server didn't report it.
	evaluatedTokens, cachedTokens int
	// Server-measured eval/gen times (ms) — exact, vs the TTFT proxy below.
	serverPromptMs, serverGenMs float64

	// TTFT (readStart→firstTokenAt) and gen (firstTokenAt→end): the rate timing
	// fallback for when the server sends no _ms.
	readStart, firstTokenAt time.Time
}

// readSSEStream consumes the event stream, forwarding deltas to the caller's
// sinks and accumulating the response. It returns on [DONE], an in-band error,
// a stream-rule hit, or end of body, and never closes the body: the caller's
// deferred Close is what stops the server generating after a rule hit.
// genChars feeds the status meter from another goroutine, hence the atomics.
func readSSEStream(body io.Reader, conn *LLMConnection, matcher *ruleMatcher, on, think func(string), onArgs func(idx int, name, delta string), genChars *int64) *streamResult {
	r := &streamResult{evaluatedTokens: -1, cachedTokens: -1}

	scanner := bufio.NewScanner(body)
	// SSE chunks can carry large tool-call argument blobs; the default 64 KB
	// line limit silently truncates. 4 MB matches common reverse-proxy caps.
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
			// Don't drop a malformed frame silently: an off-shape chunk would
			// otherwise make a backend look like a terse model with no trail.
			slog.Debug("llm: skipped unparseable SSE frame", "role", conn.Tag, "err", err, "frame", truncate(data, 200))
			continue
		}
		// In-band error: the gateway/llama.cpp can return HTTP 200 and put the
		// failure in an {"error":…} chunk (empty Choices). Capture and stop —
		// checked before the empty-Choices skip below, which would drop it and
		// leave the call looking like a silent "(empty response)".
		if msg := chunkErrorMessage(&chunk); msg != "" {
			r.streamErrMsg = msg
			break
		}
		// Usage arrives in its own trailing chunk (choices empty) when
		// stream_options.include_usage=true. Capture and keep going — there
		// may still be a [DONE] line after it.
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
		// llama.cpp timings (prefer over usage cached_tokens — it carries both
		// sides directly): prompt_n = evaluated, cache_n = reused.
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
			// Stream rule check. Deliberately AFTER on(): the tokens up to the match
			// are already on the user's screen, and hiding them would make the
			// "response discarded, retrying" notice unexplainable. Matching only
			// content (not reasoning, not tool-call arguments) is the whole design —
			// see rules.go. On a hit we stop reading the body; the caller's deferred
			// Close tears down the connection, which is what stops the server
			// generating.
			if fired := matcher.feed(delta.Content); fired != nil {
				r.firedRule = fired
				break
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
			// Surface the delta live. The name rides only the ID-bearing first
			// chunk, so read it back off the accumulator rather than from tc, which
			// is empty for every continuation.
			if onArgs != nil && tc.Function.Arguments != "" && len(r.calls) > 0 {
				onArgs(len(r.calls)-1, r.calls[len(r.calls)-1].Function.Name, tc.Function.Arguments)
			}
		}
	}
	r.scanErr = scanner.Err()
	return r
}

// recordStreamStats folds one call's server-reported usage and timings into the
// turn's running totals for the "✅ Done" line, and runs the prefix-cache rewind
// check on tool-loop calls. sid="" (probes/tests) has no session, so it is a
// no-op there; so is a stream that broke before the usage chunk, whose counts
// are all 0.
func (a *agent) recordStreamStats(sid, connLabel string, conn *LLMConnection, r *streamResult) {
	sess := a.getSession(sid)
	if sess == nil || conn.noTurnStats {
		return
	}
	// Derive evaluated (sent-but-not-cached) from cached_tokens when timings are
	// absent; -1 means no cache info reported. This is the only number we keep —
	// the gross prompt_tokens (cached prefix re-counted each call) is not summed.
	if r.evaluatedTokens < 0 && r.cachedTokens >= 0 && r.promptTokens > 0 {
		r.evaluatedTokens = r.promptTokens - r.cachedTokens
	}
	sess.addTurnTokens(r.promptTokens, r.completionTokens, r.evaluatedTokens)
	// Prefix-cache rewind check, tool-loop calls only. Each is the previous
	// call's messages plus an append, so the server should hand back
	// everything the previous call sent (cached ≈ its prompt) and evaluate
	// only the tail. A big shortfall means the prompt was re-rendered behind
	// our backs; logged per call, and reported once on the Done line.
	if conn.cacheLineage && r.promptTokens > 0 {
		render := renderKey(conn.ExtraBody)
		if conn.noPrefill {
			// A forced tool choice is a rendering on this server (see noPrefill),
			// so it belongs in the key the rewind check compares.
			render += fmt.Sprintf(" tool_choice=%v", conn.ExtraBody["tool_choice"])
		}
		rw := sess.noteCacheLineage(r.promptTokens, r.cachedTokens, render, time.Now())
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
			case rw.renderChanged && conn.noPrefill:
				cause = "Expected on this server: it keeps one prompt state per rendering and has no cache-preserving way to switch " +
					"thinking off, so a role switch re-reads what the other role appended since. The re-read is real; nothing is misconfigured."
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
				rw.tokens, r.promptTokens, r.cachedTokens, humanDuration(rw.idle.Milliseconds()), cause)
		}
	}
	// Prefer the server's measured times over our TTFT proxy (which includes
	// queue + cache-load overhead → understates pp/s).
	pMs, gMs := r.serverPromptMs, r.serverGenMs
	if pMs == 0 && gMs == 0 && !r.firstTokenAt.IsZero() {
		pMs = float64(r.firstTokenAt.Sub(r.readStart).Milliseconds())
		gMs = float64(time.Since(r.firstTokenAt).Milliseconds())
	}
	if pMs > 0 || gMs > 0 {
		sess.addTurnTiming(int64(pMs), int64(gMs))
	}
}

// streamOutcomeError turns how the stream ended into the one error llmStream
// returns. Order matters: the in-band server error is checked before the
// transport error, because gateways emit {"error":…} and THEN drop the socket,
// and "unexpected EOF" would otherwise hide "prompt exceeds n_ctx" and send a
// fatal prompt down the transient-retry path.
//   - firedRule: we abandoned the generation ourselves.
//   - finish_reason="length": AT the max_tokens cap the model is verbose or
//     looping and we bail; BELOW it, it hit the n_ctx ceiling, which is
//     recoverable (errContextCeiling: fold history and retry).
func (a *agent) streamOutcomeError(conn *LLMConnection, reqBody map[string]any, r *streamResult) error {
	switch {
	case r.firedRule != nil:
		// Checked first: we aborted this stream on purpose, so scanErr (the reader
		// stopping mid-body) and a missing finish_reason are consequences of that
		// decision, not independent failures, and either would otherwise shadow the
		// real cause. The token counts stay 0 — we never reached the usage chunk —
		// so an aborted generation is simply not attributed in the turn stats.
		return &streamRuleError{Rule: r.firedRule.Name, Reminder: r.firedRule.Reminder, Matched: r.text.String()}
	case r.streamErrMsg != "":
		return fmt.Errorf("LLM returned an error mid-stream (role=%s, model=%s): %s", conn.Tag, conn.Model, r.streamErrMsg)
	case r.scanErr != nil:
		return fmt.Errorf("reading SSE stream: %w", r.scanErr)
	case r.finishReason == "length":
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
		// A one-token request is a warm-up (prewarm, keepWarm): the single
		// token it can produce is a reasoning token, and that is not the
		// model stuck in <think>, it is the request doing its job.
		if reqMax == 1 {
			return nil
		}
		// The length limit was the n_ctx ceiling, not the cap, when EITHER the
		// generation stopped below the cap (completion < reqMax), OR the prompt
		// left less than a full generation of room (prompt + reqMax > n_ctx). The
		// second form still catches the ceiling when the server omits
		// completion_tokens (completion == 0), so it can't be compared to the cap.
		belowCap := r.completionTokens > 0 && reqMax > 0 && r.completionTokens < reqMax
		mst := a.getMainSlotTokens()
		noRoom := r.promptTokens > 0 && reqMax > 0 && mst > 0 && r.promptTokens+reqMax > mst
		switch {
		case belowCap || noRoom:
			return fmt.Errorf("generation hit the context ceiling (prompt=%d gen=%d, n_ctx=%d, role=%s): %w",
				r.promptTokens, r.completionTokens, mst, conn.Tag, errContextCeiling)
		case r.text.Len() == 0 && len(r.calls) == 0 && r.reasoning.Len() > 0 && thinkingOn(reqBody):
			// All budget went to reasoning with nothing to show — the model looped
			// in <think>. Recoverable: the tool loop retries once with thinking off
			// so it answers directly (see errStuckThinking).
			return fmt.Errorf("model stuck in <think> (%d B reasoning, 0 content/calls, role=%s): %w",
				r.reasoning.Len(), conn.Tag, errStuckThinking)
		default:
			return &capHitError{Cap: reqMax, msg: fmt.Sprintf("LLM hit max_tokens cap (role=%s, model=%s) — response truncated (%d B content, %d B reasoning, %d tool calls). Likely the model is looping or stuck in <think>; raise max_tokens in params_%s, or set chat_template_kwargs.enable_thinking=false for this role if reasoning is dominating the budget",
				conn.Tag, conn.Model, r.text.Len(), r.reasoning.Len(), len(r.calls), conn.Tag)}
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
func (a *agent) logStreamResponse(sid, connLabel string, r *streamResult, err error) {
	text, reasoning := r.text.String(), r.reasoning.String()
	var rb strings.Builder
	if r.promptTokens > 0 || r.completionTokens > 0 || r.finishReason != "" {
		// finish: "stop" = model ended; "length" = hit max_tokens (truncated);
		// "tool_calls" = ended on a tool call; "(none)" = stream broke with no
		// finish_reason (interrupted). Distinguishes a cap/interrupt from a clean end.
		fr := r.finishReason
		if fr == "" {
			fr = "(none)"
		}
		fmt.Fprintf(&rb, "tokens: prompt=%d completion=%d finish=%s", r.promptTokens, r.completionTokens, fr)
		// Cache split when the server reported it: cached = prompt tokens served
		// from the KV cache, evaluated = prompt tokens actually processed. This
		// is THE line for diagnosing prefix-cache misses ("N k uncached" in the
		// Done stats without this split is unattributable).
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

// requestLogSame opens the log entry of a request that repeats the previous
// one's bytes: the conversation is a prefix of the next request, so logging
// every body whole wrote the same 800 KB per call and a session log reached
// 6.8 GB. The entry keeps only the bytes from the first change on; digestLog
// (session_insights) reassembles a full body from the chain when it needs one.
const requestLogSame = "[same as the previous request to this connection for the first "

// requestLogText returns what to log for this request: the whole body for
// the first request on a connection, else the delta against the previous one.
func (a *agent) requestLogText(sid, connLabel string, body []byte) string {
	key := sid + "\x00" + connLabel
	a.logPrevMu.Lock()
	prev := a.logPrev[key]
	if a.logPrev == nil {
		a.logPrev = map[string][]byte{}
	}
	a.logPrev[key] = body
	a.logPrevMu.Unlock()
	return requestLogDelta(prev, body)
}

// requestLogDelta: the bytes of body from its first difference to prev, under
// a line that says how much was the same. A change early in the body is
// called out: it is the prefix cache being lost, which the log is otherwise
// silent about until the token counts come back.
func requestLogDelta(prev, body []byte) string {
	// The wrapper before "messages" (max_tokens, the prefill or thinking
	// switches, tool_choice) changes between renderings while the messages
	// do not, and the server's cache is keyed on the messages. Comparing
	// from the first byte called every rendering switch a lost prefix and
	// logged the whole body for it: 89 times in one afternoon, 2.8 MB each.
	// So the wrapper is logged as is, and the comparison starts at the
	// messages.
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

// llmStream is the core LLM call: the concurrency gate, the status meter, the
// HTTP round trip and its non-200 handling. sid scopes the session log ("" for
// probes and tests). think receives reasoning tokens separately from `on`, so
// chain-of-thought can be shown without entering the message; onArgs receives
// tool-call argument deltas, the only way to watch a terminal tool being
// written. Request building, stream reading and the passes over the result
// each live in their own function above.
//
// One retry lives here, outside the gate llmStreamOnce holds: a server that
// rejects the closed-think continuation says so on the first call that sends
// it, and that 400 is the whole detection. The connection is marked for the
// rest of the process (markNoPrefill) and the call goes again without the
// prefill. A server that accepts the shape never pays anything for this.
func (a *agent) llmStream(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, tools []map[string]any, on, think func(string), onArgs func(idx int, name, delta string)) (string, []toolCall, string, error) {
	// Connections are copied freely (the tool loop's per-subtask copy, the
	// keep-warm's, a stall retry's), and a flag set on one copy reaches none of
	// the others: marking only the copy that saw the 400 left every other copy
	// paying a refused request per call, 126 of them in one afternoon. So the
	// mark lives on the settings entry and is read here, on every call, for
	// whatever copy arrives.
	if conn != nil && conn.noThinkPrefill && !conn.noPrefill && a.prefillRejected(conn) {
		cp := *conn
		cp.noPrefill = true
		conn = &cp
	}
	text, calls, reasoning, err := a.llmStreamOnce(ctx, sid, conn, messages, tools, on, think, onArgs)
	var he *llmHTTPError
	if conn != nil && conn.noThinkPrefill && !conn.noPrefill && errors.As(err, &he) && he.Status == 400 && strings.Contains(he.Body, "continue_final_message") {
		a.markNoPrefill(conn)
		cp := *conn
		cp.noPrefill = true
		return a.llmStreamOnce(ctx, sid, &cp, messages, tools, on, think, onArgs)
	}
	return text, calls, reasoning, err
}

// markNoPrefill records, for the life of the process, that conn's server
// rejects the prefill shape (LLMConnection.noPrefill); prefillRejected reads
// it back for every later call on that server. A settings reload rebuilds the
// entries and forgets it, which costs one more rejected call.
func (a *agent) markNoPrefill(conn *LLMConnection) {
	a.cfgMu.Lock()
	defer a.cfgMu.Unlock()
	for i := range a.settings.LLM {
		e := &a.settings.LLM[i]
		if e.Server == conn.Server && e.Model == conn.Model && !e.noPrefill {
			e.noPrefill = true
			slog.Info("server rejects the closed-think continuation; thinking off is its own tool_choice / enable_thinking from here on, one prompt rendering per role",
				"server", conn.Server, "model", conn.Model)
		}
	}
}

func (a *agent) prefillRejected(conn *LLMConnection) bool {
	a.cfgMu.RLock()
	defer a.cfgMu.RUnlock()
	for i := range a.settings.LLM {
		if e := &a.settings.LLM[i]; e.Server == conn.Server && e.Model == conn.Model {
			return e.noPrefill
		}
	}
	return false
}

func (a *agent) llmStreamOnce(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, tools []map[string]any, on, think func(string), onArgs func(idx int, name, delta string)) (string, []toolCall, string, error) {
	reqBody := buildChatRequest(conn, messages, tools)
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", nil, "", fmt.Errorf("marshalling LLM request body: %w", err)
	}

	// Per-conn concurrency gate: at most `parallel` calls in flight, held for
	// this call only, so a background call gets its turn on a pool of one (the
	// wait shows as "(queued…)"). -1 means no gate (mocks, probes). The channel is
	// bound to a local under cfgMu and released on that local: a prepare phase can
	// swap a.connSems meanwhile, and re-reading it at release would block forever
	// on a new empty channel while the permit sits in the old one.
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
	// of the config). Armed only where the caller asked for it (forToolLoop).
	// See LLMConnection.streamRulesArmed for why this is opt-in and not simply
	// "any call that passes tools".
	var matcher *ruleMatcher
	if conn.streamRulesArmed && len(a.streamRules) > 0 {
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

	// Until the first generated byte the row shows "(sent… Ns)", so a busy
	// server does not read as frozen; then the live "↑bytes ↓tok" estimate.
	// genChars is written by readSSEStream while this reads it, hence atomic.
	// Registered after the status-clear defer so LIFO joins the meter first.
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
	a.logSession(sid, connLabel+" REQUEST", "%s", a.requestLogText(sid, connLabel, body))

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
				Type    string `json:"type"`
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
		return "", nil, "", &llmHTTPError{Status: resp.StatusCode, Body: msg, Type: apiErr.Error.Type, URL: resp.Request.URL.String()}
	}

	res := readSSEStream(resp.Body, conn, matcher, on, think, onArgs, &genChars)
	a.recordStreamStats(sid, connLabel, conn, res)

	// A server that quietly ignores chat_template_kwargs looks exactly like a
	// server that honours it, right up until the reasoning tokens arrive.
	a.warnChatTemplateKwargsIgnored(ctx, sid, conn, reqBody, res.reasoning.Len())

	err = a.streamOutcomeError(conn, reqBody, res)
	// A generation the caller cannot use is decode time spent for nothing, and
	// it is already inside the turn's completion total. Name it, or the Done
	// line reports the worst turns as the most productive ones.
	if err != nil && res.completionTokens > 0 {
		if sess := a.getSession(sid); sess != nil && !conn.noTurnStats {
			sess.addWastedCompletion(res.completionTokens)
		}
	}
	a.logStreamResponse(sid, connLabel, res, err)
	return res.text.String(), res.calls, res.reasoning.String(), err
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
func probeLLM(ctx context.Context, conn *LLMConnection) probeResult {
	if conn == nil {
		return probeResult{}
	}
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	r, _ := probeViaModels(probeCtx, conn)
	// Enrich via /props when reachable: it carries image + ctx metadata AND
	// llama.cpp's per-slot n_ctx / total_slots, none of which /v1/models exposes.
	// Non-llama backends 404 here (ok=false) and we keep the /v1/models result.
	p, ok := probeViaProps(probeCtx, conn, "/props")
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
		if up, upOK := probeViaProps(upCtx, conn, "/props?model="+url.QueryEscape(conn.Model)); upOK {
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

// probeGetJSON issues an authenticated GET against conn and decodes a 200 JSON
// body into v. It reports whether it got that far; false means the server is not
// usable through this endpoint, and the reason is already in the log.
//
// Both probes want exactly this, and writing it out per probe let them drift on
// the half that matters: one logged the non-200 status and body, the other
// returned silently, so a probe that said "unreachable" about a server that was
// plainly up left nothing to read. Now every failure names itself.
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

// probeViaModels asks /v1/models for the configured model: reachability and
// presence always, image support and context size only from llama-swap's
// `status.args`. Other backends return the bare OpenAI shape, which is still
// ok=true (the server is up); /props and settings.toml fill the gap. Every id
// lands in AvailableModels so a missing model can be reported by real names.
func probeViaModels(ctx context.Context, conn *LLMConnection) (probeResult, bool) {
	var models struct {
		Data []struct {
			ID     string `json:"id"`
			Status struct {
				Args []string `json:"args"`
			} `json:"status"`
		} `json:"data"`
	}
	if !probeGetJSON(ctx, conn, "/v1/models", "probeViaModels", &models) {
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
func probeViaProps(ctx context.Context, conn *LLMConnection, path string) (probeResult, bool) {
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
	if !probeGetJSON(ctx, conn, path, "probeViaProps", &props) {
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

// connFor resolves the connection for the role. Every session runs on
// LLM[0], whose KV cache owns the conversation prefix. Concurrency is enforced
// by per-conn semaphores in llmStream; MainLLM returns a value copy, safe to use
// after the lock is released.
func (a *agent) connFor(role string) *LLMConnection {
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

// summaryCooldown is how long a struck-out summariser stays out of rotation.
// Bounded, because "the rest of the run" retired a healthy server for 16 hours
// after two timeouts. A call that fails again renews it, so a server that
// really is down costs one note per cooldown.
const summaryCooldown = 10 * time.Minute

// connForBackgroundLLM returns the connection for background work: the first
// `purpose = "summary"` entry with free capacity, else LLM[0] (labelled llm[1]
// when it has two or more slots, so the meter shows the work off the foreground
// turn). The capacity peek is racy by design: falling back beats queueing
// behind somebody else's call. The second return reports that fallback, which
// is the caller's cue for a prefix-extension prompt that reuses the foreground
// cache instead of evicting it, and is what makes a single slot viable.
func (a *agent) connForBackgroundLLM() (*LLMConnection, bool) {
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
	if c != nil && a.settings.LLM[0].parallelCap() >= 2 {
		c.Slot = 1
	}
	return c, true
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
