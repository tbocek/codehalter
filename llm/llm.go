// Package llm is codehalter's client for OpenAI-compatible chat-completion
// servers: the settings of one [[llm]] entry (Conn), the request body, the SSE
// stream reader and the metadata probes. Running a call for a session (slots,
// the status line, the turn's stats, stream rules, retries) is the agent's, in
// the main package.
package llm

import (
	"bufio"
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

// DefaultMaxTokens is the max_tokens injected into an LLM request when the
// user's params block doesn't set one. Bounds a runaway completion that loops
// inside a single LLM round-trip — the per-tool-loop iteration cap can't help
// there. 8192 is generous headroom (execute ~2-4k, plan/verify <1k); override
// per-role with `max_tokens` inside params_thinking / params_execute.
const DefaultMaxTokens = 8192

// Conn describes one llama.cpp/OpenAI-compatible endpoint.
//
// Sampler params can be split by role: `params_thinking` for plan/title/
// history (higher temperature, exploratory) and `params_execute` for
// execute/verify/document/summarize (lower temperature, follow-instruction).
// `params` is the legacy single-set field — still honoured as the fallback
// when the role-specific variant is empty. Each role-specific set hits the
// SAME prefix cache on the server because sampler params never enter the KV
// cache key — only prompt tokens do.
//
// Parallel is the per-conn concurrent-call cap. Each in-flight llmStream
// acquires one of N tokens from this conn's semaphore; excess calls block
// until a token is released. Held *per LLM call*: between calls (during local
// tool dispatch) the conn is free for another caller. Optional for llama.cpp:
// probeAllLLMs auto-fills it from /props total_slots (-np) when left at 0. Set
// it explicitly only for backends that don't report slots (vLLM, OpenAI, …) or
// to cap concurrency below the server's capacity; 0 with no detection means 1.
type Conn struct {
	// Server is the base URL of the OpenAI-compatible server — host root plus
	// any reverse-proxy path prefix, e.g. "http://localhost:8080" or
	// "https://gw.example/myllm". codehalter appends the API paths itself
	// (see endpoint): /v1/chat/completions for completions, /v1/models and the
	// root-level /props for probing. Do NOT put /v1/chat/completions here.
	Server string `toml:"server"`
	APIKey string `toml:"api_key,omitempty"`
	Model  string `toml:"model"`
	Tag    string `toml:"tag,omitempty"`
	// Purpose designates which non-foreground work routes to this entry.
	// "summary" sends the per-turn summariser here instead of LLM[0].
	// Empty means no designated background work.
	//
	// Named explicitly rather than inferred as "the first free entry after
	// LLM[0]": with two extras the inferred rule sends the summariser to
	// whichever happens to be idle, a small fast model on one turn, a slow
	// reasoning model the next. Naming the entry makes the routing stable and
	// lets the summariser live on a machine picked for it. Marking LLM[0] is allowed and simply means "summarise on the main
	// conn", which is also what no marking at all yields.
	Purpose string `toml:"purpose,omitempty"`

	Parallel       *int           `toml:"parallel,omitempty"`
	Params         map[string]any `toml:"params,omitempty"`
	ParamsThinking map[string]any `toml:"params_thinking,omitempty"`
	ParamsExecute  map[string]any `toml:"params_execute,omitempty"`

	// ContextSize is the model's max prompt+output tokens. Optional — when
	// set, codehalter trusts this and skips metadata-endpoint probing for
	// ctx size. Required for backends that don't expose llama.cpp-style
	// discovery (OpenAI, Ollama, vLLM, OpenWebUI, LiteLLM, …).
	ContextSize *int `toml:"context_size,omitempty"`
	// ImageSupport declares whether the model accepts image inputs.
	// Optional — *bool so unset (probe), true (force on), and false (force
	// off) are distinct. nil falls through to discovery via /props or
	// /v1/models launch args; everywhere else the user must set it
	// explicitly to enable inline image_url blocks.
	ImageSupport *bool `toml:"image_support,omitempty"`

	// ExtraBody is the runtime alias for the role-resolved Params used by
	// llmStream when assembling the OpenAI request body. Populated by
	// connForSession so callers don't have to know which of Params /
	// ParamsThinking / ParamsExecute applies.
	ExtraBody map[string]any `toml:"-"`

	// Slot is the flat display index shown in the live meter and the session-
	// log header as llm[<Slot>]. The foreground turn runs as llm[0]; background
	// work (summariser / git-commit) runs as llm[1] — the same physical
	// connection when there's a single [[llm]] entry with parallel >= 2, a
	// distinct slot so you can see which is in use (llama.cpp assigns the real
	// KV slot). Stamped by MainLLM / ConnAt / connForBackgroundLLM; runtime-only.
	Slot int `toml:"-"`

	// NoThinkPrefill suppresses reasoning by APPENDING a closed think block to
	// the messages instead of changing chat_template_kwargs. Set (on a copy) by
	// WithThinkingDisabled. A kwargs change re-runs the template over the whole
	// conversation, so the server sees a token sequence it has never held and
	// re-prefills from zero; an appended message is an extension, so every token
	// before it still matches. Measured against ai.jos.li on a 13,972-token
	// prompt: enable_thinking=false came back cached=0, the prefill came back
	// cached=13,968 of 13,978 and suppressed reasoning just as completely.
	// Runtime-only.
	NoThinkPrefill bool

	// NoTurnStats excludes this call from the per-turn "✅ Done" usage stats.
	// Set (on a copy) by prewarm: its call logs under the real sid for
	// diagnosability, but a turn that starts while the warm is still streaming
	// resets the counters BEFORE the warm's usage lands, so without this flag
	// the warm's ~10k prefill inflates that turn's "uncached" number.
	// Runtime-only.
	NoTurnStats bool

	// StreamRulesArmed opts this call into the stream-rule check (rules.go): a
	// pattern match aborts the generation mid-token and returns a
	// streamRuleError. Opt-IN rather than on-by-default because a rule abort is
	// only useful where something catches it and re-asks — that is the tool
	// loop's retry ladder and nowhere else. The background summariser, in
	// particular, passes the foreground's full tools array (for prefix-cache
	// reasons, see summariseCall) but has no ladder: a rule firing there would
	// silently downgrade the turn's note to the raw fallback. Set on a copy by
	// runToolLoop (ForToolLoop). Runtime-only.
	StreamRulesArmed bool

	// cacheLineage folds this call into the session's prefix-cache rewind check
	// (Session.noteCacheLineage). Opt-in for the same reason: the check compares
	// this call's cached count against the PREVIOUS call's prompt size, which is
	// only meaningful when the two share a message history. The tool loop's calls
	// do (each is the last plus an append); the background summariser's do not.
	// It runs a one-shot prompt on (usually) another server, and counting it would
	// report a rewind on every turn. Set on a copy by ForToolLoop. Runtime-only.
	CacheLineage bool
}

// samplerParams are the request fields that only steer generation. They never
// reach the server's chat template, so two calls that differ only in these
// render the same tokens and share a KV prefix. Everything else in a params
// table is assumed to change the rendering.
var samplerParams = map[string]bool{
	"frequency_penalty": true, "max_tokens": true, "min_p": true,
	"n": true, "presence_penalty": true, "repeat_penalty": true,
	"seed": true, "stop": true, "temperature": true, "top_k": true, "top_p": true,
}

// RenderKey fingerprints the params that reach the server's chat template:
// everything the role configured except the samplers. Two calls with the same
// key render the same messages to the same tokens, so the second extends the
// first's KV prefix. Two different keys are two different token sequences, and
// a server with one slot can only hold one of them.
//
// Built from the role's params, not from the assembled request body: model,
// messages, tools and stream are codehalter's own and identical by
// construction. "" means "nothing that touches the template was configured".
//
// It exists so a detected rewind can NAME its cause. Without it the log can
// only list the four things that could have done it and let the user guess,
// which is what turned one real diagnosis into an offline analysis of a 397 MB
// session log.
func RenderKey(extra map[string]any) string {
	keep := map[string]any{}
	for k, v := range extra {
		if !samplerParams[k] {
			keep[k] = v
		}
	}
	if len(keep) == 0 {
		return ""
	}
	// encoding/json sorts map keys, so the same params always yield the same
	// key regardless of TOML ordering or map iteration order.
	b, err := json.Marshal(keep)
	if err != nil {
		return ""
	}
	return string(b)
}

// ParamsFor returns the sampler params for the given role, falling back to
// the legacy single `params` set when the role-specific one isn't configured.
// An empty map (nil) is fine — llmStream just won't add any extra body keys.
func (c *Conn) ParamsFor(role string) map[string]any {
	switch role {
	case "thinking":
		if len(c.ParamsThinking) > 0 {
			return c.ParamsThinking
		}
	case "execute":
		if len(c.ParamsExecute) > 0 {
			return c.ParamsExecute
		}
		return c.Params
	}
	return c.Params
}

// endpoint joins the configured server base with an API path, e.g.
// endpoint("/v1/models") → "http://host:8080/v1/models". The user configures
// only Server (the host root); codehalter owns the path layout — the
// OpenAI-compatible /v1/chat/completions and /v1/models, plus llama.cpp's
// root-level /props. Trailing slashes on Server are tolerated.
func (c *Conn) Endpoint(path string) string {
	return strings.TrimRight(c.Server, "/") + path
}

// ParallelCap returns the effective concurrent-call cap for this conn,
// defaulting to 1 when unset or invalid.
func (c *Conn) ParallelCap() int {
	if c.Parallel != nil && *c.Parallel >= 1 {
		return *c.Parallel
	}
	return 1
}

// HTTPClient is used for all LLM API calls. ResponseHeaderTimeout caps the
// wait for the first response byte so a broken TCP connection (e.g. after a
// network switch) doesn't hang for the full OS retransmission window (~15 min).
// The dialer's KeepAlive matches http.DefaultTransport so idle connections are
// probed every 30s. No Client.Timeout: streaming generations run unbounded and
// are cancelled only via the request context.
var HTTPClient = &http.Client{
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

// MetaHTTPClient serves the short metadata requests: the LLM probes, the setup
// check, the update check and download. Same dial and handshake bounds as
// HTTPClient, so a dead route fails in seconds. No ResponseHeaderTimeout on
// purpose: a llama.cpp router answers /props?model= only once the model is
// loaded, which takes minutes, and the probe's own context bounds that wait.
var MetaHTTPClient = &http.Client{
	Transport: &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout: 10 * time.Second,
		IdleConnTimeout:     90 * time.Second,
	},
}

type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type ToolCall struct {
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
			ToolCalls []ToolCall `json:"tool_calls"`
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
			// all-thinking turn look like a stalled stream to ErrStuckThinking.
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

// HTTPError is a non-200 response from the LLM endpoint. It carries the
// status code so callers can distinguish a context-overflow 400 (the tool loop
// recovers from it by summarising completed small turns, see foldHistory)
// from other failures. Error() reproduces the prior bare-string message so logs
// and UI surfaces are unchanged.
type HTTPError struct {
	Status int
	Body   string
	URL    string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("LLM returned %d: %s [URL: %s]", e.Status, e.Body, e.URL)
}

// ErrContextCeiling marks a generation that truncated at the n_ctx ceiling: the
// prompt fit (no 400) but left so little room that the model hit finish=length
// below its max_tokens cap. Same cause as a 400 (context too full), so the tool
// loop recovers it the same way — fold history and retry. Wrapped with %w so
// IsContextFull can detect it.
var ErrContextCeiling = errors.New("generation hit the context ceiling")

// IsContextFull reports whether err means the prompt filled the context: the
// server rejected it outright (HTTP 400) or a generation truncated at the n_ctx
// ceiling (ErrContextCeiling). Both are recovered by folding history and retrying.
func IsContextFull(err error) bool {
	var he *HTTPError
	if errors.As(err, &he) && he.Status == 400 {
		return true
	}
	return errors.Is(err, ErrContextCeiling)
}

// ErrStuckThinking marks a generation that spent its whole max_tokens budget on
// reasoning_content with no message text and no tool calls — the model looped in
// <think> and produced nothing usable. Recoverable: the tool loop retries once on
// a thinking-disabled copy of the connection so the model must answer directly.
// Only raised when thinking was ON, so the retry can't re-trigger it.
var ErrStuckThinking = errors.New("model stuck in reasoning")

// CapHitError marks a generation truncated AT the requested max_tokens cap with
// actual content or tool calls in flight (a reasoning-only burn classifies as
// ErrStuckThinking instead): the model needed more room than the cap allowed, or
// was looping. The partial output is unusable — truncated tool-call JSON can't be
// resumed through the chat API. Recoverable: the tool loop retries once with a
// be-concise nudge, then once more on a doubled cap, before surfacing the failure
// (see the cap ladder in runToolLoopSeeded). Cap carries the request's max_tokens
// so that retry can compute the doubled budget.
type CapHitError struct {
	Cap int
	Msg string
}

func (e *CapHitError) Error() string { return e.Msg }

// AsCapHit returns the CapHitError inside err, or nil when err isn't one.
func AsCapHit(err error) *CapHitError {
	var ce *CapHitError
	if errors.As(err, &ce) {
		return ce
	}
	return nil
}

// IsStuckThinking reports whether err is the <think>-loop stall recovered by a
// thinking-off retry (see ErrStuckThinking).
func IsStuckThinking(err error) bool { return errors.Is(err, ErrStuckThinking) }

// ThinkingOn reports whether the request had reasoning enabled. Absent
// chat_template_kwargs (or an absent enable_thinking) counts as on, since the
// stall is only classified when reasoning_content was actually produced. Two
// things count as off so a retry can't loop: the user's own
// enable_thinking=false, and our own prefilled-and-continued <think></think>.
func ThinkingOn(reqBody map[string]any) bool {
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

// NoThinkPrefillContent is the assistant prefix a thinking-off call continues
// from: an already-closed reasoning block, so the model has no way to open one.
// This is what Qwen3's own template emits for enable_thinking=false, written as
// message content instead of asked for as a template argument.
//
// It is the one model-specific literal on that path. A model whose reasoning
// delimiters are not <think>/</think> needs a different string here, where
// chat_template_kwargs would have delegated that to the server's template. That
// is the price of the append: the payoff is that the prefix cache survives both
// the switch to thinking-off and the switch back, for free.
const NoThinkPrefillContent = "<think>\n\n</think>\n\n"

// WithThinkingDisabled returns a shallow copy of the connection whose next call
// answers directly instead of reasoning. It does NOT touch ExtraBody: llmStream
// appends NoThinkPrefillContent as a trailing assistant message and asks the
// server to continue it, which leaves every earlier token identical and so
// keeps the prefix cache. Slot/Server/Model are unchanged, so it routes to the
// same connSem. The original conn is untouched.
//
// This is how codehalter turns reasoning off for the "execute" role: the
// executor and the documenter both call it, so do the summariser and the
// tool loop's <think>-stall retry. The alternative levers both cost more than
// they save. Qwen's /no_think text switch is unreliable (237 of 388 execute
// responses carrying it reasoned anyway, over one 11.6h session), and
// chat_template_kwargs.enable_thinking=false gives the two roles different
// prompt renderings, which on a one-slot server evicts the other role's KV at
// every phase switch (see ParamsFor, where that trade is costed).
//
// nil in, nil out: connForSession returns nil when no connection is configured
// and callers chain this straight onto it.
func (c *Conn) WithThinkingDisabled() *Conn {
	if c == nil {
		return nil
	}
	cp := *c
	cp.NoThinkPrefill = true
	return &cp
}

// WithMaxTokens returns a shallow copy of the connection with max_tokens
// forced to n in a copied ExtraBody (llmStream copies ExtraBody into the
// request first, so this overrides the role default). Slot/Server/Model are
// unchanged, so it routes to the same connSem. Used by prewarm to cap the
// warming call at a single generated token.
func (c *Conn) WithMaxTokens(n int) *Conn {
	cp := *c
	eb := make(map[string]any, len(c.ExtraBody)+1)
	maps.Copy(eb, c.ExtraBody)
	eb["max_tokens"] = n
	cp.ExtraBody = eb
	return &cp
}

// ForToolLoop returns a shallow copy of the connection marked as a tool-loop
// call: stream rules armed, and the call folded into the session's prefix-cache
// lineage. Both belong to the tool loop alone: it is the one caller with a retry
// ladder that can act on a rule abort, and the one caller whose successive calls
// are guaranteed to be appends to each other (see noteCacheLineage). Nothing
// about the request body changes, so the prefix cache is unaffected.
func (c *Conn) ForToolLoop() *Conn {
	cp := *c
	cp.StreamRulesArmed = true
	cp.CacheLineage = true
	return &cp
}

// WithToolChoiceNone returns a shallow copy of the connection that adds
// tool_choice="none". Used by the prefix-extension summariser: the tools
// array must still ride the request — the chat template renders it into the
// HEAD of the prompt, so omitting it changes the rendered bytes from the very
// first token and evicts the foreground's KV prefix instead of extending it —
// but generation must not be steered into a tool call; the summariser has to
// answer with the note text.
func (c *Conn) WithToolChoiceNone() *Conn {
	cp := *c
	eb := make(map[string]any, len(c.ExtraBody)+1)
	maps.Copy(eb, c.ExtraBody)
	eb["tool_choice"] = "none"
	cp.ExtraBody = eb
	return &cp
}

// BuildChatRequest assembles the OpenAI chat-completions body for one call.
// Split out of llmStream because it is the whole of what goes on the wire and
// nothing else: a pure function of the connection and the messages, so a test
// can assert the shape without standing a server up.
func BuildChatRequest(conn *Conn, messages []Message, tools []map[string]any) map[string]any {
	// Seed with extra_body (per-role sampler/reasoning overrides), then write
	// core fields last so model/messages/stream/tools can't be hijacked from
	// settings.toml.
	reqBody := map[string]any{}
	maps.Copy(reqBody, conn.ExtraBody)
	if _, ok := reqBody["max_tokens"]; !ok {
		reqBody["max_tokens"] = DefaultMaxTokens
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
	if conn.NoThinkPrefill {
		// Append the closed think block and tell the server to continue that
		// message rather than open a fresh assistant turn. Written straight onto
		// reqBody and not into ExtraBody on purpose: RenderKey reads ExtraBody to
		// decide whether two calls asked for different renderings, and this pair
		// must not read as one. Verified streaming against llama.cpp: the
		// continued prefix is not echoed back in the deltas, so the content
		// arrives clean and needs no stripping.
		withPrefill := make([]Message, len(messages), len(messages)+1)
		copy(withPrefill, messages)
		reqBody["messages"] = append(withPrefill, Message{Role: "assistant", Content: NoThinkPrefillContent})
		reqBody["add_generation_prompt"] = false
		reqBody["continue_final_message"] = true
	}
	if tools != nil {
		reqBody["tools"] = tools
	}
	return reqBody
}

// StreamResult is everything one SSE stream produced: the reconstructed
// response, the server's own accounting of it, and how it ended. It exists so
// the three passes that run after the scan — turn stats, outcome
// classification, the RESPONSE log — can each take one value instead of a
// dozen arguments.
type StreamResult struct {
	Text      strings.Builder
	Reasoning strings.Builder
	Calls     []ToolCall

	// finishReason is the server's own word for how generation ended: "stop",
	// "length", "tool_calls", or "" when the stream broke before it said.
	FinishReason string
	// streamErrMsg holds an error the server delivered in-band (HTTP 200, an
	// {"error":…} SSE chunk). Surfaced as the call error so it isn't swallowed
	// as an empty response.
	StreamErrMsg string
	// scanErr is set when the stream broke mid-flight with no in-band error to
	// explain it.
	ScanErr error

	PromptTokens, CompletionTokens int
	// Server-reported cache split (see sseChunk.Timings / PromptTokensDetails).
	// evaluatedTokens = prompt tokens actually run through the model this call;
	// cachedTokens = reused from KV cache. -1 = the server didn't report it.
	EvaluatedTokens, CachedTokens int
	// Server-measured eval/gen times (ms) — exact, vs the TTFT proxy below.
	ServerPromptMs, ServerGenMs float64

	// TTFT (readStart→firstTokenAt) and gen (firstTokenAt→end): the rate timing
	// fallback for when the server sends no _ms.
	ReadStart, FirstTokenAt time.Time
}

// ReadSSEStream consumes the chat-completions event stream, forwarding deltas to
// the caller's sinks as they arrive and accumulating the reconstructed response.
// It returns on [DONE], on an in-band error chunk, when stop says so (checked
// with each content delta), or when the body ends. It never closes the body: the caller's deferred Close is what
// tears the connection down, which is what stops the server generating after a
// rule hit.
//
// genChars counts generated bytes for the live status meter, which reads it from
// another goroutine — hence the pointer and the atomics.
func ReadSSEStream(body io.Reader, conn *Conn, stop func(delta string) bool, on, think func(string), onArgs func(idx int, name, delta string), genChars *int64) *StreamResult {
	r := &StreamResult{EvaluatedTokens: -1, CachedTokens: -1}

	scanner := bufio.NewScanner(body)
	// SSE chunks can carry large tool-call argument blobs; the default 64 KB
	// line limit silently truncates. 4 MB matches common reverse-proxy caps.
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	r.ReadStart = time.Now()
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
			slog.Debug("llm: skipped unparseable SSE frame", "role", conn.Tag, "err", err, "frame", clip(data, 200))
			continue
		}
		// In-band error: the gateway/llama.cpp can return HTTP 200 and put the
		// failure in an {"error":…} chunk (empty Choices). Capture and stop —
		// checked before the empty-Choices skip below, which would drop it and
		// leave the call looking like a silent "(empty response)".
		if msg := chunkErrorMessage(&chunk); msg != "" {
			r.StreamErrMsg = msg
			break
		}
		// Usage arrives in its own trailing chunk (choices empty) when
		// stream_options.include_usage=true. Capture and keep going — there
		// may still be a [DONE] line after it.
		if chunk.Usage != nil {
			if chunk.Usage.PromptTokens > 0 {
				r.PromptTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens > 0 {
				r.CompletionTokens = chunk.Usage.CompletionTokens
			}
			if d := chunk.Usage.PromptTokensDetails; d != nil {
				r.CachedTokens = d.CachedTokens
			}
		}
		// llama.cpp timings (prefer over usage cached_tokens — it carries both
		// sides directly): prompt_n = evaluated, cache_n = reused.
		if chunk.Timings != nil {
			r.EvaluatedTokens = chunk.Timings.PromptN
			r.CachedTokens = chunk.Timings.CacheN
			r.ServerPromptMs = chunk.Timings.PromptMs
			r.ServerGenMs = chunk.Timings.PredictedMs
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		if fr := chunk.Choices[0].FinishReason; fr != "" {
			r.FinishReason = fr
		}

		delta := chunk.Choices[0].Delta
		if delta.ReasoningContent == "" { // fold vLLM's spelling into the standard one
			delta.ReasoningContent = delta.Reasoning
		}

		if r.FirstTokenAt.IsZero() && (delta.Content != "" || delta.ReasoningContent != "" || len(delta.ToolCalls) > 0) {
			r.FirstTokenAt = time.Now()
		}

		if delta.ReasoningContent != "" {
			r.Reasoning.WriteString(delta.ReasoningContent)
			atomic.AddInt64(genChars, int64(len(delta.ReasoningContent)))
			if think != nil {
				think(delta.ReasoningContent)
			}
		}

		if delta.Content != "" {
			r.Text.WriteString(delta.Content)
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
			if stop != nil && stop(delta.Content) {
				break
			}
		}

		for _, tc := range delta.ToolCalls {
			atomic.AddInt64(genChars, int64(len(tc.Function.Name)+len(tc.Function.Arguments)))
			if tc.ID != "" {
				r.Calls = append(r.Calls, tc)
			} else if len(r.Calls) > 0 {
				last := &r.Calls[len(r.Calls)-1]
				last.Function.Arguments += tc.Function.Arguments
			}
			// Surface the delta live. The name rides only the ID-bearing first
			// chunk, so read it back off the accumulator rather than from tc, which
			// is empty for every continuation.
			if onArgs != nil && tc.Function.Arguments != "" && len(r.Calls) > 0 {
				onArgs(len(r.Calls)-1, r.Calls[len(r.Calls)-1].Function.Name, tc.Function.Arguments)
			}
		}
	}
	r.ScanErr = scanner.Err()
	return r
}

// ProbeResult is what a single LLM probe call yields: server reachability,
// model presence (when the endpoint enumerates models), image support, and
// (when discoverable) the model's context window in tokens. Empty value
// means "we could not reach the server at all."
type ProbeResult struct {
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

// Probe checks reachability, model presence, and (best-effort) image
// support + context size via cheap metadata endpoints. /v1/models is the
// universal reachability endpoint — every OpenAI-compatible backend exposes
// it. /props is a llama.cpp-specific enrichment step that fills in image /
// ctx metadata when /v1/models returns the bare OpenAI shape. Backends
// without /props (OpenAI, Ollama, vLLM, OpenWebUI, LiteLLM, …) leave those
// fields unset and the user supplies them via settings.toml (probeAllLLMs
// applies that precedence).
func Probe(ctx context.Context, conn *Conn) ProbeResult {
	if conn == nil {
		return ProbeResult{}
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
func probeGetJSON(ctx context.Context, conn *Conn, path, who string, v any) bool {
	url := conn.Endpoint(path)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		slog.Info(who+": unusable request URL", "url", url, "err", err)
		return false
	}
	if conn.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+conn.APIKey)
	}
	resp, err := MetaHTTPClient.Do(req)
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

// probeViaModels asks /v1/models for the configured model. Always confirms
// reachability + model presence; image_support / context_size only land
// when the response carries llama-swap-style `status.args` (--mmproj /
// --ctx-size). OpenAI/Ollama/vLLM/LiteLLM all 200 here but return the bare
// OpenAI shape, so the caller's /props enrichment + settings.toml fallback
// fills the gap. ok=false only on network / non-200 — a bare response still
// returns ok=true so the caller knows the server is up. Records every
// enumerated id in AvailableModels so renderLLMStatus can show the real names
// when the configured model isn't found.
func probeViaModels(ctx context.Context, conn *Conn) (ProbeResult, bool) {
	var models struct {
		Data []struct {
			ID     string `json:"id"`
			Status struct {
				Args []string `json:"args"`
			} `json:"status"`
		} `json:"data"`
	}
	if !probeGetJSON(ctx, conn, "/v1/models", "probeViaModels", &models) {
		return ProbeResult{}, false
	}
	r := ProbeResult{Reachable: true, ModelKnown: true}
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
func probeViaProps(ctx context.Context, conn *Conn, path string) (ProbeResult, bool) {
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
		return ProbeResult{}, false
	}
	r := ProbeResult{Reachable: true, ContextSize: props.NCtx, TotalSlots: props.TotalSlots}
	if props.Modalities != nil {
		r.ImageSupport = props.Modalities.Vision
	}
	if props.DefaultGenerationSettings != nil {
		r.SlotCtx = props.DefaultGenerationSettings.NCtx
	}
	return r, true
}

// clip shortens s to n bytes for a log line.
func clip(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
