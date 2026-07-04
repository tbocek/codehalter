package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// crafterHTTPClient mirrors the main package's llmHTTPClient conservatively:
// a generous ResponseHeaderTimeout so a broken TCP path fails instead of
// hanging for the OS retransmission window, but no overall Client.Timeout —
// a 27B model generating a long answer can legitimately run for minutes, and
// the per-call context.Context is the real bound. All calls stream, so the
// header timeout covers only model load (a router holds the request while it
// loads/switches the named model) + prompt processing — generation time is
// unbounded by it.
var crafterHTTPClient = &http.Client{
	Transport: &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		ResponseHeaderTimeout: 300 * time.Second,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	},
}

// llmLog is the wire-level trace: every chat completion (judge and targets,
// success or failure) appends one JSON line with the full messages as sent,
// the raw visible reply (or error), and timing. results.jsonl holds the
// per-claim outcome; this holds what actually went over the wire — the thing
// you read when a verdict looks wrong and you need to see the exact prompts.
// Nil until openLLMLog; logging is then a no-op (e.g. in tests).
var llmLog struct {
	mu sync.Mutex
	f  *os.File
}

// llmLogEntry is one traced call — or, with Chunk set, one in-flight slice of
// it. Chunk lines land roughly every 1KB of generated text so `tail -f` shows
// a long generation live instead of one line at the end; they carry only the
// text delta (repeating Messages every KB would bloat the file), and the final
// line still carries the complete call as before.
type llmLogEntry struct {
	Time       string        `json:"time"`
	DurationMs int64         `json:"duration_ms"`
	Name       string        `json:"name"`  // ModelSpec.Name ("judge", "gemma-4-31b", …)
	Model      string        `json:"model"` // wire model id
	Chunk      bool          `json:"chunk,omitempty"`
	Messages   []chatMessage `json:"messages,omitempty"`
	MaxTokens  *int          `json:"max_tokens,omitempty"`
	Response   string        `json:"response,omitempty"`
	Reasoning  string        `json:"reasoning,omitempty"` // thinking-channel output, if the model emitted any
	Error      string        `json:"error,omitempty"`
}

// openLLMLog opens (append) the trace file. Empty path disables tracing.
func openLLMLog(path string) error {
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open llm log %s: %w", path, err)
	}
	llmLog.f = f
	return nil
}

// writeTrace appends one trace line. A trace-write failure must not fail the
// run (the call itself succeeded), but it must not be silent either — warn on
// stderr.
func writeTrace(e llmLogEntry) {
	row, err := json.Marshal(e)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warn: llm log: marshal trace: %v\n", err)
		return
	}
	llmLog.mu.Lock()
	defer llmLog.mu.Unlock()
	if _, err := fmt.Fprintln(llmLog.f, string(row)); err != nil {
		fmt.Fprintf(os.Stderr, "warn: llm log: write trace: %v\n", err)
	}
}

// logLLM traces one completed call (final line: full request + full reply).
func logLLM(m ModelSpec, req chatRequest, content, reasoning string, callErr error, d time.Duration) {
	if llmLog.f == nil {
		return
	}
	e := llmLogEntry{
		Time:       time.Now().Format(time.RFC3339),
		DurationMs: d.Milliseconds(),
		Name:       m.Name,
		Model:      m.Model,
		Messages:   req.Messages,
		MaxTokens:  req.MaxTokens,
		Response:   content,
		Reasoning:  reasoning,
	}
	if callErr != nil {
		e.Error = callErr.Error()
	}
	writeTrace(e)
}

// logLLMChunk traces one in-flight slice of a streaming call: just the text
// delta since the previous chunk line, stamped with elapsed time since the
// call started so pace is visible.
func logLLMChunk(m ModelSpec, content, reasoning string, elapsed time.Duration) {
	if llmLog.f == nil {
		return
	}
	writeTrace(llmLogEntry{
		Time:       time.Now().Format(time.RFC3339),
		DurationMs: elapsed.Milliseconds(),
		Name:       m.Name,
		Model:      m.Model,
		Chunk:      true,
		Response:   content,
		Reasoning:  reasoning,
	})
}

// chatMessage is one OpenAI chat message. Content is a plain string here — the
// crafter never sends images, so we don't need the polymorphic `any` content
// the main package carries.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatRequest is the request body for POST /v1/chat/completions. Stream is
// always true: with stream=false llama.cpp sends no response headers until the
// ENTIRE generation is done, so any header timeout silently caps generation
// time (measured: a long judge call died at 300s "timeout awaiting response
// headers"). Streaming gets headers right after prompt processing and lets
// generation run as long as it needs.
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature *float64      `json:"temperature,omitempty"`
	TopP        *float64      `json:"top_p,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
}

// streamChunk is one SSE frame of a streamed completion — the delta shapes
// llama.cpp emits, plus the two error shapes that can arrive mid-stream under
// an HTTP 200 (nested OpenAI style and bare top-level message).
type streamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
			// ReasoningContent is the split-off chain-of-thought thinking
			// backends emit. Accumulated separately: it feeds the llm.jsonl
			// trace and the repetition guard, but never the returned answer.
			ReasoningContent string `json:"reasoning_content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
	Message string `json:"message"`
}

// chunkError extracts an in-stream error from an SSE frame, or "" for a normal
// frame. A bare {"message":…} counts only when the frame carries no choices,
// mirroring the main package's caution.
func chunkError(ch *streamChunk) string {
	if ch.Error != nil && ch.Error.Message != "" {
		return ch.Error.Message
	}
	if ch.Message != "" && len(ch.Choices) == 0 {
		return ch.Message
	}
	return ""
}

// Repetition guard: a looping model generates near-identical text windows
// back-to-back. Word-bag Jaccard over consecutive fixed-size windows (same
// measure the main package uses for its stuck-output ladder, utils.go) trips
// after repStrikes consecutive windows ≥ repSimilarity, aborting the call
// instead of streaming garbage until the context fills.
//
// Windows are counted in WORDS, not chars: char-based windows cut words at
// their boundaries ("again" → "a"+"gain"), and on a small-vocabulary loop
// those phantom tokens alone push Jaccard under any sane threshold — measured:
// a 5-word loop scored ~0.7 on 512-char windows purely from boundary splits.
//
// repSimilarity matches the main package's stuckOutputSimilarity: these are
// raw output windows, not short reason strings — long text shares vocabulary
// easily, so only near-identical windows may count. Three 64-word strikes ≈
// 250 words of self-similar output before aborting, and never fires on the
// repetitive-but-progressing JSON the segmenter legitimately emits (each
// claims object shares keys but differs in content words, so consecutive
// windows stay well under 0.9).
const (
	repWindowWords = 64
	repSimilarity  = 0.9
	repStrikes     = 3
)

// errRepetition marks a stream aborted by the repetition guard. chat retries
// such a call once with a "conclude now" nudge appended (measured failure:
// Qwen3.5 thinking looped "Wait, I need to make sure…" verbatim for 14KB) —
// a stream can't be steered mid-generation, so abort-and-renudge is the only
// lever. errors.Is on this decides that retry.
var errRepetition = errors.New("model output looping")

// repetitionNudge is appended to the user message on the retry after a
// repetition abort.
const repetitionNudge = "\n\n(Note: your previous attempt on this task was aborted because its reasoning kept repeating itself without concluding. Keep reasoning brief and decisive — come to a conclusion and output the final answer.)"

// repetitionGuard tokenises the stream into lowercase alphanumeric words
// (same tokenisation as the main package's issueBag), folds them into
// fixed-size word windows, and counts consecutive high-similarity neighbours.
// The partial builder carries a word split across two streamed chunks.
type repetitionGuard struct {
	partial strings.Builder
	window  map[string]bool
	count   int // words in the current window
	total   int // words seen overall (for the error message)
	prevBag map[string]bool
	strikes int
}

// add feeds streamed text and returns an error once repStrikes consecutive
// windows are ≥ repSimilarity Jaccard-similar to their predecessor.
func (g *repetitionGuard) add(s string) error {
	for _, r := range strings.ToLower(s) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			g.partial.WriteRune(r)
		default:
			if err := g.flushWord(); err != nil {
				return err
			}
		}
	}
	return nil
}

// flushWord completes the pending word, and when the window is full compares
// it against the previous one.
func (g *repetitionGuard) flushWord() error {
	if g.partial.Len() == 0 {
		return nil
	}
	if g.window == nil {
		g.window = make(map[string]bool)
	}
	g.window[g.partial.String()] = true
	g.partial.Reset()
	g.count++
	g.total++
	if g.count < repWindowWords {
		return nil
	}
	bag := g.window
	g.window = nil
	g.count = 0
	if g.prevBag != nil && jaccard(bag, g.prevBag) >= repSimilarity {
		g.strikes++
		if g.strikes >= repStrikes {
			return fmt.Errorf("%w after %d words (%d consecutive %d-word windows ≥ %.2f Jaccard)",
				errRepetition, g.total, repStrikes, repWindowWords, repSimilarity)
		}
	} else {
		g.strikes = 0
	}
	g.prevBag = bag
	return nil
}

// wordBag tokenises text into a set of distinct lowercase alphanumeric words —
// punctuation, casing and ordering discarded (same tokenisation as the main
// package's issueBag).
func wordBag(s string) map[string]bool {
	bag := make(map[string]bool)
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			bag[cur.String()] = true
			cur.Reset()
		}
	}
	for _, r := range strings.ToLower(s) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			cur.WriteRune(r)
		default:
			flush()
		}
	}
	flush()
	return bag
}

// jaccard returns |A ∩ B| / |A ∪ B| for two word sets. 1.0 = identical,
// 0.0 = disjoint. Two empty bags are treated as identical.
func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	inter := 0
	for w := range a {
		if b[w] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// doChat POSTs one streamed chat completion and returns the accumulated
// visible content and reasoning-channel text. It surfaces transport,
// HTTP-status, decode, and in-band server errors — the checks that mean "this
// endpoint/model/auth is broken" — but says nothing about the answer's
// content. chat and ping layer their own content rules on top.
//
// Every request names m.Model in the body: that is what makes a router in
// front of several models switch to the right one. Measured against the live
// llama-server router: it simply HOLDS the request while it loads the model
// and then answers — the ResponseHeaderTimeout absorbs that.
//
// Every call is traced to llm.jsonl on every exit path (named returns so the
// defer sees exactly what the caller gets).
func doChat(ctx context.Context, m ModelSpec, reqBody chatRequest) (content, reasoning string, err error) {
	start := time.Now()
	defer func() { logLLM(m, reqBody, content, reasoning, err, time.Since(start)) }()

	// One slot per Parallel: wait HERE, not in the server's queue — llama.cpp
	// holds a queued request without headers, which the transport's header
	// timeout would kill. Held for the whole stream (a slot is busy until the
	// generation ends). Nil sem (unit tests) skips the gate.
	if m.sem != nil {
		select {
		case m.sem <- struct{}{}:
			defer func() { <-m.sem }()
		case <-ctx.Done():
			return "", "", fmt.Errorf("call %s (%s): cancelled while waiting for a free slot: %w", m.Name, m.Model, ctx.Err())
		}
	}

	// Assemble the wire body as a map so m.Params merges verbatim on top of
	// the named fields (a params key wins over Temperature/TopP/MaxTokens).
	reqBody.Stream = true
	bodyMap := map[string]any{
		"model":    reqBody.Model,
		"messages": reqBody.Messages,
		"stream":   true,
	}
	if reqBody.Temperature != nil {
		bodyMap["temperature"] = *reqBody.Temperature
	}
	if reqBody.TopP != nil {
		bodyMap["top_p"] = *reqBody.TopP
	}
	if reqBody.MaxTokens != nil {
		bodyMap["max_tokens"] = *reqBody.MaxTokens
	}
	for k, v := range m.Params {
		bodyMap[k] = v
	}
	body, err := json.Marshal(bodyMap)
	if err != nil {
		return "", "", fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", m.endpoint("/v1/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return "", "", fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if m.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+m.APIKey)
	}

	resp, err := crafterHTTPClient.Do(httpReq)
	if err != nil {
		return "", "", fmt.Errorf("call %s (%s): %w", m.Name, m.Model, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", "", fmt.Errorf("call %s (%s): http %d: %s", m.Name, m.Model, resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var visible, thinking strings.Builder
	var guard repetitionGuard
	// Live trace: flush a chunk line to llm.jsonl every ~traceChunkSize bytes
	// of generated text, so a long generation is observable (`tail -f`) while
	// it runs, not only after. pendResp/pendReason hold the not-yet-flushed
	// deltas; the final logLLM line still carries the complete call.
	const traceChunkSize = 1024
	var pendResp, pendReason strings.Builder
	flushChunk := func() {
		if pendResp.Len()+pendReason.Len() == 0 {
			return
		}
		logLLMChunk(m, pendResp.String(), pendReason.String(), time.Since(start))
		pendResp.Reset()
		pendReason.Reset()
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}
		var ch streamChunk
		if err := json.Unmarshal([]byte(payload), &ch); err != nil {
			return visible.String(), thinking.String(),
				fmt.Errorf("call %s (%s): bad SSE chunk: %w (chunk: %s)", m.Name, m.Model, err, truncate(payload, 200))
		}
		if msg := chunkError(&ch); msg != "" {
			return visible.String(), thinking.String(),
				fmt.Errorf("call %s (%s): server error mid-stream: %s", m.Name, m.Model, msg)
		}
		for _, c := range ch.Choices {
			visible.WriteString(c.Delta.Content)
			thinking.WriteString(c.Delta.ReasoningContent)
			pendResp.WriteString(c.Delta.Content)
			pendReason.WriteString(c.Delta.ReasoningContent)
			if pendResp.Len()+pendReason.Len() >= traceChunkSize {
				flushChunk()
			}
			if err := guard.add(c.Delta.Content + c.Delta.ReasoningContent); err != nil {
				// Returning closes resp.Body, which aborts the server-side
				// generation — the loop stops burning GPU time too.
				flushChunk()
				return visible.String(), thinking.String(), fmt.Errorf("call %s (%s): %w", m.Name, m.Model, err)
			}
		}
	}
	flushChunk()
	if err := sc.Err(); err != nil {
		return visible.String(), thinking.String(), fmt.Errorf("call %s (%s): stream read: %w", m.Name, m.Model, err)
	}
	return visible.String(), thinking.String(), nil
}

// ping verifies a model endpoint is reachable, authorized, and actually serving
// the configured model id, by sending a one-token chat completion. Empty
// content still counts as reachable — a thinking model may burn its single
// token in the reasoning channel; we only care that the call round-trips
// without a transport, HTTP, or server error. Used by the preflight check.
func ping(ctx context.Context, m ModelSpec) error {
	one := 1
	_, _, err := doChat(ctx, m, chatRequest{
		Model:     m.Model,
		Messages:  []chatMessage{{Role: "user", Content: "ping"}},
		MaxTokens: &one,
	})
	return err
}

// chat sends one chat completion to the model and returns the visible
// assistant text. It is the single LLM entry point for the tool's real work —
// the judge and every target model go through here, differing only in ModelSpec.
//
// A call aborted by the repetition guard is not fatal: chat says so on stderr
// and retries once with a nudge to conclude appended to the user message. Only
// a second loop surfaces as an error.
func chat(ctx context.Context, m ModelSpec, system, user string) (string, error) {
	build := func(userText string) []chatMessage {
		var msgs []chatMessage
		if strings.TrimSpace(system) != "" {
			msgs = append(msgs, chatMessage{Role: "system", Content: system})
		}
		return append(msgs, chatMessage{Role: "user", Content: userText})
	}
	req := chatRequest{
		Model:       m.Model,
		Temperature: m.Temperature,
		TopP:        m.TopP,
		MaxTokens:   m.MaxTokens,
	}

	req.Messages = build(user)
	content, reasoning, err := doChat(ctx, m, req)
	if errors.Is(err, errRepetition) {
		fmt.Fprintf(os.Stderr, "warn: %s (%s): %v — nudging model to conclude and retrying once\n", m.Name, m.Model, err)
		req.Messages = build(user + repetitionNudge)
		content, reasoning, err = doChat(ctx, m, req)
	}
	if err != nil {
		return "", err
	}
	content = strings.TrimSpace(content)
	if content == "" {
		// A thinking model that spent its whole budget in the reasoning channel
		// leaves visible content empty; surface that specifically so it isn't
		// mistaken for a network fault.
		if reasoning != "" {
			return "", fmt.Errorf("call %s (%s): empty answer (all output went to the reasoning channel)", m.Name, m.Model)
		}
		return "", fmt.Errorf("call %s (%s): empty answer", m.Name, m.Model)
	}
	return content, nil
}

// chatJSON calls chat and unwraps a JSON object from the reply into out. Weak
// models routinely wrap JSON in ```json fences or add a sentence of preamble,
// so it extracts the outermost {...} span rather than requiring a clean body.
func chatJSON(ctx context.Context, m ModelSpec, system, user string, out any) error {
	reply, err := chat(ctx, m, system, user)
	if err != nil {
		return err
	}
	js := extractJSON(reply)
	if js == "" {
		return fmt.Errorf("no JSON object found in %s reply: %s", m.Name, truncate(reply, 400))
	}
	if err := json.Unmarshal([]byte(js), out); err != nil {
		return fmt.Errorf("parse JSON from %s: %w (extracted: %s)", m.Name, err, truncate(js, 400))
	}
	return nil
}

// extractJSON returns the outermost balanced {...} span in s, ignoring braces
// inside JSON string literals (so a `}` inside a value doesn't end it early),
// or "" if there is no balanced object. Good enough for the fenced / prefixed
// output weak models produce; not a general JSON scanner.
func extractJSON(s string) string {
	start := strings.IndexByte(s, '{')
	if start < 0 {
		return ""
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
	return ""
}

// truncate shortens s to n bytes with an ellipsis, for error messages that
// embed a model's raw reply.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
