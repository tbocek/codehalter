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
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

// onToken is called for each text token. nil means discard.
type onToken func(token string)

// llmStream is the core LLM call. Streams SSE, collects text and tool calls.
// sid scopes the debug log: req body and raw SSE response are appended to
// .codehalter/session_<sid>.log so a single file captures everything that
// went over the wire for a session. Pass "" to disable logging (used by tests
// and pre-session probes).
func (a *agent) llmStream(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, tools []map[string]any, on onToken) (string, []toolCall, error) {
	// Seed with extra_body (per-role sampler/reasoning overrides), then write
	// core fields last so model/messages/stream/tools can't be hijacked from
	// settings.toml.
	reqBody := map[string]any{}
	for k, v := range conn.ExtraBody {
		reqBody[k] = v
	}
	reqBody["model"] = conn.Model
	reqBody["stream"] = true
	reqBody["messages"] = messages
	if tools != nil {
		reqBody["tools"] = tools
	}

	body, _ := json.Marshal(reqBody)

	// Surface "(thinking…)" on the active phase entry while any LLM call is in
	// flight — execute and verify can stall on cache misses too, not just the
	// thinking role. notifyPhaseSuffix is a no-op when no phase is active so
	// summarisation calls between phases don't flash anything.
	a.notifyPhaseSuffix(ctx, sid, " (thinking…)")
	defer a.notifyPhaseSuffix(ctx, sid, "")

	// Per-session log: request header + body. The raw SSE wire is too noisy
	// to read (one event per token, each repeating the full envelope), so we
	// parse the stream and write a single aggregated RESPONSE block at the
	// end. Mid-stream events (HTTP errors, partial state on transport
	// failure) are appended inline below. Closed on return.
	logF := a.sessionLog(sid)
	if logF != nil {
		defer logF.Close()
		fmt.Fprintf(logF, "\n=== %s [%s] REQUEST ===\n%s\n",
			time.Now().Format(time.RFC3339), conn.Tag, string(body))
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
	var calls []toolCall

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
		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta

		if delta.Content != "" {
			fullText.WriteString(delta.Content)
			if on != nil {
				on(delta.Content)
			}
		}

		for _, tc := range delta.ToolCalls {
			if tc.ID != "" {
				calls = append(calls, tc)
			} else if len(calls) > 0 {
				last := &calls[len(calls)-1]
				last.Function.Arguments += tc.Function.Arguments
			}
		}
	}
	if err := scanner.Err(); err != nil {
		writeResponseLog(logF, fullText.String(), calls, err)
		return fullText.String(), calls, fmt.Errorf("reading SSE stream: %w", err)
	}

	writeResponseLog(logF, fullText.String(), calls, nil)
	return fullText.String(), calls, nil
}

// writeResponseLog emits one compact RESPONSE block per LLM call. The raw SSE
// wire is one event per token (~100 lines for a short reply, thousands for a
// tool-call argument blob) — useless for skimming. This collapses all the
// deltas into the reconstructed text and tool calls so the log reads like a
// transcript instead of a packet capture.
func writeResponseLog(logF *os.File, text string, calls []toolCall, streamErr error) {
	if logF == nil {
		return
	}
	fmt.Fprintln(logF, "=== RESPONSE ===")
	if text != "" {
		fmt.Fprintf(logF, "content:\n%s\n", text)
	}
	for i, c := range calls {
		fmt.Fprintf(logF, "tool_call[%d] %s id=%s args=%s\n", i, c.Function.Name, c.ID, c.Function.Arguments)
	}
	if text == "" && len(calls) == 0 {
		fmt.Fprintln(logF, "(empty response)")
	}
	if streamErr != nil {
		fmt.Fprintf(logF, "[stream error] %v\n", streamErr)
	}
}

// probeResult is what a single LLM probe call yields: server reachability,
// model presence (when the endpoint enumerates models), and image support.
// Empty value means "we could not reach the server at all."
type probeResult struct {
	Reachable    bool // got 200 from any probe endpoint
	ModelKnown   bool // /v1/models enumerated models — ModelLoaded is meaningful
	ModelLoaded  bool // the configured model was in the enumeration
	ImageSupport bool
}

// probeLLM checks reachability, model presence, and image support in a single
// HTTP call against cheap metadata endpoints (no inference). Works against two
// llama.cpp topologies:
//
//  1. A llama-swap-style router in front of multiple llama-server instances —
//     GET /v1/models lists every model with its launch args; vision models
//     were started with --mmproj.
//  2. A single llama-server — GET /props reports modalities.vision once an
//     mmproj is loaded.
//
// /v1/models is preferred (it identifies the specific configured model even
// when the router hasn't loaded it yet); /props is the fallback when the
// router endpoint is missing.
func (a *agent) probeLLM(ctx context.Context, conn *LLMConnection) probeResult {
	if conn == nil {
		return probeResult{}
	}
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if r, ok := a.probeViaModels(probeCtx, conn); ok {
		slog.Info("probeLLM: /v1/models", "model", conn.Model, "reachable", r.Reachable, "loaded", r.ModelLoaded, "image", r.ImageSupport)
		return r
	}
	if r, ok := a.probeViaProps(probeCtx, conn); ok {
		slog.Info("probeLLM: /props", "model", conn.Model, "reachable", r.Reachable, "image", r.ImageSupport)
		return r
	}
	slog.Info("probeLLM: unreachable", "url", conn.URL, "model", conn.Model)
	return probeResult{}
}

// probeViaModels asks the router's /v1/models for the configured model and
// (when present) its launch args. `--mmproj` in those args means the server
// loaded a multimodal projector. Returns (result, ok) — ok=false means this
// endpoint could not answer at all (network error / non-200 / model present
// but launch args missing); caller should fall back to /props.
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
		if len(m.Status.Args) == 0 {
			// Plain llama-server: /v1/models returns the model but no args.
			// Fall back to /props for image support.
			return probeResult{}, false
		}
		r.ModelLoaded = true
		for _, arg := range m.Status.Args {
			if arg == "--mmproj" {
				r.ImageSupport = true
				break
			}
		}
		return r, true
	}
	// 200 OK, server enumerated models, but the configured one was not in
	// the list — that's a definitive "not loaded", no fallback needed.
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
	}
	if err := json.NewDecoder(resp.Body).Decode(&props); err != nil {
		return probeResult{}, false
	}
	r := probeResult{Reachable: true}
	if props.Modalities != nil {
		r.ImageSupport = props.Modalities.Vision
	}
	return r, true
}

// pickAvailable resolves a connection for the session and role. Tier is
// derived from the session's depth (depth>0 → subagent → [[subllm]],
// otherwise → main → [llm]). Subagent sessions carry a PinnedSubLLMIdx
// assigned at creation by launch_subagent; when set we route every call from
// that session to settings.SubLLM[PinnedSubLLMIdx] regardless of role tag,
// so the slot's prefix cache stays warm across plan/execute/verify switches.
// Unpinned (main, tests) falls through to LLMCandidates + round-robin across
// alive servers. Returns nil only when no connections exist.
func (a *agent) pickAvailable(ctx context.Context, sid SessionId, role string) *LLMConnection {
	sess := a.getSession(sid)

	// Subagent with a pinned [[subllm]] entry: hard route, ignore tag match.
	// Cache consistency wins over per-role sampler tuning — the parent's
	// launch_subagent already picked a slot for this session, and bouncing
	// across entries (the old tag-match behaviour) evicted the prefix cache
	// on every plan/execute transition.
	if sess != nil && sess.Depth > 0 && sess.PinnedSubLLMIdx >= 0 && sess.PinnedSubLLMIdx < len(a.settings.SubLLM) {
		c := a.settings.SubLLM[sess.PinnedSubLLMIdx]
		c.ExtraBody = c.paramsFor(role)
		c.Tag = role
		return &c
	}

	tier := "main"
	if sess != nil && sess.Depth > 0 {
		tier = "subagent"
	}

	cs := a.settings.LLMCandidates(role, tier)
	if len(cs) == 0 {
		return nil
	}

	// Drop connections the startup probe couldn't reach so we don't burn a
	// /slots timeout per call. If every candidate is dead, return the first
	// so the caller surfaces a clear "LLM unreachable" error instead of
	// waiting 2s per server.
	alive := cs
	if len(a.connReachable) > 0 {
		alive = alive[:0:0]
		for _, c := range cs {
			if a.connReachable[connKey(&c)] {
				alive = append(alive, c)
			}
		}
		if len(alive) == 0 {
			return &cs[0]
		}
	}

	start := int(a.pickRotor.Add(1)-1) % len(alive)
	for i := 0; i < len(alive); i++ {
		idx := (start + i) % len(alive)
		if a.hasSlot(ctx, &alive[idx]) {
			return &alive[idx]
		}
	}
	return &alive[start]
}

// hasSlot returns true if the server has at least one idle slot, or if slot
// state cannot be determined (no /slots endpoint, unparseable response).
// Returns false only when /slots definitively reports every slot busy or the
// server is unreachable. /slots requires `--slots` on llama-server; older
// builds and cloud endpoints return 404/501 which we treat as "unknown =
// available" so they keep working without slot-awareness.
func (a *agent) hasSlot(ctx context.Context, conn *LLMConnection) bool {
	slotsURL, ok := deriveServerURL(conn.URL, "/slots")
	if !ok {
		return true
	}
	probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	resp, err := getWithAuth(probeCtx, slotsURL, conn.APIKey)
	if err != nil {
		slog.Info("hasSlot: unreachable", "url", slotsURL, "err", err)
		return false
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusNotFound, http.StatusNotImplemented, http.StatusForbidden, http.StatusUnauthorized:
		return true
	}
	if resp.StatusCode != http.StatusOK {
		slog.Info("hasSlot: non-OK", "url", slotsURL, "status", resp.StatusCode)
		return false
	}
	var slots []struct {
		IsProcessing bool `json:"is_processing"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&slots); err != nil {
		return true
	}
	if len(slots) == 0 {
		return true
	}
	for _, s := range slots {
		if !s.IsProcessing {
			return true
		}
	}
	slog.Info("hasSlot: all busy", "url", slotsURL, "slots", len(slots))
	return false
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
func (a *agent) llmSimple(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage) (string, error) {
	text, _, err := a.llmStream(ctx, sid, conn, messages, nil, func(token string) {
		fmt.Fprint(os.Stderr, token)
	})
	fmt.Fprintln(os.Stderr)
	return text, err
}

const maxParallel = 10

// parallel runs fn for each index [0, n) with up to `cap` concurrent
// goroutines. cap<=0 falls back to maxParallel. Callers that know an upper
// bound (e.g. launch_subagent capping at the number of [[subllm]] entries)
// pass it explicitly so excess work queues instead of contending for slots.
func parallel(n, cap int, fn func(i int)) {
	if cap <= 0 {
		cap = maxParallel
	}
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
	// StartedAt is when the first llmStream call of this loop began.
	// DurationMs is the cumulative wall-clock time spent in llmStream calls
	// across all iterations (excludes tool execution). Phase is the pipeline
	// stage tag passed in by the caller ("plan", "execute", "verify",
	// "document", "subagent"). Callers that own the final assistant message
	// (prompt.go, runSubagent) use these to stamp the message they create
	// after the loop returns.
	StartedAt  time.Time
	DurationMs int64
	Phase      string
}

// maxToolLoopIterations bounds runToolLoop so a model that keeps producing
// "different enough" tool calls (e.g. path variations to dodge perceived
// repetition) can't spin forever. One iteration is one LLM round-trip; a
// complex execute pass is usually 10-20, so 50 leaves comfortable headroom
// while still bailing on genuine runaways.
const maxToolLoopIterations = 50

// runToolLoopJSON runs the tool loop and unmarshals the response into dst.
// Small models routinely produce JSON with surrounding prose; on parse
// failure we send one corrective follow-up ("reply with ONLY JSON") and try
// again before giving up. The returned toolLoopResult always carries the
// merged tool uses from both passes so callers don't lose visibility.
//
// Text tokens never stream to the UI here — JSON is internal machinery, and
// the caller renders the parsed result (renderSteps for plans, verify
// outcome for verify). Tool calls still surface as cards so the user sees
// what the planner/verifier is probing.
func (a *agent) runToolLoopJSON(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, filter toolFilter, phase string, dst any) (toolLoopResult, error) {
	noop := func(string) {}
	res, err := a.runToolLoopOn(ctx, sid, conn, messages, filter, phase, noop)
	if err != nil {
		return res, err
	}
	if err := json.Unmarshal([]byte(trimJSON(res.Text)), dst); err == nil {
		return res, nil
	}
	slog.Info("JSON parse failed; retrying with corrective", "snippet", truncate(res.Text, 200))
	fixMsgs := append([]llmMessage(nil), messages...)
	fixMsgs = append(fixMsgs,
		llmMessage{Role: "assistant", Content: res.Text},
		llmMessage{Role: "user", Content: "Your previous reply was not valid JSON. Reply with ONLY the JSON object — first character `{`, last character `}`, nothing before or after. No prose, no markdown fences."},
	)
	res2, err := a.runToolLoopOn(ctx, sid, conn, fixMsgs, filter, phase, noop)
	res.Text = res2.Text
	res.ToolUses = append(res.ToolUses, res2.ToolUses...)
	res.DurationMs += res2.DurationMs
	if err != nil {
		return res, err
	}
	if err := json.Unmarshal([]byte(trimJSON(res2.Text)), dst); err != nil {
		return res, fmt.Errorf("non-JSON after retry: %w", err)
	}
	return res, nil
}

// runToolLoop runs the agentic tool loop with the default token-streaming
// callback so the user sees execute / document phases live. The internal
// JSON phases (planner, verifier) call runToolLoopOn directly with a no-op.
func (a *agent) runToolLoop(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, filter toolFilter, phase string) (toolLoopResult, error) {
	on := func(token string) {
		if sid != "" {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(token)))
		}
	}
	return a.runToolLoopOn(ctx, sid, conn, messages, filter, phase, on)
}

// runToolLoopOn is the core agentic tool loop: send to LLM, execute tool
// calls, repeat. Callers supply the per-token callback (default streaming
// for runToolLoop, no-op for runToolLoopJSON). The phase string ("plan",
// "execute", "verify", "document", "subagent") flows onto the trailing
// assistant message via MarkLastAssistantTiming together with the
// cumulative llmStream wall-clock — so session.toml records who ran the
// turn and how much of its time was model generation vs tool execution.
func (a *agent) runToolLoopOn(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, filter toolFilter, phase string, on func(string)) (toolLoopResult, error) {
	tools := llmToolDefinitionsFiltered(filter)

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
	for iter := 0; ; iter++ {
		if iter >= maxToolLoopIterations {
			res.Text = allText.String()
			stampTiming()
			return res, fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations)
		}
		streamStart := time.Now()
		if res.StartedAt.IsZero() {
			res.StartedAt = streamStart
		}
		text, calls, err := a.llmStream(ctx, sid, conn, messages, tools, on)
		genElapsed += time.Since(streamStart)
		if err != nil {
			res.Text = allText.String()
			stampTiming()
			return res, err
		}
		allText.WriteString(text)

		if len(calls) == 0 {
			res.Text = allText.String()
			stampTiming()
			return res, nil
		}

		messages = append(messages, llmMessage{
			Role:      "assistant",
			Content:   text,
			ToolCalls: calls,
		})

		for _, tc := range calls {
			// Subagent sessions don't surface their own UI (Zed doesn't know
			// their sid), so forward a one-liner to the parent before each
			// tool call. Gives the user a live feed of what each subagent is
			// up to instead of just "Starting…" → 5 minutes → "Done".
			if sess := a.getSession(sid); sess != nil && sess.ParentID != "" && sess.DisplayLabel != "" {
				a.sendUpdate(ctx, sess.ParentID, AgentMessageChunk(TextBlock(
					fmt.Sprintf("[%s] %s %s\n\n", sess.DisplayLabel, tc.Function.Name, truncate(tc.Function.Arguments, 80)))))
			}
			started := time.Now()
			result, failed := a.executeTool(ctx, sid, tc)
			tu := ToolUse{
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
			// Small models routinely keep retrying a failing run_command
			// without consulting the SKILL-*.md docs that were loaded into
			// the system prompt at session start. Re-surface them at the
			// moment the model would benefit most: right after the failure.
			// Hint is appended to the live tool result only — not to the
			// stored ToolUse.Output — so session.toml stays clean.
			content := result
			if failed {
				if hint := a.failureSkillHint(sid, tc.Function.Name); hint != "" {
					content = result + "\n\n" + hint
				}
			}
			messages = append(messages, llmMessage{
				Role:       "tool",
				Content:    content,
				ToolCallID: tc.ID,
			})
		}
	}
}
