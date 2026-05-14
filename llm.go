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

	// Per-session log: request header + body, then we'll tee the raw SSE
	// response into the same file as it streams. Closed on return.
	logF := a.sessionLog(sid)
	if logF != nil {
		defer logF.Close()
		fmt.Fprintf(logF, "\n=== %s [%s] REQUEST ===\n%s\n=== RESPONSE ===\n",
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

	// Tee SSE bytes into the session log so the unparsed wire output is
	// preserved. Parser still reads the same stream because TeeReader splits
	// every Read between the consumer and the file.
	var respReader io.Reader = resp.Body
	if logF != nil {
		respReader = io.TeeReader(resp.Body, logF)
	}
	scanner := bufio.NewScanner(respReader)
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
		return fullText.String(), calls, fmt.Errorf("reading SSE stream: %w", err)
	}

	return fullText.String(), calls, nil
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

// pickAvailable picks the first LLMConnection in role/tier preference order
// whose server reports a free slot. If all candidates are busy or
// unreachable, falls back to the first preference — the caller's LLM call
// will then queue server-side or surface a clear error. Returns nil only
// when no connections are configured.
func (a *agent) pickAvailable(ctx context.Context, role, tier string) *LLMConnection {
	cs := a.settings.LLMCandidates(role, tier)
	if len(cs) == 0 {
		return nil
	}
	// Drop connections the startup probe couldn't reach so they don't burn
	// a /slots timeout every call. If every candidate is marked dead, skip
	// slot probing too — return the first preference so the caller surfaces
	// a clear "LLM unreachable" error instead of waiting 2s per server.
	if len(a.connReachable) > 0 {
		var alive []LLMConnection
		for _, c := range cs {
			if a.connReachable[connKey(&c)] {
				alive = append(alive, c)
			}
		}
		if len(alive) == 0 {
			return &cs[0]
		}
		cs = alive
	}
	for i := range cs {
		if a.hasSlot(ctx, &cs[i]) {
			return &cs[i]
		}
	}
	return &cs[0]
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

// parallel runs fn for each index [0, n) with up to maxParallel goroutines.
func parallel(n int, fn func(i int)) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, maxParallel)
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

// trimJSON strips markdown code fences from LLM JSON responses.
func trimJSON(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	return strings.TrimSpace(s)
}

type toolLoopResult struct {
	Text     string
	ToolUses []ToolUse
}

// runToolLoop runs the agentic tool loop: send to LLM, execute tool calls, repeat.
// When filter.readOnly is true, only read-only tools are sent and tokens are
// not streamed to Zed.
func (a *agent) runToolLoop(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, filter toolFilter) (toolLoopResult, error) {
	tools := llmToolDefinitionsFiltered(filter)
	var on onToken
	if !filter.readOnly {
		on = func(token string) {
			if sid != "" {
				a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(token)))
			}
		}
	}

	var res toolLoopResult
	var allText strings.Builder
	for {
		text, calls, err := a.llmStream(ctx, sid, conn, messages, tools, on)
		if err != nil {
			res.Text = allText.String()
			return res, err
		}
		allText.WriteString(text)

		if len(calls) == 0 {
			res.Text = allText.String()
			return res, nil
		}

		messages = append(messages, llmMessage{
			Role:      "assistant",
			Content:   text,
			ToolCalls: calls,
		})

		for _, tc := range calls {
			result := a.executeTool(ctx, sid, tc)
			tu := ToolUse{
				Name:   tc.Function.Name,
				Input:  tc.Function.Arguments,
				Output: result,
			}
			res.ToolUses = append(res.ToolUses, tu)

			// Save incrementally so tool results survive crashes.
			if sess := a.getSession(sid); sess != nil {
				sess.AppendToolUse(tu)
				_ = sess.Save()
			}
			messages = append(messages, llmMessage{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}
}
