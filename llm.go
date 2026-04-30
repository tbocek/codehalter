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

// probeImageSupport determines whether the configured model accepts images,
// using cheap metadata endpoints (no inference). Works against two llama.cpp
// topologies:
//
//  1. A llama-swap-style router in front of multiple llama-server instances —
//     GET /v1/models lists every model with its launch args; vision models
//     were started with --mmproj.
//  2. A single llama-server — GET /props reports modalities.vision once an
//     mmproj is loaded.
//
// We try /v1/models first (it identifies the specific configured model even
// when the router hasn't loaded it yet) and fall back to /props.
func (a *agent) probeImageSupport(ctx context.Context, conn *LLMConnection) bool {
	if conn == nil {
		return false
	}
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if supported, known := a.probeViaModels(probeCtx, conn); known {
		slog.Info("probeImageSupport: via /v1/models", "model", conn.Model, "vision", supported)
		return supported
	}
	if supported, known := a.probeViaProps(probeCtx, conn); known {
		slog.Info("probeImageSupport: via /props", "model", conn.Model, "vision", supported)
		return supported
	}
	slog.Info("probeImageSupport: indeterminate", "model", conn.Model)
	return false
}

// probeViaModels asks the router's /v1/models for the configured model's
// launch args. `--mmproj` in those args means the server loaded a multimodal
// projector. Returns (supported, known) — known=false means this endpoint
// could not answer (e.g. a plain llama-server whose /v1/models omits args).
func (a *agent) probeViaModels(ctx context.Context, conn *LLMConnection) (bool, bool) {
	modelsURL, ok := deriveServerURL(conn.URL, "/v1/models")
	if !ok {
		return false, false
	}
	resp, err := getWithAuth(ctx, modelsURL, conn.APIKey)
	if err != nil {
		slog.Info("probeViaModels: request failed", "url", modelsURL, "err", err)
		return false, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false, false
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
		return false, false
	}
	for _, m := range models.Data {
		if m.ID != conn.Model {
			continue
		}
		if len(m.Status.Args) == 0 {
			return false, false
		}
		for _, arg := range m.Status.Args {
			if arg == "--mmproj" {
				return true, true
			}
		}
		return false, true
	}
	return false, false
}

// probeViaProps reads llama-server's /props and checks modalities.vision.
func (a *agent) probeViaProps(ctx context.Context, conn *LLMConnection) (bool, bool) {
	propsURL, ok := deriveServerURL(conn.URL, "/props")
	if !ok {
		return false, false
	}
	resp, err := getWithAuth(ctx, propsURL, conn.APIKey)
	if err != nil {
		slog.Info("probeViaProps: request failed", "url", propsURL, "err", err)
		return false, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 256))
		slog.Info("probeViaProps: non-OK", "url", propsURL, "status", resp.StatusCode, "body", string(body))
		return false, false
	}
	var props struct {
		Modalities *struct {
			Vision bool `json:"vision"`
		} `json:"modalities"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&props); err != nil {
		return false, false
	}
	if props.Modalities == nil {
		return false, false
	}
	return props.Modalities.Vision, true
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
