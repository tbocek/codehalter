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
func (a *agent) llmStream(ctx context.Context, conn *LLMConnection, messages []llmMessage, tools []map[string]any, on onToken) (string, []toolCall, error) {
	reqBody := map[string]any{
		"model":    conn.Model,
		"stream":   true,
		"messages": messages,
	}
	if tools != nil {
		reqBody["tools"] = tools
	}

	body, _ := json.Marshal(reqBody)
	if f, err := os.OpenFile("/tmp/codehalter_debug.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
		fmt.Fprintf(f, "\n=== %s ===\n%s\n", time.Now().Format(time.RFC3339), string(body))
		f.Close()
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", conn.URL, bytes.NewReader(body))
	if err != nil {
		return "", nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if conn.Token != "" {
		httpReq.Header.Set("Authorization", "Bearer "+conn.Token)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", nil, fmt.Errorf("LLM returned %d: %s", resp.StatusCode, b)
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
	resp, err := getWithAuth(ctx, modelsURL, conn.Token)
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
	resp, err := getWithAuth(ctx, propsURL, conn.Token)
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

// llmSimple sends a no-tools LLM call, logs to stderr.
func (a *agent) llmSimple(ctx context.Context, conn *LLMConnection, messages []llmMessage) (string, error) {
	text, _, err := a.llmStream(ctx, conn, messages, nil, func(token string) {
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

// runToolLoop runs the agentic tool loop: send to LLM, execute tool calls, repeat.
// If readOnly is true, only read-only tools are sent and tokens are not streamed to Zed.
type toolLoopResult struct {
	Text     string
	ToolUses []ToolUse
}

func (a *agent) runToolLoop(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, readOnly bool) (toolLoopResult, error) {
	return a.runToolLoopFiltered(ctx, sid, conn, messages, toolFilter{readOnly: readOnly})
}

func (a *agent) runToolLoopFiltered(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, filter toolFilter) (toolLoopResult, error) {
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
		text, calls, err := a.llmStream(ctx, conn, messages, tools, on)
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
