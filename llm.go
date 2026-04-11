package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// LLM message types for the OpenAI API.

type llmMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
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

	return fullText.String(), calls, nil
}

// llmSimple sends a no-tools LLM call, logs to stderr.
func (a *agent) llmSimple(ctx context.Context, conn *LLMConnection, messages []llmMessage) (string, error) {
	text, _, err := a.llmStream(ctx, conn, messages, nil, func(token string) {
		fmt.Fprint(os.Stderr, token)
	})
	fmt.Fprintln(os.Stderr)
	return text, err
}


const maxToolLoopIterations = 20

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
	for i := 0; i < maxToolLoopIterations; i++ {
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
	res.Text = allText.String()
	return res, fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations)
}
