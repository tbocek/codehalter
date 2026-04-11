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
)

// LLM message types for the OpenAI API.

type llmMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
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

// llmRequest sends a request with tools, streams text to Zed.
func (a *agent) llmRequest(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage) (string, []toolCall, error) {
	a.mu.Lock()
	readOnly := a.mode == "discussion"
	a.mu.Unlock()

	return a.llmStream(ctx, conn, messages, llmToolDefinitions(readOnly), func(token string) {
		if sid != "" {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(token)))
		}
	})
}

const maxToolLoopIterations = 10

// runToolLoop runs the agentic tool loop: send to LLM, execute tool calls, repeat.
func (a *agent) runToolLoop(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage) (string, error) {
	var allText strings.Builder
	for i := 0; i < maxToolLoopIterations; i++ {
		text, calls, err := a.llmRequest(ctx, sid, conn, messages)
		if err != nil {
			return allText.String(), err
		}
		allText.WriteString(text)

		if len(calls) == 0 {
			return allText.String(), nil
		}

		messages = append(messages, llmMessage{
			Role:      "assistant",
			Content:   text,
			ToolCalls: calls,
		})

		for _, tc := range calls {
			result := a.executeTool(ctx, sid, tc)
			messages = append(messages, llmMessage{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}
	return allText.String(), fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations)
}
