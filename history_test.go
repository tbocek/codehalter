package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// mockLLM stands up an httptest server that accepts OpenAI chat-completions
// requests and returns a queued SSE response for each call. Tests queue one
// response per LLM call they expect; an unexpected call fails the test.
type mockLLM struct {
	ts    *httptest.Server
	resps []string

	mu   sync.Mutex
	reqs []map[string]any // captured request bodies, in order
	idx  atomic.Int32
	t    *testing.T
}

func newMockLLM(t *testing.T, responses ...string) *mockLLM {
	t.Helper()
	m := &mockLLM{resps: responses, t: t}
	m.ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Runtime callers probe /slots before each LLM call. Mock doesn't
		// implement it — 404 lets connForSession treat the server as "unknown,
		// assume available" so the chat-completions path still runs.
		if r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("mockLLM: decode request body: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		m.mu.Lock()
		m.reqs = append(m.reqs, body)
		m.mu.Unlock()

		i := int(m.idx.Add(1)) - 1
		if i >= len(m.resps) {
			t.Errorf("mockLLM: unexpected call %d (only %d responses queued)", i+1, len(m.resps))
			http.Error(w, "no response queued", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(m.resps[i]))
	}))
	return m
}

func (m *mockLLM) Close() { m.ts.Close() }

func (m *mockLLM) conn(name string) *LLMConnection {
	return &LLMConnection{Tag: name, Server: m.ts.URL, Model: "test-model"}
}

func (m *mockLLM) callCount() int { return int(m.idx.Load()) }

func (m *mockLLM) request(i int) map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	if i < 0 || i >= len(m.reqs) {
		return nil
	}
	return m.reqs[i]
}

// sseText builds an SSE body with a single text-delta chunk followed by [DONE].
func sseText(text string) string {
	chunk := map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{"content": text},
		}},
	}
	data, _ := json.Marshal(chunk)
	return fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", data)
}

// sseToolCall builds an SSE body that emits a single tool call with the given
// name + JSON args, then [DONE]. First chunk carries the tool-call id (triggers
// append); the second delta extends arguments (per the llmStream protocol).
func sseToolCall(id, name, args string) string {
	var b strings.Builder
	first := map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{
				"tool_calls": []map[string]any{{
					"id":   id,
					"type": "function",
					"function": map[string]any{
						"name":      name,
						"arguments": args,
					},
				}},
			},
		}},
	}
	d, _ := json.Marshal(first)
	fmt.Fprintf(&b, "data: %s\n\n", d)
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

// TestSessionRoundtrip verifies the TOML schema for Session is stable: what we
// write in memory comes back byte-for-byte on reload.
func TestSessionRoundtrip(t *testing.T) {
	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("hello")
	s.AddAssistant("hi there")
	s.AppendToolUse(ToolUse{Name: "read_file", Input: `{"path":"x.go"}`, Output: "file content"})
	s.Summary = "earlier summary"
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := loadSession(dir, s.ID)
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if got := len(loaded.Messages); got != 2 {
		t.Fatalf("Messages len: got %d, want 2", got)
	}
	if loaded.Messages[0].Role != "user" || loaded.Messages[0].Content != "hello" {
		t.Errorf("Messages[0]: got %+v", loaded.Messages[0])
	}
	if loaded.Messages[1].Role != "assistant" || loaded.Messages[1].Content != "hi there" {
		t.Errorf("Messages[1]: got %+v", loaded.Messages[1])
	}
	if got := len(loaded.Messages[1].ToolUses); got != 1 {
		t.Fatalf("ToolUses len: got %d, want 1", got)
	}
	tu := loaded.Messages[1].ToolUses[0]
	if tu.Name != "read_file" || tu.Output != "file content" {
		t.Errorf("ToolUse: got %+v", tu)
	}
	if loaded.Summary != "earlier summary" {
		t.Errorf("Summary mismatch: got %q, want %q", loaded.Summary, "earlier summary")
	}
}

// TestAppendToolUseCreatesAssistantMessage verifies that recording a tool use
// when the last message is a user turn creates a new empty assistant message
// to hold it (rather than attaching to the user).
func TestAppendToolUseCreatesAssistantMessage(t *testing.T) {
	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("do a thing")
	s.AppendToolUse(ToolUse{Name: "read_file", Input: "{}", Output: "ok"})

	if got := len(s.Messages); got != 2 {
		t.Fatalf("Messages len: got %d, want 2", got)
	}
	if s.Messages[1].Role != "assistant" {
		t.Errorf("expected assistant message to be created, got role %q", s.Messages[1].Role)
	}
	if len(s.Messages[1].ToolUses) != 1 {
		t.Errorf("tool use not appended to assistant message: %+v", s.Messages[1])
	}

	// A second tool use must stay on the same assistant message.
	s.AppendToolUse(ToolUse{Name: "write_file", Input: "{}", Output: "ok"})
	if got := len(s.Messages); got != 2 {
		t.Fatalf("Messages len after second AppendToolUse: got %d, want 2", got)
	}
	if got := len(s.Messages[1].ToolUses); got != 2 {
		t.Fatalf("ToolUses len: got %d, want 2", got)
	}
}

// TestLLMStreamParsesTextAndTools verifies the SSE parser collects streamed
// text and tool calls correctly from the mock server.
func TestLLMStreamParsesTextAndTools(t *testing.T) {
	// Build an SSE body that mixes text + a tool call, split across chunks.
	var b strings.Builder
	// Chunk 1: text delta.
	c1, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{"content": "Hello "},
		}},
	})
	fmt.Fprintf(&b, "data: %s\n\n", c1)
	// Chunk 2: more text.
	c2, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{"content": "world"},
		}},
	})
	fmt.Fprintf(&b, "data: %s\n\n", c2)
	// Chunk 3: tool call start (has id).
	c3, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{
				"tool_calls": []map[string]any{{
					"id":       "call_1",
					"type":     "function",
					"function": map[string]any{"name": "read_file", "arguments": `{"pa`},
				}},
			},
		}},
	})
	fmt.Fprintf(&b, "data: %s\n\n", c3)
	// Chunk 4: tool args continuation (no id → appends to last call).
	c4, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{
				"tool_calls": []map[string]any{{
					"function": map[string]any{"arguments": `th":"x.go"}`},
				}},
			},
		}},
	})
	fmt.Fprintf(&b, "data: %s\n\n", c4)
	b.WriteString("data: [DONE]\n\n")

	mock := newMockLLM(t, b.String())
	defer mock.Close()

	a := &agent{}
	var collected strings.Builder
	text, calls, err := a.llmStream(
		context.Background(),
		"", // unscoped: no session log
		mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "hi"}},
		nil,
		func(tok string) { collected.WriteString(tok) },
		nil,
	)
	if err != nil {
		t.Fatalf("llmStream: %v", err)
	}
	if text != "Hello world" {
		t.Errorf("text: got %q, want %q", text, "Hello world")
	}
	if collected.String() != "Hello world" {
		t.Errorf("onToken: got %q, want %q", collected.String(), "Hello world")
	}
	if len(calls) != 1 {
		t.Fatalf("calls: got %d, want 1", len(calls))
	}
	if calls[0].Function.Name != "read_file" {
		t.Errorf("tool name: got %q", calls[0].Function.Name)
	}
	if calls[0].Function.Arguments != `{"path":"x.go"}` {
		t.Errorf("tool args: got %q", calls[0].Function.Arguments)
	}
}

// TestToolLoopRecordsToolUses verifies that when the LLM returns a tool call,
// the tool loop executes it, appends the ToolUse to the session, and persists
// to disk — before the second LLM turn produces the final text.
func TestToolLoopRecordsToolUses(t *testing.T) {
	// Isolate from the package-level registry so the synthetic `respond` tool
	// (registered in tool_respond.go init) isn't in scope — its presence would
	// flip the loop's empty-tool-call branch from "exit with allText" to a
	// nudge, which is a different code path tested elsewhere.
	withFreshToolRegistry(t)
	// Register a stub tool inline so we don't depend on filesystem tool
	// implementations. This runs at init in package-level registeredTools.
	const testToolName = "test_echo_tool_9d7f"
	RegisterTool(Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        testToolName,
				"description": "echoes the input.msg field (test only)",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"msg": map[string]any{"type": "string"}},
				},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			args := parseArgs(rawArgs)
			return "echo: " + args["msg"], false
		},
	})

	mock := newMockLLM(t,
		// Turn 1: LLM asks to call the stub tool.
		sseToolCall("call_1", testToolName, `{"msg":"hello"}`),
		// Turn 2: LLM produces the final assistant text.
		sseText("All done."),
	)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("please echo hello")
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
	}

	res, err := a.runToolLoop(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "please echo hello"}}, toolFilter{}, "execute", 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}

	if res.Text != "All done." {
		t.Errorf("final text: got %q, want %q", res.Text, "All done.")
	}
	if len(res.ToolUses) != 1 {
		t.Fatalf("ToolUses: got %d, want 1", len(res.ToolUses))
	}
	if res.ToolUses[0].Output != "echo: hello" {
		t.Errorf("ToolUse output: got %q", res.ToolUses[0].Output)
	}

	// The session should carry the recorded tool use, attached to a freshly
	// created assistant message (since the last message was user).
	if got := len(s.Messages); got != 2 {
		t.Fatalf("session messages: got %d, want 2", got)
	}
	if s.Messages[1].Role != "assistant" {
		t.Errorf("expected trailing assistant message, got role %q", s.Messages[1].Role)
	}
	if got := len(s.Messages[1].ToolUses); got != 1 {
		t.Fatalf("session tool uses: got %d, want 1", got)
	}

	// Persistence: the tool use should already be on disk from the
	// incremental Save in the tool loop, even before the caller adds the
	// final assistant text.
	loaded, err := loadSession(dir, s.ID)
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if got := len(loaded.Messages); got != 2 {
		t.Fatalf("persisted messages: got %d, want 2", got)
	}
	if got := len(loaded.Messages[1].ToolUses); got != 1 {
		t.Fatalf("persisted tool uses: got %d, want 1", got)
	}
	if loaded.Messages[1].ToolUses[0].Output != "echo: hello" {
		t.Errorf("persisted output mismatch")
	}

	if mock.callCount() != 2 {
		t.Errorf("LLM call count: got %d, want 2", mock.callCount())
	}
}

// TestToolLoopRespondExits verifies that when the model calls the synthetic
// `respond` terminal tool, the loop exits with the message arg as res.Text on
// the same iteration — no second LLM round-trip to "produce final text".
// This is the post-respond exit semantic (vs the legacy "empty tool_calls
// means done" path covered by TestToolLoopRecordsToolUses with a fresh
// registry).
func TestToolLoopRespondExits(t *testing.T) {
	// Use the real package registry so respond is in scope (no withFreshToolRegistry).
	mock := newMockLLM(t,
		sseToolCall("call_1", respondToolName, `{"message":"final answer"}`),
	)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("answer me")
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	a := &agent{sessions: map[string]*Session{s.ID: s}}

	res, err := a.runToolLoop(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "answer me"}}, toolFilter{}, "execute", 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}

	if res.Text != "final answer" {
		t.Errorf("res.Text: got %q, want %q", res.Text, "final answer")
	}
	if mock.callCount() != 1 {
		t.Errorf("LLM call count: got %d, want 1 (respond should exit on the same iteration)", mock.callCount())
	}
	if len(res.ToolUses) != 1 || res.ToolUses[0].Name != respondToolName {
		t.Errorf("ToolUses: want one respond call, got %+v", res.ToolUses)
	}
}

// TestToolLoopRespondExcludedFromJSONPhases verifies that the plan/verify/
// document filter (excluding respond) keeps the legacy text-only exit: a
// no-tool-calls turn returns immediately instead of nudging for respond. This
// is what lets runPlanLLM parse the assistant text as JSON.
func TestToolLoopRespondExcludedFromJSONPhases(t *testing.T) {
	mock := newMockLLM(t,
		sseText(`{"clear": true, "steps": ["do x"]}`),
	)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("plan something")
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	a := &agent{sessions: map[string]*Session{s.ID: s}}

	res, err := a.runToolLoop(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "plan something"}},
		toolFilter{exclude: map[string]bool{respondToolName: true}}, "plan", 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if mock.callCount() != 1 {
		t.Errorf("LLM call count: got %d, want 1 (no nudge when respond is excluded)", mock.callCount())
	}
	if !strings.Contains(res.Text, "do x") {
		t.Errorf("res.Text missing JSON content: got %q", res.Text)
	}
}

// TestToolLoopNoDedup verifies that identical tool calls in the same tool
// loop each execute the underlying tool. The dedup cache used to suppress
// the second call, which broke read-after-write: a mutator (sed via
// run_command) would change state, then a re-issued read returned the
// pre-mutation cached value and the model concluded the mutation failed.
// Now every call executes; the model gets a fresh result every time.
func TestToolLoopNoDedup(t *testing.T) {
	withFreshToolRegistry(t)
	const readName = "test_read"
	const writeName = "test_write"
	var reads, writes int
	RegisterTool(Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": readName, "description": "read",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			reads++
			return "read-ok", false
		},
	})
	RegisterTool(Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": writeName, "description": "write",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			writes++
			return "wrote", false
		},
	})

	mock := newMockLLM(t,
		sseToolCall("c1", readName, `{}`),
		sseToolCall("c2", readName, `{}`),
		sseToolCall("c3", writeName, `{}`),
		sseToolCall("c4", writeName, `{}`),
		sseText("done"),
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	res, err := a.runToolLoop(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, toolFilter{}, "execute", 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if reads != 2 {
		t.Errorf("read tool executed %d times, want 2 (no dedup)", reads)
	}
	if writes != 2 {
		t.Errorf("write tool executed %d times, want 2 (no dedup)", writes)
	}
	if len(res.ToolUses) != 4 {
		t.Fatalf("ToolUses: got %d, want 4", len(res.ToolUses))
	}
	for i, tu := range res.ToolUses {
		if strings.HasPrefix(tu.Output, "[deduped:") {
			t.Errorf("ToolUses[%d] should not be deduped, got %q", i, tu.Output)
		}
	}
}

// TestToolLoopRepeatNudgeAndBail covers the repetition recovery path: the
// second consecutive identical tool call still executes (read-after-write
// must keep working) but appends a user-role nudge; the third identical
// call bails before executing. ~3 iterations of stuck behavior fails
// fast instead of waiting for the 50-iter cap.
func TestToolLoopRepeatNudgeAndBail(t *testing.T) {
	withFreshToolRegistry(t)
	const toolName = "test_probe_a3f"
	var execs int
	RegisterTool(Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": toolName, "description": "probe",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			execs++
			return "result", false
		},
	})

	mock := newMockLLM(t,
		sseToolCall("c1", toolName, `{}`),
		sseToolCall("c2", toolName, `{}`),
		sseToolCall("c3", toolName, `{}`),
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	_, err := a.runToolLoop(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, toolFilter{}, "execute", 0)
	if err == nil {
		t.Fatalf("runToolLoop: want error, got nil")
	}
	if !strings.Contains(err.Error(), "stuck on identical call") {
		t.Errorf("error: got %v, want substring 'stuck on identical call'", err)
	}
	if execs != 2 {
		t.Errorf("tool execs: got %d, want 2 (iter 1 + iter 2; iter 3 bails)", execs)
	}
	if mock.callCount() != 3 {
		t.Errorf("LLM calls: got %d, want 3", mock.callCount())
	}
	// The 3rd LLM request should carry the nudge message appended after
	// iter 2's tool result.
	req3 := mock.request(2)
	msgs, _ := req3["messages"].([]any)
	var found bool
	for _, m := range msgs {
		mm, _ := m.(map[string]any)
		if mm["role"] == "user" {
			if s, _ := mm["content"].(string); strings.Contains(s, "repeated the same tool call") {
				found = true
				break
			}
		}
	}
	if !found {
		t.Errorf("3rd LLM request did not contain the repeat-nudge user message")
	}
}

// TestToolLoopDoesNotEscalateOnDistinctArgs verifies that legitimate fan-out
// across distinct arguments (e.g. read_file on go.mod, examples/go.mod, …
// when surveying a multi-module repo) does NOT trip the per-name escalation.
// Only redundant calls — same (name, args) pair seen before — count toward
// toolNameEscalateThreshold. Sampler must stay on the execute role for the
// whole loop.
func TestToolLoopDoesNotEscalateOnDistinctArgs(t *testing.T) {
	withFreshToolRegistry(t)
	const toolName = "test_grep_q9z"
	var execs int
	RegisterTool(Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": toolName, "description": "grep",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			execs++
			return "no match", false
		},
	})

	// 5 tool calls with *different* args (the surveying-pattern that used
	// to trip the old per-name counter). With distinct-args counting, none
	// of these count as redundant, so no escalation should fire.
	mock := newMockLLM(t,
		sseToolCall("c1", toolName, `{"q":"x1"}`),
		sseToolCall("c2", toolName, `{"q":"x2"}`),
		sseToolCall("c3", toolName, `{"q":"x3"}`),
		sseToolCall("c4", toolName, `{"q":"x4"}`),
		sseToolCall("c5", toolName, `{"q":"x5"}`),
		sseText("done."),
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{
		LLM: []LLMConnection{{
			Server:         mock.ts.URL,
			Model:          "test-model",
			ParamsExecute:  map[string]any{"temperature": 0.3},
			ParamsThinking: map[string]any{"temperature": 1.0},
		}},
	}

	conn := a.connForSession(context.Background(), s.ID, "execute")
	if conn == nil {
		t.Fatalf("connForSession(execute) returned nil")
	}
	_, err := a.runToolLoop(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, toolFilter{}, "execute", 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}

	if execs != 5 {
		t.Errorf("tool execs: got %d, want 5", execs)
	}
	if mock.callCount() != 6 {
		t.Errorf("LLM calls: got %d, want 6", mock.callCount())
	}

	// Every call must still use the execute sampler — no escalation.
	for i := 0; i < mock.callCount(); i++ {
		req := mock.request(i)
		if req == nil {
			t.Fatalf("request %d missing", i)
		}
		temp, _ := req["temperature"].(float64)
		if temp != 0.3 {
			t.Errorf("request %d temperature: got %v, want 0.3 (no escalation expected on distinct args)", i, temp)
		}
	}
}

// TestToolLoopEscalatesOnRepeatedArgs covers the per-name escalation under
// distinct-args counting: when the same (name, arguments) pair is revisited
// enough times in one loop, the connection swaps to the "thinking" role's
// sampler. We interleave two arg values to avoid two consecutive byte-for-
// byte identical sigs (which would trip the signature nudge / bail first).
func TestToolLoopEscalatesOnRepeatedArgs(t *testing.T) {
	withFreshToolRegistry(t)
	const toolName = "test_grep_q9z"
	var execs int
	RegisterTool(Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": toolName, "description": "grep",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			execs++
			return "no match", false
		},
	})

	// Alternate two args: A B A B A B A. Sigs differ each iter so the
	// signature nudge never fires, but every call after the first
	// occurrence of each arg counts as redundant. The per-name counter
	// (aggregated across all args of the same tool) reaches
	// toolNameEscalateThreshold (5) at iter 6 — escalating the 8th LLM
	// call's sampler.
	mock := newMockLLM(t,
		sseToolCall("a1", toolName, `{"q":"A"}`),
		sseToolCall("b1", toolName, `{"q":"B"}`),
		sseToolCall("a2", toolName, `{"q":"A"}`),
		sseToolCall("b2", toolName, `{"q":"B"}`),
		sseToolCall("a3", toolName, `{"q":"A"}`),
		sseToolCall("b3", toolName, `{"q":"B"}`),
		sseToolCall("a4", toolName, `{"q":"A"}`),
		sseText("giving up."),
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{
		LLM: []LLMConnection{{
			Server:         mock.ts.URL,
			Model:          "test-model",
			ParamsExecute:  map[string]any{"temperature": 0.3},
			ParamsThinking: map[string]any{"temperature": 1.0},
		}},
	}

	conn := a.connForSession(context.Background(), s.ID, "execute")
	if conn == nil {
		t.Fatalf("connForSession(execute) returned nil")
	}
	_, err := a.runToolLoop(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, toolFilter{}, "execute", 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}

	if execs != 7 {
		t.Errorf("tool execs: got %d, want 7", execs)
	}
	if mock.callCount() != 8 {
		t.Errorf("LLM calls: got %d, want 8", mock.callCount())
	}

	// Calls 1..7 used execute sampler (temperature 0.3) — escalation runs
	// at the END of iter 6 (the 7th call), so the call itself still went
	// out under execute params; the 8th is the first one with thinking.
	for i := 0; i < 7; i++ {
		req := mock.request(i)
		if req == nil {
			t.Fatalf("request %d missing", i)
		}
		temp, _ := req["temperature"].(float64)
		if temp != 0.3 {
			t.Errorf("request %d temperature: got %v, want 0.3", i, temp)
		}
	}
	// Call 8 must have escalated to thinking (temperature 1.0).
	req8 := mock.request(7)
	if req8 == nil {
		t.Fatalf("request 8 missing")
	}
	temp8, _ := req8["temperature"].(float64)
	if temp8 != 1.0 {
		t.Errorf("request 8 temperature: got %v, want 1.0 (escalation should have flipped sampler)", temp8)
	}
}

// TestCompressHistoryRecordsSummary is the headline history test: once the
// server-reported prompt_tokens crosses the trigger, compressHistory should
// rotate the session — freeze the pre-rotation state to a "session_archive_*"
// file, push the drained shadow entries into Summary, trim raw messages to
// the single most-recent one, and persist everything. The mock LLM has zero
// responses queued: the shadow fast path is fully local and any LLM call
// would fail the test.
func TestCompressHistoryRecordsSummary(t *testing.T) {
	mock := newMockLLM(t)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	// Build 10 user+assistant pairs so the rotation trims 19 messages into
	// the summary and leaves exactly 1 verbatim (the trailing assistant reply).
	filler := strings.Repeat("lorem ipsum ", 100)
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user msg %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst msg %d %s", i, filler))
	}
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	originalMsgCount := len(s.Messages)

	// Seed shadow with two notes covering the older turns plus one anchor
	// that must stay in the buffer for the next compaction.
	s.appendShadow("Goal: ship feature\nProgress: scaffolded module")
	s.appendShadow("Goal: ship feature\nProgress: wired up handler")
	s.appendShadow("Goal: ship feature\nProgress: anchor entry")

	// mainSlotTokens = 90_000 → trigger at 72_000. Set LastCompletePromptTokens
	// past the trigger to simulate the server's most recent /usage chunk.
	s.SetLastCompletePromptTokens(80_000)

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	a.compressHistory(context.Background(), s)

	if !strings.Contains(s.Summary, "scaffolded module") || !strings.Contains(s.Summary, "wired up handler") {
		t.Errorf("summary missing drained shadow entries; got %q", s.Summary)
	}
	if strings.Contains(s.Summary, "anchor entry") {
		t.Errorf("anchor entry leaked into Summary; should still be in shadow: %q", s.Summary)
	}
	if len(s.Messages) >= originalMsgCount {
		t.Errorf("Messages not trimmed: before %d, after %d", originalMsgCount, len(s.Messages))
	}
	if len(s.Messages) != 1 {
		t.Errorf("Messages after trim: got %d, want 1", len(s.Messages))
	}

	// An archive file should exist holding the pre-rotation full state, and
	// the live session must keep its original ID + path.
	archives, err := filepath.Glob(filepath.Join(dir, sessionDir, "session_archive_*.toml"))
	if err != nil {
		t.Fatalf("glob archives: %v", err)
	}
	if len(archives) != 1 {
		t.Fatalf("expected 1 archive file, got %d: %v", len(archives), archives)
	}
	archiveID := strings.TrimSuffix(strings.TrimPrefix(filepath.Base(archives[0]), "session_"), ".toml")
	archived, err := loadSession(dir, archiveID)
	if err != nil {
		t.Fatalf("loadSession archive: %v", err)
	}
	if len(archived.Messages) != originalMsgCount {
		t.Errorf("archive Messages: got %d, want %d", len(archived.Messages), originalMsgCount)
	}

	// Persistence — the drained shadow must survive a reload.
	loaded, err := loadSession(dir, s.ID)
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if !strings.Contains(loaded.Summary, "wired up handler") {
		t.Errorf("persisted summary missing drained shadow entries; got %q", loaded.Summary)
	}

	if mock.callCount() != 0 {
		t.Errorf("LLM calls: got %d, want 0 (shadow fast path is fully local)", mock.callCount())
	}
}

// TestConcurrentSessionWritesAreRaceFree exercises the Session mutex by
// racing session mutations against concurrent Save() calls. Run with -race to
// catch regressions in the locking that fix #4 introduced.
func TestConcurrentSessionWritesAreRaceFree(t *testing.T) {
	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	const workers = 8
	const perWorker = 50
	var wg sync.WaitGroup
	wg.Add(workers * 3)

	// Writers: exercise each mutator type concurrently.
	for i := 0; i < workers; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < perWorker; j++ {
				s.AddUser("u")
			}
		}()
		go func() {
			defer wg.Done()
			for j := 0; j < perWorker; j++ {
				s.AppendToolUse(ToolUse{Name: "x", Input: "{}", Output: "ok"})
			}
		}()
		go func() {
			defer wg.Done()
			for j := 0; j < perWorker; j++ {
				_ = s.Save()
			}
		}()
	}
	wg.Wait()

	// Loosely assert that all AddUser calls landed — exact count verifies
	// that no append was lost to a concurrent slice-grow race.
	userCount := 0
	for _, m := range s.Messages {
		if m.Role == "user" {
			userCount++
		}
	}
	if want := workers * perWorker; userCount != want {
		t.Errorf("user message count: got %d, want %d", userCount, want)
	}
}

// TestCapReadContent verifies the default-cap logic in read_file: explicit
// limits are honoured without warnings, implicit reads get a truncation note
// when exceeding defaultReadLines, and the byte cap fires for long-line files.
func TestCapReadContent(t *testing.T) {
	tests := []struct {
		name          string
		lines         int
		explicit      bool
		startLine     string
		wantNote      bool
		wantKeyword   string // substring expected in note
		wantTruncated bool
	}{
		{name: "small implicit read", lines: 10, explicit: false, wantNote: false},
		{name: "small explicit read", lines: 10, explicit: true, wantNote: false},
		{name: "exact boundary implicit", lines: defaultReadLines, explicit: false, wantNote: true, wantKeyword: "truncated"},
		{name: "over default implicit", lines: defaultReadLines + 500, explicit: false, wantNote: true, wantKeyword: "truncated"},
		{name: "over default explicit", lines: defaultReadLines + 500, explicit: true, wantNote: false},
		{name: "implicit with startLine", lines: defaultReadLines + 1, explicit: false, startLine: "50", wantNote: true, wantKeyword: "line 50"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			content := strings.Repeat("x\n", tc.lines)
			_, note := capReadContent(content, tc.explicit, tc.startLine)
			if tc.wantNote && note == "" {
				t.Errorf("expected a truncation note, got empty")
			}
			if !tc.wantNote && note != "" {
				t.Errorf("expected no note, got %q", note)
			}
			if tc.wantKeyword != "" && !strings.Contains(note, tc.wantKeyword) {
				t.Errorf("note missing %q: %q", tc.wantKeyword, note)
			}
		})
	}

	// Byte-cap path: one enormous single line.
	long := strings.Repeat("a", maxReadBytes+1024)
	out, note := capReadContent(long, false, "")
	if len(out) != maxReadBytes {
		t.Errorf("byte cap: got length %d, want %d", len(out), maxReadBytes)
	}
	if !strings.Contains(note, "bytes") {
		t.Errorf("byte cap note missing 'bytes': %q", note)
	}
}

// TestPrefixStableAcrossTurns is the cache-correctness contract: a second
// Prompt() turn must reproduce the previous turn's wire bytes byte-for-byte
// for every message that's already on record. llama.cpp / vLLM / etc. only
// reuse their KV cache when the leading tokens of the new request match the
// leading tokens of the old one, so any drift in the prefix bytes silently
// reprocesses the entire history each turn.
//
// The append-only transcript model makes this trivial: each phase pushes a
// new user/assistant pair onto sess.Messages and never mutates earlier
// entries, so buildLLMContext replays the same bytes. The test exercises the
// load-bearing case — the first turn populates sess.SystemPrompt (skills +
// project context), and that leading message must remain identical on later
// turns so the LLM's prefix cache keeps hitting.
func TestPrefixStableAcrossTurns(t *testing.T) {
	dir := t.TempDir()

	// Seed a SKILL file so loadSkills returns a non-empty system prompt —
	// otherwise the bug (sysPrompt set turn 1, dropped turn 2) is invisible
	// because sysPrompt is empty.
	cfgDir := filepath.Join(dir, ".codehalter")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(cfgDir, "SKILL-go.md"), []byte("# Go skill\nsome conventions\n"), 0o644); err != nil {
		t.Fatalf("write SKILL: %v", err)
	}

	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	a := &agent{sessions: map[string]*Session{s.ID: s}}

	// Sanity: systemPrompt must be non-empty so the bug we're guarding
	// against (sysPrompt set turn 1, dropped turn 2) is observable.
	sysPrompt, _ := a.systemPrompt(s.ID)
	if sysPrompt == "" {
		t.Fatal("expected non-empty systemPrompt — SKILL seed didn't take effect")
	}

	// --- Turn 1: first prompt of the session ---
	// Prompt() sets sess.SystemPrompt (emitted by buildLLMContext as the
	// leading user message); subsequent phases (PLAN/EXECUTE/VERIFY/
	// DOCUMENT.md) get their own user turns appended to Messages.
	s.SystemPrompt = sysPrompt
	s.AddUser("first prompt")
	msgs1 := a.buildLLMContext(s)

	// Assistant replies (planner JSON, executor text, etc. — collapsed to
	// one assistant message here since the test only cares about the
	// user/assistant alternation that lands in history).
	s.UpsertLastAssistant("done with turn 1")

	// --- Turn 2: a follow-up prompt ---
	// Subsequent user turns store just the raw text — sysPrompt is already
	// in sess.SystemPrompt from turn 1.
	s.AddUser("second prompt")
	msgs2 := a.buildLLMContext(s)

	if len(msgs2) <= len(msgs1) {
		t.Fatalf("turn 2 should extend turn 1's history; got len1=%d len2=%d",
			len(msgs1), len(msgs2))
	}
	// Every message turn 1 sent must reappear byte-identically as the prefix
	// of turn 2's wire — this is exactly what the prefix cache keys on.
	for i := range msgs1 {
		b1, _ := json.Marshal(msgs1[i])
		b2, _ := json.Marshal(msgs2[i])
		if !bytes.Equal(b1, b2) {
			t.Errorf("prefix message %d drifted between turns:\n  turn 1: %s\n  turn 2: %s",
				i, b1, b2)
		}
	}
}

// TestLoadSkillsDeterministic verifies loadSkills sorts entries so the
// concatenated system-prompt prefix is byte-stable across calls — a moving
// SKILL order would invalidate the cache on every session start.
func TestLoadSkillsDeterministic(t *testing.T) {
	dir := t.TempDir()
	cfgDir := filepath.Join(dir, ".codehalter")
	if err := os.MkdirAll(cfgDir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// Write in non-alphabetical order; readdir order is filesystem-dependent.
	files := map[string]string{
		"SKILL-ts.md":   "# TS\n",
		"SKILL-go.md":   "# Go\n",
		"SKILL-bash.md": "# Bash\n",
		"SKILL-java.md": "# Java\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(cfgDir, name), []byte(body), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	first := loadSkills(dir)
	for i := 0; i < 5; i++ {
		got := loadSkills(dir)
		if got != first {
			t.Errorf("loadSkills run %d differs from run 0:\n  run 0: %q\n  run %d: %q", i, first, i, got)
		}
	}
	// Bash should come first alphabetically; TS should be last. Looking at
	// the order via index ensures we catch a swap, not just presence.
	idx := func(needle string) int { return strings.Index(first, needle) }
	if idx("# Bash") != 0 {
		t.Errorf("expected loadSkills to start with Bash; got %q", truncate(first, 80))
	}
	if !(idx("# Bash") < idx("# Go") && idx("# Go") < idx("# Java") && idx("# Java") < idx("# TS")) {
		t.Errorf("loadSkills not alphabetical:\n%s", first)
	}
}

// TestCompressHistoryNoopWhenBelowBudget verifies that sessions with a
// prompt_tokens reading below the trigger don't call the LLM at all — no
// summary.
func TestCompressHistoryNoopWhenBelowBudget(t *testing.T) {
	mock := newMockLLM(t) // zero responses queued → any call fails the test.
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("short")
	s.AddAssistant("also short")
	// Well below the 72_000 trigger.
	s.SetLastCompletePromptTokens(5_000)

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	a.compressHistory(context.Background(), s)

	if s.Summary != "" {
		t.Errorf("expected empty summary, got %q", s.Summary)
	}
	if len(s.Messages) != 2 {
		t.Errorf("expected messages untouched, got %d", len(s.Messages))
	}
	if mock.callCount() != 0 {
		t.Errorf("expected no LLM calls, got %d", mock.callCount())
	}
}

// TestCompressHistoryShadowFastPath verifies the background-summariser fast
// path: when shadow buffer already has structured notes (populated during the
// turns), compaction installs them directly as Summary and skips the
// synchronous summarize LLM call. With retitle removed, compaction makes
// zero LLM calls in this path.
func TestCompressHistoryShadowFastPath(t *testing.T) {
	mock := newMockLLM(t) // zero responses queued → any call fails the test.
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	filler := strings.Repeat("lorem ipsum ", 100)
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user msg %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst msg %d %s", i, filler))
	}
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	s.SetLastCompletePromptTokens(80_000)

	s.appendShadow("Goal: do thing\nProgress: did thing")
	s.appendShadow("Goal: do thing\nProgress: refined thing")
	s.appendShadow("Goal: do thing\nProgress: anchor entry")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	a.compressHistory(context.Background(), s)

	// Older entries fold into Summary; the most-recent stays in the buffer
	// as the anchor covering the verbatim trailing message.
	if !strings.Contains(s.Summary, "did thing") || !strings.Contains(s.Summary, "refined thing") {
		t.Errorf("expected Summary to contain the older shadow chunks, got %q", s.Summary)
	}
	if strings.Contains(s.Summary, "anchor entry") {
		t.Errorf("Summary swallowed the anchor entry; should still be in the shadow buffer: %q", s.Summary)
	}
	if mock.callCount() != 0 {
		t.Errorf("LLM calls: got %d, want 0 (shadow fast path is fully local)", mock.callCount())
	}
	// The anchor entry is still in the buffer; a follow-up peek must see it.
	if peek := s.peekShadow(); !strings.Contains(peek, "anchor entry") {
		t.Errorf("anchor entry missing from shadow buffer after compaction; got %q", peek)
	}
}

// contentString asserts that an llmMessage carries a string payload and
// returns it; it fails the test if the content was something else.
func contentString(t *testing.T, m llmMessage) string {
	t.Helper()
	s, ok := m.Content.(string)
	if !ok {
		t.Fatalf("expected string content, got %T", m.Content)
	}
	return s
}

// TestBuildLLMHistoryShape verifies the header injection when a summary
// exists (one leading user message) and that stored messages follow in
// order. The no-summary case verifies the header is omitted.
func TestBuildLLMHistoryShape(t *testing.T) {
	a := &agent{}
	s := &Session{
		Summary: "earlier summary",
		Messages: []Message{
			{Role: "user", Content: "q1"},
			{Role: "assistant", Content: "a1"},
			{Role: "user", Content: "q2"},
		},
	}

	msgs := a.buildLLMContext(s)

	if len(msgs) != 4 {
		t.Fatalf("got %d messages, want 4: %+v", len(msgs), msgs)
	}
	if msgs[0].Role != "user" {
		t.Errorf("header role: got %q, want user", msgs[0].Role)
	}
	if !strings.Contains(contentString(t, msgs[0]), "earlier summary") {
		t.Errorf("intro missing summary content: %q", contentString(t, msgs[0]))
	}
	if got := contentString(t, msgs[1]); got != "q1" {
		t.Errorf("msgs[1]: got %q, want q1", got)
	}
	if got := contentString(t, msgs[2]); got != "a1" {
		t.Errorf("msgs[2]: got %q, want a1", got)
	}
	if got := contentString(t, msgs[3]); got != "q2" {
		t.Errorf("msgs[3]: got %q, want q2", got)
	}

	// No summary → no header; stored messages pass through unchanged.
	s2 := &Session{Messages: []Message{
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "a1"},
	}}
	msgs2 := a.buildLLMContext(s2)
	if len(msgs2) != 2 {
		t.Fatalf("no-summary case: got %d, want 2", len(msgs2))
	}
	if got := contentString(t, msgs2[0]); got != "q1" {
		t.Errorf("no-summary case msgs2[0]: got %q, want q1", got)
	}
	if got := contentString(t, msgs2[1]); got != "a1" {
		t.Errorf("no-summary case msgs2[1]: got %q, want a1", got)
	}
}

// TestBuildLLMHistoryToolUseProtocolShape verifies that a stored assistant
// message with ToolUses is rebuilt in the OpenAI protocol shape — assistant
// with ToolCalls field, followed by one tool-role message per call carrying
// the (truncated) output and a ToolCallID pointer back. tu.ID is reused as
// tool_call_id. Also covers the empty-assistant-content path (model emitted
// only tool calls) and the long-output truncation path.
func TestBuildLLMHistoryToolUseProtocolShape(t *testing.T) {
	long := strings.Repeat("X", truncateThreshold+500)

	a := &agent{}
	s := &Session{Messages: []Message{
		{Role: "user", Content: "do it"},
		{Role: "assistant", Content: "running", ToolUses: []ToolUse{
			{ID: "tu_1", Name: "read_file", Input: `{"path":"x"}`, Output: "file content"},
			{ID: "tu_2", Name: "search", Input: `{"q":"foo"}`, Output: long},
		}},
		{Role: "user", Content: "thanks"},
		{Role: "assistant", Content: "", ToolUses: []ToolUse{
			{ID: "tu_3", Name: "respond", Input: `{}`, Output: "done"},
		}},
	}}

	msgs := a.buildLLMContext(s)
	// user + (assistant + 2 tool) + user + (assistant + 1 tool) = 7
	if len(msgs) != 7 {
		t.Fatalf("got %d messages, want 7: %+v", len(msgs), msgs)
	}

	if msgs[0].Role != "user" || contentString(t, msgs[0]) != "do it" {
		t.Errorf("msgs[0] wrong: %+v", msgs[0])
	}

	asst := msgs[1]
	if asst.Role != "assistant" || contentString(t, asst) != "running" {
		t.Errorf("assistant Role/Content wrong: %+v", asst)
	}
	if len(asst.ToolCalls) != 2 {
		t.Fatalf("assistant ToolCalls len: got %d, want 2", len(asst.ToolCalls))
	}
	if asst.ToolCalls[0].ID != "tu_1" || asst.ToolCalls[0].Type != "function" ||
		asst.ToolCalls[0].Function.Name != "read_file" || asst.ToolCalls[0].Function.Arguments != `{"path":"x"}` {
		t.Errorf("ToolCalls[0] wrong: %+v", asst.ToolCalls[0])
	}
	if asst.ToolCalls[1].ID != "tu_2" || asst.ToolCalls[1].Function.Name != "search" {
		t.Errorf("ToolCalls[1] wrong: %+v", asst.ToolCalls[1])
	}

	if msgs[2].Role != "tool" || msgs[2].ToolCallID != "tu_1" || contentString(t, msgs[2]) != "file content" {
		t.Errorf("tool message [0] wrong: %+v", msgs[2])
	}
	// Long output runs through truncateForLLM — the wire copy is shorter than
	// the stored Output and carries the view_output hint.
	tool2Content := contentString(t, msgs[3])
	if msgs[3].Role != "tool" || msgs[3].ToolCallID != "tu_2" {
		t.Errorf("tool message [1] Role/ID wrong: %+v", msgs[3])
	}
	if len(tool2Content) >= len(long) {
		t.Errorf("long tool output not truncated: wire len %d >= stored len %d", len(tool2Content), len(long))
	}
	if !strings.Contains(tool2Content, "view_output id=\"tu_2\"") {
		t.Errorf("truncated tool message missing view_output hint: %q", tool2Content)
	}

	if msgs[4].Role != "user" || contentString(t, msgs[4]) != "thanks" {
		t.Errorf("msgs[4] wrong: %+v", msgs[4])
	}
	// Empty-content assistant turn (model emitted only tool calls) still
	// gets a properly-shaped assistant message.
	if msgs[5].Role != "assistant" || contentString(t, msgs[5]) != "" || len(msgs[5].ToolCalls) != 1 {
		t.Errorf("empty-content assistant wrong: %+v", msgs[5])
	}
	if msgs[6].Role != "tool" || msgs[6].ToolCallID != "tu_3" {
		t.Errorf("tool message [2] wrong: %+v", msgs[6])
	}
}

// TestBuildLLMHistoryImageHandling covers the imagesSupported branch and the
// cache-consistency rule: every stored image gets its bytes inlined every turn
// — there is no trailing-vs-older split. After compaction the image lives in
// Summary as a reference and view_image fetches it on demand.
func TestBuildLLMHistoryImageHandling(t *testing.T) {
	dir := t.TempDir()
	bytes1 := []byte("pngbytes-1")
	bytes2 := []byte("pngbytes-2-different")
	id1 := "img_test_a"
	id2 := "img_test_b"
	if err := writeImageFile(dir, id1, "image/png", bytes1); err != nil {
		t.Fatalf("writeImageFile id1: %v", err)
	}
	if err := writeImageFile(dir, id2, "image/png", bytes2); err != nil {
		t.Fatalf("writeImageFile id2: %v", err)
	}
	img1 := ImageData{ID: id1, MimeType: "image/png"}
	img2 := ImageData{ID: id2, MimeType: "image/png"}

	// Images NOT supported → text fallback with per-image placeholders.
	a := &agent{imagesSupported: false}
	s := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "look at this", Images: []ImageData{img1, img2}}}}
	out := a.buildLLMContext(s)
	if len(out) != 1 {
		t.Fatalf("expected 1 message, got %d", len(out))
	}
	got := contentString(t, out[0])
	if !strings.Contains(got, "[Image "+id1) || !strings.Contains(got, "[Image "+id2) {
		t.Errorf("expected per-image placeholders in %q", got)
	}
	if !strings.Contains(got, "view_image id="+id1) || !strings.Contains(got, "view_image id="+id2) {
		t.Errorf("expected view_image hints in %q", got)
	}

	// Images supported → []any with one text block + N image_url blocks
	// containing the actual data: URLs read from disk.
	a.imagesSupported = true
	out = a.buildLLMContext(s)
	parts, ok := out[0].Content.([]any)
	if !ok {
		t.Fatalf("expected []any content, got different type")
	}
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts (text + 2 images), got %d", len(parts))
	}
	text, _ := parts[0].(map[string]any)
	if text["type"] != "text" || text["text"] != "look at this" {
		t.Errorf("parts[0] wrong: %+v", text)
	}
	for i, part := range parts[1:] {
		block, _ := part.(map[string]any)
		if block["type"] != "image_url" {
			t.Errorf("parts[%d] type: got %v, want image_url", i+1, block["type"])
		}
		url, _ := block["image_url"].(map[string]string)
		if !strings.HasPrefix(url["url"], "data:image/png;base64,") {
			t.Errorf("parts[%d] url prefix wrong: %q", i+1, url["url"])
		}
	}

	// Cache-consistency: two messages with images both inline bytes every
	// turn. No trailing/older split — the older image is NOT degraded to a
	// text placeholder.
	s4 := &Session{Cwd: dir, Messages: []Message{
		{Role: "user", Content: "earlier", Images: []ImageData{img1}},
		{Role: "user", Content: "now look at this one", Images: []ImageData{img2}},
	}}
	out = a.buildLLMContext(s4)
	if len(out) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(out))
	}
	olderParts, ok := out[0].Content.([]any)
	if !ok || len(olderParts) != 2 {
		t.Fatalf("older: expected []any of len 2 (text + image_url), got %+v", out[0].Content)
	}
	olderImg, _ := olderParts[1].(map[string]any)
	if olderImg["type"] != "image_url" {
		t.Errorf("older parts[1] type: got %v, want image_url", olderImg["type"])
	}
	trailingParts, ok := out[1].Content.([]any)
	if !ok || len(trailingParts) != 2 {
		t.Fatalf("trailing: expected []any of len 2, got %+v", out[1].Content)
	}

	// Wire bytes for the same stored message must be identical turn-over-turn.
	first := a.buildLLMContext(s4)
	second := a.buildLLMContext(s4)
	firstJSON, err := json.Marshal(first)
	if err != nil {
		t.Fatalf("marshal first: %v", err)
	}
	secondJSON, err := json.Marshal(second)
	if err != nil {
		t.Fatalf("marshal second: %v", err)
	}
	if string(firstJSON) != string(secondJSON) {
		t.Errorf("cache consistency: wire bytes differ between consecutive rebuilds\nfirst:  %s\nsecond: %s", firstJSON, secondJSON)
	}

	// Missing file on disk → text fallback for that image only, no panic and
	// the rest of the request still goes through.
	imgMissing := ImageData{ID: "img_deadbeef00000000", MimeType: "image/png"}
	sMissing := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "missing", Images: []ImageData{imgMissing}}}}
	outMissing := a.buildLLMContext(sMissing)
	missingParts, ok := outMissing[0].Content.([]any)
	if !ok || len(missingParts) != 2 {
		t.Fatalf("missing: expected []any of len 2 (text + fallback), got %+v", outMissing[0].Content)
	}
	fallback, _ := missingParts[1].(map[string]any)
	if fallback["type"] != "text" {
		t.Errorf("missing image fallback type: got %v, want text", fallback["type"])
	}
	if fallbackText, _ := fallback["text"].(string); !strings.Contains(fallbackText, "img_deadbeef00000000") {
		t.Errorf("missing image fallback missing id reference: %q", fallbackText)
	}

	// Message with no images → plain string, untouched.
	s2 := &Session{Cwd: dir, Messages: []Message{{Role: "user", Content: "no imgs"}}}
	if got := contentString(t, a.buildLLMContext(s2)[0]); got != "no imgs" {
		t.Errorf("plain: got %q, want 'no imgs'", got)
	}

	// Combined: tool uses + images on a message. The image is inlined as an
	// image_url part; the tool use lives in ToolCalls on the assistant message
	// and produces a follow-up tool-role message.
	combined := Message{
		Role:     "assistant",
		Content:  "done",
		Images:   []ImageData{img1},
		ToolUses: []ToolUse{{ID: "tu_77", Name: "read_file", Input: `{"path":"x"}`, Output: "ok"}},
	}
	s3 := &Session{Cwd: dir, Messages: []Message{combined}}
	outCombined := a.buildLLMContext(s3)
	if len(outCombined) != 2 {
		t.Fatalf("combined: expected 2 messages (assistant + tool), got %d: %+v", len(outCombined), outCombined)
	}
	parts, ok = outCombined[0].Content.([]any)
	if !ok || len(parts) != 2 {
		t.Fatalf("combined assistant Content: expected []any of len 2, got %+v", outCombined[0].Content)
	}
	text, _ = parts[0].(map[string]any)
	textStr, _ := text["text"].(string)
	if textStr != "done" {
		t.Errorf("combined text block: got %q, want %q", textStr, "done")
	}
	if len(outCombined[0].ToolCalls) != 1 || outCombined[0].ToolCalls[0].ID != "tu_77" || outCombined[0].ToolCalls[0].Function.Name != "read_file" {
		t.Errorf("combined ToolCalls wrong: %+v", outCombined[0].ToolCalls)
	}
	if outCombined[1].Role != "tool" || outCombined[1].ToolCallID != "tu_77" || outCombined[1].Content.(string) != "ok" {
		t.Errorf("combined tool message wrong: %+v", outCombined[1])
	}
}

// TestBackgroundSummariseAppendsImageRefsThroughCompaction is the end-to-end
// post-compaction view_image story: a user turn with images + an assistant
// turn → backgroundSummarise produces a shadow chunk that includes the
// `Attached images:` ref block → compressHistory folds it into Session.Summary
// so the handle survives even after the original message rotates out.
func TestBackgroundSummariseAppendsImageRefsThroughCompaction(t *testing.T) {
	mock := newMockLLM(t, sseText("Goal: inspect screenshot\nProgress: looked at it"))
	defer mock.Close()

	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, ".codehalter"), 0o755); err != nil {
		t.Fatalf("mkdir .codehalter: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".codehalter", "SUMMARISE.md"), []byte("SUMMARISE PROMPT\n"), 0o644); err != nil {
		t.Fatalf("write SUMMARISE.md: %v", err)
	}

	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	bytes := []byte("PNG screenshot bytes")
	imgID := "img_test_compaction"
	if err := writeImageFile(dir, imgID, "image/png", bytes); err != nil {
		t.Fatalf("writeImageFile: %v", err)
	}
	s.AddUserWithImages("look at this", []ImageData{{ID: imgID, MimeType: "image/png"}})
	s.AddAssistant("I see a screenshot.")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	a.backgroundSummarise(s)

	deadline := time.Now().Add(2 * time.Second)
	for {
		if peek := s.peekShadow(); strings.Contains(peek, "Attached images:") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("backgroundSummarise never appended image refs to shadow; got %q", s.peekShadow())
		}
		time.Sleep(10 * time.Millisecond)
	}

	peek := s.peekShadow()
	if !strings.Contains(peek, "Goal: inspect screenshot") {
		t.Errorf("shadow chunk missing summariser output: %q", peek)
	}
	if !strings.Contains(peek, "view_image id="+imgID) {
		t.Errorf("shadow chunk missing view_image handle %q: %q", imgID, peek)
	}

	// Drive the full compaction. drainShadow requires 2+ entries (the last
	// stays as anchor), so append a second chunk; then exercise compressHistory
	// and confirm the image reference lands in Summary.
	s.appendShadow("Goal: anchor\nProgress: follow-up turn")
	s.SetLastCompletePromptTokens(80_000)
	a.compressHistory(context.Background(), s)

	if !strings.Contains(s.Summary, "view_image id="+imgID) {
		t.Errorf("Summary lost the image handle after compaction; got %q", s.Summary)
	}
}

// TestCompressHistoryShadowPreservesPriorSummary verifies that when the
// shadow fast path runs and a previous Summary is already in place, the
// previous Summary is kept and the shadow is appended after it.
func TestCompressHistoryShadowPreservesPriorSummary(t *testing.T) {
	mock := newMockLLM(t) // shadow fast path is fully local — no LLM calls expected.
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.Summary = "PRIOR SUMMARY FROM AN EARLIER COMPACTION"

	filler := strings.Repeat("lorem ipsum ", 100)
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst %d %s", i, filler))
	}
	s.SetLastCompletePromptTokens(80_000)

	s.appendShadow("Goal: x\nProgress: y")
	s.appendShadow("Goal: x\nProgress: anchor")

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}},
		},
		mainSlotTokens: 90_000,
	}

	a.compressHistory(context.Background(), s)

	if !strings.Contains(s.Summary, "PRIOR SUMMARY") {
		t.Errorf("prior Summary dropped during shadow fast path; got %q", s.Summary)
	}
	if !strings.Contains(s.Summary, "Goal: x") {
		t.Errorf("shadow chunk missing from new Summary; got %q", s.Summary)
	}
}
