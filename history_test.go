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
		// implement it — 404 lets pickAvailable treat the server as "unknown,
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
	return &LLMConnection{Tag: name, URL: m.ts.URL, Model: "test-model"}
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
	s.Title = "round trip"
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
	if loaded.Title != "round trip" {
		t.Errorf("Title: got %q, want %q", loaded.Title, "round trip")
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
					"id":   "call_1",
					"type": "function",
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
		Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
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
		sessions: map[SessionId]*Session{s.ID: s},
	}

	res, err := a.runToolLoop(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "please echo hello"}}, toolFilter{}, "execute")
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

	a := &agent{sessions: map[SessionId]*Session{s.ID: s}}

	res, err := a.runToolLoop(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "answer me"}}, toolFilter{}, "execute")
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
// is what lets runToolLoopJSON parse the assistant text as JSON.
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

	a := &agent{sessions: map[SessionId]*Session{s.ID: s}}

	res, err := a.runToolLoop(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "plan something"}},
		toolFilter{exclude: map[string]bool{respondToolName: true}}, "plan")
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
		Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
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
		Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
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
		[]llmMessage{{Role: "user", Content: "go"}}, toolFilter{}, "execute")
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
		Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
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
		[]llmMessage{{Role: "user", Content: "go"}}, toolFilter{}, "execute")
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
		Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
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
			URL:            mock.ts.URL,
			Model:          "test-model",
			ParamsExecute:  map[string]any{"temperature": 0.3},
			ParamsThinking: map[string]any{"temperature": 1.0},
		}},
	}

	conn := a.pickAvailable(context.Background(), s.ID, "execute")
	if conn == nil {
		t.Fatalf("pickAvailable(execute) returned nil")
	}
	_, err := a.runToolLoop(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, toolFilter{}, "execute")
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
		Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
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
			URL:            mock.ts.URL,
			Model:          "test-model",
			ParamsExecute:  map[string]any{"temperature": 0.3},
			ParamsThinking: map[string]any{"temperature": 1.0},
		}},
	}

	conn := a.pickAvailable(context.Background(), s.ID, "execute")
	if conn == nil {
		t.Fatalf("pickAvailable(execute) returned nil")
	}
	_, err := a.runToolLoop(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, toolFilter{}, "execute")
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

// TestCompressHistoryRecordsSummary is the headline history test: once raw
// messages exceed rawBufferTokens, compressHistory should rotate the session
// — freeze the pre-rotation state to a "session_archive_*" file, push the
// new Summary into the live session, trim raw messages to the recent 20%,
// retitle via the thinking LLM, and persist everything.
func TestCompressHistoryRecordsSummary(t *testing.T) {
	mock := newMockLLM(t,
		sseText("SUMMARY-OF-OLD-MESSAGES"),
		sseText("NEW TITLE"),
	)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	// rawBufferTokens (fallback when n_ctx is unknown) = 60_000, charsPerToken = 4
	// → need > 240_000 chars. 20 messages × ~22_000 chars = ~440_000 chars =
	// ~110_000 tokens, comfortably over the trigger.
	filler := strings.Repeat("lorem ipsum ", 1834) // ~22_000 chars
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user msg %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst msg %d %s", i, filler))
	}
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	originalMsgCount := len(s.Messages)

	a := &agent{
		sessions: map[SessionId]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{URL: mock.ts.URL, Model: "m"}},
		},
	}

	a.compressHistory(context.Background(), s)

	if s.Summary != "SUMMARY-OF-OLD-MESSAGES" {
		t.Errorf("summary: got %q, want %q", s.Summary, "SUMMARY-OF-OLD-MESSAGES")
	}
	if len(s.Messages) >= originalMsgCount {
		t.Errorf("Messages not trimmed: before %d, after %d", originalMsgCount, len(s.Messages))
	}
	// 20/80 split: only the recent 20% stays raw.
	if want := originalMsgCount - (originalMsgCount*4)/5; len(s.Messages) != want {
		t.Errorf("Messages after trim: got %d, want %d", len(s.Messages), want)
	}
	if s.Title != "NEW TITLE" {
		t.Errorf("Title after retitle: got %q, want %q", s.Title, "NEW TITLE")
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
	archived, err := loadSession(dir, SessionId(archiveID))
	if err != nil {
		t.Fatalf("loadSession archive: %v", err)
	}
	if len(archived.Messages) != originalMsgCount {
		t.Errorf("archive Messages: got %d, want %d", len(archived.Messages), originalMsgCount)
	}

	// Persistence.
	loaded, err := loadSession(dir, s.ID)
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if loaded.Summary != "SUMMARY-OF-OLD-MESSAGES" {
		t.Errorf("persisted summary: got %q", loaded.Summary)
	}
	if loaded.Title != "NEW TITLE" {
		t.Errorf("persisted title: got %q", loaded.Title)
	}

	// Exactly two LLM calls (summarize + retitle).
	if mock.callCount() != 2 {
		t.Errorf("LLM calls: got %d, want 2", mock.callCount())
	}

	// The summary prompt should reference the older half of the conversation.
	if body := mock.request(0); body != nil {
		msgs, _ := body["messages"].([]any)
		if len(msgs) == 0 {
			t.Error("summary request had no messages")
		} else {
			m0, _ := msgs[0].(map[string]any)
			content, _ := m0["content"].(string)
			if !strings.Contains(content, "user msg 0") {
				t.Errorf("summary prompt missing oldest user message; got snippet: %q",
					truncate(content, 200))
			}
		}
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
	wg.Add(workers * 4)

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
				s.SetTitle(fmt.Sprintf("title-%d", j))
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
// The test mirrors what runTaskCycle does: stash a stored-message string,
// build the wire content (with sysPrompt prepended on the first turn only),
// then build messages for a second turn and compare.
func TestPrefixStableAcrossTurns(t *testing.T) {
	dir := t.TempDir()

	// Seed a SKILL file so loadSkills returns a non-empty system prompt —
	// otherwise the bug (sysPrompt prepended turn 1, dropped turn 2) is
	// invisible because sysPrompt is empty.
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
	a := &agent{sessions: map[SessionId]*Session{s.ID: s}}

	// Sanity: systemPrompt must be non-empty so the bug we're guarding
	// against (sysPrompt prepended turn 1, dropped turn 2) is observable.
	if sp, _ := a.systemPrompt(s.ID); sp == "" {
		t.Fatal("expected non-empty systemPrompt — SKILL seed didn't take effect")
	}

	// --- Turn 1: first prompt of the session ---
	s.AddUser("first prompt")
	idx1 := len(s.Messages) - 1

	stored1, err := a.composeUserContent(s.ID, "first prompt", nil, nil, true)
	if err != nil {
		t.Fatalf("composeUserContent turn 1: %v", err)
	}
	s.UpdateLastMessageContent(idx1, stored1)

	msgs1 := a.buildLLMHistory(s, idx1)
	msgs1 = append(msgs1, a.buildUserMessage(stored1, nil))

	// LLM replies; runTaskCycle persists the assistant turn.
	s.UpsertLastAssistant("done with turn 1")

	// --- Turn 2: a follow-up prompt ---
	s.AddUser("second prompt")
	idx2 := len(s.Messages) - 1
	stored2, err := a.composeUserContent(s.ID, "second prompt", nil, nil, false)
	if err != nil {
		t.Fatalf("composeUserContent turn 2: %v", err)
	}
	s.UpdateLastMessageContent(idx2, stored2)

	msgs2 := a.buildLLMHistory(s, idx2)
	msgs2 = append(msgs2, a.buildUserMessage(stored2, nil))

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

// TestCompressHistoryNoopWhenBelowBudget verifies that small sessions don't
// call the LLM at all — no summary, no retitle.
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

	a := &agent{
		sessions: map[SessionId]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{URL: mock.ts.URL, Model: "m"}},
		},
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
// synchronous summarize LLM call. Only the retitle call should fire.
func TestCompressHistoryShadowFastPath(t *testing.T) {
	mock := newMockLLM(t,
		sseText("RETITLED FROM SHADOW"),
	)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}

	filler := strings.Repeat("lorem ipsum ", 1834)
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user msg %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst msg %d %s", i, filler))
	}
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	s.appendShadow("Goal: do thing\nProgress: did thing")
	s.appendShadow("Goal: do thing\nProgress: refined thing")

	a := &agent{
		sessions: map[SessionId]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{URL: mock.ts.URL, Model: "m"}},
		},
	}

	a.compressHistory(context.Background(), s)

	if !strings.Contains(s.Summary, "did thing") || !strings.Contains(s.Summary, "refined thing") {
		t.Errorf("expected Summary to contain both shadow chunks, got %q", s.Summary)
	}
	// Exactly one LLM call (retitle only — no synchronous summarize).
	if mock.callCount() != 1 {
		t.Errorf("LLM calls: got %d, want 1 (retitle only)", mock.callCount())
	}
	// Drain again — buffer must be empty after compaction.
	if remaining := s.drainShadow(); remaining != "" {
		t.Errorf("shadow buffer not drained after compaction: %q", remaining)
	}
}

// TestCompressHistoryShadowPreservesPriorSummary verifies that when the
// shadow fast path runs and a previous Summary is already in place, the
// previous Summary is kept and the shadow is appended after it.
func TestCompressHistoryShadowPreservesPriorSummary(t *testing.T) {
	mock := newMockLLM(t,
		sseText("RETITLED"),
	)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.Summary = "PRIOR SUMMARY FROM AN EARLIER COMPACTION"

	filler := strings.Repeat("lorem ipsum ", 1834)
	for i := 0; i < 10; i++ {
		s.AddUser(fmt.Sprintf("user %d %s", i, filler))
		s.AddAssistant(fmt.Sprintf("asst %d %s", i, filler))
	}

	s.appendShadow("Goal: x\nProgress: y")

	a := &agent{
		sessions: map[SessionId]*Session{s.ID: s},
		settings: Settings{
			LLM: []LLMConnection{{URL: mock.ts.URL, Model: "m"}},
		},
	}

	a.compressHistory(context.Background(), s)

	if !strings.Contains(s.Summary, "PRIOR SUMMARY") {
		t.Errorf("prior Summary dropped during shadow fast path; got %q", s.Summary)
	}
	if !strings.Contains(s.Summary, "Goal: x") {
		t.Errorf("shadow chunk missing from new Summary; got %q", s.Summary)
	}
}
