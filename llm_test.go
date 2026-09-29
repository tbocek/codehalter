package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestTrimJSON(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{name: "plain", in: `{"ok":true}`, want: `{"ok":true}`},
		{name: "leading whitespace", in: "  \n{\"ok\":true}\n  ", want: `{"ok":true}`},
		{name: "json fence", in: "```json\n{\"ok\":true}\n```", want: `{"ok":true}`},
		{name: "bare fence", in: "```\n{\"ok\":true}\n```", want: `{"ok":true}`},
		{name: "prose prefix", in: "Sure, here's the JSON:\n{\"ok\":true}", want: `{"ok":true}`},
		{name: "prose suffix", in: "{\"ok\":true}\nLet me know if you need more.", want: `{"ok":true}`},
		{name: "prose both sides", in: "Here you go: {\"ok\":true} — that's it!", want: `{"ok":true}`},
		{name: "nested", in: "noise {\"a\":{\"b\":1}} noise", want: `{"a":{"b":1}}`},
		{name: "brace in string", in: `{"s":"} not the end"}`, want: `{"s":"} not the end"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := trimJSON(tc.in); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// Only a `purpose = "summary"` entry takes background work off llm[0]; an unmarked
// extra entry must NOT absorb the summariser.
func TestBackgroundSlotLabel(t *testing.T) {
	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: "u", Model: "m", Parallel: ptr(2)}}}}
	a.buildConnSems()
	if fg := a.settings.ConnAt(0, "execute"); fg == nil || fg.Slot != 0 {
		t.Fatalf("ConnAt(0).Slot = %v, want 0", fg)
	}
	bg, onMain := a.connForBackgroundLLM()
	if bg == nil || bg.Slot != 1 || bg.Server != "u" || bg.Model != "m" || !onMain {
		t.Fatalf("connForBackgroundLLM = %+v onMain=%v, want Slot 1 on u/m, onMain", bg, onMain)
	}

	a1 := &agent{settings: Settings{LLM: []LLMConnection{{Server: "u", Model: "m", Parallel: ptr(1)}}}}
	a1.buildConnSems()
	if bg, onMain := a1.connForBackgroundLLM(); bg == nil || bg.Slot != 0 || !onMain {
		t.Fatalf("single-slot connForBackgroundLLM = %+v onMain=%v, want Slot 0, onMain", bg, onMain)
	}

	// Two entries, neither designated: background stays on llm[0].
	a2 := &agent{settings: Settings{LLM: []LLMConnection{
		{Server: "u0", Model: "m0", Parallel: ptr(1)},
		{Server: "u1", Model: "m1", Parallel: ptr(1)},
	}}}
	a2.buildConnSems()
	if bg, onMain := a2.connForBackgroundLLM(); bg == nil || bg.Server != "u0" || !onMain {
		t.Fatalf("undesignated extra entry = %+v onMain=%v, want u0 onMain — it must not absorb the summariser", bg, onMain)
	}

	a3 := &agent{settings: Settings{LLM: []LLMConnection{
		{Server: "u0", Model: "m0", Parallel: ptr(1)},
		{Server: "u1", Model: "m1", Parallel: ptr(1)},
		{Server: "u2", Model: "m2", Parallel: ptr(1), Purpose: "summary"},
	}}}
	a3.buildConnSems()
	if bg, onMain := a3.connForBackgroundLLM(); bg == nil || bg.Slot != 2 || bg.Server != "u2" || onMain {
		t.Fatalf("designated connForBackgroundLLM = %+v onMain=%v, want Slot 2 on u2, NOT onMain", bg, onMain)
	}

	// Designated but saturated → fall back to llm[0] rather than queue behind it.
	a3.connSems[2] <- struct{}{}
	if bg, onMain := a3.connForBackgroundLLM(); bg == nil || bg.Server != "u0" || !onMain {
		t.Fatalf("saturated summary conn = %+v onMain=%v, want u0 onMain", bg, onMain)
	}

	// purpose on llm[0] is the same as no purpose: the fallback already lands there.
	a4 := &agent{settings: Settings{LLM: []LLMConnection{{Server: "u", Model: "m", Parallel: ptr(2), Purpose: "summary"}}}}
	a4.buildConnSems()
	if bg, onMain := a4.connForBackgroundLLM(); bg == nil || bg.Slot != 1 || !onMain {
		t.Fatalf("purpose on llm[0] = %+v onMain=%v, want Slot 1 onMain", bg, onMain)
	}
}

// An in-flight llmStream releases on the channel it acquired, so an unchanged
// reload must not swap it.
func TestBuildConnSemsIdempotent(t *testing.T) {
	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: "s", Model: "m"}}}}
	a.buildConnSems()
	first := a.connSems[0]

	a.buildConnSems() // same shape → must reuse the same channel
	if a.connSems[0] != first {
		t.Fatal("buildConnSems swapped the channel on an unchanged reload — would orphan in-flight permits")
	}

	v := cap(first) + 3
	a.settings.LLM[0].Parallel = &v
	a.buildConnSems()
	if a.connSems[0] == first || cap(a.connSems[0]) != cap(first)+3 {
		t.Errorf("cap change should rebuild: got cap %d, want %d", cap(a.connSems[0]), cap(first)+3)
	}
}

// Only meaningful under -race.
func TestCfgConcurrentReloadAndRead(t *testing.T) {
	a, _ := newTestAgent(t)
	reload := func(p int) {
		a.cfgMu.Lock()
		a.settings = Settings{LLM: []LLMConnection{
			{Server: "http://a", Model: "m0", Parallel: ptr(p)},
			{Server: "http://b", Model: "m1", Parallel: ptr(1)},
		}}
		a.buildConnSems()
		a.cfgMu.Unlock()
	}
	reload(2)

	var wg sync.WaitGroup
	stop := make(chan struct{})

	wg.Add(1)
	go func() { // writer: prepare reloading settings repeatedly
		defer wg.Done()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
				reload((i % 3) + 1)
			}
		}
	}()

	for r := 0; r < 4; r++ { // readers: background conn resolution
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					_, _ = a.connForBackgroundLLM()
					_ = a.connFor("execute")
				}
			}
		}()
	}

	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// A 400 is a full context only when the body says so.
func TestIsContextFull(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"llama.cpp overflow", &llmHTTPError{Status: 400, Body: "request (262314 tokens) exceeds the available context size (262144 tokens), try increasing it"}, true},
		{"llama.cpp structured type", &llmHTTPError{Status: 400, Type: "exceed_context_size_error", Body: "n/a"}, true},
		{"halogen: prompt + max_tokens over the window", &llmHTTPError{Status: 400, Body: "max_tokens 32768 does not fit: prompt is 255003 tokens and the context is 262144, leaving room for 7141"}, true},
		{"vLLM / OpenAI wording", &llmHTTPError{Status: 400, Body: "This model's maximum context length is 8192 tokens. However, you requested 9000 tokens"}, true},
		{"413", &llmHTTPError{Status: 413, Body: "payload too large"}, true},
		{"ceiling truncation", errContextCeiling, true},
		{"llama.cpp cannot continue", &llmHTTPError{Status: 400, Body: "Cannot continue an assistant message that contains tool calls."}, false},
		{"halogen prefill shape", &llmHTTPError{Status: 400, Body: "continue_final_message cannot be combined with a forced tool choice: one resumes the assistant turn already in the history, the other starts a new call"}, false},
		{"500 mentioning context", &llmHTTPError{Status: 500, Body: "context size"}, false},
		{"bare 400, no reason given", &llmHTTPError{Status: 400}, false},
		{"500 error", &llmHTTPError{Status: 500, Body: "boom"}, false},
		{"wrapped ceiling", fmt.Errorf("ceiling: %w", errContextCeiling), true},
		{"transport", errors.New("dial tcp: connection refused"), false},
		{"nil", nil, false},
	} {
		if got := isContextFull(tc.err); got != tc.want {
			t.Errorf("%s: isContextFull = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// BELOW the cap is the n_ctx ceiling; AT the cap, reasoning or content, is a cap hit.
func TestFinishLengthClassification(t *testing.T) {
	run := func(sse string) error {
		t.Helper()
		mock := newMockLLM(t, sse)
		defer mock.Close()
		a, _ := newTestAgent(t)
		a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
		a.mainSlotTokens.Store(85248)
		conn := a.connFor("thinking")
		if conn == nil {
			t.Fatalf("connFor returned nil")
		}
		_, _, _, err := a.llmStream(context.Background(), "", conn, []llmMessage{{Role: "user", Content: "go"}}, nil, nil, nil, nil)
		return err
	}

	var ce *capHitError
	if err := run(sseTruncated("thinking", 82393, 2854)); err == nil || !isContextFull(err) || errors.As(err, &ce) {
		t.Errorf("below-cap truncation should be a context ceiling, got: %v", err)
	}
	// completion_tokens omitted, but prompt+max_tokens overruns n_ctx: still the ceiling.
	if err := run(sseTruncated("thinking", 80000, 0)); err == nil || !isContextFull(err) {
		t.Errorf("no-room truncation with unreported completion should be a ceiling, got: %v", err)
	}
	for name, sse := range map[string]string{
		"reasoning only": sseTruncated("thinking", 1000, defaultMaxTokens),
		"content":        sseTruncatedContent("verbose output", 1000, defaultMaxTokens),
	} {
		err := run(sse)
		if !errors.As(err, &ce) || isContextFull(err) {
			t.Errorf("%s at the cap should be a cap hit, got: %v", name, err)
		} else if ce.Cap != defaultMaxTokens {
			t.Errorf("%s: capHitError.Cap = %d, want %d", name, ce.Cap, defaultMaxTokens)
		}
	}
}

// llama.cpp sends reasoning_content, vLLM sends reasoning.
func TestReasoningArrivesUnderEitherSpelling(t *testing.T) {
	sse := func(field string) string {
		c, _ := json.Marshal(map[string]any{"choices": []map[string]any{{
			"delta": map[string]any{field: "weighing the options"},
		}}})
		return fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", c)
	}
	for _, field := range []string{"reasoning_content", "reasoning"} {
		mock := newMockLLM(t, sse(field))
		a, _ := newTestAgent(t)
		a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
		conn := a.connFor("thinking")
		if conn == nil {
			mock.Close()
			t.Fatalf("%s: connFor returned nil", field)
		}
		var streamed strings.Builder
		_, _, reasoning, err := a.llmStream(context.Background(), "", conn,
			[]llmMessage{{Role: "user", Content: "go"}}, nil, nil,
			func(s string) { streamed.WriteString(s) }, nil)
		mock.Close()
		if err != nil {
			t.Fatalf("%s: %v", field, err)
		}
		if reasoning != "weighing the options" {
			t.Errorf("%s: accumulated reasoning = %q, want it kept", field, reasoning)
		}
		if streamed.String() != "weighing the options" {
			t.Errorf("%s: streamed to the thought channel = %q, want it forwarded", field, streamed.String())
		}
	}
}

func TestIsTransientStreamError(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"io.EOF", io.EOF, true},
		{"unexpected EOF", io.ErrUnexpectedEOF, true},
		{"wrapped SSE EOF", fmt.Errorf("reading SSE stream: %w", io.ErrUnexpectedEOF), true},
		{"connection reset string", errors.New(`Post "http://x": read: connection reset by peer`), true},
		{"broken pipe string", errors.New("write: broken pipe"), true},
		{"net error", &net.OpError{Op: "read", Err: errors.New("reset")}, true},
		{"context canceled", context.Canceled, false},
		{"user cancelled", errUserCancelled, false},
		{"deadline exceeded", context.DeadlineExceeded, false},
		{"clean LLM error", errors.New("model returned no plan"), false},
		{"nil", nil, false},
	}
	for _, c := range cases {
		if got := isTransientStreamError(c.err); got != c.want {
			t.Errorf("%s: isTransientStreamError=%v, want %v", c.name, got, c.want)
		}
	}
}

func TestThinkingOn(t *testing.T) {
	cases := []struct {
		name string
		body map[string]any
		want bool
	}{
		{"no kwargs", map[string]any{}, true},
		{"empty kwargs", map[string]any{"chat_template_kwargs": map[string]any{}}, true},
		{"enabled", map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": true}}, true},
		{"disabled", map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}, false},
		{"prefill continued", map[string]any{"continue_final_message": true}, false},
		{"prefill, kwargs say on", map[string]any{
			"continue_final_message": true,
			"chat_template_kwargs":   map[string]any{"preserve_thinking": true},
		}, false},
	}
	for _, c := range cases {
		if got := thinkingOn(c.body); got != c.want {
			t.Errorf("%s: thinkingOn=%v, want %v", c.name, got, c.want)
		}
	}
}

// ExtraBody must stay untouched: renderKey fingerprints it, and writing
// chat_template_kwargs there would re-render the whole prompt.
func TestWithThinkingDisabled(t *testing.T) {
	orig := &LLMConnection{Server: "s", Model: "m", Slot: 2, ExtraBody: map[string]any{
		"temperature":          0.7,
		"chat_template_kwargs": map[string]any{"preserve_thinking": true},
	}}
	off := orig.withThinkingDisabled()

	if off.Server != "s" || off.Model != "m" || off.Slot != 2 {
		t.Errorf("routing fields changed: %+v", off)
	}
	if !off.noThinkPrefill {
		t.Error("withThinkingDisabled did not arm the prefill")
	}
	if renderKey(off.ExtraBody) != renderKey(orig.ExtraBody) {
		t.Errorf("the retry changed the rendering: %s vs %s",
			renderKey(off.ExtraBody), renderKey(orig.ExtraBody))
	}
	if _, set := off.ExtraBody["chat_template_kwargs"].(map[string]any)["enable_thinking"]; set {
		t.Errorf("the retry still writes enable_thinking: %+v", off.ExtraBody)
	}
	if off.ExtraBody["temperature"] != 0.7 {
		t.Errorf("sibling params dropped: %+v", off.ExtraBody)
	}
	if orig.noThinkPrefill {
		t.Error("withThinkingDisabled mutated the original conn")
	}
}

// Earlier messages and the caller's slice stay untouched: an append keeps the prefix cache.
func TestPrefillIsAppendedNotRendered(t *testing.T) {
	mock := newMockLLM(t, sseText("done"))
	defer mock.Close()
	a, sess := newTestAgent(t)

	conn := mock.conn("test").withThinkingDisabled()
	msgs := []llmMessage{{Role: "user", Content: "hi"}}
	if _, _, _, err := a.llmStream(context.Background(), sess.ID, conn, msgs, nil, nil, nil, nil); err != nil {
		t.Fatalf("llmStream: %v", err)
	}
	if len(msgs) != 1 {
		t.Errorf("llmStream mutated the caller's messages: %+v", msgs)
	}
	body := mock.request(0)
	if body["add_generation_prompt"] != false || body["continue_final_message"] != true {
		t.Errorf("continuation flags missing: add_generation_prompt=%v continue_final_message=%v",
			body["add_generation_prompt"], body["continue_final_message"])
	}
	if _, set := body["chat_template_kwargs"]; set {
		t.Errorf("the retry re-rendered the prompt instead of appending: %v", body["chat_template_kwargs"])
	}
	sent, _ := body["messages"].([]any)
	if len(sent) != 2 {
		t.Fatalf("messages: got %d, want the original plus the prefill: %v", len(sent), sent)
	}
	first, _ := sent[0].(map[string]any)
	if first["role"] != "user" || first["content"] != "hi" {
		t.Errorf("the original message changed: %v", first)
	}
	last, _ := sent[1].(map[string]any)
	if last["role"] != "assistant" || last["content"] != noThinkPrefillContent {
		t.Errorf("prefill: got %v, want an assistant %q", last, noThinkPrefillContent)
	}
}

// The copy overrides the role default; the original keeps its own cap.
func TestWithBody(t *testing.T) {
	orig := &LLMConnection{Server: "s", Model: "m", Slot: 1, ExtraBody: map[string]any{
		"max_tokens":  8192,
		"temperature": 0.7,
	}}
	capped := orig.withBody("max_tokens", 1)

	if capped.Server != "s" || capped.Model != "m" || capped.Slot != 1 {
		t.Errorf("routing fields changed: %+v", capped)
	}
	if capped.ExtraBody["max_tokens"] != 1 || capped.ExtraBody["temperature"] != 0.7 {
		t.Errorf("ExtraBody: got %+v, want max_tokens=1 + temperature kept", capped.ExtraBody)
	}
	if orig.ExtraBody["max_tokens"] != 8192 {
		t.Error("withBody mutated the original conn")
	}
}

// Fires once per Server+Model, and only for a user-set enable_thinking=false.
func TestWarnsWhenServerIgnoresThinkingOff(t *testing.T) {
	reasoned, _ := json.Marshal(map[string]any{"choices": []map[string]any{{
		"delta": map[string]any{"reasoning_content": "still thinking about it"},
	}}})
	sseReasoned := fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", reasoned)

	warned := func(t *testing.T, role, body string) bool {
		t.Helper()
		mock := newMockLLM(t, body)
		defer mock.Close()
		a, s := newTestAgent(t)
		a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m",
			ParamsExecute: map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}}}}
		conn := a.connFor(role)
		if conn == nil {
			t.Fatalf("connFor(%s) returned nil", role)
		}
		if _, _, _, err := a.llmStream(context.Background(), s.ID, conn,
			[]llmMessage{{Role: "user", Content: "go"}}, nil, nil, nil, nil); err != nil {
			t.Fatalf("llmStream(%s): %v", role, err)
		}
		_, seen := a.ctkIgnored.Load(conn.Server + "\x00" + conn.Model)
		return seen
	}

	// execute carries enable_thinking=false, and the server reasoned regardless.
	if !warned(t, "execute", sseReasoned) {
		t.Error("server ignored enable_thinking=false and nothing was reported")
	}
	// thinking asked for no such thing, so reasoning is exactly what was ordered.
	if warned(t, "thinking", sseReasoned) {
		t.Error("warned about reasoning on the role that requested it")
	}
	if warned(t, "execute", sseText("done")) {
		t.Error("warned although the server produced no reasoning")
	}
}

func TestRejectedChatTemplateKwargsNamesTheSetting(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"Unrecognized request argument supplied: chat_template_kwargs"}}`, http.StatusBadRequest)
	}))
	defer ts.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Server: ts.URL, Model: "gpt-x",
		ParamsExecute: map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}}}}
	conn := a.connFor("execute")
	if conn == nil {
		t.Fatal("connFor returned nil")
	}
	_, _, _, err := a.llmStream(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, nil, nil, nil, nil)
	if err == nil {
		t.Fatal("a 400 should be an error")
	}
	if !strings.Contains(err.Error(), "params_execute") {
		t.Errorf("rejection error does not say where the field came from:\n%v", err)
	}
}

func TestLLMStreamParsesTextAndTools(t *testing.T) {
	var b strings.Builder
	c1, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{"content": "Hello "},
		}},
	})
	fmt.Fprintf(&b, "data: %s\n\n", c1)
	c2, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{"content": "world"},
		}},
	})
	fmt.Fprintf(&b, "data: %s\n\n", c2)
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
	// No id: appends to the last call.
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
	text, calls, _, err := a.llmStream(
		context.Background(),
		"", // unscoped: no session log
		mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "hi"}},
		nil,
		func(tok string) { collected.WriteString(tok) },
		nil,
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

// An {"error":…} chunk under HTTP 200 must raise the server's message, not read as empty.
func TestLLMStreamSurfacesInStreamError(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{
			name: "nested openai shape",
			body: `data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}` + "\n\n" +
				`data: {"error":{"message":"The number of tokens to keep from the initial prompt is greater than the context length"}}` + "\n\n" +
				"data: [DONE]\n\n",
		},
		{
			name: "bare top-level message",
			body: `data: {"message":"The number of tokens to keep from the initial prompt is greater than the context length"}` + "\n\n" +
				"data: [DONE]\n\n",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mock := newMockLLM(t, tc.body)
			defer mock.Close()

			a := &agent{}
			_, _, _, err := a.llmStream(
				context.Background(), "", mock.conn("execute"),
				[]llmMessage{{Role: "user", Content: "hi"}}, nil, nil, nil, nil,
			)
			if err == nil {
				t.Fatal("llmStream: got nil error, want the in-stream error surfaced")
			}
			if !strings.Contains(err.Error(), "greater than the context length") {
				t.Errorf("error must carry the server's message verbatim, got: %v", err)
			}
			if !strings.Contains(err.Error(), "mid-stream") {
				t.Errorf("error should be framed as a mid-stream failure, got: %v", err)
			}
		})
	}
}

// Unreachable at the probe, or struck out until summaryCooldown: back to llm[0].
func TestBackgroundSkipsDeadSummariser(t *testing.T) {
	newAgent := func() *agent {
		a := &agent{settings: Settings{LLM: []LLMConnection{
			{Server: "u0", Model: "m0", Parallel: ptr(1)},
			{Server: "sum", Model: "ms", Parallel: ptr(1), Purpose: "summary"},
		}}}
		a.buildConnSems()
		return a
	}
	wantSummariser := func(t *testing.T, a *agent, why string) {
		t.Helper()
		if bg, onMain := a.connForBackgroundLLM(); bg == nil || bg.Server != "sum" || onMain {
			t.Fatalf("%s: got %+v onMain=%v, want the summariser at sum", why, bg, onMain)
		}
	}
	wantMain := func(t *testing.T, a *agent, why string) {
		t.Helper()
		if bg, onMain := a.connForBackgroundLLM(); bg == nil || bg.Server != "u0" || !onMain {
			t.Fatalf("%s: got %+v onMain=%v, want llm[0] at u0 with onMain", why, bg, onMain)
		}
	}

	// No probe result yet (nil map) is not evidence of a dead server.
	wantSummariser(t, newAgent(), "unprobed")

	a := newAgent()
	a.connProbe = map[string]probeResult{"sum\x00ms": {Reachable: true}}
	wantSummariser(t, a, "probe says reachable")

	a = newAgent()
	a.connProbe = map[string]probeResult{"sum\x00ms": {Reachable: false}}
	wantMain(t, a, "probe says unreachable")

	// A probe result for some OTHER endpoint must not take this one out.
	a = newAgent()
	a.connProbe = map[string]probeResult{"elsewhere\x00mx": {Reachable: false}}
	wantSummariser(t, a, "unrelated unreachable endpoint")

	a = newAgent()
	a.summaryStrikes.Store(summaryMaxStrikes)
	a.summaryStruckAt.Store(time.Now().UnixNano())
	wantMain(t, a, "struck out")

	// Struck out is not forever: after the cooldown it gets another call.
	a = newAgent()
	a.summaryStrikes.Store(summaryMaxStrikes)
	a.summaryStruckAt.Store(time.Now().Add(-summaryCooldown - time.Second).UnixNano())
	wantSummariser(t, a, "cooldown over")

	a = newAgent()
	a.summaryStrikes.Store(summaryMaxStrikes - 1)
	wantSummariser(t, a, "one strike short")
}

func TestKeepWarmRefreshesUntilStopped(t *testing.T) {
	mock := newMockLLM(t, sseText("."), sseText("."), sseText("."), sseText("."))
	defer mock.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}, KeepWarm: "20ms"}
	msgs := []llmMessage{{Role: "user", Content: "the conversation so far"}}

	stop := a.keepWarm(s, mock.conn("execute"), func() []llmMessage { return msgs })
	deadline := time.Now().Add(2 * time.Second)
	for mock.callCount() == 0 {
		if time.Now().After(deadline) {
			stop()
			t.Fatal("keepWarm never refreshed the prefix")
		}
		time.Sleep(5 * time.Millisecond)
	}
	stop()

	// More than one token costs generation; anything but the conversation seeds another prefix.
	body := mock.request(0)
	if body["max_tokens"] != float64(1) {
		t.Errorf("refresh max_tokens = %v, want 1", body["max_tokens"])
	}
	sent, _ := body["messages"].([]any)
	if len(sent) != 1 {
		t.Fatalf("refresh sent %d messages, want the conversation", len(sent))
	}
	if m, _ := sent[0].(map[string]any); m["content"] != "the conversation so far" {
		t.Errorf("refresh sent %v, want the conversation", m)
	}

	settled := mock.callCount()
	time.Sleep(80 * time.Millisecond)
	if got := mock.callCount(); got != settled {
		t.Errorf("stop() did not end the refreshes: %d -> %d", settled, got)
	}
	// A refresh is not the turn's work, so it must not land in the turn stats.
	if r := s.turnStats(); r.completion != 0 || r.evaluatedPrompt != 0 {
		t.Errorf("a refresh was counted against the turn: %+v", r)
	}
}

// A hosted endpoint caches on its own and bills per request.
func TestKeepWarmOff(t *testing.T) {
	mock := newMockLLM(t, sseText("."))
	defer mock.Close()
	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}, KeepWarm: "off"}
	stop := a.keepWarm(s, mock.conn("execute"), func() []llmMessage { return []llmMessage{{Role: "user", Content: "x"}} })
	defer stop()
	time.Sleep(60 * time.Millisecond)
	if mock.callCount() != 0 {
		t.Errorf("keep_warm=off still called the model %d time(s)", mock.callCount())
	}
}

// Halogen: the refused continuation is retried with the forced tool choice alone,
// and the entry remembers, so a later call that forces nothing uses enable_thinking.
func TestLLMStreamDropsPrefillWhenRejected(t *testing.T) {
	var mu sync.Mutex
	var reqs []map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode: %v", err)
		}
		mu.Lock()
		reqs = append(reqs, body)
		mu.Unlock()
		// Like Halogen: the prefill shape is refused every time, not just once.
		if body["continue_final_message"] == true {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"continue_final_message cannot be combined with a forced tool choice: one resumes the assistant turn already in the history, the other starts a new call","type":"invalid_request_error"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(sseText("ok")))
	}))
	defer ts.Close()

	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: ts.URL, Model: "m"}}}}
	conn := a.settings.ConnAt(0, "execute").withThinkingDisabled().withBody("tool_choice", "required")
	msgs := []llmMessage{{Role: "user", Content: "go"}}
	if _, _, _, err := a.llmStream(context.Background(), "", conn, msgs, nil, nil, nil, nil); err != nil {
		t.Fatalf("llmStream: %v", err)
	}
	if len(reqs) != 2 {
		t.Fatalf("calls = %d, want the rejected one and the retry", len(reqs))
	}
	first, retry := reqs[0], reqs[1]
	if first["continue_final_message"] != true || first["add_generation_prompt"] != false {
		t.Errorf("first call did not carry the prefill shape: %v", first)
	}
	if _, has := retry["continue_final_message"]; has {
		t.Error("the retry still carried continue_final_message")
	}
	if retry["tool_choice"] != "required" {
		t.Errorf("the retry lost tool_choice: %v", retry["tool_choice"])
	}
	if _, has := retry["chat_template_kwargs"]; has {
		t.Error("the retry added a thinking flag beside a forced choice, a third rendering")
	}
	if msgsOut := retry["messages"].([]any); len(msgsOut) != len(msgs) {
		t.Errorf("the retry still carried the prefill message: %d messages", len(msgsOut))
	}

	// Every copy must know, even one taken from the original pointer before the 400.
	for name, c := range map[string]*LLMConnection{"the reused connection": conn, "a fresh copy of the original": conn.withBody("max_tokens", 1)} {
		before := len(reqs)
		if _, _, _, err := a.llmStream(context.Background(), "", c, msgs, nil, nil, nil, nil); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(reqs) != before+1 {
			t.Fatalf("%s: %d requests, want 1 (must skip the prefill without a refusal)", name, len(reqs)-before)
		}
		if _, has := reqs[len(reqs)-1]["continue_final_message"]; has {
			t.Errorf("%s sent the prefill again", name)
		}
	}

	// Remembered on the entry: a call that forces nothing now uses the flag.
	next := a.settings.ConnAt(0, "thinking").withThinkingDisabled()
	req := buildChatRequest(next, msgs, nil)
	if _, has := req["continue_final_message"]; has {
		t.Error("the entry did not remember the rejection")
	}
	if kw, _ := req["chat_template_kwargs"].(map[string]any); kw["enable_thinking"] != false {
		t.Errorf("a call that forces nothing must use enable_thinking=false here, got %v", req["chat_template_kwargs"])
	}
	// And thinking ON stays exactly what it was: no flag, no continuation.
	on := buildChatRequest(a.settings.ConnAt(0, "thinking"), msgs, nil)
	if _, has := on["chat_template_kwargs"]; has {
		t.Errorf("a thinking-on call grew a flag: %v", on["chat_template_kwargs"])
	}
}

func TestRequestLogDelta(t *testing.T) {
	first := []byte(`{"max_tokens":8192,"messages":[{"role":"user","content":"hi"}]}`)
	if got := requestLogDelta(nil, first); got != string(first) {
		t.Errorf("first request logged as %q", got)
	}
	// A new wrapper over the same messages plus one is not a lost prefix.
	second := []byte(`{"chat_template_kwargs":{"enable_thinking":false},"max_tokens":16384,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}`)
	got := requestLogDelta(first, second)
	if !strings.HasPrefix(got, `{"chat_template_kwargs":{"enable_thinking":false},"max_tokens":16384,`+requestLogSame+"42 of ") ||
		!strings.HasSuffix(got, `,{"role":"assistant","content":"yo"}]}`) || strings.Contains(got, "ONLY") {
		t.Errorf("appended request logged as %q", got)
	}
	changed := []byte(`{"max_tokens":8192,"messages":[{"role":"system","content":"new"}]}`)
	if got := requestLogDelta(second, changed); !strings.Contains(got, "ONLY: a change this early") {
		t.Errorf("an early change was not called out: %q", got)
	}
}

// Halogen's picture limit, refused in the stream, is recovered like a full context.
func TestIsContextFullSeesThePictureLimit(t *testing.T) {
	if !isContextFull(&llmStreamError{Msg: "the engine refused this request: IMG count outside 0..64"}) {
		t.Error("the picture-count refusal was not taken for a full context")
	}
	if isContextFull(&llmStreamError{Msg: "the engine refused this request: bad grammar"}) {
		t.Error("another in-stream refusal was taken for a full context")
	}
}
