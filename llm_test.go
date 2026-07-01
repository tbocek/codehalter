package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
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

// TestBackgroundSlotLabel pins the display-slot routing AND the onMain flag
// that switches background prompts to prefix-extension mode: the foreground
// turn reads llm[0]; background work on a single [[llm]] entry falls back to
// llm[0] (onMain=true — its KV holds the conversation, so extend it), labelled
// llm[1] for display when parallel >= 2; a second entry routes background to
// llm[1] proper (onMain=false — fresh prompt, no cache to protect there).
func TestBackgroundSlotLabel(t *testing.T) {
	// Single entry, parallel=2 → foreground llm[0], background llm[1] (same conn).
	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: "u", Model: "m", Parallel: ptr(2)}}}}
	a.buildConnSems()
	if fg := a.settings.MainLLM("execute"); fg == nil || fg.Slot != 0 {
		t.Fatalf("MainLLM.Slot = %v, want 0", fg)
	}
	bg, onMain := a.connForBackgroundLLM()
	if bg == nil || bg.Slot != 1 || bg.Server != "u" || bg.Model != "m" || !onMain {
		t.Fatalf("connForBackgroundLLM = %+v onMain=%v, want Slot 1 on u/m, onMain", bg, onMain)
	}

	// Single entry, parallel=1 → no second slot to label; background stays llm[0].
	a1 := &agent{settings: Settings{LLM: []LLMConnection{{Server: "u", Model: "m", Parallel: ptr(1)}}}}
	a1.buildConnSems()
	if bg, onMain := a1.connForBackgroundLLM(); bg == nil || bg.Slot != 0 || !onMain {
		t.Fatalf("single-slot connForBackgroundLLM = %+v onMain=%v, want Slot 0, onMain", bg, onMain)
	}

	// Two entries → background routes to the second entry, llm[1].
	a2 := &agent{settings: Settings{LLM: []LLMConnection{
		{Server: "u0", Model: "m0", Parallel: ptr(1)},
		{Server: "u1", Model: "m1", Parallel: ptr(1)},
	}}}
	a2.buildConnSems()
	if bg, onMain := a2.connForBackgroundLLM(); bg == nil || bg.Slot != 1 || bg.Server != "u1" || onMain {
		t.Fatalf("two-entry connForBackgroundLLM = %+v onMain=%v, want Slot 1 on u1, NOT onMain", bg, onMain)
	}
}

// TestBuildConnSemsIdempotent pins the fix for the connSems release deadlock: an
// unchanged settings reload must NOT swap the slot channels, or an in-flight
// llmStream (which captured the old channel at acquire) would release into a new
// empty channel and block forever. A real cap change DOES rebuild.
func TestBuildConnSemsIdempotent(t *testing.T) {
	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: "s", Model: "m"}}}}
	a.buildConnSems()
	first := a.connSems[0]

	a.buildConnSems() // same shape → must reuse the same channel
	if a.connSems[0] != first {
		t.Fatal("buildConnSems swapped the channel on an unchanged reload — would orphan in-flight permits")
	}

	// A cap change rebuilds.
	v := cap(first) + 3
	a.settings.LLM[0].Parallel = &v
	a.buildConnSems()
	if a.connSems[0] == first || cap(a.connSems[0]) != cap(first)+3 {
		t.Errorf("cap change should rebuild: got cap %d, want %d", cap(a.connSems[0]), cap(first)+3)
	}
}

// TestCfgConcurrentReloadAndRead exercises the cfgMu guard: a foreground "prepare"
// reassigns a.settings + rebuilds a.connSems while background goroutines resolve
// connections through connForBackgroundLLM / connForSession (as the summariser and
// git-commit drafter do). Before cfgMu these raced the settings struct and the
// connSems slice header; the test is meaningful under -race, where an
// unsynchronised access on either side reports a failure.
func TestCfgConcurrentReloadAndRead(t *testing.T) {
	a, s := newTestAgent(t)
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
					_ = a.connForSession(context.Background(), s.ID, "execute")
				}
			}
		}()
	}

	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()
}

func TestIsContextFull(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"400 reject", &llmHTTPError{Status: 400}, true},
		{"context ceiling", fmt.Errorf("ceiling: %w", errContextCeiling), true},
		{"500 error", &llmHTTPError{Status: 500}, false},
		{"plain error", errors.New("boom"), false},
		{"nil", nil, false},
	}
	for _, c := range cases {
		if got := isContextFull(c.err); got != c.want {
			t.Errorf("%s: isContextFull=%v, want %v", c.name, got, c.want)
		}
	}
}

// TestFinishLengthClassification pins the finish=length split: truncation BELOW
// the cap is the n_ctx ceiling (fold + retry); reasoning-only AT the cap is a
// <think> stall (thinking-off retry); content AT the cap is a genuine, not-
// recoverable verbose/looping cap.
func TestFinishLengthClassification(t *testing.T) {
	run := func(sse string) error {
		t.Helper()
		mock := newMockLLM(t, sse)
		defer mock.Close()
		a, s := newTestAgent(t)
		a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
		a.mainSlotTokens = 85248
		conn := a.connForSession(context.Background(), s.ID, "thinking")
		if conn == nil {
			t.Fatalf("connForSession returned nil")
		}
		// sid="" disables session logging; the finish=length classification works
		// off the locally-parsed usage tokens regardless.
		_, _, _, err := a.llmStream(context.Background(), "", conn, []llmMessage{{Role: "user", Content: "go"}}, nil, nil, nil)
		return err
	}

	// Reasoning truncated BELOW the cap → n_ctx ceiling (recover by folding).
	if err := run(sseTruncated("thinking", 82393, 2854)); err == nil || !isContextFull(err) || isStuckThinking(err) {
		t.Errorf("below-cap truncation should be a context ceiling, got: %v", err)
	}
	// completion_tokens omitted (0) but prompt+max_tokens overruns n_ctx (85248)
	// → still the ceiling, detected from prompt size, not the missing completion.
	if err := run(sseTruncated("thinking", 80000, 0)); err == nil || !isContextFull(err) {
		t.Errorf("no-room truncation with unreported completion should be a ceiling, got: %v", err)
	}
	// Reasoning-only AT the cap → stuck in <think> (recover by a thinking-off retry).
	if err := run(sseTruncated("thinking", 1000, defaultMaxTokens)); err == nil || !isStuckThinking(err) || isContextFull(err) {
		t.Errorf("reasoning-only cap should be a stuck-thinking stall, got: %v", err)
	}
	// Message CONTENT at the cap → verbose/looping output, genuinely not recoverable.
	if err := run(sseTruncatedContent("verbose output", 1000, defaultMaxTokens)); err == nil || isStuckThinking(err) || isContextFull(err) {
		t.Errorf("content at the cap should be a non-recoverable cap, got: %v", err)
	}
}

// TestIsTransientStreamError pins the mid-response-drop detector that drives the
// retry: EOF / reset / network errors are transient (retry), while a deliberate
// cancel, a clean LLM error, and nil are not.
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

// TestThinkingOn pins the guard deciding whether a <think> stall is recoverable:
// thinking counts as ON unless chat_template_kwargs.enable_thinking is explicitly
// false (a thinking-off retry), so the retry can't loop.
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
	}
	for _, c := range cases {
		if got := thinkingOn(c.body); got != c.want {
			t.Errorf("%s: thinkingOn=%v, want %v", c.name, got, c.want)
		}
	}
}

// TestWithThinkingDisabled pins the retry conn copy: enable_thinking is forced
// false, sibling params and routing fields survive, and the original is untouched.
func TestWithThinkingDisabled(t *testing.T) {
	orig := &LLMConnection{Server: "s", Model: "m", Slot: 2, ExtraBody: map[string]any{
		"temperature":          0.7,
		"chat_template_kwargs": map[string]any{"enable_thinking": true, "keep": 1},
	}}
	off := orig.withThinkingDisabled()

	if off.Server != "s" || off.Model != "m" || off.Slot != 2 {
		t.Errorf("routing fields changed: %+v", off)
	}
	ctk := off.ExtraBody["chat_template_kwargs"].(map[string]any)
	if ctk["enable_thinking"] != false || ctk["keep"] != 1 {
		t.Errorf("kwargs: got %+v, want enable_thinking=false + keep=1", ctk)
	}
	if off.ExtraBody["temperature"] != 0.7 {
		t.Errorf("sibling params dropped: %+v", off.ExtraBody)
	}
	if orig.ExtraBody["chat_template_kwargs"].(map[string]any)["enable_thinking"] != true {
		t.Error("withThinkingDisabled mutated the original conn")
	}
}

// TestWithMaxTokens pins the prewarm conn copy: max_tokens is forced in a
// copied ExtraBody (overriding the role default, since llmStream copies
// ExtraBody into the request first), sibling params and routing fields
// survive, and the original conn keeps its own cap.
func TestWithMaxTokens(t *testing.T) {
	orig := &LLMConnection{Server: "s", Model: "m", Slot: 1, ExtraBody: map[string]any{
		"max_tokens":  8192,
		"temperature": 0.7,
	}}
	capped := orig.withMaxTokens(1)

	if capped.Server != "s" || capped.Model != "m" || capped.Slot != 1 {
		t.Errorf("routing fields changed: %+v", capped)
	}
	if capped.ExtraBody["max_tokens"] != 1 || capped.ExtraBody["temperature"] != 0.7 {
		t.Errorf("ExtraBody: got %+v, want max_tokens=1 + temperature kept", capped.ExtraBody)
	}
	if orig.ExtraBody["max_tokens"] != 8192 {
		t.Error("withMaxTokens mutated the original conn")
	}
}
