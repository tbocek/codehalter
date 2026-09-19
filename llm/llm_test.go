package llm

import (
	"errors"
	"fmt"
	"testing"
)

func TestIsContextFull(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"400 reject", &HTTPError{Status: 400}, true},
		{"context ceiling", fmt.Errorf("ceiling: %w", ErrContextCeiling), true},
		{"500 error", &HTTPError{Status: 500}, false},
		{"plain error", errors.New("boom"), false},
		{"nil", nil, false},
	}
	for _, c := range cases {
		if got := IsContextFull(c.err); got != c.want {
			t.Errorf("%s: IsContextFull=%v, want %v", c.name, got, c.want)
		}
	}
}

// TestThinkingOn pins the guard deciding whether a <think> stall is recoverable:
// thinking counts as ON unless the request already suppressed it, so the retry
// can't loop. Two ways it can be suppressed: the user's own
// chat_template_kwargs.enable_thinking=false, or codehalter's own retry
// continuing a prefilled closed <think></think> (continue_final_message).
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
		if got := ThinkingOn(c.body); got != c.want {
			t.Errorf("%s: ThinkingOn=%v, want %v", c.name, got, c.want)
		}
	}
}

// TestWithThinkingDisabled pins the retry conn copy: it arms the prefill and
// leaves ExtraBody alone, so the two calls still render the same way. Routing
// fields survive and the original is untouched.
//
// ExtraBody is what RenderKey fingerprints. If the retry wrote
// chat_template_kwargs (as it once did), the server would re-render the whole
// conversation and hand back cached=0; the append leaves every earlier token in
// place. Measured against ai.jos.li on the same 13,972-token prompt: kwargs
// cached=0, prefill cached=13,968 of 13,978, reasoning suppressed either way.
func TestWithThinkingDisabled(t *testing.T) {
	orig := &Conn{Server: "s", Model: "m", Slot: 2, ExtraBody: map[string]any{
		"temperature":          0.7,
		"chat_template_kwargs": map[string]any{"preserve_thinking": true},
	}}
	off := orig.WithThinkingDisabled()

	if off.Server != "s" || off.Model != "m" || off.Slot != 2 {
		t.Errorf("routing fields changed: %+v", off)
	}
	if !off.NoThinkPrefill {
		t.Error("Conn.WithThinkingDisabled did not arm the prefill")
	}
	if RenderKey(off.ExtraBody) != RenderKey(orig.ExtraBody) {
		t.Errorf("the retry changed the rendering: %s vs %s",
			RenderKey(off.ExtraBody), RenderKey(orig.ExtraBody))
	}
	if _, set := off.ExtraBody["chat_template_kwargs"].(map[string]any)["enable_thinking"]; set {
		t.Errorf("the retry still writes enable_thinking: %+v", off.ExtraBody)
	}
	if off.ExtraBody["temperature"] != 0.7 {
		t.Errorf("sibling params dropped: %+v", off.ExtraBody)
	}
	if orig.NoThinkPrefill {
		t.Error("Conn.WithThinkingDisabled mutated the original conn")
	}
}

// TestWithMaxTokens pins the prewarm conn copy: max_tokens is forced in a
// copied ExtraBody (overriding the role default, since llmStream copies
// ExtraBody into the request first), sibling params and routing fields
// survive, and the original conn keeps its own cap.
func TestWithMaxTokens(t *testing.T) {
	orig := &Conn{Server: "s", Model: "m", Slot: 1, ExtraBody: map[string]any{
		"max_tokens":  8192,
		"temperature": 0.7,
	}}
	capped := orig.WithMaxTokens(1)

	if capped.Server != "s" || capped.Model != "m" || capped.Slot != 1 {
		t.Errorf("routing fields changed: %+v", capped)
	}
	if capped.ExtraBody["max_tokens"] != 1 || capped.ExtraBody["temperature"] != 0.7 {
		t.Errorf("ExtraBody: got %+v, want max_tokens=1 + temperature kept", capped.ExtraBody)
	}
	if orig.ExtraBody["max_tokens"] != 8192 {
		t.Error("Conn.WithMaxTokens mutated the original conn")
	}
}

// TestParamsForInventsNoChatTemplateKwargs pins that codehalter never puts a
// chat-template argument on the wire that the user did not write. It is
// tempting to: turning reasoning off for the execute role is worth ~50 minutes
// of decode on an 11.6h session against Qwen3.8-27B.
//
// But it gives the two roles different renderings, and on a one-slot server
// they evict each other on every plan -> execute switch: 35 switches in that
// session carrying 2414262 prompt tokens, at 483 tok/s prefill, so 83.3 minutes
// of re-prefill against the 50 it saves. codehalter banks that decode win by
// appending a closed <think></think> instead (Conn.WithThinkingDisabled), which is
// a suffix and leaves the rendering alone. The kwargs field stays the user's
// call per connection (res/settings.toml costs it), so Conn.ParamsFor hands back
// exactly what was configured.
func TestParamsForInventsNoChatTemplateKwargs(t *testing.T) {
	c := Conn{
		ParamsThinking: map[string]any{"temperature": 1.0},
		ParamsExecute:  map[string]any{"temperature": 0.6},
	}
	for _, role := range []string{"thinking", "execute"} {
		if _, set := c.ParamsFor(role)["chat_template_kwargs"]; set {
			t.Errorf("%s: codehalter invented chat_template_kwargs: %v", role, c.ParamsFor(role))
		}
	}
	// The legacy single `params` set gets the same treatment.
	legacy := Conn{Params: map[string]any{"temperature": 0.7}}
	if _, set := legacy.ParamsFor("execute")["chat_template_kwargs"]; set {
		t.Errorf("legacy params: codehalter invented chat_template_kwargs: %v", legacy.ParamsFor("execute"))
	}

	// What the user DID write is passed through untouched, on the role they put
	// it on and only that role. The divergence is theirs to place.
	opted := Conn{
		ParamsThinking: map[string]any{"temperature": 1.0},
		ParamsExecute: map[string]any{
			"temperature":          0.6,
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
		},
	}
	ctk, _ := opted.ParamsFor("execute")["chat_template_kwargs"].(map[string]any)
	if ctk["enable_thinking"] != false {
		t.Errorf("the user's own kwargs did not survive: %v", opted.ParamsFor("execute"))
	}
	if _, set := opted.ParamsFor("thinking")["chat_template_kwargs"]; set {
		t.Error("params_execute kwargs leaked onto the thinking role")
	}
	if got := opted.ParamsFor("execute")["temperature"]; got != 0.6 {
		t.Errorf("temperature = %v, want 0.6", got)
	}
}
