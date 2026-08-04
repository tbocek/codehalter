package main

import (
	"reflect"
	"testing"

	"github.com/BurntSushi/toml"
)

// samplerParams are the request fields that only steer generation. They never
// reach the server's chat template, so the two roles may differ in them freely.
// Anything else is assumed to change the rendered prompt.
var samplerParams = map[string]bool{
	"frequency_penalty": true, "max_tokens": true, "min_p": true,
	"n": true, "presence_penalty": true, "repeat_penalty": true,
	"seed": true, "stop": true, "temperature": true, "top_k": true, "top_p": true,
}

// TestSeedSettingsRolesDifferInSamplersOnly pins the rule that made a turn pay a
// full prefill twice: params_execute carried chat_template_kwargs =
// { enable_thinking = false } and params_thinking did not. That field is an
// argument to the server's chat template, not a sampler, so the plan → execute
// role switch re-rendered the entire prompt — 9088 of 14436 tokens
// re-evaluated, 15s, with the next call (same field) hot again. Reasoning is
// turned off for execute in the message text now (noThinkSwitch), which the
// cache doesn't see. Sampler differences stay allowed: they don't enter the KV
// cache key.
func TestSeedSettingsRolesDifferInSamplersOnly(t *testing.T) {
	var s Settings
	if _, err := toml.Decode(defaultSettingsTOML, &s); err != nil {
		t.Fatalf("decode res/settings.toml: %v", err)
	}
	if len(s.LLM) == 0 {
		t.Fatal("res/settings.toml declares no [[llm]] entry")
	}
	for i, c := range s.LLM {
		keys := map[string]bool{}
		for k := range c.ParamsThinking {
			keys[k] = true
		}
		for k := range c.ParamsExecute {
			keys[k] = true
		}
		for k := range keys {
			if samplerParams[k] {
				continue
			}
			// DeepEqual, not ==: a nested TOML table decodes to a map, and == on two
			// maps of the same type panics.
			tv, ev := c.ParamsThinking[k], c.ParamsExecute[k]
			if !reflect.DeepEqual(tv, ev) {
				t.Errorf("llm[%d] %q differs between roles (thinking=%v execute=%v): non-sampler params re-render the prompt and break the prefix cache on every phase switch", i, k, tv, ev)
			}
		}
	}
}
