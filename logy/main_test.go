package main

import (
	"strings"
	"testing"
)

// TestParseEntries pins the log framing: `=== time [tag] ===` headers open an
// entry, everything until the next header is its body.
func TestParseEntries(t *testing.T) {
	log := `=== 2026-07-03T20:16:03Z [llm[0] thinking REQUEST] ===
{"messages":[{"role":"user","content":"hi"}]}
=== 2026-07-03T20:16:11Z [llm[0] thinking RESPONSE] ===
tokens: prompt=10 completion=2 finish=stop cached=8 evaluated=2
content:
ok
=== 2026-07-03T20:16:12Z [PREWARM] ===
done in 8s err=<nil>`
	entries := parseEntries(log)
	if len(entries) != 3 {
		t.Fatalf("entries = %d, want 3", len(entries))
	}
	if entries[0].tag != "llm[0] thinking REQUEST" || !strings.Contains(entries[0].body, `"role":"user"`) {
		t.Errorf("entry 0 wrong: %+v", entries[0])
	}
	if entries[2].tag != "PREWARM" || entries[2].body != "done in 8s err=<nil>" {
		t.Errorf("entry 2 wrong: %+v", entries[2])
	}
}

// TestCanonAndPrefix pins the diff core: canonicalised messages compare
// equal across identical requests, a pure extension shares the full previous
// length, and an in-place change (an unpersisted nudge replaced by the
// assistant reply — the real-world divergence logy exists to catch) shares
// only the messages before it.
func TestCanonAndPrefix(t *testing.T) {
	prev := canonMessages([]any{
		map[string]any{"role": "user", "content": "sys"},
		map[string]any{"role": "user", "content": "nudge: call a tool"},
	})
	ext := canonMessages([]any{
		map[string]any{"role": "user", "content": "sys"},
		map[string]any{"role": "user", "content": "nudge: call a tool"},
		map[string]any{"role": "assistant", "content": "", "tool_calls": []any{
			map[string]any{"id": "c1", "function": map[string]any{"name": "respond", "arguments": `{"message":"done"}`}},
		}},
	})
	if p := commonPrefix(prev, ext); p != 2 {
		t.Errorf("pure extension prefix = %d, want 2", p)
	}
	// The nudge vanishes from the rebuilt history; the assistant reply takes
	// its slot — only message 0 is still shared.
	rebuilt := canonMessages([]any{
		map[string]any{"role": "user", "content": "sys"},
		map[string]any{"role": "assistant", "content": "", "tool_calls": []any{
			map[string]any{"id": "c1", "function": map[string]any{"name": "respond", "arguments": `{"message":"done"}`}},
		}},
	})
	if p := commonPrefix(prev, rebuilt); p != 1 {
		t.Errorf("in-place change prefix = %d, want 1", p)
	}
	// Tool calls must be part of the canonical form — two assistants with the
	// same (empty) content but different calls must not compare equal.
	other := canonMessages([]any{
		map[string]any{"role": "user", "content": "sys"},
		map[string]any{"role": "assistant", "content": "", "tool_calls": []any{
			map[string]any{"id": "c2", "function": map[string]any{"name": "read_file", "arguments": `{"path":"a"}`}},
		}},
	})
	if p := commonPrefix(rebuilt, other); p != 1 {
		t.Errorf("differing tool calls should not match, prefix = %d, want 1", p)
	}
}
