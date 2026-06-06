package main

import "testing"

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

// TestBackgroundSlotLabel pins the display-slot routing: the foreground turn
// reads llm[0], background work reads llm[1] even on a single [[llm]] entry
// with parallel >= 2 (same connection, distinct display slot), and a second
// entry routes background to llm[1] proper.
func TestBackgroundSlotLabel(t *testing.T) {
	// Single entry, parallel=2 → foreground llm[0], background llm[1] (same conn).
	a := &agent{settings: Settings{LLM: []LLMConnection{{Server: "u", Model: "m", Parallel: 2}}}}
	a.buildConnSems()
	if fg := a.settings.MainLLM("execute"); fg == nil || fg.Slot != 0 {
		t.Fatalf("MainLLM.Slot = %v, want 0", fg)
	}
	bg := a.connForBackgroundLLM()
	if bg == nil || bg.Slot != 1 || bg.Server != "u" || bg.Model != "m" {
		t.Fatalf("connForBackgroundLLM = %+v, want Slot 1 on u/m", bg)
	}

	// Single entry, parallel=1 → no second slot to label; background stays llm[0].
	a1 := &agent{settings: Settings{LLM: []LLMConnection{{Server: "u", Model: "m", Parallel: 1}}}}
	a1.buildConnSems()
	if bg := a1.connForBackgroundLLM(); bg == nil || bg.Slot != 0 {
		t.Fatalf("single-slot connForBackgroundLLM.Slot = %v, want 0", bg)
	}

	// Two entries → background routes to the second entry, llm[1].
	a2 := &agent{settings: Settings{LLM: []LLMConnection{
		{Server: "u0", Model: "m0", Parallel: 1},
		{Server: "u1", Model: "m1", Parallel: 1},
	}}}
	a2.buildConnSems()
	if bg := a2.connForBackgroundLLM(); bg == nil || bg.Slot != 1 || bg.Server != "u1" {
		t.Fatalf("two-entry connForBackgroundLLM = %+v, want Slot 1 on u1", bg)
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
	a.settings.LLM[0].Parallel = cap(first) + 3
	a.buildConnSems()
	if a.connSems[0] == first || cap(a.connSems[0]) != cap(first)+3 {
		t.Errorf("cap change should rebuild: got cap %d, want %d", cap(a.connSems[0]), cap(first)+3)
	}
}
