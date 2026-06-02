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
	a := &agent{settings: Settings{LLM: []LLMConnection{{URL: "u", Model: "m", Parallel: 2}}}}
	a.buildSlotSems()
	if fg := a.settings.MainLLM("execute"); fg == nil || fg.Slot != 0 {
		t.Fatalf("MainLLM.Slot = %v, want 0", fg)
	}
	bg := a.pickBackgroundLLM()
	if bg == nil || bg.Slot != 1 || bg.URL != "u" || bg.Model != "m" {
		t.Fatalf("pickBackgroundLLM = %+v, want Slot 1 on u/m", bg)
	}

	// Single entry, parallel=1 → no second slot to label; background stays llm[0].
	a1 := &agent{settings: Settings{LLM: []LLMConnection{{URL: "u", Model: "m", Parallel: 1}}}}
	a1.buildSlotSems()
	if bg := a1.pickBackgroundLLM(); bg == nil || bg.Slot != 0 {
		t.Fatalf("single-slot pickBackgroundLLM.Slot = %v, want 0", bg)
	}

	// Two entries → background routes to the second entry, llm[1].
	a2 := &agent{settings: Settings{LLM: []LLMConnection{
		{URL: "u0", Model: "m0", Parallel: 1},
		{URL: "u1", Model: "m1", Parallel: 1},
	}}}
	a2.buildSlotSems()
	if bg := a2.pickBackgroundLLM(); bg == nil || bg.Slot != 1 || bg.URL != "u1" {
		t.Fatalf("two-entry pickBackgroundLLM = %+v, want Slot 1 on u1", bg)
	}
}

// TestFmtKB pins the upload-meter format: bytes → two-decimal KiB with a "kb"
// suffix, no MB rollover so the cumulative total reads consistently.
func TestFmtKB(t *testing.T) {
	cases := map[int64]string{
		0:       "0.00kb",
		512:     "0.50kb",
		12636:   "12.34kb",
		1048576: "1024.00kb",
	}
	for in, want := range cases {
		if got := fmtKB(in); got != want {
			t.Errorf("fmtKB(%d) = %q, want %q", in, got, want)
		}
	}
}
