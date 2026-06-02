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
