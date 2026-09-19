package main

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// TestSliceWebBodyRunes pins that a model-supplied offset/limit range read never
// yields invalid UTF-8, regardless of where the cuts land.
func TestSliceWebBodyRunes(t *testing.T) {
	body := "aé€bc" // 1 + 2 + 3 + 1 + 1 = 8 bytes across mixed-width runes
	for off := 0; off <= len(body)+1; off++ {
		for lim := 0; lim <= len(body)+1; lim++ {
			if got := sliceWebBody(body, off, lim); !utf8.ValidString(got) {
				t.Fatalf("sliceWebBody(%q, %d, %d) = %q is not valid UTF-8", body, off, lim, got)
			}
		}
	}
}

// TestWebReadNeedsStandaloneQuestion pins web_read's contract: its reader is a
// separate LLM call that sees only the page and the question, so a call without
// one is refused before any browser starts, and only web_read (not the raw
// variant) requires it.
func TestWebReadNeedsStandaloneQuestion(t *testing.T) {
	a, s := newTestAgent(t)
	out, failed := makeWebRead(true)(t.Context(), a, s.ID, `{"url":"https://example.com"}`)
	if !failed || !strings.Contains(out, "question is required") {
		t.Errorf("web_read without a question = (%q, failed=%v), want a refusal", out, failed)
	}

	required := func(def map[string]any) []string {
		return def["function"].(map[string]any)["parameters"].(map[string]any)["required"].([]string)
	}
	if got := required(webReadDef("web_read", "d", true)); len(got) != 2 || got[1] != "question" {
		t.Errorf("web_read required = %v, want url and question", got)
	}
	if got := required(webReadDef("web_read_raw", "d", false)); len(got) != 1 {
		t.Errorf("web_read_raw required = %v, want url only", got)
	}
}
