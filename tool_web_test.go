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

// TestWebReadQuestionPicksTheMode pins the one-tool contract: `question` is
// optional and is what chooses between an answered read and the raw text, and
// the schema still spells out that the reader sees only the page and that
// question (it is a separate LLM call with no view of the conversation).
func TestWebReadQuestionPicksTheMode(t *testing.T) {
	def := webReadDef()["function"].(map[string]any)
	params := def["parameters"].(map[string]any)
	if got := params["required"].([]string); len(got) != 1 || got[0] != "url" {
		t.Errorf("web_read required = %v, want url only: the question is what picks the mode", got)
	}
	q, ok := params["properties"].(map[string]any)["question"].(map[string]any)
	if !ok || !strings.Contains(q["description"].(string), "STANDALONE") {
		t.Errorf("web_read must still ask for a standalone question, got %v", q)
	}
	if !strings.Contains(def["description"].(string), "OMIT `question`") {
		t.Errorf("the description must say how to get raw text, got %q", def["description"])
	}
}
