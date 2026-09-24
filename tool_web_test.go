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

// TestWebReadIsRawTextOnly pins web_read's shape after the question mode went:
// the page's text, paged with offset/limit, and no `question` parameter. The
// model reads the page itself; a second reader was a round trip that bought
// nothing the model could not do.
func TestWebReadIsRawTextOnly(t *testing.T) {
	def := webReadDef()
	fn := def["function"].(map[string]any)
	props := fn["parameters"].(map[string]any)["properties"].(map[string]any)
	if _, has := props["question"]; has {
		t.Error("web_read still offers a question parameter")
	}
	for _, p := range []string{"url", "offset", "limit"} {
		if _, has := props[p]; !has {
			t.Errorf("web_read lost its %q parameter", p)
		}
	}
	desc, _ := fn["description"].(string)
	if !strings.Contains(desc, "offset") || !strings.Contains(desc, "cached") {
		t.Errorf("the description must say the body is cached and paged with offset/limit: %q", desc)
	}
}
