package main

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// A range read never yields invalid UTF-8, wherever the cuts land.
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

// No `question` parameter: the model reads the page itself.
func TestWebReadIsRawTextOnly(t *testing.T) {
	var def map[string]any
	for _, tool := range webTools {
		if toolName(tool) == "web_read" {
			def = tool.Def
		}
	}
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
