package main

import (
	"context"
	"strconv"
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
}

// A cached page is served without a fetch: a repeat gets the same first page, and
// any `limit`, or an offset, asks for a slice.
func TestWebReadFromCache(t *testing.T) {
	a, s := newTestAgent(t)
	body := "HEAD" + strings.Repeat("x", maxRawPageChars) + "TAIL"
	s.rememberWebBody("https://example.org/p", body)
	read := func(args string) string {
		t.Helper()
		out, failed := webReadExecute(context.Background(), a, s.ID, args)
		if failed {
			t.Fatalf("%s failed: %s", args, out)
		}
		return out
	}
	if got := read(`{"url":"https://example.org/p"}`); got != firstPage(body) {
		t.Errorf("a repeat without a range is not the first page: %d chars", len(got))
	}
	if got := read(`{"url":"https://example.org/p","limit":4}`); got != "HEAD" {
		t.Errorf("limit alone = %q, want the first 4 chars", got)
	}
	if got := read(`{"url":"https://example.org/p","offset":` + strconv.Itoa(len(body)-4) + `}`); got != "TAIL" {
		t.Errorf("offset to the end = %q, want TAIL", got)
	}
}
