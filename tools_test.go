package main

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestLiveVsHistoryTruncation pins the split that caused (then fixed) the
// history bloat that pushed requests past n_ctx: read_file / continue_read /
// search_text pass through whole LIVE (they manage their own size), but the
// SAME output is byte-clipped when re-rendered from history, so an old read
// can't sit at full size in every later request.
func TestLiveVsHistoryTruncation(t *testing.T) {
	big := strings.Repeat("x", truncateThreshold*3)

	for _, tool := range []string{"read_file", "continue_read", "search_text"} {
		if got := liveToolOutput("tu_1", tool, "{}", big); got != big {
			t.Errorf("liveToolOutput(%s) clipped a size-managing tool (len %d, want %d)", tool, len(got), len(big))
		}
	}
	if got := liveToolOutput("tu_1", "run_command", "{}", big); got == big || !strings.Contains(got, "chars omitted") {
		t.Errorf("liveToolOutput(run_command) should clip a non-exempt tool")
	}
	// History re-render clips everything, including the live-exempt tools.
	if got := truncateForLLM("tu_1", "read_file", "{}", big); got == big || !strings.Contains(got, "chars omitted") {
		t.Errorf("truncateForLLM must clip read_file in history (regression guard)")
	}
	if got := truncateForLLM("tu_1", "run_command", "{}", "small"); got != "small" {
		t.Errorf("short content should pass through, got %q", got)
	}
}

func toolDef(name string) map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": name, "description": "test tool",
			"parameters": map[string]any{"type": "object"},
		},
	}
}

func toolNames(defs []map[string]any) []string {
	names := make([]string, 0, len(defs))
	for _, d := range defs {
		fn, _ := d["function"].(map[string]any)
		n, _ := fn["name"].(string)
		names = append(names, n)
	}
	return names
}

// TestResolvePath covers the sandbox boundary: relative paths are joined to
// sess.Cwd; absolute paths must already live inside it; `..` that would escape
// the root is rejected.
func TestResolvePath(t *testing.T) {
	a, s := newTestAgent(t)
	cwd := s.Cwd

	cases := []struct {
		name    string
		in      string
		want    string // expected resolved value; "" means expect error
		wantErr bool
	}{
		{name: "relative file", in: "foo.go", want: filepath.Join(cwd, "foo.go")},
		{name: "relative nested", in: "a/b/c.go", want: filepath.Join(cwd, "a/b/c.go")},
		{name: "dot", in: ".", want: cwd},
		{name: "empty", in: "", want: cwd},
		{name: "absolute inside cwd", in: filepath.Join(cwd, "x.go"), want: filepath.Join(cwd, "x.go")},
		{name: "escape via dotdot", in: "../../../etc/passwd", wantErr: true},
		{name: "absolute outside cwd", in: "/etc/passwd", wantErr: true},
		{name: "trailing dotdot escape", in: "sub/../../escaped", wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := a.resolvePath(s.ID, tc.in)
			if tc.wantErr {
				if err == nil {
					t.Errorf("expected error for %q, got %q", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}

	if _, err := a.resolvePath("missing", "foo.go"); err == nil {
		t.Error("expected error for unknown session id, got nil")
	}
}

// TestToolFilter pins the exclude semantics AND the sort: an empty filter passes
// every registered tool, the exclude set drops the named tools, and the result
// is sorted by name (NOT registration order) so the rendered `tools` block is
// byte-stable turn-over-turn for the LLM's prefix cache. (Registered out of
// order on purpose below.)
func TestToolFilter(t *testing.T) {
	withFreshToolRegistry(t)
	RegisterTool(Tool{Def: toolDef("read")})
	RegisterTool(Tool{Def: toolDef("write")})
	RegisterTool(Tool{Def: toolDef("other")})

	cases := []struct {
		name   string
		filter toolFilter
		want   []string
	}{
		{"no filter", toolFilter{}, []string{"other", "read", "write"}},
		{"exclude read", toolFilter{exclude: map[string]bool{"read": true}}, []string{"other", "write"}},
		{"exclude two", toolFilter{exclude: map[string]bool{"read": true, "other": true}}, []string{"write"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := toolNames(llmToolDefinitionsFiltered(tc.filter))
			if !slices.Equal(got, tc.want) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestParseArgs(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want map[string]string
	}{
		{name: "valid flat", in: `{"key":"value","a":"b"}`, want: map[string]string{"key": "value", "a": "b"}},
		{name: "empty object", in: `{}`, want: map[string]string{}},
		{name: "empty string", in: ``, want: map[string]string{}},
		{name: "invalid JSON", in: `not json`, want: map[string]string{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := parseArgs(tc.in)
			if got == nil {
				t.Fatalf("parseArgs must never return nil")
			}
			if len(got) != len(tc.want) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
			for k, v := range tc.want {
				if got[k] != v {
					t.Errorf("key %q: got %q, want %q", k, got[k], v)
				}
			}
		})
	}
}
