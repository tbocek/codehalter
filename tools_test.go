package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// An in-tree symlink to an out-of-tree target is rejected; in-tree files and
// in-tree symlinks still resolve.
func TestResolvePathSymlinkEscape(t *testing.T) {
	a, s := newTestAgent(t)

	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(s.Cwd, "escape")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if _, err := a.resolvePath(s.ID, "escape/secret.txt"); err == nil {
		t.Error("resolvePath followed a symlink out of cwd (sandbox escape)")
	}

	if err := os.WriteFile(filepath.Join(s.Cwd, "ok.txt"), []byte("y"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := a.resolvePath(s.ID, "ok.txt"); err != nil {
		t.Errorf("in-tree file rejected: %v", err)
	}

	if err := os.MkdirAll(filepath.Join(s.Cwd, "real"), 0o755); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(s.Cwd, "real", "f.txt"), []byte("z"), 0o644)
	if err := os.Symlink(filepath.Join(s.Cwd, "real"), filepath.Join(s.Cwd, "alias")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if _, err := a.resolvePath(s.ID, "alias/f.txt"); err != nil {
		t.Errorf("in-tree symlink to an in-tree path was rejected: %v", err)
	}
}

// A string, even "", passes; every non-string JSON value is flagged.
func TestArgsWrongType(t *testing.T) {
	cases := []struct {
		raw  string
		want bool
	}{
		{`{"content":"hi"}`, false},
		{`{"content":""}`, false},
		{`{"content":123}`, true},
		{`{"content":null}`, true},
		{`{"content":{"a":1}}`, true},
		{`{"content":["x"]}`, true},
		{`{"content":true}`, true},
		{`{"path":"x"}`, false},
		{`not json`, false},
	}
	for _, c := range cases {
		if got := parseArgs(c.raw).wrongType("content"); got != c.want {
			t.Errorf("parseArgs(%q).wrongType(\"content\") = %v, want %v", c.raw, got, c.want)
		}
	}
}

// Schema-correct JSON numbers and bools decode, and so do the quoted forms.
func TestArgsTypedAccessors(t *testing.T) {
	t.Run("num from JSON number", func(t *testing.T) {
		a := parseArgs(`{"path":"x.go","line":42,"limit":10}`)
		if got := a.str("path"); got != "x.go" {
			t.Errorf("str(path) = %q, want x.go", got)
		}
		if got, ok := a.num("line"); !ok || got != 42 {
			t.Errorf("num(line) = %d,%v, want 42,true", got, ok)
		}
		if got, ok := a.num("limit"); !ok || got != 10 {
			t.Errorf("num(limit) = %d,%v, want 10,true", got, ok)
		}
	})
	t.Run("num from quoted digits", func(t *testing.T) {
		a := parseArgs(`{"line":"42"}`)
		if got, ok := a.num("line"); !ok || got != 42 {
			t.Errorf("num(line) = %d,%v, want 42,true", got, ok)
		}
	})
	t.Run("num absent or unparseable", func(t *testing.T) {
		a := parseArgs(`{"line":"abc"}`)
		if _, ok := a.num("line"); ok {
			t.Error("num(line) on non-numeric text: ok = true, want false")
		}
		if _, ok := a.num("missing"); ok {
			t.Error("num(missing): ok = true, want false")
		}
	})
	t.Run("flag from JSON bool and string", func(t *testing.T) {
		for _, raw := range []string{`{"regex":true}`, `{"regex":"true"}`, `{"regex":"TRUE"}`} {
			if !parseArgs(raw).flag("regex") {
				t.Errorf("flag(regex) on %s = false, want true", raw)
			}
		}
		for _, raw := range []string{`{"regex":false}`, `{"regex":"no"}`, `{}`} {
			if parseArgs(raw).flag("regex") {
				t.Errorf("flag(regex) on %s = true, want false", raw)
			}
		}
	})
	t.Run("str refuses to coerce", func(t *testing.T) {
		if got := parseArgs(`{"content":123}`).str("content"); got != "" {
			t.Errorf("str on a number = %q, want \"\" (the caller rejects via wrongType)", got)
		}
	})
	t.Run("has is type-blind", func(t *testing.T) {
		a := parseArgs(`{"limit":0}`)
		if !a.has("limit") {
			t.Error("has(limit) = false for a supplied zero, want true")
		}
		if a.has("offset") {
			t.Error("has(offset) = true for an absent key, want false")
		}
	})
	t.Run("one bad key does not drop the others", func(t *testing.T) {
		a := parseArgs(`{"path":"x.go","line":42}`)
		if a.str("path") != "x.go" {
			t.Errorf("str(path) = %q, want x.go", a.str("path"))
		}
	})
}

// Content-retrieval tools pass through whole; every other tool gets the clip.
func TestLiveToolOutput(t *testing.T) {
	big := strings.Repeat("x", truncateThreshold*3)

	for _, tool := range []string{"read_file", "continue_read", "web_search", "web_read"} {
		if got := liveToolOutput(tool, "{}", big); got != big {
			t.Errorf("liveToolOutput(%s) clipped a size-managing tool (len %d, want %d)", tool, len(got), len(big))
		}
	}
	if got := liveToolOutput("run_command", "{}", big); got == big || !strings.Contains(got, "chars omitted") {
		t.Errorf("liveToolOutput(run_command) should clip a non-exempt tool")
	}
	if got := truncateForLLM("run_command", "{}", big); got == big || !strings.Contains(got, "chars omitted") {
		t.Errorf("truncateForLLM must clip a long non-exempt output")
	}
	if got := truncateForLLM("run_command", "{}", "small"); got != "small" {
		t.Errorf("short content should pass through, got %q", got)
	}
}

// An oversized exempt output is clipped at a line boundary, never mid-line.
func TestLiveExemptCap(t *testing.T) {
	line := strings.Repeat("a", 80) + "\n"
	huge := strings.Repeat(line, liveExemptCap/len(line)+50)

	got := liveToolOutput("read_file", "{}", huge)
	if got == huge {
		t.Fatal("oversized read_file output should be capped, not passed whole")
	}
	if !strings.Contains(got, "chars omitted") || !strings.Contains(got, "capped") {
		t.Errorf("capped output must report the omission, got tail %q", got[max(0, len(got)-160):])
	}
	if len(got) > liveExemptCap+512 { // body ≤ cap, plus the short hint footer
		t.Errorf("capped output too large: %d (cap %d)", len(got), liveExemptCap)
	}
	body := got[:strings.Index(got, "\n\n[...")]
	if !strings.HasSuffix(body, "a") { // last kept char is real content, cut on a line boundary
		t.Errorf("clip not line-aware: body ends %q", body[max(0, len(body)-5):])
	}
}

func TestWebSearchRefineHint(t *testing.T) {
	hint := truncationHint("web_search", `{"query":"x"}`)
	if !strings.Contains(hint, "refine the query") {
		t.Errorf("web_search hint should suggest refining the query, got %q", hint)
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

// Relative paths join cwd, absolute ones must be inside it, and `..` escapes fail.
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

// defs is sorted by name, not registration order (registered out of order here).
func TestAllToolDefinitions(t *testing.T) {
	a := &agent{}
	withTools(a, Tool{Def: toolDef("read")}, Tool{Def: toolDef("write")}, Tool{Def: toolDef("other")})

	if got, want := toolNames(a.tools.defs()), []string{"other", "read", "write"}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v (all tools, sorted)", got, want)
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
				if got.str(k) != v {
					t.Errorf("key %q: got %q, want %q", k, got.str(k), v)
				}
			}
		})
	}
}
