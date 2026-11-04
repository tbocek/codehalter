package main

import (
	"context"
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

	writeTree(t, s.Cwd, map[string]string{"real/f.txt": "z"})
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

// Numbers and flags arrive as JSON values or as quoted strings; anything else,
// bad JSON included, reads as absent.
func TestArgsTypedAccessors(t *testing.T) {
	for _, c := range []struct {
		raw   string
		num   int
		numOK bool
		flag  bool
	}{
		{`{"v":42}`, 42, true, false},
		{`{"v":"42"}`, 42, true, false},
		{`{"v":"abc"}`, 0, false, false},
		{`{}`, 0, false, false},
		{`not json`, 0, false, false},
		{`{"v":true}`, 0, false, true},
		{`{"v":"TRUE"}`, 0, false, true},
		{`{"v":false}`, 0, false, false},
		{`{"v":"no"}`, 0, false, false},
	} {
		a := parseArgs(c.raw)
		if n, ok := a.num("v"); n != c.num || ok != c.numOK {
			t.Errorf("%s: num = %d,%v, want %d,%v", c.raw, n, ok, c.num, c.numOK)
		}
		if got := a.flag("v"); got != c.flag {
			t.Errorf("%s: flag = %v, want %v", c.raw, got, c.flag)
		}
	}
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

// A card a tool opens names the tool the model called; a card no tool opened names none.
func TestToolCardsNameTheirTool(t *testing.T) {
	h := newTerminalHarness(t)
	h.agent.tools.add(Tool{Def: map[string]any{"type": "function", "function": map[string]any{"name": "lookup", "parameters": map[string]any{"type": "object"}}},
		Execute: func(ctx context.Context, a *agent, sid, args string) (string, bool) {
			a.StartToolCall(ctx, sid, "Looking it up", "read", nil)
			return "ok", false
		}})
	callTool(t, h.agent, h.sess.ID, "lookup", `{}`)
	h.agent.StartToolCall(context.Background(), h.sess.ID, "Clarification needed", "think", nil)
	got := map[any]any{}
	h.waitFor(func() bool {
		for _, u := range h.updatesOfKind("tool_call") {
			got[u["title"]] = u["name"]
		}
		return len(got) == 2
	})
	if got["Looking it up"] != "lookup" {
		t.Errorf("the tool's card names %v, want lookup", got["Looking it up"])
	}
	if name, ok := got["Clarification needed"]; !ok || name != nil {
		t.Errorf("a card no tool opened names %v, want none", name)
	}
}

// An argument the schema declares a list or an object but that came as a string holding one is
// decoded before the tool runs, at any depth the schema describes; a string that is not exactly
// the declared JSON, or where the schema allows text, stays a string; the history keeps the call
// as the model sent it, and arguments with nothing to fit reach the tool byte for byte.
func TestToolArgsFitTheirSchema(t *testing.T) {
	a, sess := newTestAgent(t)
	obj := func(props map[string]any) map[string]any {
		return map[string]any{"type": "object", "properties": props}
	}
	list := func(items map[string]any) map[string]any { return map[string]any{"type": "array", "items": items} }
	var got string
	withTools(a, Tool{Def: map[string]any{"type": "function", "function": map[string]any{
		"name": "plan_trip",
		"parameters": obj(map[string]any{
			"stops": list(obj(map[string]any{"name": map[string]any{"type": "string"}, "hours": list(map[string]any{"type": "integer"})})),
			"meta":  obj(map[string]any{"tags": list(map[string]any{"type": "string"})}),
			"note":  map[string]any{"type": "string"},
			"any":   map[string]any{},
			"maybe": map[string]any{"type": []any{"array", "string"}},
		}),
	}}, Execute: func(_ context.Context, _ *agent, _ string, args string) (string, bool) {
		got = args
		return "ok", false
	}})
	call := func(args string) string {
		t.Helper()
		var tc toolCall
		tc.ID, tc.Function.Name, tc.Function.Arguments = "c", "plan_trip", args
		if tu, _ := a.runToolCall(context.Background(), sess.ID, tc); tu.Input != args {
			t.Errorf("the history holds %s, not the call as sent %s", tu.Input, args)
		}
		return got
	}

	sent := `{"stops": "[{\"name\": \"a\", \"hours\": \"[1, 2]\"}]", "meta": "{\"tags\": [\"x\"]}", "note": "[1, 2]", "any": "[3]", "maybe": "[4]"}`
	if out := call(sent); out != `{"any":"[3]","maybe":"[4]","meta":{"tags":["x"]},"note":"[1, 2]","stops":[{"hours":[1,2],"name":"a"}]}` {
		t.Errorf("fitted to\n%s", out)
	}
	for _, args := range []string{
		`{"stops": "[{\"name\": \"a\"}, {\"name\":"}`, // cut off: the tool says what is wrong
		`{"stops": "{\"name\": \"a\"}"}`,              // an object where a list is declared
		`{"stops": "a", "note": "x"}`,
		`{"note":  "spaced",   "stops": []}`,
	} {
		if out := call(args); out != args {
			t.Errorf("%s reached the tool as %s", args, out)
		}
	}
}
