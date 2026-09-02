package main

import (
	"testing"
)

// TestDiagArgsFillsDeclaredShape covers the adapter that lets codehalter call a
// diagnostics tool it did not write: each bridge names the file argument
// differently and declares a different type for it, so the args are built from
// the server's own schema rather than from one hardcoded convention.
func TestDiagArgsFillsDeclaredShape(t *testing.T) {
	const abs, rel, root = "/w/proj/main.go", "main.go", "/w/proj"

	tests := []struct {
		name   string
		schema map[string]any
		want   map[string]any
	}{
		{
			// gopls: go_diagnostics takes an array of absolute paths.
			name: "array of files",
			schema: map[string]any{"properties": map[string]any{
				"files": map[string]any{"type": "array"},
			}},
			want: map[string]any{"files": []string{abs}},
		},
		{
			// lsmcp: a project-relative path plus the workspace root.
			name: "relative path with root",
			schema: map[string]any{"properties": map[string]any{
				"relativePath": map[string]any{"type": "string"},
				"root":         map[string]any{"type": "string"},
			}},
			want: map[string]any{"relativePath": rel, "root": root},
		},
		{
			name: "uri form",
			schema: map[string]any{"properties": map[string]any{
				"uri": map[string]any{"type": "string"},
			}},
			want: map[string]any{"uri": "file://" + abs},
		},
		{
			name: "plain path string",
			schema: map[string]any{"properties": map[string]any{
				"path": map[string]any{"type": "string"},
			}},
			want: map[string]any{"path": abs},
		},
		{
			name: "rootUri gets the uri form too",
			schema: map[string]any{"properties": map[string]any{
				"filePath": map[string]any{"type": "string"},
				"rootUri":  map[string]any{"type": "string"},
			}},
			want: map[string]any{"filePath": abs, "rootUri": "file://" + root},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := diagArgs(tc.schema, abs, rel, root)
			if !ok {
				t.Fatalf("diagArgs returned ok=false for %+v", tc.schema)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %d args %+v, want %d %+v", len(got), got, len(tc.want), tc.want)
			}
			for k, want := range tc.want {
				switch w := want.(type) {
				case []string:
					g, isSlice := got[k].([]string)
					if !isSlice || len(g) != len(w) || g[0] != w[0] {
						t.Errorf("arg %q = %#v, want %#v", k, got[k], w)
					}
				default:
					if got[k] != want {
						t.Errorf("arg %q = %#v, want %#v", k, got[k], want)
					}
				}
			}
		})
	}
}

// TestDiagArgsRefusesUnknownSchema pins the skip path: a diagnostics tool whose
// schema names no file argument must be declined, not called with a guess. A
// wrong guess spends a round-trip and logs an error on every write.
func TestDiagArgsRefusesUnknownSchema(t *testing.T) {
	for _, schema := range []map[string]any{
		{},
		{"properties": map[string]any{}},
		{"properties": map[string]any{"query": map[string]any{"type": "string"}}},
	} {
		if got, ok := diagArgs(schema, "/w/a.go", "a.go", "/w"); ok {
			t.Errorf("diagArgs(%+v) = %+v, ok=true; want a refusal", schema, got)
		}
	}
}

// TestDiagHasFindings pins the zero-cost-when-clean contract: every way a
// server can say "nothing wrong" must produce no output, because a clean file
// is the common case and must not spend context.
func TestDiagHasFindings(t *testing.T) {
	clean := []string{"", "   ", "[]", "{}", "null", "No diagnostics found.", "0 diagnostics", "no errors in main.go"}
	for _, s := range clean {
		if diagHasFindings(s) {
			t.Errorf("diagHasFindings(%q) = true, want false", s)
		}
	}
	dirty := []string{
		"main.go:12:5: undefined: parseEntry",
		"[{\"severity\":1,\"message\":\"expected ';'\"}]",
	}
	for _, s := range dirty {
		if !diagHasFindings(s) {
			t.Errorf("diagHasFindings(%q) = false, want true", s)
		}
	}
}

// TestFindDiagnosticsToolPrefersLanguageMatch covers server selection when a
// project has both a language-specific bridge and a generic one wired: the
// specific one wins for its own extension, and must not be handed a file it
// can't read.
func TestFindDiagnosticsToolPrefersLanguageMatch(t *testing.T) {
	a, _ := newTestAgent(t)
	a.mcp.clients = map[string]*MCPClient{
		"gopls": {name: "gopls", tools: []mcpTool{
			{Name: "go_definition"},
			{Name: "go_diagnostics", InputSchema: map[string]any{"properties": map[string]any{"files": map[string]any{"type": "array"}}}},
		}},
		"lsmcp": {name: "lsmcp", tools: []mcpTool{
			{Name: "lsp_get_diagnostics", InputSchema: map[string]any{"properties": map[string]any{"relativePath": map[string]any{"type": "string"}}}},
		}},
	}

	c, tool, ok := a.findDiagnosticsTool("/w/proj/main.go")
	if !ok || c.name != "gopls" || tool.Name != "go_diagnostics" {
		t.Errorf("main.go routed to %v/%v (ok=%v), want gopls/go_diagnostics", c, tool.Name, ok)
	}
	// A .ts file must NOT reach gopls: go_diagnostics is language-scoped, so it
	// falls through to the generic bridge.
	c, tool, ok = a.findDiagnosticsTool("/w/proj/app.ts")
	if !ok || c.name != "lsmcp" || tool.Name != "lsp_get_diagnostics" {
		t.Errorf("app.ts routed to %v/%v (ok=%v), want lsmcp/lsp_get_diagnostics", c, tool.Name, ok)
	}
}

// TestFindDiagnosticsToolNoServer pins that the common case — no MCP server, or
// one with no diagnostics tool — is a clean miss rather than a panic or a
// wrong-tool call.
func TestFindDiagnosticsToolNoServer(t *testing.T) {
	a, _ := newTestAgent(t)
	if _, _, ok := a.findDiagnosticsTool("/w/proj/main.go"); ok {
		t.Error("found a diagnostics tool with no MCP servers configured")
	}
	a.mcp.clients = map[string]*MCPClient{
		"gopls": {name: "gopls", tools: []mcpTool{{Name: "go_definition"}, {Name: "go_hover"}}},
	}
	if _, _, ok := a.findDiagnosticsTool("/w/proj/main.go"); ok {
		t.Error("found a diagnostics tool on a server that exposes none")
	}
	// Language-scoped tool, unrelated extension, no generic fallback → miss.
	a.mcp.clients["gopls"].tools = append(a.mcp.clients["gopls"].tools,
		mcpTool{Name: "go_diagnostics", InputSchema: map[string]any{"properties": map[string]any{"files": map[string]any{"type": "array"}}}})
	if _, _, ok := a.findDiagnosticsTool("/w/proj/app.py"); ok {
		t.Error("routed a .py file to gopls")
	}
}

// TestPostWriteDiagnosticsSkips covers the cheap exits taken before any MCP
// round-trip: the settings toggle, prose extensions, and an unknown session.
// These are what keep the feature free when it has nothing to contribute.
func TestPostWriteDiagnosticsSkips(t *testing.T) {
	a, s := newTestAgent(t)
	a.mcp.clients = map[string]*MCPClient{
		"gopls": {name: "gopls", tools: []mcpTool{
			{Name: "go_diagnostics", InputSchema: map[string]any{"properties": map[string]any{"files": map[string]any{"type": "array"}}}},
		}},
	}
	ctx := t.Context()

	if got := a.postWriteDiagnostics(ctx, s.ID, "/w/proj/README.md", "# doc"); got != "" {
		t.Errorf("markdown got diagnostics: %q", got)
	}
	if got := a.postWriteDiagnostics(ctx, "no-such-session", "/w/proj/main.go", "package main"); got != "" {
		t.Errorf("unknown session got diagnostics: %q", got)
	}
	off := false
	a.settings.Diagnostics = &off
	if got := a.postWriteDiagnostics(ctx, s.ID, "/w/proj/main.go", "package main"); got != "" {
		t.Errorf("diagnostics=false still ran: %q", got)
	}
}
