package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/BurntSushi/toml"
)

// Pins POST, session-id capture and echo, SSE past leading notifications, and DELETE on close.
func TestHTTPTransportRoundTrip(t *testing.T) {
	var (
		postCount      atomic.Int32
		deleteCount    atomic.Int32
		seenSessionIds []string
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			n := postCount.Add(1)
			seenSessionIds = append(seenSessionIds, r.Header.Get("Mcp-Session-Id"))
			if got := r.Header.Get("X-Test"); got != "yes" {
				t.Errorf("custom header missing: got %q", got)
			}

			var env mcpRequest
			if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
				t.Errorf("decode request body: %v", err)
				return
			}

			if n == 1 {
				w.Header().Set("Mcp-Session-Id", "session-xyz")
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				resp := mcpResponse{JSONRPC: "2.0", ID: &env.ID, Result: json.RawMessage(`{"ok":1}`)}
				_ = json.NewEncoder(w).Encode(resp)
				return
			}

			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			notif := `{"jsonrpc":"2.0","method":"notifications/progress","params":{"progress":1}}`
			resp := mcpResponse{JSONRPC: "2.0", ID: &env.ID, Result: json.RawMessage(`{"ok":2}`)}
			respBytes, _ := json.Marshal(resp)
			_, _ = io.WriteString(w, "event: message\ndata: "+notif+"\n\n")
			_, _ = io.WriteString(w, "data: "+string(respBytes)+"\n\n")

		case http.MethodDelete:
			deleteCount.Add(1)
			seenSessionIds = append(seenSessionIds, "DEL:"+r.Header.Get("Mcp-Session-Id"))
			w.WriteHeader(http.StatusNoContent)

		default:
			t.Errorf("unexpected method %s", r.Method)
		}
	}))
	defer srv.Close()

	tr := &httpTransport{
		name:    "test",
		url:     srv.URL,
		headers: map[string]string{"X-Test": "yes"},
		client:  &http.Client{},
	}

	ctx := context.Background()

	resp1, err := tr.send(ctx, mcpRequest{JSONRPC: "2.0", ID: 1, Method: "ping"})
	if err != nil {
		t.Fatalf("first send: %v", err)
	}
	if string(resp1.Result) != `{"ok":1}` {
		t.Fatalf("first result = %s", string(resp1.Result))
	}

	resp2, err := tr.send(ctx, mcpRequest{JSONRPC: "2.0", ID: 2, Method: "ping"})
	if err != nil {
		t.Fatalf("second send: %v", err)
	}
	if string(resp2.Result) != `{"ok":2}` {
		t.Fatalf("sse result = %s", string(resp2.Result))
	}

	tr.close()

	if postCount.Load() != 2 {
		t.Fatalf("posts = %d, want 2", postCount.Load())
	}
	if deleteCount.Load() != 1 {
		t.Fatalf("deletes = %d, want 1", deleteCount.Load())
	}

	if seenSessionIds[0] != "" {
		t.Fatalf("first session id = %q, want empty", seenSessionIds[0])
	}
	if seenSessionIds[1] != "session-xyz" {
		t.Fatalf("second session id = %q, want session-xyz", seenSessionIds[1])
	}
	if !strings.HasSuffix(seenSessionIds[2], ":session-xyz") {
		t.Fatalf("delete session id = %q", seenSessionIds[2])
	}
}

// No initialize, modern _meta and mirrored headers on every request, Mcp-Name and Mcp-Param-* on
// tools/call, and an MRTR "input_required" result surfacing as an error.
func TestStartMCPClientModernHTTP(t *testing.T) {
	var sawInitialize, sawSessionID atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("unexpected http method %s", r.Method)
			return
		}
		if r.Header.Get("Mcp-Session-Id") != "" {
			sawSessionID.Store(true)
		}
		body, _ := io.ReadAll(r.Body)
		var env mcpRequest
		if err := json.Unmarshal(body, &env); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if got := r.Header.Get("MCP-Protocol-Version"); got != "2026-07-28" {
			t.Errorf("%s: MCP-Protocol-Version = %q", env.Method, got)
		}
		if got := r.Header.Get("Mcp-Method"); got != env.Method {
			t.Errorf("Mcp-Method = %q, want %q", got, env.Method)
		}
		var params struct {
			Meta      map[string]any `json:"_meta"`
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		praw, _ := json.Marshal(env.Params)
		_ = json.Unmarshal(praw, &params)
		if params.Meta["io.modelcontextprotocol/protocolVersion"] != "2026-07-28" {
			t.Errorf("%s: missing modern _meta: %s", env.Method, body)
		}

		w.Header().Set("Content-Type", "application/json")
		reply := func(result string) {
			resp := mcpResponse{JSONRPC: "2.0", ID: &env.ID, Result: json.RawMessage(result)}
			_ = json.NewEncoder(w).Encode(resp)
		}
		switch env.Method {
		case "server/discover":
			reply(`{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{}}}`)
		case "initialize":
			sawInitialize.Store(true)
			reply(`{}`)
		case "tools/list":
			reply(`{"resultType":"complete","ttlMs":60000,"cacheScope":"private","tools":[` +
				`{"name":"execute_sql","description":"d","inputSchema":{"type":"object","properties":` +
				`{"region":{"type":"string","x-mcp-header":"Region"},"query":{"type":"string"}}}},` +
				`{"name":"needs_input","description":"d","inputSchema":{"type":"object"}}]}`)
		case "tools/call":
			if params.Name == "needs_input" {
				reply(`{"resultType":"input_required","inputRequests":[]}`)
				return
			}
			if got := r.Header.Get("Mcp-Name"); got != "execute_sql" {
				t.Errorf("Mcp-Name = %q", got)
			}
			if got := r.Header.Get("Mcp-Param-Region"); got != "us-west1" {
				t.Errorf("Mcp-Param-Region = %q", got)
			}
			reply(`{"resultType":"complete","content":[{"type":"text","text":"42 rows"}]}`)
		default:
			t.Errorf("unexpected rpc %q", env.Method)
			reply(`{}`)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	c, err := StartMCPClient(ctx, MCPServerConfig{Name: "m", URL: srv.URL}, t.TempDir())
	if err != nil {
		t.Fatalf("StartMCPClient: %v", err)
	}
	defer c.Close()
	if !c.modern {
		t.Fatal("modern server not detected as modern")
	}
	if sawInitialize.Load() {
		t.Error("legacy initialize handshake sent to a modern server")
	}

	tools, err := c.listTools(ctx)
	if err != nil {
		t.Fatalf("listTools: %v", err)
	}
	c.tools = c.vetTools(tools)
	if len(c.tools) != 2 || len(c.tools[0].headerParams) != 1 {
		t.Fatalf("vetted tools = %+v", c.tools)
	}

	out, isErr, err := c.callTool(ctx, "execute_sql", json.RawMessage(`{"region":"us-west1","query":"SELECT 1"}`))
	if err != nil || isErr {
		t.Fatalf("callTool: err=%v isErr=%v", err, isErr)
	}
	if out != "42 rows" {
		t.Fatalf("out = %q", out)
	}

	if _, _, err := c.callTool(ctx, "needs_input", nil); err == nil || !strings.Contains(err.Error(), "MRTR") {
		t.Errorf("input_required result must error, got err=%v", err)
	}
	if sawSessionID.Load() {
		t.Error("client echoed a session id in modern mode")
	}
}

// A 2025-06-18 server answers the probe with a bare 400; the client falls back to initialize.
func TestStartMCPClientLegacyHTTPFallback(t *testing.T) {
	var calls []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			calls = append(calls, "DELETE:"+r.Header.Get("Mcp-Session-Id"))
			w.WriteHeader(http.StatusNoContent)
			return
		}
		var env mcpRequest
		_ = json.NewDecoder(r.Body).Decode(&env)
		calls = append(calls, env.Method)
		switch env.Method {
		case "server/discover":
			// What a legacy server does with an unknown method and no session.
			http.Error(w, "Bad Request: no valid session ID provided", http.StatusBadRequest)
		case "initialize":
			w.Header().Set("Mcp-Session-Id", "s1")
			w.Header().Set("Content-Type", "application/json")
			resp := mcpResponse{JSONRPC: "2.0", ID: &env.ID, Result: json.RawMessage(`{"protocolVersion":"2025-06-18"}`)}
			_ = json.NewEncoder(w).Encode(resp)
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			if got := r.Header.Get("Mcp-Session-Id"); got != "s1" {
				t.Errorf("session id = %q, want s1", got)
			}
			if got := r.Header.Get("MCP-Protocol-Version"); got != "2025-06-18" {
				t.Errorf("MCP-Protocol-Version = %q, want the negotiated legacy version", got)
			}
			praw, _ := json.Marshal(env.Params)
			if strings.Contains(string(praw), "io.modelcontextprotocol") {
				t.Errorf("modern _meta leaked onto a legacy request: %s", praw)
			}
			w.Header().Set("Content-Type", "application/json")
			resp := mcpResponse{JSONRPC: "2.0", ID: &env.ID, Result: json.RawMessage(`{"tools":[{"name":"t1"}]}`)}
			_ = json.NewEncoder(w).Encode(resp)
		default:
			t.Errorf("unexpected rpc %q", env.Method)
		}
	}))
	defer srv.Close()

	c, err := StartMCPClient(context.Background(), MCPServerConfig{Name: "l", URL: srv.URL}, t.TempDir())
	if err != nil {
		t.Fatalf("StartMCPClient: %v", err)
	}
	if c.modern {
		t.Fatal("legacy server misdetected as modern")
	}
	if tools, err := c.listTools(context.Background()); err != nil || len(tools) != 1 {
		t.Fatalf("listTools = %v, %v", tools, err)
	}
	c.Close()

	want := []string{"server/discover", "initialize", "notifications/initialized", "tools/list", "DELETE:s1"}
	if len(calls) != len(want) {
		t.Fatalf("calls = %v, want %v", calls, want)
	}
	for i := range want {
		if calls[i] != want[i] {
			t.Fatalf("calls[%d] = %q, want %q (all: %v)", i, calls[i], want[i], calls)
		}
	}
}

// Legacy SDKs answer server/discover with -32601.
func TestStdioEraDetection(t *testing.T) {
	t.Run("modern", func(t *testing.T) {
		// The ids align only if no handshake was sent.
		script := `read l; printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"resultType":"complete","supportedVersions":["2026-07-28"]}}'
read l; printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"resultType":"complete","tools":[{"name":"t1"}]}}'
read l`
		c, err := StartMCPClient(context.Background(), MCPServerConfig{Name: "mod", Command: "sh", Args: []string{"-c", script}}, t.TempDir())
		if err != nil {
			t.Fatalf("StartMCPClient: %v", err)
		}
		defer c.Close()
		if !c.modern {
			t.Fatal("modern stdio server not detected as modern")
		}
		if tools, err := c.listTools(context.Background()); err != nil || len(tools) != 1 {
			t.Fatalf("listTools = %v, %v", tools, err)
		}
	})
	t.Run("legacy", func(t *testing.T) {
		// discover (id 1) gets -32601, initialize (id 2), initialized (no reply), tools/list (id 3).
		script := `read l; printf '%s\n' '{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}'
read l; printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"protocolVersion":"2025-06-18"}}'
read l
read l; printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"tools":[{"name":"t1"}]}}'
read l`
		c, err := StartMCPClient(context.Background(), MCPServerConfig{Name: "leg", Command: "sh", Args: []string{"-c", script}}, t.TempDir())
		if err != nil {
			t.Fatalf("StartMCPClient: %v", err)
		}
		defer c.Close()
		if c.modern {
			t.Fatal("legacy stdio server misdetected as modern")
		}
		if tools, err := c.listTools(context.Background()); err != nil || len(tools) != 1 {
			t.Fatalf("listTools = %v, %v", tools, err)
		}
	})
}

func TestCollectHeaderParams(t *testing.T) {
	valid := map[string]any{"type": "object", "properties": map[string]any{
		"region": map[string]any{"type": "string", "x-mcp-header": "Region"},
		"opts": map[string]any{"type": "object", "properties": map[string]any{
			"limit": map[string]any{"type": "integer", "x-mcp-header": "Limit"},
		}},
	}}
	hp, err := collectHeaderParams(valid)
	if err != nil {
		t.Fatalf("valid schema rejected: %v", err)
	}
	if len(hp) != 2 {
		t.Fatalf("params = %+v, want 2", hp)
	}

	for name, schema := range map[string]map[string]any{
		"inside items": {"type": "object", "properties": map[string]any{
			"xs": map[string]any{"type": "array", "items": map[string]any{"type": "string", "x-mcp-header": "X"}},
		}},
		"inside anyOf": {"type": "object", "anyOf": []any{
			map[string]any{"properties": map[string]any{"a": map[string]any{"type": "string", "x-mcp-header": "A"}}},
		}},
		"number type": {"type": "object", "properties": map[string]any{
			"n": map[string]any{"type": "number", "x-mcp-header": "N"},
		}},
		"invalid token": {"type": "object", "properties": map[string]any{
			"a": map[string]any{"type": "string", "x-mcp-header": "bad header"},
		}},
		"duplicate case-insensitive": {"type": "object", "properties": map[string]any{
			"a": map[string]any{"type": "string", "x-mcp-header": "Region"},
			"b": map[string]any{"type": "string", "x-mcp-header": "region"},
		}},
	} {
		if _, err := collectHeaderParams(schema); err == nil {
			t.Errorf("%s: schema accepted, want rejection", name)
		}
	}

	// A property literally named x-mcp-header is data, not an annotation.
	benign := map[string]any{"type": "object", "properties": map[string]any{
		"x-mcp-header": map[string]any{"type": "string"},
	}}
	if _, err := collectHeaderParams(benign); err != nil {
		t.Errorf("property named x-mcp-header rejected: %v", err)
	}
}

func TestEncodeMCPHeaderValue(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"us-west1", "us-west1"},
		{"Hello, 世界", "=?base64?SGVsbG8sIOS4lueVjA==?="},
		{" padded ", "=?base64?IHBhZGRlZCA=?="},
		{"line1\nline2", "=?base64?bGluZTEKbGluZTI=?="},
		{"=?base64?literal?=", "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="},
	} {
		if got := encodeMCPHeaderValue(tc.in); got != tc.want {
			t.Errorf("encodeMCPHeaderValue(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestStartMCPClientRejectsCommandAndURL(t *testing.T) {
	_, err := StartMCPClient(context.Background(), MCPServerConfig{
		Name:    "bad",
		Command: "echo",
		URL:     "http://localhost",
	}, t.TempDir())
	if err == nil {
		t.Fatal("expected error when both command and url set")
	}
	if !strings.Contains(err.Error(), "command and url") {
		t.Fatalf("error %q didn't mention command/url conflict", err)
	}
}

// A crashing stdio child's stderr must surface and a send must fail fast instead of blocking forever.
func TestStdioTransportSurfacesCrash(t *testing.T) {
	cfg := MCPServerConfig{Name: "crashy", Command: "sh", Args: []string{"-c", "echo 'boom node:sqlite' >&2; exit 1"}}
	tr, err := newStdioTransport(cfg, "")
	if err != nil {
		t.Fatalf("newStdioTransport: %v", err)
	}
	defer tr.close()

	select {
	case <-tr.done:
	case <-time.After(3 * time.Second):
		t.Fatal("child exit not detected: done never closed")
	}
	if se := tr.stderr.String(); !strings.Contains(se, "boom") {
		t.Errorf("stderr was not captured (would be swallowed): %q", se)
	}
	if _, err := tr.send(context.Background(), mcpRequest{JSONRPC: "2.0", ID: 1, Method: "initialize"}); err == nil {
		t.Error("send to a dead server should error fast, not hang")
	}
}

func offerFixture(t *testing.T, a *agent, s *Session) {
	t.Helper()
	s.mcpOffer = []acpMCPServer{
		{Name: "scad", Command: "node", Args: []string{"/srv/scad.js"}, Env: []acpNameValue{{Name: "SCAD_HOME", Value: "/opt/scad"}}},
		{Type: "http", Name: "gmail", URL: "https://mcp.example/gmail", Headers: []acpNameValue{{Name: "X-Api-Key", Value: "secret"}}},
	}
	if err := os.MkdirAll(filepath.Join(s.Cwd, sessionDir), 0o755); err != nil {
		t.Fatal(err)
	}
}

// liveServers parses mcp.toml the way reconcileMCP does, so tests assert on what would start.
func liveServers(t *testing.T, cwd string) []MCPServerConfig {
	t.Helper()
	cfgs, _, err := readMCPConfig(mcpConfigPath(cwd))
	if err != nil {
		t.Fatalf("mcp.toml does not parse: %v", err)
	}
	return cfgs
}

// A client without forms still gets its servers recorded (commented out), so the offer is one-shot.
func TestOfferMCPImportWithoutElicitation(t *testing.T) {
	a, s := newTestAgent(t)
	offerFixture(t, a, s)

	a.offerMCPImport(context.Background(), s.Cwd, s.ID)

	if live := liveServers(t, s.Cwd); len(live) != 0 {
		t.Errorf("nothing was picked, so no server should be live, got %+v", live)
	}
	raw, err := os.ReadFile(mcpConfigPath(s.Cwd))
	if err != nil {
		t.Fatalf("mcp.toml not written: %v", err)
	}
	for _, name := range []string{"scad", "gmail"} {
		if !mcpNameInFile(string(raw), name) {
			t.Errorf("%q not recorded in mcp.toml, so it would be offered again:\n%s", name, raw)
		}
	}

	a.offerMCPImport(context.Background(), s.Cwd, s.ID)
	again, err := os.ReadFile(mcpConfigPath(s.Cwd))
	if err != nil {
		t.Fatal(err)
	}
	if string(again) != string(raw) {
		t.Errorf("second offer rewrote the file:\n%s", again)
	}
}

func TestOfferMCPImportAdoptsPicked(t *testing.T) {
	a, s, br, peerW := elicitingAgent(t)
	offerFixture(t, a, s)

	done := make(chan struct{})
	go func() {
		defer close(done)
		a.offerMCPImport(context.Background(), s.Cwd, s.ID)
	}()

	line, err := br.ReadString('\n')
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var req struct {
		ID     *json.RawMessage `json:"id"`
		Method string           `json:"method"`
		Params struct {
			Mode            string `json:"mode"`
			RequestedSchema struct {
				Properties map[string]struct {
					Type  string `json:"type"`
					Items struct {
						AnyOf []struct {
							Const string `json:"const"`
							Title string `json:"title"`
						} `json:"anyOf"`
					} `json:"items"`
				} `json:"properties"`
			} `json:"requestedSchema"`
		} `json:"params"`
	}
	if err := json.Unmarshal([]byte(line), &req); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if req.Method != "elicitation/create" || req.Params.Mode != "form" {
		t.Fatalf("method=%q mode=%q, want an elicitation form", req.Method, req.Params.Mode)
	}
	prop, ok := req.Params.RequestedSchema.Properties[elicitMCPKey]
	if !ok {
		t.Fatalf("form has no %q property: %s", elicitMCPKey, line)
	}
	if prop.Type != "array" {
		t.Errorf("property type = %q, want array: this is a multi-select", prop.Type)
	}
	if len(prop.Items.AnyOf) != 2 || prop.Items.AnyOf[0].Const != "scad" || prop.Items.AnyOf[1].Const != "gmail" {
		t.Errorf("options = %+v, want both offered servers", prop.Items.AnyOf)
	}
	if !strings.Contains(prop.Items.AnyOf[1].Title, "https://mcp.example/gmail") {
		t.Errorf("option title = %q, want it to show what the server is", prop.Items.AnyOf[1].Title)
	}

	reply := jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{
		"action":  "accept",
		"content": map[string]any{elicitMCPKey: []string{"scad"}},
	}}
	b, _ := json.Marshal(reply)
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("offerMCPImport did not return")
	}

	live := liveServers(t, s.Cwd)
	if len(live) != 1 || live[0].Name != "scad" {
		t.Fatalf("live servers = %+v, want only the picked one", live)
	}
	if live[0].Command != "node" || len(live[0].Args) != 1 || live[0].Args[0] != "/srv/scad.js" {
		t.Errorf("stdio fields lost in translation: %+v", live[0])
	}
	if live[0].Env["SCAD_HOME"] != "/opt/scad" {
		t.Errorf("env = %v, want the editor's env preserved", live[0].Env)
	}
	raw, err := os.ReadFile(mcpConfigPath(s.Cwd))
	if err != nil {
		t.Fatal(err)
	}
	if !mcpNameInFile(string(raw), "gmail") {
		t.Errorf("declined server not recorded, so it would be offered again:\n%s", raw)
	}
}

// Commented-out entries count; a name that merely contains another does not.
func TestMCPNameInFile(t *testing.T) {
	raw := "[[server]]\nname = \"live\"\ncommand = \"x\"\n\n# [[server]]\n# name = \"declined\"\n# url = \"http://x\"\n"
	for _, tc := range []struct {
		name string
		want bool
	}{
		{"live", true},
		{"declined", true},
		{"liv", false},
		{"live2", false},
		{"absent", false},
	} {
		if got := mcpNameInFile(raw, tc.name); got != tc.want {
			t.Errorf("mcpNameInFile(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// The commented form must stay syntactically inert.
func TestMCPTOMLEntryRoundTrips(t *testing.T) {
	stdio := acpMCPServer{Name: "a", Command: "node", Args: []string{"x.js", "--flag"}, Env: []acpNameValue{{Name: "K", Value: "v"}}}
	remote := acpMCPServer{Type: "http", Name: "b", URL: "http://x/mcp", Headers: []acpNameValue{{Name: "X-Api-Key", Value: "s"}}}

	var f struct {
		Server []MCPServerConfig `toml:"server"`
	}
	text := mcpTOMLEntry(stdio, false) + mcpTOMLEntry(remote, false) + mcpTOMLEntry(stdio, true)
	if _, err := toml.Decode(text, &f); err != nil {
		t.Fatalf("rendered toml does not parse: %v\n%s", err, text)
	}
	if len(f.Server) != 2 {
		t.Fatalf("got %d servers, want 2: the commented entry must be inert:\n%s", len(f.Server), text)
	}
	if f.Server[0].Command != "node" || f.Server[0].Args[1] != "--flag" || f.Server[0].Env["K"] != "v" {
		t.Errorf("stdio entry = %+v", f.Server[0])
	}
	if f.Server[1].URL != "http://x/mcp" || f.Server[1].Headers["X-Api-Key"] != "s" {
		t.Errorf("http entry = %+v", f.Server[1])
	}
}

// TestReconcileMCPStoppedStartIsRetried: a start cut short by the user's Stop
// is not reported as failed, and the next prompt tries it again although
// mcp.toml was not edited.
func TestReconcileMCPStoppedStartIsRetried(t *testing.T) {
	a, s := newTestAgent(t)
	dir := filepath.Join(s.Cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := "[[server]]\nname = \"gone\"\ncommand = \"/nonexistent/mcp-server\"\n"
	if err := os.WriteFile(filepath.Join(dir, "mcp.toml"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	stopped, cancel := context.WithCancel(context.Background())
	cancel()
	if changes := a.reconcileMCP(stopped, s.Cwd); len(changes) != 0 {
		t.Errorf("a stopped start reported %+v, want nothing", changes)
	}
	changes := a.reconcileMCP(context.Background(), s.Cwd)
	if len(changes) != 1 || changes[0].action != "failed" || changes[0].name != "gone" {
		t.Errorf("the next prompt did not retry the start: %+v", changes)
	}
	if again := a.reconcileMCP(context.Background(), s.Cwd); len(again) != 0 {
		t.Errorf("a real failure is retried only after an edit, got %+v", again)
	}
}

// A Stop during the restart of an edited server keeps the old one running, and
// the next prompt still applies the edit.
// fakeMCPServer is a modern HTTP server with one tool "t"; header holds the last X-Cfg sent.
func fakeMCPServer(t *testing.T) (srv *httptest.Server, header *atomic.Value) {
	t.Helper()
	header = &atomic.Value{}
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var env mcpRequest
		if err := json.NewDecoder(r.Body).Decode(&env); err != nil {
			return
		}
		header.Store(r.Header.Get("X-Cfg"))
		result := `{}`
		switch env.Method {
		case "server/discover":
			result = `{"resultType":"complete","supportedVersions":["2026-07-28"],"capabilities":{"tools":{}}}`
		case "tools/list":
			result = `{"resultType":"complete","tools":[{"name":"t","description":"d","inputSchema":{"type":"object"}}]}`
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mcpResponse{JSONRPC: "2.0", ID: &env.ID, Result: json.RawMessage(result)})
	}))
	t.Cleanup(srv.Close)
	return srv, header
}

// writeMCPConfig writes mcp.toml aged by age: reconcileMCP diffs only on a changed mtime.
func writeMCPConfig(t *testing.T, cwd, cfg string, age time.Duration) {
	t.Helper()
	path := mcpConfigPath(cwd)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	at := time.Now().Add(-age)
	if err := os.Chtimes(path, at, at); err != nil {
		t.Fatal(err)
	}
}

// A server dropped from the file is stopped and its tools leave the registry.
func TestReconcileMCPStopsARemovedServer(t *testing.T) {
	a, s := newTestAgent(t)
	withTools(a)
	srv, _ := fakeMCPServer(t)
	defer a.shutdownMCP()

	writeMCPConfig(t, s.Cwd, "[[server]]\nname = \"srv\"\nurl = \""+srv.URL+"\"\n", time.Hour)
	if changes := a.reconcileMCP(context.Background(), s.Cwd); len(changes) != 1 || changes[0].action != "started" {
		t.Fatalf("first reconcile = %+v", changes)
	}
	writeMCPConfig(t, s.Cwd, "# nothing configured\n", 0)
	changes := a.reconcileMCP(context.Background(), s.Cwd)
	if len(changes) != 1 || changes[0].action != "stopped" || changes[0].name != "srv" {
		t.Errorf("removing the entry reported %+v, want srv stopped", changes)
	}
	if slices.ContainsFunc(a.tools.defs(), func(d map[string]any) bool {
		return strings.HasPrefix(d["function"].(map[string]any)["name"].(string), "srv__")
	}) {
		t.Error("the stopped server's tools are still registered")
	}
	if len(a.mcp.clients) != 0 || len(a.mcp.applied) != 0 {
		t.Errorf("still tracked: clients %v, applied %v", a.mcp.clients, a.mcp.applied)
	}
}

// A broken file is reported once, not on every prompt, and again after the next edit.
func TestReconcileMCPReportsAParseErrorOncePerEdit(t *testing.T) {
	a, s := newTestAgent(t)
	writeMCPConfig(t, s.Cwd, "[[server]\nname = ", time.Hour)
	for i, want := range []int{1, 0} {
		changes := a.reconcileMCP(context.Background(), s.Cwd)
		if len(changes) != want || (want == 1 && changes[0].action != "parse_error") {
			t.Errorf("reconcile %d = %+v, want %d parse_error", i+1, changes, want)
		}
	}
	writeMCPConfig(t, s.Cwd, "[[server]\nname = \"still broken\"", 0)
	if changes := a.reconcileMCP(context.Background(), s.Cwd); len(changes) != 1 || changes[0].action != "parse_error" {
		t.Errorf("after an edit = %+v, want the parse_error again", changes)
	}
}

func TestReconcileMCPStoppedRestartIsRetried(t *testing.T) {
	a, s := newTestAgent(t)
	withTools(a)
	srv, header := fakeMCPServer(t)
	defer a.shutdownMCP()

	write := func(v string, age time.Duration) {
		writeMCPConfig(t, s.Cwd, "[[server]]\nname = \"srv\"\nurl = \""+srv.URL+"\"\nheaders = { X-Cfg = \""+v+"\" }\n", age)
	}
	write("1", time.Hour)
	if changes := a.reconcileMCP(context.Background(), s.Cwd); len(changes) != 1 || changes[0].action != "started" {
		t.Fatalf("first reconcile = %+v", changes)
	}
	write("2", 0)
	stopped, cancel := context.WithCancel(context.Background())
	cancel()
	a.reconcileMCP(stopped, s.Cwd)
	changes := a.reconcileMCP(context.Background(), s.Cwd)
	if len(changes) != 1 || changes[0].action != "restarted" {
		t.Errorf("the next prompt did not retry the restart: %+v", changes)
	}
	if got, _ := header.Load().(string); got != "2" {
		t.Errorf("the running server still has the old config: X-Cfg = %q", got)
	}
}
