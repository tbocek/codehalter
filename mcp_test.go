package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/BurntSushi/toml"
)

// TestHTTPTransportRoundTrip covers the parts of the Streamable HTTP transport
// the spec actually pins: the request goes out as POST, the server's
// Mcp-Session-Id is captured and echoed on subsequent calls, an SSE response
// is parsed past leading progress notifications, and DELETE is issued on
// close. It uses a single httptest.Server that fans out by method/path so a
// real round-trip plus close runs end-to-end.
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

			// First POST: assign a session id and return a JSON envelope.
			if n == 1 {
				w.Header().Set("Mcp-Session-Id", "session-xyz")
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				resp := mcpResponse{JSONRPC: "2.0", ID: &env.ID, Result: json.RawMessage(`{"ok":1}`)}
				_ = json.NewEncoder(w).Encode(resp)
				return
			}

			// Second POST: SSE stream with a leading progress notification
			// (no id) followed by the actual response.
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

	// First POST sees no session id; the second must echo what the server
	// assigned on the first; the DELETE must carry the same id.
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

// TestStartMCPClientRejectsCommandAndURL pins the mutual-exclusion contract
// on MCPServerConfig: each entry is one transport, not a hybrid.
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

// TestStdioTransportSurfacesCrash pins the fix for the swallowed-MCP-error hang:
// a stdio child that crashes on startup (here: writes to stderr and exits, like
// lsmcp's "No such built-in module: node:sqlite" on Node < 22) must have its
// stderr CAPTURED, its exit DETECTED, and a send fail fast — not block forever
// waiting on a response from a dead process.
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
		t.Fatal("child exit not detected — done never closed")
	}
	if se := tr.stderr.String(); !strings.Contains(se, "boom") {
		t.Errorf("stderr was not captured (would be swallowed): %q", se)
	}
	if _, err := tr.send(context.Background(), mcpRequest{JSONRPC: "2.0", ID: 1, Method: "initialize"}); err == nil {
		t.Error("send to a dead server should error fast, not hang")
	}
}

// ---------------------------------------------------------------------------
// Importing the editor's MCP servers
// ---------------------------------------------------------------------------

// offerFixture is the two servers (one stdio, one HTTP) an editor sends on
// session/new, wired onto a session ready for offerMCPImport.
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

// liveServers parses mcp.toml the way reconcileMCP does, so a test asserts on
// what the reconciler would actually start rather than on the file text.
func liveServers(t *testing.T, cwd string) []MCPServerConfig {
	t.Helper()
	var f struct {
		Server []MCPServerConfig `toml:"server"`
	}
	if _, err := toml.DecodeFile(mcpConfigPath(cwd), &f); err != nil {
		t.Fatalf("mcp.toml does not parse: %v", err)
	}
	return f.Server
}

// TestOfferMCPImportWithoutElicitation pins the fallback path: a client that
// can't show a form still gets its servers recorded, commented out. That is
// what makes the offer one-shot — a second bootstrap must find the names in the
// file and stay silent, instead of appending duplicates every session.
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

// TestOfferMCPImportAdoptsPicked pins the accept path end to end: the picked
// server becomes a live [[server]] the reconciler will start, the unpicked one
// is recorded commented out, and the form itself offers both by name.
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
		t.Errorf("property type = %q, want array — this is a multi-select", prop.Type)
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

// TestMCPNameInFile pins the "have we already offered this?" check. It has to
// see commented-out entries (that's how a declined server is remembered) and
// must not match on a name that merely contains another.
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

// TestMCPTOMLEntryRoundTrips pins that what we write is what the reconciler
// reads back, for both transports. The commented form must also stay
// syntactically inert.
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
		t.Fatalf("got %d servers, want 2 — the commented entry must be inert:\n%s", len(f.Server), text)
	}
	if f.Server[0].Command != "node" || f.Server[0].Args[1] != "--flag" || f.Server[0].Env["K"] != "v" {
		t.Errorf("stdio entry = %+v", f.Server[0])
	}
	if f.Server[1].URL != "http://x/mcp" || f.Server[1].Headers["X-Api-Key"] != "s" {
		t.Errorf("http entry = %+v", f.Server[1])
	}
}

// TestMCPFlushCoalesces pins the scheduler contract: at most one flush runs at
// a time, and any number of requests arriving while one runs collapse into
// exactly ONE follow-up. Without the collapse a run of turns that each touch
// mcp.toml would stack reconciles, and every one of them rewrites the tools
// array the whole conversation is rendered behind.
func TestMCPFlushCoalesces(t *testing.T) {
	var m mcpState
	var runs atomic.Int32
	entered := make(chan struct{}, 8)
	release := make(chan struct{})
	run := func() {
		runs.Add(1)
		entered <- struct{}{}
		<-release
	}

	m.schedule(run)
	<-entered // the first flush is now in the middle of its work
	for i := 0; i < 5; i++ {
		m.schedule(run)
	}
	close(release)
	m.wait()

	if got := runs.Load(); got != 2 {
		t.Fatalf("ran %d flushes, want 2 (the in-flight one plus a single coalesced follow-up)", got)
	}
	// Idle again, so the next request starts its own run rather than being
	// swallowed by the finished one.
	m.schedule(func() { runs.Add(1) })
	m.wait()
	if got := runs.Load(); got != 3 {
		t.Fatalf("ran %d flushes, want 3 — schedule after the queue drained must start a fresh run", got)
	}
}

// TestMCPFlushWaitBlocks covers the other half: a turn starting while a flush
// is still bringing a child up has to wait it out, so tools never appear
// half-registered in the middle of a turn.
func TestMCPFlushWaitBlocks(t *testing.T) {
	var m mcpState
	started, release := make(chan struct{}), make(chan struct{})
	m.schedule(func() { close(started); <-release })
	<-started

	done := make(chan struct{})
	go func() { m.wait(); close(done) }()
	select {
	case <-done:
		t.Fatal("wait returned while a flush was still running")
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("wait did not return after the flush finished")
	}
	m.wait() // idle wait is a no-op, not a hang
}

// TestMCPTakePending: what a background flush parked (it runs between turns,
// where nothing it says belongs to a turn) is handed to the next checkMCP
// exactly once, so a notice is not repeated and a card is not offered twice.
func TestMCPTakePending(t *testing.T) {
	var m mcpState
	if notes, fixes := m.takePending(); notes != nil || fixes != nil {
		t.Fatalf("takePending on an idle state = %v / %v, want nil / nil", notes, fixes)
	}
	m.flushNotes = append(m.flushNotes, "gopls started")
	m.flushFixes = append(m.flushFixes, fixProblem{desc: "boom"})

	notes, fixes := m.takePending()
	if len(notes) != 1 || notes[0] != "gopls started" || len(fixes) != 1 || fixes[0].desc != "boom" {
		t.Fatalf("takePending = %v / %v, want the queued notice and card", notes, fixes)
	}
	if notes, fixes := m.takePending(); notes != nil || fixes != nil {
		t.Fatalf("takePending twice = %v / %v, want nil / nil", notes, fixes)
	}
}
