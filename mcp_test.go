package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
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
