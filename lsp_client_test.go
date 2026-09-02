package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeLSPServer speaks just enough of the protocol to exercise the client:
// initialize, the server-initiated workspace/configuration request, didOpen, and
// the textDocument/diagnostic pull.
//
// The one behaviour worth reproducing exactly is the ordering trap: like tsgo,
// it will not answer a pull until its own configuration request has been
// answered. A client that ignores server-initiated requests deadlocks here, the
// same way it deadlocks against the real server.
type fakeLSPServer struct {
	t *testing.T
	r *bufio.Reader
	w io.Writer

	// wmu serialises sends: the delayed pull reply below answers from its own
	// goroutine, so two writers can otherwise interleave their frames.
	wmu sync.Mutex

	mu     sync.Mutex
	opened map[string]string // uri -> text last seen in didOpen
	closed []string          // uris seen in didClose, in order

	configAnswered chan struct{}
}

// startFakeLSP wires a client to a fake server over two in-memory pipes and
// returns both. The client's read loop is already running.
func startFakeLSP(t *testing.T) (*lspClient, *fakeLSPServer) {
	t.Helper()
	clientReads, serverWrites := io.Pipe()
	serverReads, clientWrites := io.Pipe()
	srv := &fakeLSPServer{
		t:              t,
		r:              bufio.NewReader(serverReads),
		w:              serverWrites,
		opened:         map[string]string{},
		configAnswered: make(chan struct{}),
	}
	c := newLSPClient("fake", "/w/proj", clientWrites, clientReads)
	go srv.serve()
	t.Cleanup(func() {
		_ = clientWrites.Close()
		_ = serverWrites.Close()
	})
	return c, srv
}

func (s *fakeLSPServer) send(m map[string]any) {
	body, err := json.Marshal(m)
	if err != nil {
		s.t.Errorf("fake server: marshalling %v: %v", m, err)
		return
	}
	s.wmu.Lock()
	defer s.wmu.Unlock()
	fmt.Fprintf(s.w, "Content-Length: %d\r\n\r\n%s", len(body), body)
}

func (s *fakeLSPServer) serve() {
	for {
		m, err := readLSPMessage(s.r)
		if err != nil {
			return // pipe closed: the test is over
		}
		switch {
		case m.Method == "initialize":
			s.send(map[string]any{"jsonrpc": "2.0", "id": json.RawMessage(m.ID),
				"result": map[string]any{"capabilities": map[string]any{
					"diagnosticProvider": map[string]any{"interFileDependencies": true},
				}}})
			// String id on purpose: tsgo uses "ts1", and a client that assumes
			// numeric ids cannot answer it.
			s.send(map[string]any{"jsonrpc": "2.0", "id": "cfg1", "method": "workspace/configuration",
				"params": map[string]any{"items": []map[string]any{{"section": "typescript"}, {"section": "editor"}}}})
		case m.Method == "textDocument/didOpen":
			var p struct {
				TextDocument struct {
					URI  string `json:"uri"`
					Text string `json:"text"`
				} `json:"textDocument"`
			}
			if err := json.Unmarshal(m.Params, &p); err != nil {
				s.t.Errorf("fake server: didOpen params: %v", err)
				continue
			}
			s.mu.Lock()
			s.opened[p.TextDocument.URI] = p.TextDocument.Text
			s.mu.Unlock()
		case m.Method == "textDocument/didClose":
			var p struct {
				TextDocument struct {
					URI string `json:"uri"`
				} `json:"textDocument"`
			}
			if err := json.Unmarshal(m.Params, &p); err != nil {
				s.t.Errorf("fake server: didClose params: %v", err)
				continue
			}
			s.mu.Lock()
			s.closed = append(s.closed, p.TextDocument.URI)
			s.mu.Unlock()
		case m.Method == "textDocument/diagnostic":
			// Answer from a goroutine, because this reply waits on the client and
			// the client's reply arrives on this same pipe: blocking the read loop
			// here would deadlock the fake, not the client. A real server reads
			// continuously for exactly this reason.
			go func(id json.RawMessage) {
				select {
				case <-s.configAnswered:
				case <-time.After(5 * time.Second):
					s.t.Error("fake server: no reply to workspace/configuration — a real server would stall here")
					s.send(map[string]any{"jsonrpc": "2.0", "id": id,
						"error": map[string]any{"code": -32603, "message": "configuration never answered"}})
					return
				}
				s.send(map[string]any{"jsonrpc": "2.0", "id": id,
					"result": json.RawMessage(`{"kind":"full","items":[
						{"range":{"start":{"line":0,"character":6},"end":{"line":0,"character":7}},"severity":1,"code":2322,"source":"ts","message":"Type 'string' is not assignable to type 'number'."},
						{"range":{"start":{"line":4,"character":0},"end":{"line":4,"character":3}},"severity":3,"code":6133,"source":"ts","message":"'dead' is declared but never read."}
					]}`)})
			}(m.ID)
		case m.Method == "" && len(m.ID) > 0:
			// A response from the client — the only one it owes us is the
			// configuration reply.
			var got []any
			if err := json.Unmarshal(m.Result, &got); err != nil {
				s.t.Errorf("fake server: configuration reply is not an array: %s", m.Result)
			} else if len(got) != 2 {
				s.t.Errorf("fake server: configuration reply has %d entries, want one per requested section (2)", len(got))
			}
			close(s.configAnswered)
		}
	}
}

// TestLSPDiagnoseRoundTrip is the whole feature end to end against the fake
// server: handshake, the server request the client must answer, the document it
// opens, the pull, and the report the model ends up reading.
func TestLSPDiagnoseRoundTrip(t *testing.T) {
	c, srv := startFakeLSP(t)
	ctx := t.Context()

	if _, err := c.call(ctx, "initialize", map[string]any{"rootUri": fileURI("/w/proj")}); err != nil {
		t.Fatalf("initialize: %v", err)
	}

	const src = "const x: number = \"nope\";\n"
	out, err := c.diagnose(ctx, "/w/proj/src/app.ts", src, "typescript")
	if err != nil {
		t.Fatalf("diagnose: %v", err)
	}
	// Path relative to the root, 1-based line/column, TS code kept, and the
	// severity-3 hint dropped.
	want := "src/app.ts:1:7: error: Type 'string' is not assignable to type 'number'. [ts2322]\n"
	if out != want {
		t.Errorf("report:\n got %q\nwant %q", out, want)
	}

	srv.mu.Lock()
	text, ok := srv.opened[fileURI("/w/proj/src/app.ts")]
	srv.mu.Unlock()
	if !ok {
		t.Fatal("server never saw a didOpen for the written file")
	}
	// The content that was written, not a re-read of the path: over ACP the file
	// on disk may still be an unflushed editor buffer.
	if text != src {
		t.Errorf("didOpen text = %q, want the content just written %q", text, src)
	}

	// Second write to the same file: the document has to be closed before it is
	// reopened, or the server holds two versions of one uri.
	if _, err := c.diagnose(ctx, "/w/proj/src/app.ts", src, "typescript"); err != nil {
		t.Fatalf("second diagnose: %v", err)
	}
	srv.mu.Lock()
	closed := append([]string(nil), srv.closed...)
	srv.mu.Unlock()
	if len(closed) != 1 || closed[0] != fileURI("/w/proj/src/app.ts") {
		t.Errorf("didClose before reopen = %v, want exactly the one uri", closed)
	}
}

// TestLSPDeadServerUnblocksCallers pins that a child dying mid-request fails the
// caller instead of parking it until its own timeout, and that the client then
// reports itself unusable so the next write starts a fresh one.
func TestLSPDeadServerUnblocksCallers(t *testing.T) {
	clientReads, serverWrites := io.Pipe()
	serverReads, clientWrites := io.Pipe()
	c := newLSPClient("fake", "/w/proj", clientWrites, clientReads)
	go func() {
		// Consume the request, then die without answering.
		if _, err := readLSPMessage(bufio.NewReader(serverReads)); err != nil {
			t.Errorf("fake server: reading the request: %v", err)
		}
		_ = serverWrites.Close()
	}()

	done := make(chan error, 1)
	go func() {
		_, err := c.call(t.Context(), "initialize", map[string]any{})
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("call against a dead server returned no error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("call did not return after the server died")
	}
	if c.alive() {
		t.Error("client still reports itself alive after its server died")
	}
	_ = clientWrites.Close()
}

// TestLspServerForGating covers who gets a language server at all. The gates are
// the whole reason this costs nothing on projects it cannot help: a plain-JS repo
// with no tsconfig gets no server, which is exactly the shape that used to
// trigger a multi-minute lsmcp setup turn.
func TestLspServerForGating(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the fake tsgo is a shell script")
	}
	dir := t.TempDir()
	bin := filepath.Join(dir, "node_modules", ".bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	tsgo := filepath.Join(bin, "tsgo")
	if err := os.WriteFile(tsgo, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	// No tsconfig yet: a TypeScript file still gets nothing.
	if name, _, _ := lspServerFor(dir, filepath.Join(dir, "src/app.ts")); name != "" {
		t.Errorf("got server %q for a project with no tsconfig.json", name)
	}
	if err := os.WriteFile(filepath.Join(dir, "tsconfig.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		path     string
		want     string
		wantLang string
	}{
		{"src/app.ts", "tsgo", "typescript"},
		{"src/app.tsx", "tsgo", "typescriptreact"},
		{"src/app.mts", "tsgo", "typescript"},
		{"src/app.js", "", ""}, // no type checking to do
		{"README.md", "", ""},  // prose
		{"main.go", "", ""},    // gopls' business, over MCP
	}
	for _, tc := range tests {
		name, argv, lang := lspServerFor(dir, filepath.Join(dir, tc.path))
		if name != tc.want || lang != tc.wantLang {
			t.Errorf("%s → (%q, %q), want (%q, %q)", tc.path, name, lang, tc.want, tc.wantLang)
			continue
		}
		if name == "" {
			continue
		}
		// The project's own pinned binary wins over anything on PATH.
		if argv[0] != tsgo {
			t.Errorf("%s → %v, want the project-local %s", tc.path, argv, tsgo)
		}
		if !strings.Contains(strings.Join(argv, " "), "--lsp") {
			t.Errorf("%s → %v, want the server started in LSP mode", tc.path, argv)
		}
	}
}
