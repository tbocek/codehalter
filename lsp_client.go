package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// A minimal LSP client, spoken for exactly one purpose: post-write diagnostics
// from a language server that has no MCP bridge.
//
// Why not go through an MCP bridge like lsmcp: the bridge is a second process
// in front of the real server, and every one of its costs lands on the session
// open. lsmcp is fetched by `npx` (a cold fetch the first time, a resolve every
// time), needs its own node runtime version, and has to be installed AND
// registered in mcp.toml before a single diagnostic appears — a setup the model
// was being asked to plan and perform, which measured at minutes. tsgo, the
// native TypeScript compiler, speaks LSP itself: `tsgo --lsp -stdio`. Talking
// to it directly removes the bridge, the npx cold start, and the mcp.toml
// round trip.
//
// Only the diagnostics slice of the protocol is implemented: initialize,
// didOpen/didClose, and the textDocument/diagnostic pull. Definitions,
// references and hover are deliberately absent — the model does not call
// navigation tools even when they are wired (see the note at the top of lsp.go),
// so they would be dead weight in the tools array and a live cost to the prompt
// prefix. Everything here is best-effort: any failure returns "" and the write
// that triggered it behaves exactly as it did before.

const (
	// lspStartTimeout bounds spawn + initialize, which the first write to a
	// TypeScript file waits on. Measured at 50ms for spawn + handshake + two
	// pulls against tsgo on a warm page cache, so this is two orders of
	// magnitude of headroom for a cold one, and still short enough that a server
	// which never handshakes costs one write rather than the session.
	lspStartTimeout = 10 * time.Second

	// lspRetryDelay throttles restarts after a failed start or a crashed child.
	// Without it, a server that cannot run (missing runtime, unreadable
	// tsconfig) would pay its full startup timeout on every single write.
	lspRetryDelay = 60 * time.Second
)

// lspServerFor returns the language server to ask about path, its argv, and the
// LSP languageId for the file. name == "" means "no server for this file", which
// is the answer for most files and always a silent no-op.
//
// The tsconfig.json gate is not a heuristic: without a project, tsgo type-checks
// the single file against default compiler options and reports errors the
// project's own build would never produce. A repo of plain .js with no tsconfig
// (the shape that made the old lsmcp setup card fire on projects with nothing to
// type-check) therefore gets no server at all.
func lspServerFor(cwd, path string) (name string, argv []string, langID string) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".ts", ".mts", ".cts":
		langID = "typescript"
	case ".tsx":
		langID = "typescriptreact"
	default:
		return "", nil, ""
	}
	if st, err := os.Stat(filepath.Join(cwd, "tsconfig.json")); err != nil || st.IsDir() {
		return "", nil, ""
	}
	// Prefer the tsgo the project pins in its devDeps (the version its tsconfig
	// was written against) over a global one. npx is deliberately not used, for
	// the same reason prettierBin refuses it: its cold start would tax every
	// edit.
	bin := filepath.Join(cwd, "node_modules", ".bin", "tsgo")
	if st, err := os.Stat(bin); err != nil || st.IsDir() {
		found, err := exec.LookPath("tsgo")
		if err != nil {
			return "", nil, ""
		}
		bin = found
	}
	return "tsgo", []string{bin, "--lsp", "-stdio"}, langID
}

// lspMessage is one JSON-RPC frame in either direction. ID is kept raw because
// a server's own requests carry ids of its own choosing — tsgo sends the string
// "ts1" — and the reply has to echo the id back in the exact shape it arrived.
type lspMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *lspError       `json:"error,omitempty"`
}

type lspError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *lspError) Error() string { return fmt.Sprintf("lsp error %d: %s", e.Code, e.Message) }

// lspClient is one language-server child and the JSON-RPC bookkeeping for it.
// The child is not stopped explicitly: it reads our stdin, so it exits when
// codehalter does. Same posture as the MCP children.
type lspClient struct {
	name string
	root string

	w io.WriteCloser

	mu      sync.Mutex
	nextID  int
	pending map[int]chan *lspMessage
	opened  map[string]bool
	dead    error
}

// newLSPClient wires a client to an already-running server's pipes and starts
// the read loop. Split from startLSPClient so a test can drive the protocol over
// an in-memory pipe instead of spawning a language server.
func newLSPClient(name, root string, w io.WriteCloser, r io.Reader) *lspClient {
	c := &lspClient{
		name:    name,
		root:    root,
		w:       w,
		pending: map[int]chan *lspMessage{},
		opened:  map[string]bool{},
	}
	go c.readLoop(r)
	return c
}

// startLSPClient spawns argv, completes the initialize handshake, and returns a
// client ready for diagnostics calls.
func startLSPClient(ctx context.Context, name, root string, argv []string) (*lspClient, error) {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = root
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting %s: %w", name, err)
	}
	// A server that refuses to run explains itself on stderr and then exits;
	// without this the only symptom would be a handshake timeout with no reason.
	go func() {
		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			slog.Debug("lsp: stderr", "server", name, "line", sc.Text())
		}
	}()

	c := newLSPClient(name, root, stdin, stdout)
	initCtx, cancel := context.WithTimeout(ctx, lspStartTimeout)
	defer cancel()
	if _, err := c.call(initCtx, "initialize", map[string]any{
		"processId": os.Getpid(),
		"rootUri":   fileURI(root),
		"workspaceFolders": []map[string]any{
			{"uri": fileURI(root), "name": filepath.Base(root)},
		},
		"capabilities": map[string]any{
			// configuration:true is what licenses the server to ask us for
			// settings; answering that request is mandatory, see answer().
			"workspace": map[string]any{"workspaceFolders": true, "configuration": true},
			"textDocument": map[string]any{
				"synchronization":    map[string]any{},
				"diagnostic":         map[string]any{},
				"publishDiagnostics": map[string]any{},
			},
		},
	}); err != nil {
		c.markDead(err)
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("%s initialize: %w", name, err)
	}
	if err := c.notify("initialized", map[string]any{}); err != nil {
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("%s initialized: %w", name, err)
	}
	slog.Debug("lsp: server ready", "server", name, "root", root)
	return c, nil
}

// alive reports whether the client is still usable. A dead one is replaced on
// the next request rather than repaired: the child is gone with its state.
func (c *lspClient) alive() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.dead == nil
}

// markDead records why the client stopped working and wakes every waiter, so a
// caller blocked on a request that will never be answered fails now instead of
// at its own timeout.
func (c *lspClient) markDead(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dead != nil {
		return
	}
	c.dead = err
	for id, ch := range c.pending {
		close(ch)
		delete(c.pending, id)
	}
	_ = c.w.Close()
}

// write frames one message and sends it. Serialised: two goroutines interleaving
// their bytes on the pipe would corrupt the stream (the read loop answers server
// requests while a caller may be sending its own).
func (c *lspClient) write(m map[string]any) error {
	body, err := json.Marshal(m)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dead != nil {
		return c.dead
	}
	if _, err := fmt.Fprintf(c.w, "Content-Length: %d\r\n\r\n%s", len(body), body); err != nil {
		c.dead = err
		return err
	}
	return nil
}

func (c *lspClient) notify(method string, params any) error {
	return c.write(map[string]any{"jsonrpc": "2.0", "method": method, "params": params})
}

// call sends a request and waits for its response, the context, or the child
// dying, whichever comes first.
func (c *lspClient) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	c.mu.Lock()
	if c.dead != nil {
		err := c.dead
		c.mu.Unlock()
		return nil, err
	}
	c.nextID++
	id := c.nextID
	ch := make(chan *lspMessage, 1)
	c.pending[id] = ch
	c.mu.Unlock()

	if err := c.write(map[string]any{"jsonrpc": "2.0", "id": id, "method": method, "params": params}); err != nil {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
		return nil, err
	}
	select {
	case <-ctx.Done():
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
		return nil, ctx.Err()
	case m, ok := <-ch:
		if !ok {
			return nil, fmt.Errorf("%s: server stopped", c.name)
		}
		if m.Error != nil {
			return nil, m.Error
		}
		return m.Result, nil
	}
}

// readLoop demultiplexes the server's output until the pipe closes.
func (c *lspClient) readLoop(r io.Reader) {
	br := bufio.NewReader(r)
	for {
		m, err := readLSPMessage(br)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				slog.Debug("lsp: read loop ended", "server", c.name, "err", err)
			}
			c.markDead(fmt.Errorf("%s: %w", c.name, err))
			return
		}
		switch {
		case m.Method != "" && len(m.ID) > 0:
			c.answer(m)
		case len(m.ID) > 0:
			var id int
			if err := json.Unmarshal(m.ID, &id); err != nil {
				slog.Debug("lsp: response with unusable id", "server", c.name, "id", string(m.ID))
				continue
			}
			c.mu.Lock()
			ch := c.pending[id]
			delete(c.pending, id)
			c.mu.Unlock()
			if ch != nil {
				ch <- m
			}
		default:
			// Notifications (window/logMessage, publishDiagnostics for files we
			// did not ask about). Nothing here consumes them: diagnostics are
			// pulled per file, so a push for some other file is not our answer.
			slog.Debug("lsp: notification", "server", c.name, "method", m.Method)
		}
	}
}

// answer replies to a server-initiated request. Every one of them must be
// answered, including the ones we have nothing to say to: tsgo sends
// workspace/configuration during initialize and then waits for it, and an
// unanswered request stalls every later request behind it (observed as a pull
// that never returns).
func (c *lspClient) answer(m *lspMessage) {
	var result any
	if m.Method == "workspace/configuration" {
		// One entry per requested section. Null means "no setting configured",
		// which is exactly true: codehalter carries no editor settings.
		var p struct {
			Items []json.RawMessage `json:"items"`
		}
		if err := json.Unmarshal(m.Params, &p); err != nil {
			slog.Debug("lsp: unparsable configuration request", "server", c.name, "err", err)
		}
		result = make([]any, len(p.Items))
	}
	if err := c.write(map[string]any{"jsonrpc": "2.0", "id": m.ID, "result": result}); err != nil {
		slog.Debug("lsp: replying to server request failed", "server", c.name, "method", m.Method, "err", err)
	}
}

// readLSPMessage reads one Content-Length framed JSON-RPC message.
func readLSPMessage(r *bufio.Reader) (*lspMessage, error) {
	length := -1
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if v, ok := strings.CutPrefix(line, "Content-Length:"); ok {
			n, err := strconv.Atoi(strings.TrimSpace(v))
			if err != nil {
				return nil, fmt.Errorf("bad Content-Length %q: %w", v, err)
			}
			length = n
		}
	}
	if length < 0 {
		return nil, errors.New("header block without Content-Length")
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	var m lspMessage
	if err := json.Unmarshal(buf, &m); err != nil {
		return nil, fmt.Errorf("undecodable message: %w", err)
	}
	return &m, nil
}

// fileURI renders a path as a file:// URI. url.URL is used rather than string
// concatenation so a path with spaces or non-ASCII survives the trip.
func fileURI(path string) string {
	return (&url.URL{Scheme: "file", Path: path}).String()
}

// lspDiagItem is the part of an LSP Diagnostic that reaches the model.
type lspDiagItem struct {
	Range struct {
		Start struct {
			Line      int `json:"line"`
			Character int `json:"character"`
		} `json:"start"`
	} `json:"range"`
	Severity int             `json:"severity"`
	Code     json.RawMessage `json:"code"`
	Message  string          `json:"message"`
}

// diagnose type-checks one file and returns the findings, already formatted.
// text is the content codehalter just wrote rather than a re-read of the path:
// over ACP the write may still be an unflushed editor buffer, and an open
// document's text is what the protocol is built to carry.
func (c *lspClient) diagnose(ctx context.Context, path, text, langID string) (string, error) {
	uri := fileURI(path)
	c.mu.Lock()
	wasOpen := c.opened[uri]
	c.mu.Unlock()
	// Close and reopen rather than didChange: the server advertises incremental
	// sync, and a full replacement is one notification with no range arithmetic
	// to get wrong. The document stays open afterwards so the server keeps the
	// project's inter-file state warm for the next write.
	if wasOpen {
		if err := c.notify("textDocument/didClose", map[string]any{
			"textDocument": map[string]any{"uri": uri},
		}); err != nil {
			return "", err
		}
	}
	if err := c.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{
			"uri": uri, "languageId": langID, "version": 1, "text": text,
		},
	}); err != nil {
		return "", err
	}
	c.mu.Lock()
	c.opened[uri] = true
	c.mu.Unlock()

	raw, err := c.call(ctx, "textDocument/diagnostic", map[string]any{
		"textDocument": map[string]any{"uri": uri},
	})
	if err != nil {
		return "", err
	}
	var res struct {
		Items []lspDiagItem `json:"items"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return "", fmt.Errorf("undecodable diagnostic report: %w", err)
	}
	rel, relErr := filepath.Rel(c.root, path)
	if relErr != nil {
		rel = path
	}
	return formatLSPDiagnostics(rel, res.Items), nil
}

// formatLSPDiagnostics renders the report the way a compiler would, which is the
// shape the model already reads fluently from build output.
//
// Only errors and warnings are kept. Hints and information (severity 3 and 4)
// are style nudges from the editor's point of view — unused-import greying,
// "prefer const" — and every one of them spends context the write itself is
// competing for.
func formatLSPDiagnostics(rel string, items []lspDiagItem) string {
	var b strings.Builder
	for _, it := range items {
		level := ""
		switch it.Severity {
		case 1:
			level = "error"
		case 2:
			level = "warning"
		default:
			continue
		}
		fmt.Fprintf(&b, "%s:%d:%d: %s: %s", rel,
			it.Range.Start.Line+1, it.Range.Start.Character+1, level, it.Message)
		if code := strings.Trim(string(it.Code), `"`); code != "" && code != "null" {
			fmt.Fprintf(&b, " [ts%s]", code)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

// lspState holds the language servers started for post-write diagnostics, one
// per (server, project root). Its own mutex, not the agent's: a start blocks on
// a handshake and must not hold a lock the whole agent needs.
type lspState struct {
	mu      sync.Mutex
	clients map[string]*lspClient
	lastTry map[string]time.Time
}

// clientFor returns a ready client for path's language, starting one if needed.
// nil means "no diagnostics for this file", which covers every uninteresting
// case: no server for the extension, no tsconfig, binary not installed, a start
// that failed recently.
func (a *agent) clientFor(ctx context.Context, cwd, path string) (*lspClient, string) {
	name, argv, langID := lspServerFor(cwd, path)
	if name == "" {
		return nil, ""
	}
	key := name + "\x00" + cwd

	a.lsp.mu.Lock()
	defer a.lsp.mu.Unlock()
	if c := a.lsp.clients[key]; c != nil {
		if c.alive() {
			return c, langID
		}
		delete(a.lsp.clients, key)
	}
	if last, ok := a.lsp.lastTry[key]; ok && time.Since(last) < lspRetryDelay {
		return nil, ""
	}
	if a.lsp.lastTry == nil {
		a.lsp.lastTry = map[string]time.Time{}
	}
	a.lsp.lastTry[key] = time.Now()

	c, err := startLSPClient(ctx, name, cwd, argv)
	if err != nil {
		// Debug, not Warn: a project whose devDeps aren't installed yet is the
		// ordinary case, and the write it rode along with succeeded.
		slog.Debug("lsp: start failed", "server", name, "root", cwd, "err", err)
		return nil, ""
	}
	if a.lsp.clients == nil {
		a.lsp.clients = map[string]*lspClient{}
	}
	a.lsp.clients[key] = c
	return c, langID
}
