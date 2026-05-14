package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
)

// Gopls drives a long-lived `gopls serve` child over LSP/JSON-RPC. One
// instance per workspace, shared across tool calls so the index isn't rebuilt
// on every query. Concurrent Send calls are serialised on writes but multiplex
// on responses via the pending map.
type Gopls struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser

	writeMu sync.Mutex
	nextID  atomic.Int64

	pendingMu sync.Mutex
	pending   map[int64]chan json.RawMessage

	openMu sync.Mutex
	opened map[string]bool // abs path → didOpen sent
}

type lspRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type lspNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type lspResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *lspError       `json:"error,omitempty"`
}

type lspError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// StartGopls spawns gopls, runs initialize/initialized, and returns a ready
// client. Caller owns Close().
func StartGopls(ctx context.Context, cwd string) (*Gopls, error) {
	path, err := exec.LookPath("gopls")
	if err != nil {
		return nil, fmt.Errorf("gopls not found in PATH; install with `go install golang.org/x/tools/gopls@latest`")
	}

	cmd := exec.Command(path, "serve")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("gopls stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("gopls stdout: %w", err)
	}
	// gopls prints chatty logs to stderr (memory stats, watcher events). Drop
	// them — they'd otherwise land in the Zed agent log alongside our own.
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting gopls: %w", err)
	}

	g := &Gopls{
		cmd:     cmd,
		stdin:   stdin,
		stdout:  stdout,
		pending: make(map[int64]chan json.RawMessage),
		opened:  make(map[string]bool),
	}
	go g.readLoop()

	if err := g.initialize(ctx, cwd); err != nil {
		g.Close()
		return nil, fmt.Errorf("gopls initialize: %w", err)
	}
	slog.Info("gopls ready", "pid", cmd.Process.Pid, "cwd", cwd)
	return g, nil
}

func (g *Gopls) initialize(ctx context.Context, cwd string) error {
	root := pathToURI(cwd)
	_, err := g.Send(ctx, "initialize", map[string]any{
		"processId":    os.Getpid(),
		"rootUri":      root,
		"capabilities": map[string]any{},
		"workspaceFolders": []map[string]any{
			{"uri": root, "name": filepath.Base(cwd)},
		},
	})
	if err != nil {
		return err
	}
	return g.Notify("initialized", map[string]any{})
}

// readLoop reads framed messages forever. Responses with an id route to the
// pending caller; notifications and server-originated requests are dropped
// (we never registered for them).
func (g *Gopls) readLoop() {
	br := bufio.NewReader(g.stdout)
	for {
		body, err := readFrame(br)
		if err != nil {
			slog.Debug("gopls read error", "error", err)
			return
		}
		var resp lspResponse
		if err := json.Unmarshal(body, &resp); err != nil {
			continue
		}
		if resp.ID == nil {
			// Notification or server-originated request — ignore.
			continue
		}
		g.pendingMu.Lock()
		ch, ok := g.pending[*resp.ID]
		g.pendingMu.Unlock()
		if !ok {
			continue
		}
		if resp.Error != nil {
			// Surface the error to the waiter as a raw JSON envelope; Send
			// re-decodes and turns it into a Go error.
			ch <- body
		} else {
			ch <- resp.Result
		}
	}
}

// Send issues an LSP request and waits for the matching response.
func (g *Gopls) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := g.nextID.Add(1)
	ch := make(chan json.RawMessage, 1)
	g.pendingMu.Lock()
	g.pending[id] = ch
	g.pendingMu.Unlock()
	defer func() {
		g.pendingMu.Lock()
		delete(g.pending, id)
		g.pendingMu.Unlock()
	}()

	if err := g.writeFrame(lspRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params}); err != nil {
		return nil, err
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case raw := <-ch:
		// On error responses the envelope itself is delivered; decode to
		// detect that case.
		var probe lspResponse
		if err := json.Unmarshal(raw, &probe); err == nil && probe.Error != nil {
			return nil, fmt.Errorf("lsp error %d: %s", probe.Error.Code, probe.Error.Message)
		}
		return raw, nil
	}
}

// Notify sends a one-way LSP notification (no id, no response).
func (g *Gopls) Notify(method string, params any) error {
	return g.writeFrame(lspNotification{JSONRPC: "2.0", Method: method, Params: params})
}

func (g *Gopls) writeFrame(msg any) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	g.writeMu.Lock()
	defer g.writeMu.Unlock()
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))
	if _, err := io.WriteString(g.stdin, header); err != nil {
		return err
	}
	_, err = g.stdin.Write(data)
	return err
}

// readFrame consumes one LSP message: a `Content-Length: N` header block
// followed by exactly N bytes of body.
func readFrame(br *bufio.Reader) ([]byte, error) {
	var length int
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if strings.HasPrefix(line, "Content-Length:") {
			fmt.Sscanf(line, "Content-Length: %d", &length)
		}
	}
	if length <= 0 {
		return nil, fmt.Errorf("missing Content-Length")
	}
	body := make([]byte, length)
	if _, err := io.ReadFull(br, body); err != nil {
		return nil, err
	}
	return body, nil
}

// didOpen tells gopls to track a file. Required before per-position requests
// (references, definition, hover) on files gopls hasn't loaded yet. Idempotent
// — we send didOpen once per path and remember it.
func (g *Gopls) didOpen(absPath string) error {
	g.openMu.Lock()
	if g.opened[absPath] {
		g.openMu.Unlock()
		return nil
	}
	g.opened[absPath] = true
	g.openMu.Unlock()

	body, err := os.ReadFile(absPath)
	if err != nil {
		return err
	}
	return g.Notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{
			"uri":        pathToURI(absPath),
			"languageId": "go",
			"version":    1,
			"text":       string(body),
		},
	})
}

// SymbolInfo is the subset of workspace/symbol result we surface to tools.
type SymbolInfo struct {
	Name string
	Kind string
	File string // absolute path
	Line int    // 1-indexed for display
	Col  int    // 1-indexed for display
}

// WorkspaceSymbol runs workspace/symbol with the given query. Returns up to
// 50 matches; gopls itself caps fairly aggressively so the limit rarely bites.
func (g *Gopls) WorkspaceSymbol(ctx context.Context, query string) ([]SymbolInfo, error) {
	raw, err := g.Send(ctx, "workspace/symbol", map[string]any{"query": query})
	if err != nil {
		return nil, err
	}
	var items []struct {
		Name     string `json:"name"`
		Kind     int    `json:"kind"`
		Location struct {
			URI   string `json:"uri"`
			Range lspRange `json:"range"`
		} `json:"location"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, err
	}
	out := make([]SymbolInfo, 0, len(items))
	for _, it := range items {
		out = append(out, SymbolInfo{
			Name: it.Name,
			Kind: symbolKindName(it.Kind),
			File: uriToPath(it.Location.URI),
			Line: it.Location.Range.Start.Line + 1,
			Col:  it.Location.Range.Start.Character + 1,
		})
	}
	return out, nil
}

// Reference is one hit from textDocument/references.
type Reference struct {
	File string
	Line int
	Col  int
}

// References returns every callsite/usage of the symbol at the given
// 1-indexed file position. line/col are 1-indexed for caller convenience and
// converted to LSP's 0-indexed positions internally.
func (g *Gopls) References(ctx context.Context, absPath string, line, col int) ([]Reference, error) {
	if err := g.didOpen(absPath); err != nil {
		return nil, fmt.Errorf("didOpen %s: %w", absPath, err)
	}
	raw, err := g.Send(ctx, "textDocument/references", map[string]any{
		"textDocument": map[string]any{"uri": pathToURI(absPath)},
		"position":     map[string]any{"line": line - 1, "character": col - 1},
		"context":      map[string]any{"includeDeclaration": false},
	})
	if err != nil {
		return nil, err
	}
	var items []struct {
		URI   string   `json:"uri"`
		Range lspRange `json:"range"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, err
	}
	out := make([]Reference, 0, len(items))
	for _, it := range items {
		out = append(out, Reference{
			File: uriToPath(it.URI),
			Line: it.Range.Start.Line + 1,
			Col:  it.Range.Start.Character + 1,
		})
	}
	return out, nil
}

type lspPosition struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

type lspRange struct {
	Start lspPosition `json:"start"`
	End   lspPosition `json:"end"`
}

// Close shuts gopls down cleanly: shutdown request → exit notification →
// stdin EOF → kill if it overstays its welcome.
func (g *Gopls) Close() {
	if g.stdin != nil {
		// Best-effort graceful shutdown. We don't wait on the response — if
		// gopls is wedged we'd rather kill -9 than hang Close().
		_ = g.writeFrame(lspRequest{JSONRPC: "2.0", ID: g.nextID.Add(1), Method: "shutdown"})
		_ = g.Notify("exit", nil)
		g.stdin.Close()
	}
	if g.cmd != nil && g.cmd.Process != nil {
		g.cmd.Process.Kill()
		g.cmd.Wait()
	}
	slog.Info("gopls closed")
}

// LSP SymbolKind enum (subset we render — others fall back to a number).
func symbolKindName(k int) string {
	switch k {
	case 5:
		return "class"
	case 6:
		return "method"
	case 7:
		return "property"
	case 8:
		return "field"
	case 11:
		return "interface"
	case 12:
		return "function"
	case 13:
		return "variable"
	case 14:
		return "constant"
	case 22:
		return "struct"
	case 23:
		return "event"
	default:
		return fmt.Sprintf("kind%d", k)
	}
}

func pathToURI(p string) string {
	// LSP URI spec: file://<absolute path with each segment percent-encoded>.
	abs, err := filepath.Abs(p)
	if err != nil {
		abs = p
	}
	u := &url.URL{Scheme: "file", Path: abs}
	return u.String()
}

func uriToPath(u string) string {
	parsed, err := url.Parse(u)
	if err != nil {
		return u
	}
	return parsed.Path
}
