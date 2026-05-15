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
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// ---------------------------------------------------------------------------
// LSP client
// ---------------------------------------------------------------------------

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

type lspPosition struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

type lspRange struct {
	Start lspPosition `json:"start"`
	End   lspPosition `json:"end"`
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
		Location struct {
			URI   string   `json:"uri"`
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
			File: uriToPath(it.Location.URI),
			Line: it.Location.Range.Start.Line + 1,
			Col:  it.Location.Range.Start.Character + 1,
		})
	}
	return out, nil
}

// Location is one file:line:col returned by references/definition/implementation.
type Location struct {
	File string
	Line int
	Col  int
}

// References returns every callsite/usage of the symbol at the given
// 1-indexed file position. line/col are 1-indexed for caller convenience and
// converted to LSP's 0-indexed positions internally.
func (g *Gopls) References(ctx context.Context, absPath string, line, col int) ([]Location, error) {
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
	out := make([]Location, 0, len(items))
	for _, it := range items {
		out = append(out, Location{
			File: uriToPath(it.URI),
			Line: it.Range.Start.Line + 1,
			Col:  it.Range.Start.Character + 1,
		})
	}
	return out, nil
}

// Definition returns the declaration site(s) for the symbol at the given
// 1-indexed position. Useful when the model has a USE site and wants to jump
// to where the symbol is defined (handles aliases, embedded methods, and
// interface methods that workspace/symbol may not surface cleanly).
func (g *Gopls) Definition(ctx context.Context, absPath string, line, col int) ([]Location, error) {
	return g.locationRequest(ctx, "textDocument/definition", absPath, line, col)
}

// Implementations returns concrete implementations of the interface method
// (or interface type) at the given position. Go-specific gold — the inverse
// of "find references" when you have an interface and want to enumerate its
// implementors.
func (g *Gopls) Implementations(ctx context.Context, absPath string, line, col int) ([]Location, error) {
	return g.locationRequest(ctx, "textDocument/implementation", absPath, line, col)
}

func (g *Gopls) locationRequest(ctx context.Context, method, absPath string, line, col int) ([]Location, error) {
	if err := g.didOpen(absPath); err != nil {
		return nil, fmt.Errorf("didOpen %s: %w", absPath, err)
	}
	raw, err := g.Send(ctx, method, map[string]any{
		"textDocument": map[string]any{"uri": pathToURI(absPath)},
		"position":     map[string]any{"line": line - 1, "character": col - 1},
	})
	if err != nil {
		return nil, err
	}
	// LSP returns either a single Location, an array, or null. Probe.
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var items []struct {
		URI   string   `json:"uri"`
		Range lspRange `json:"range"`
	}
	if raw[0] == '[' {
		if err := json.Unmarshal(raw, &items); err != nil {
			return nil, err
		}
	} else {
		var one struct {
			URI   string   `json:"uri"`
			Range lspRange `json:"range"`
		}
		if err := json.Unmarshal(raw, &one); err != nil {
			return nil, err
		}
		items = append(items, one)
	}
	out := make([]Location, 0, len(items))
	for _, it := range items {
		out = append(out, Location{
			File: uriToPath(it.URI),
			Line: it.Range.Start.Line + 1,
			Col:  it.Range.Start.Character + 1,
		})
	}
	return out, nil
}

// Hover returns the markdown signature + doc comment for the symbol at the
// given position. Empty string when gopls has nothing to say.
func (g *Gopls) Hover(ctx context.Context, absPath string, line, col int) (string, error) {
	if err := g.didOpen(absPath); err != nil {
		return "", fmt.Errorf("didOpen %s: %w", absPath, err)
	}
	raw, err := g.Send(ctx, "textDocument/hover", map[string]any{
		"textDocument": map[string]any{"uri": pathToURI(absPath)},
		"position":     map[string]any{"line": line - 1, "character": col - 1},
	})
	if err != nil {
		return "", err
	}
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	// Hover.contents is a MarkupContent {kind, value} for modern LSP.
	var resp struct {
		Contents struct {
			Kind  string `json:"kind"`
			Value string `json:"value"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", err
	}
	return resp.Contents.Value, nil
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

// ---------------------------------------------------------------------------
// Tool wrappers
// ---------------------------------------------------------------------------

// ensureGopls starts the shared gopls client on first use, rooted at cwd.
// Subsequent calls return the existing client (or the recorded start error).
func (a *agent) ensureGopls(ctx context.Context, cwd string) (*Gopls, error) {
	a.goplsOnce.Do(func() {
		g, err := StartGopls(ctx, cwd)
		if err != nil {
			a.goplsErr = err
			return
		}
		a.gopls = g
	})
	if a.goplsErr != nil {
		return nil, a.goplsErr
	}
	return a.gopls, nil
}

// goplsInlineMax caps how many lines we render in the tool-call panel content.
// The full list still flows to the model as the tool's return value.
const goplsInlineMax = 20

func renderLocs(cwd string, locs []Location) []string {
	lines := make([]string, 0, len(locs))
	for _, l := range locs {
		rel, err := filepath.Rel(cwd, l.File)
		if err != nil {
			rel = l.File
		}
		lines = append(lines, fmt.Sprintf("%s:%d:%d", rel, l.Line, l.Col))
	}
	return lines
}

func truncateBlock(lines []string) string {
	if len(lines) <= goplsInlineMax {
		return strings.Join(lines, "\n")
	}
	head := strings.Join(lines[:goplsInlineMax], "\n")
	return fmt.Sprintf("%s\n(... %d more)", head, len(lines)-goplsInlineMax)
}

// positionTool factories a tool that takes (file, line, col) and runs a
// gopls request. The run callback returns (modelText, panelText, titleSuffix, err):
//   - modelText goes back to the LLM as the tool's return value (full result).
//   - panelText is shown in the Zed tool-call panel (may be truncated).
//   - titleSuffix is appended to the tool title at completion, so the user
//     sees the result preview without expanding the disclosure (e.g.
//     "go_references router_stub.go:19:6 → 1 hit: cmd/build/.../main.go:22").
func positionTool(name, description string, run func(ctx context.Context, g *Gopls, cwd, absPath string, line, col int) (modelText, panelText, titleSuffix string, err error)) Tool {
	return Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        name,
			"description": description,
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"file", "line", "col"},
				"properties": map[string]any{
					"file": map[string]any{"type": "string", "description": "Path to the Go file (absolute or relative to project root)."},
					"line": map[string]any{"type": "string", "description": "1-indexed line number where the symbol appears."},
					"col":  map[string]any{"type": "string", "description": "1-indexed column where the symbol name starts."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
		args := parseArgs(rawArgs)
		sess := a.getSession(sid)
		if sess == nil {
			return "error: no session"
		}
		file := args["file"]
		if file == "" {
			return "error: file is empty"
		}
		line, err := strconv.Atoi(args["line"])
		if err != nil || line < 1 {
			return "error: invalid line (must be 1-indexed integer)"
		}
		col, err := strconv.Atoi(args["col"])
		if err != nil || col < 1 {
			return "error: invalid col (must be 1-indexed integer)"
		}
		absPath, err := a.resolvePath(sid, file)
		if err != nil {
			return "error: " + err.Error()
		}

		title := fmt.Sprintf("%s: %s:%d:%d", name, file, line, col)
		tcId := a.StartToolCall(ctx, sid, title, "search", nil)
		g, err := a.ensureGopls(ctx, sess.Cwd)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}
		modelText, panelText, titleSuffix, err := run(ctx, g, sess.Cwd, absPath, line, col)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}
		if panelText == "" {
			panelText = modelText
		}
		finalTitle := title
		if titleSuffix != "" {
			finalTitle = title + " → " + titleSuffix
		}
		a.CompleteToolCallTitled(ctx, sid, tcId, finalTitle, []ToolCallContent{TextContent(panelText)})
		return modelText
	}}
}

// previewLocs builds a short title suffix from a list of locations.
// Examples:
//
//	1 hit              → "router.go:27"
//	2 hits             → "router.go:27 (+1)"
//	many               → "router.go:27 (+9)"
func previewLocs(lines []string) string {
	if len(lines) == 0 {
		return "no hits"
	}
	if len(lines) == 1 {
		return lines[0]
	}
	return fmt.Sprintf("%s (+%d)", lines[0], len(lines)-1)
}

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": "go_symbols",
			"description": "Find Go symbols (functions, types, methods, vars, consts) by NAME across the workspace. " +
				"Returns lines like `NewRouter router.go:27:6` — name, then file:line:col (1-indexed). " +
				"Use this FIRST when the user mentions a Go symbol by name and you need to locate it. " +
				"Faster and more accurate than search_text for Go code: handles methods on types, embedded fields, and case-insensitive substring match. " +
				"After this, use `go_references` (find callers), `go_hover` (read signature + doc), or `go_definition` (jump from a use to its declaration).",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"query"},
				"properties": map[string]any{
					"query": map[string]any{"type": "string", "description": "Symbol name or fragment. Case-insensitive substring match."},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
		args := parseArgs(rawArgs)
		sess := a.getSession(sid)
		if sess == nil {
			return "error: no session"
		}
		query := args["query"]
		if query == "" {
			return "error: query is empty"
		}

		tcId := a.StartToolCall(ctx, sid, "go_symbols: "+query, "search", nil)
		g, err := a.ensureGopls(ctx, sess.Cwd)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}
		syms, err := g.WorkspaceSymbol(ctx, query)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}
		if len(syms) == 0 {
			a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("no symbols found")})
			return "no symbols found"
		}

		lines := make([]string, 0, len(syms))
		previews := make([]string, 0, len(syms))
		for _, s := range syms {
			rel, err := filepath.Rel(sess.Cwd, s.File)
			if err != nil {
				rel = s.File
			}
			lines = append(lines, fmt.Sprintf("%s %s:%d:%d", s.Name, rel, s.Line, s.Col))
			previews = append(previews, fmt.Sprintf("%s:%d", rel, s.Line))
		}
		title := "go_symbols: " + query + " → " + previewLocs(previews)
		a.CompleteToolCallTitled(ctx, sid, tcId, title, []ToolCallContent{TextContent(truncateBlock(lines))})
		return strings.Join(lines, "\n")
	}})

	RegisterTool(positionTool("go_references",
		"Find every callsite/usage of the Go symbol at the given file position. "+
			"Returns lines like `cmd/build/main.go:22:14` (file:line:col, 1-indexed), one per call site. "+
			"The declaration itself is excluded — only USES are returned. "+
			"Workflow: call `go_symbols` first to get the symbol's file:line:col, then pass those here. "+
			"Use this when the user asks 'what calls X', 'who uses X', or before renaming/changing a function's signature.",
		func(ctx context.Context, g *Gopls, cwd, absPath string, line, col int) (string, string, string, error) {
			locs, err := g.References(ctx, absPath, line, col)
			if err != nil {
				return "", "", "", err
			}
			if len(locs) == 0 {
				return "no references found", "", "no hits", nil
			}
			lines := renderLocs(cwd, locs)
			return strings.Join(lines, "\n"), truncateBlock(lines), previewLocs(lines), nil
		},
	))

	RegisterTool(positionTool("go_definition",
		"Jump from a USE of a Go symbol to its DECLARATION. "+
			"Pass the file:line:col of the call/reference; returns the file:line:col where the symbol is defined. "+
			"Use this when you're reading code and see a symbol you need to look up. "+
			"Handles cases workspace/symbol misses: aliased types, embedded methods, methods promoted from anonymous fields. "+
			"For finding a top-level symbol by NAME (not by position), prefer `go_symbols`.",
		func(ctx context.Context, g *Gopls, cwd, absPath string, line, col int) (string, string, string, error) {
			locs, err := g.Definition(ctx, absPath, line, col)
			if err != nil {
				return "", "", "", err
			}
			if len(locs) == 0 {
				return "no definition found", "", "no hits", nil
			}
			lines := renderLocs(cwd, locs)
			return strings.Join(lines, "\n"), truncateBlock(lines), previewLocs(lines), nil
		},
	))

	RegisterTool(positionTool("go_implementations",
		"Find concrete IMPLEMENTATIONS of the Go interface (or interface method) at the given position. "+
			"Returns file:line:col of each type that satisfies the interface. "+
			"Use this when the user has an interface and asks 'what types implement this' or 'who provides this method'. "+
			"The inverse of `go_definition` when you're at an interface method and want to see all the concrete bodies.",
		func(ctx context.Context, g *Gopls, cwd, absPath string, line, col int) (string, string, string, error) {
			locs, err := g.Implementations(ctx, absPath, line, col)
			if err != nil {
				return "", "", "", err
			}
			if len(locs) == 0 {
				return "no implementations found", "", "no hits", nil
			}
			lines := renderLocs(cwd, locs)
			return strings.Join(lines, "\n"), truncateBlock(lines), previewLocs(lines), nil
		},
	))

	RegisterTool(positionTool("go_hover",
		"Get the signature + doc comment for the Go symbol at the given position. "+
			"Returns markdown: the symbol's declaration line and any doc comment above it. "+
			"Use this when you need to know what a function does, what arguments it takes, or what fields a struct has — without reading the whole file. "+
			"Cheap and precise. Run `go_symbols` first to find the position if you only have a name.",
		func(ctx context.Context, g *Gopls, cwd, absPath string, line, col int) (string, string, string, error) {
			text, err := g.Hover(ctx, absPath, line, col)
			if err != nil {
				return "", "", "", err
			}
			text = strings.TrimSpace(text)
			if text == "" {
				return "no hover info", "", "no info", nil
			}
			// First non-blank line is usually the signature — perfect preview.
			preview := text
			if i := strings.IndexByte(text, '\n'); i >= 0 {
				preview = text[:i]
			}
			preview = strings.TrimPrefix(preview, "```go")
			preview = strings.TrimPrefix(preview, "```")
			preview = strings.TrimSpace(preview)
			if len(preview) > 80 {
				preview = preview[:77] + "..."
			}
			return text, text, preview, nil
		},
	))
}
