package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/BurntSushi/toml"
)

// ---------------------------------------------------------------------------
// MCP settings
// ---------------------------------------------------------------------------

// MCPServerConfig is one [[server]] entry from .codehalter/mcp.toml. Each
// entry is exactly one of stdio (command/args/env) OR HTTP (url/headers) —
// the reconciler rejects entries that set both. There is no `enabled` field;
// commenting out an entry is the only way to disable it. Keeping the schema
// small means fewer states the reconciler can interpret.
type MCPServerConfig struct {
	Name    string            `toml:"name"`
	Command string            `toml:"command"`
	Args    []string          `toml:"args"`
	Env     map[string]string `toml:"env"`
	URL     string            `toml:"url"`
	Headers map[string]string `toml:"headers"`
}

type mcpSettingsFile struct {
	Server []MCPServerConfig `toml:"server"`
}

// loadMCPSettings reads .codehalter/mcp.toml from cwd. Missing file returns
// (nil, zero time, nil) — MCP is opt-in, so absence is silently fine. The
// returned mtime lets reconcileMCP skip the diff when the file is unchanged
// since the last pass, so a persistent start failure doesn't re-emit the
// same failed card on every prompt.
func loadMCPSettings(cwd string) ([]MCPServerConfig, time.Time, error) {
	path := filepath.Join(cwd, sessionDir, "mcp.toml")
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, time.Time{}, nil
		}
		return nil, time.Time{}, err
	}
	var f mcpSettingsFile
	if _, err := toml.DecodeFile(path, &f); err != nil {
		return nil, info.ModTime(), fmt.Errorf("loading %s: %w", path, err)
	}
	return f.Server, info.ModTime(), nil
}

// ---------------------------------------------------------------------------
// JSON-RPC envelopes (MCP uses 2.0 over its chosen transport)
// ---------------------------------------------------------------------------

type mcpRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type mcpNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type mcpResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *mcpError       `json:"error,omitempty"`
}

type mcpError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

// ---------------------------------------------------------------------------
// Transport abstraction
// ---------------------------------------------------------------------------

// mcpTransport hides whether the server is reached over stdio or Streamable
// HTTP from the rest of the client. send blocks until the response arrives
// (or the context is cancelled); notify is fire-and-forget; close shuts the
// transport down (and for HTTP, releases the session id).
type mcpTransport interface {
	send(ctx context.Context, req mcpRequest) (mcpResponse, error)
	notify(ctx context.Context, n mcpNotification) error
	close()
}

// ---------------------------------------------------------------------------
// MCP client
// ---------------------------------------------------------------------------

// MCPClient drives a long-lived MCP server. One instance per [[server]]
// entry, shared across tool calls so initialize cost is paid once. send is
// safe to call concurrently — the underlying transport serialises writes
// and demultiplexes responses by id.
type MCPClient struct {
	name      string
	transport mcpTransport
	nextID    atomic.Int64
}

// StartMCPClient brings up the server described by cfg and runs the MCP
// initialize handshake. The returned client is ready for tools/list and
// tools/call. cwd scopes a stdio child's working directory; HTTP transports
// ignore it.
func StartMCPClient(ctx context.Context, cfg MCPServerConfig, cwd string) (*MCPClient, error) {
	if cfg.Command != "" && cfg.URL != "" {
		return nil, fmt.Errorf("mcp config %q sets both command and url — pick one", cfg.Name)
	}
	var t mcpTransport
	switch {
	case cfg.URL != "":
		t = newHTTPTransport(cfg)
	case cfg.Command != "":
		st, err := newStdioTransport(cfg, cwd)
		if err != nil {
			return nil, err
		}
		t = st
	default:
		return nil, fmt.Errorf("mcp config %q has neither command nor url", cfg.Name)
	}

	c := &MCPClient{name: cfg.Name, transport: t}
	if err := c.initialize(ctx); err != nil {
		c.Close()
		return nil, fmt.Errorf("initialize %s: %w", cfg.Name, err)
	}
	slog.Info("mcp ready", "name", cfg.Name)
	return c, nil
}

// initialize completes the MCP handshake: send `initialize`, wait for the
// server's capabilities response, then send the `notifications/initialized`
// notification. After that the server is ready for tools/list and tools/call.
func (c *MCPClient) initialize(ctx context.Context) error {
	_, err := c.send(ctx, "initialize", map[string]any{
		"protocolVersion": "2025-06-18",
		"capabilities":    map[string]any{},
		"clientInfo": map[string]any{
			"name":    "codehalter",
			"version": "0.1.0",
		},
	})
	if err != nil {
		return err
	}
	return c.transport.notify(ctx, mcpNotification{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
		Params:  map[string]any{},
	})
}

func (c *MCPClient) send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := c.nextID.Add(1)
	resp, err := c.transport.send(ctx, mcpRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params})
	if err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, fmt.Errorf("mcp error %d: %s", resp.Error.Code, resp.Error.Message)
	}
	return resp.Result, nil
}

func (c *MCPClient) Close() {
	c.transport.close()
	slog.Info("mcp closed", "name", c.name)
}

// ---------------------------------------------------------------------------
// stdioTransport — line-delimited JSON-RPC over a child's stdin/stdout
// ---------------------------------------------------------------------------

type stdioTransport struct {
	name   string
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser

	writeMu sync.Mutex

	pendingMu sync.Mutex
	pending   map[int64]chan mcpResponse
}

func newStdioTransport(cfg MCPServerConfig, cwd string) (*stdioTransport, error) {
	bin, err := exec.LookPath(cfg.Command)
	if err != nil {
		return nil, fmt.Errorf("command %q not found in PATH", cfg.Command)
	}

	cmd := exec.Command(bin, cfg.Args...)
	cmd.Dir = cwd
	if len(cfg.Env) > 0 {
		cmd.Env = os.Environ()
		for k, v := range cfg.Env {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp stdout: %w", err)
	}
	// Drop stderr — MCP servers tend to log liveness chatter that would
	// otherwise drown the agent log.
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting %s: %w", cfg.Command, err)
	}
	t := &stdioTransport{
		name:    cfg.Name,
		cmd:     cmd,
		stdin:   stdin,
		stdout:  stdout,
		pending: make(map[int64]chan mcpResponse),
	}
	go t.readLoop()
	return t, nil
}

// readLoop reads one JSON-RPC message per line, forever. Responses with an id
// route to the pending caller; notifications and server-originated requests
// are dropped (we don't register for any).
func (t *stdioTransport) readLoop() {
	br := bufio.NewReader(t.stdout)
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			if !errors.Is(err, io.EOF) {
				slog.Debug("mcp read error", "name", t.name, "error", err)
			}
			return
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			continue
		}
		var resp mcpResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			slog.Debug("mcp decode error", "name", t.name, "error", err, "line", truncate(line, 200))
			continue
		}
		if resp.ID == nil {
			continue
		}
		t.pendingMu.Lock()
		ch, ok := t.pending[*resp.ID]
		t.pendingMu.Unlock()
		if !ok {
			continue
		}
		ch <- resp
	}
}

func (t *stdioTransport) send(ctx context.Context, req mcpRequest) (mcpResponse, error) {
	ch := make(chan mcpResponse, 1)
	t.pendingMu.Lock()
	t.pending[req.ID] = ch
	t.pendingMu.Unlock()
	defer func() {
		t.pendingMu.Lock()
		delete(t.pending, req.ID)
		t.pendingMu.Unlock()
	}()

	if err := t.writeMessage(req); err != nil {
		return mcpResponse{}, err
	}
	select {
	case <-ctx.Done():
		return mcpResponse{}, ctx.Err()
	case resp := <-ch:
		return resp, nil
	}
}

func (t *stdioTransport) notify(_ context.Context, n mcpNotification) error {
	return t.writeMessage(n)
}

func (t *stdioTransport) writeMessage(msg any) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	t.writeMu.Lock()
	defer t.writeMu.Unlock()
	if _, err := t.stdin.Write(data); err != nil {
		return err
	}
	_, err = t.stdin.Write([]byte{'\n'})
	return err
}

// close shuts the child down. We don't bother with a graceful shutdown
// request — closing stdin signals EOF to a well-behaved MCP server, and a
// hard Kill catches any that don't take the hint.
func (t *stdioTransport) close() {
	if t.stdin != nil {
		t.stdin.Close()
	}
	if t.cmd != nil && t.cmd.Process != nil {
		done := make(chan struct{})
		go func() {
			t.cmd.Wait()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(500 * time.Millisecond):
			t.cmd.Process.Kill()
			<-done
		}
	}
}

// ---------------------------------------------------------------------------
// httpTransport — MCP Streamable HTTP (spec 2025-06-18)
// ---------------------------------------------------------------------------

// httpTransport speaks MCP over HTTP per the 2025-06-18 spec. Each request
// is a POST whose body is the JSON-RPC envelope; the server replies with
// either a single JSON object or an SSE stream. The first server response
// carries the Mcp-Session-Id header (if the server is session-aware), and
// every subsequent request must echo it. close DELETEs the URL to release
// the server-side session before returning.
type httpTransport struct {
	name      string
	url       string
	headers   map[string]string
	client    *http.Client
	sessionMu sync.Mutex
	sessionId string
}

func newHTTPTransport(cfg MCPServerConfig) *httpTransport {
	return &httpTransport{
		name:    cfg.Name,
		url:     cfg.URL,
		headers: cfg.Headers,
		// Per-server timeout protects against a hung server holding up the
		// agent indefinitely; tools/call for long-running operations should
		// be wrapped in the request ctx for finer control.
		client: &http.Client{Timeout: 60 * time.Second},
	}
}

func (t *httpTransport) send(ctx context.Context, req mcpRequest) (mcpResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return mcpResponse{}, err
	}
	resp, err := t.do(ctx, body)
	if err != nil {
		return mcpResponse{}, err
	}
	return resp, nil
}

func (t *httpTransport) notify(ctx context.Context, n mcpNotification) error {
	body, err := json.Marshal(n)
	if err != nil {
		return err
	}
	// Per spec the server may return 202 Accepted with an empty body for
	// notifications. We still issue the round trip so the session id flows
	// through; an empty/202 response decodes to a zero mcpResponse, which
	// we just discard.
	_, err = t.do(ctx, body)
	return err
}

// do issues one HTTP request and decodes the response. Handles both JSON
// and SSE response bodies and captures any Mcp-Session-Id the server emits.
func (t *httpTransport) do(ctx context.Context, body []byte) (mcpResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, t.url, bytes.NewReader(body))
	if err != nil {
		return mcpResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	for k, v := range t.headers {
		httpReq.Header.Set(k, v)
	}
	t.sessionMu.Lock()
	if t.sessionId != "" {
		httpReq.Header.Set("Mcp-Session-Id", t.sessionId)
	}
	t.sessionMu.Unlock()

	httpResp, err := t.client.Do(httpReq)
	if err != nil {
		return mcpResponse{}, err
	}
	defer httpResp.Body.Close()

	if sid := httpResp.Header.Get("Mcp-Session-Id"); sid != "" {
		t.sessionMu.Lock()
		t.sessionId = sid
		t.sessionMu.Unlock()
	}

	if httpResp.StatusCode == http.StatusAccepted || httpResp.StatusCode == http.StatusNoContent {
		return mcpResponse{}, nil
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		buf, _ := io.ReadAll(io.LimitReader(httpResp.Body, 2048))
		return mcpResponse{}, fmt.Errorf("mcp http %d: %s", httpResp.StatusCode, string(buf))
	}

	contentType := httpResp.Header.Get("Content-Type")
	if strings.HasPrefix(contentType, "text/event-stream") {
		return t.readSSEResponse(httpResp.Body)
	}
	// Default: a single JSON-RPC envelope in the body.
	var resp mcpResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		if errors.Is(err, io.EOF) {
			return mcpResponse{}, nil
		}
		return mcpResponse{}, fmt.Errorf("decode response: %w", err)
	}
	return resp, nil
}

// readSSEResponse drains an event stream looking for the final JSON-RPC
// response. Per spec the server may emit progress notifications first; we
// keep reading "data:" frames and return the first envelope that carries an
// id (i.e. is a response, not a notification).
func (t *httpTransport) readSSEResponse(body io.Reader) (mcpResponse, error) {
	br := bufio.NewReader(body)
	var data strings.Builder
	for {
		line, err := br.ReadString('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return mcpResponse{}, fmt.Errorf("read sse: %w", err)
		}
		trimmed := strings.TrimRight(line, "\r\n")
		if trimmed == "" {
			// End of event — try to decode whatever we've accumulated.
			if data.Len() > 0 {
				var env mcpResponse
				if jerr := json.Unmarshal([]byte(data.String()), &env); jerr == nil {
					if env.ID != nil {
						return env, nil
					}
				}
				data.Reset()
			}
			if errors.Is(err, io.EOF) {
				return mcpResponse{}, fmt.Errorf("sse stream ended without response")
			}
			continue
		}
		if strings.HasPrefix(trimmed, "data:") {
			data.WriteString(strings.TrimPrefix(strings.TrimPrefix(trimmed, "data:"), " "))
		}
		// Ignore event:, id:, retry: lines.
	}
}

// close releases the server-side session by DELETEing the endpoint with the
// session id, then drops the local id. Best-effort: errors are logged but
// not surfaced — reconcile is already replacing this transport.
func (t *httpTransport) close() {
	t.sessionMu.Lock()
	sid := t.sessionId
	t.sessionId = ""
	t.sessionMu.Unlock()
	if sid == "" {
		return
	}
	req, err := http.NewRequest(http.MethodDelete, t.url, nil)
	if err != nil {
		return
	}
	req.Header.Set("Mcp-Session-Id", sid)
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	if resp, err := t.client.Do(req); err == nil {
		resp.Body.Close()
	} else {
		slog.Debug("mcp http delete failed", "name", t.name, "err", err)
	}
}

// ---------------------------------------------------------------------------
// Tool discovery & registration
// ---------------------------------------------------------------------------

// mcpTool is one entry from the tools/list response.
type mcpTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type mcpToolsListResult struct {
	Tools      []mcpTool `json:"tools"`
	NextCursor string    `json:"nextCursor,omitempty"`
}

// listTools enumerates every tool the server exposes. Some servers paginate
// via `nextCursor`; we follow until empty.
func (c *MCPClient) listTools(ctx context.Context) ([]mcpTool, error) {
	var all []mcpTool
	var cursor string
	for {
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		raw, err := c.send(ctx, "tools/list", params)
		if err != nil {
			return nil, err
		}
		var page mcpToolsListResult
		if err := json.Unmarshal(raw, &page); err != nil {
			return nil, fmt.Errorf("tools/list decode: %w", err)
		}
		all = append(all, page.Tools...)
		if page.NextCursor == "" {
			break
		}
		cursor = page.NextCursor
	}
	return all, nil
}

// mcpCallResultContent is one block from a tools/call result. We only care
// about the text variant — image/resource blocks are summarised down to a
// "[non-text content of type X]" placeholder.
type mcpCallResultContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type mcpCallResult struct {
	Content []mcpCallResultContent `json:"content"`
	IsError bool                   `json:"isError,omitempty"`
}

// callTool invokes a remote tool and renders the response into a single
// string suitable for surfacing back to the LLM.
func (c *MCPClient) callTool(ctx context.Context, name string, args json.RawMessage) (string, bool, error) {
	argsField := any(map[string]any{})
	if len(args) > 0 && string(args) != "null" {
		var parsed any
		if err := json.Unmarshal(args, &parsed); err == nil {
			argsField = parsed
		}
	}
	raw, err := c.send(ctx, "tools/call", map[string]any{
		"name":      name,
		"arguments": argsField,
	})
	if err != nil {
		return "", false, err
	}
	var result mcpCallResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", false, fmt.Errorf("tools/call decode: %w", err)
	}
	var b strings.Builder
	for _, block := range result.Content {
		switch block.Type {
		case "text":
			b.WriteString(block.Text)
		default:
			fmt.Fprintf(&b, "[non-text content of type %q]", block.Type)
		}
		b.WriteString("\n")
	}
	out := strings.TrimRight(b.String(), "\n")
	return out, result.IsError, nil
}

// registerMCPTools registers the given tools into codehalter's tool
// registry, prefixed with `<server>__` to avoid collisions across servers.
// The tool's description and JSON schema flow through verbatim — the MCP
// server is the source of truth for both. Caller must have already fetched
// the list via listTools so any startup failure is observed before
// registration (avoiding partial-state if tools/list errors mid-flight).
func registerMCPTools(c *MCPClient, tools []mcpTool) {
	for _, t := range tools {
		toolName := c.name + "__" + t.Name
		description := t.Description
		if description == "" {
			description = "(no description provided by MCP server " + c.name + ")"
		}
		params := t.InputSchema
		if params == nil {
			params = map[string]any{"type": "object"}
		}
		client := c
		remoteName := t.Name
		RegisterTool(Tool{
			Def: map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        toolName,
					"description": description,
					"parameters":  params,
				},
			},
			Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
				tcId := a.StartToolCall(ctx, sid, toolName, "search", nil)
				output, isErr, err := client.callTool(ctx, remoteName, json.RawMessage(rawArgs))
				if err != nil {
					a.FailToolCall(ctx, sid, tcId, err.Error())
					return "error: " + err.Error(), false
				}
				if isErr {
					a.FailToolCall(ctx, sid, tcId, output)
					return output, true
				}
				a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(output)})
				return output, false
			},
		})
	}
	slog.Info("mcp tools registered", "server", c.name, "count", len(tools))
}

// ---------------------------------------------------------------------------
// Lifecycle wired into the agent
// ---------------------------------------------------------------------------

// mcpConfigsEqual compares two server configs by all fields that matter for
// runtime behavior. A `false` here means the server must be restarted.
func mcpConfigsEqual(a, b MCPServerConfig) bool {
	if a.Name != b.Name || a.Command != b.Command || a.URL != b.URL {
		return false
	}
	if len(a.Args) != len(b.Args) {
		return false
	}
	for i := range a.Args {
		if a.Args[i] != b.Args[i] {
			return false
		}
	}
	if !stringMapsEqual(a.Env, b.Env) {
		return false
	}
	if !stringMapsEqual(a.Headers, b.Headers) {
		return false
	}
	return true
}

func stringMapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

// mcpChange describes one outcome of a reconciliation pass. The reconciler
// turns these into tool-call cards in the chat so the user sees additions,
// removals, restarts, and failures distinctly.
type mcpChange struct {
	action string // "started" | "stopped" | "restarted" | "failed" | "parse_error"
	name   string // server name; "" for parse_error
	err    error  // populated when action == "failed" or "parse_error"
}

// reconcileMCP brings the running MCP clients in line with .codehalter/mcp.toml.
// It is idempotent and safe to call on every Prompt(): if the file is
// unchanged at the semantic level, no UI is emitted. Failures don't block the
// caller — the user's turn proceeds with whatever set of tools is currently
// registered.
//
// Restart semantics are start-then-stop: a config change brings up the new
// process first and only kills the old one after the new client is verified
// (initialize + tools/list both succeeded). That way a typo in args doesn't
// take down a working server.
func (a *agent) reconcileMCP(ctx context.Context, cwd string) []mcpChange {
	a.mcpReconcileMu.Lock()
	defer a.mcpReconcileMu.Unlock()

	cfgs, mtime, err := loadMCPSettings(cwd)
	if err != nil {
		// Parse errors are reported once per mtime change. If the user's
		// editor saved a half-written file at t0, we surface it once; if
		// they don't touch it again, we don't keep nagging on every prompt.
		if !mtime.IsZero() && mtime.Equal(a.mcpAppliedMtime) {
			return nil
		}
		a.mcpAppliedMtime = mtime
		return []mcpChange{{action: "parse_error", err: err}}
	}
	// File unchanged since last reconcile — skip the diff entirely. This
	// also suppresses re-emitting a failed-start card every turn when the
	// user has a server configured incorrectly; they have to actually edit
	// the file (which bumps mtime) to trigger another attempt.
	if !mtime.IsZero() && mtime.Equal(a.mcpAppliedMtime) {
		return nil
	}
	a.mcpAppliedMtime = mtime

	// Last-write-wins on duplicate names. The mcp.toml schema doesn't define
	// behavior here, and the user probably meant the second entry to override.
	desired := make(map[string]MCPServerConfig, len(cfgs))
	for _, c := range cfgs {
		if c.Name == "" {
			continue
		}
		if c.Command == "" && c.URL == "" {
			continue
		}
		desired[c.Name] = c
	}

	applied := make(map[string]MCPServerConfig, len(a.mcpApplied))
	for _, c := range a.mcpApplied {
		applied[c.Name] = c
	}

	var changes []mcpChange

	// Pass 1: start brand-new + restart changed. Start-then-stop, so we
	// verify the new client works before tearing down the old one.
	for name, want := range desired {
		old, existed := applied[name]
		if existed && mcpConfigsEqual(old, want) {
			continue // no-op
		}

		newClient, err := StartMCPClient(ctx, want, cwd)
		if err != nil {
			changes = append(changes, mcpChange{action: "failed", name: name, err: err})
			continue
		}
		tools, err := newClient.listTools(ctx)
		if err != nil {
			newClient.Close()
			changes = append(changes, mcpChange{action: "failed", name: name, err: fmt.Errorf("tools/list: %w", err)})
			continue
		}

		// New client is ready. Atomically swap: unregister old tools, register
		// new tools, replace the client handle, close the old client.
		a.mu.Lock()
		if a.mcpClients == nil {
			a.mcpClients = make(map[string]*MCPClient)
		}
		oldClient := a.mcpClients[name]
		a.mcpClients[name] = newClient
		a.mu.Unlock()

		if oldClient != nil {
			UnregisterToolsByPrefix(name + "__")
		}
		registerMCPTools(newClient, tools)
		if oldClient != nil {
			oldClient.Close()
			changes = append(changes, mcpChange{action: "restarted", name: name})
		} else {
			changes = append(changes, mcpChange{action: "started", name: name})
		}
	}

	// Pass 2: stop entries that disappeared from the file (or were disabled).
	for name := range applied {
		if _, stillWanted := desired[name]; stillWanted {
			continue
		}
		a.mu.Lock()
		oldClient := a.mcpClients[name]
		delete(a.mcpClients, name)
		a.mu.Unlock()

		UnregisterToolsByPrefix(name + "__")
		if oldClient != nil {
			oldClient.Close()
		}
		changes = append(changes, mcpChange{action: "stopped", name: name})
	}

	// Snapshot the current applied set for the next diff. Only entries that
	// are actually running go in — a server that failed to start stays out,
	// so when the user fixes the file (bumping mtime) the next reconcile
	// sees it as "missing" and retries the start.
	a.mcpApplied = a.mcpApplied[:0]
	for _, c := range cfgs {
		if c.Name == "" {
			continue
		}
		if c.Command == "" && c.URL == "" {
			continue
		}
		a.mu.Lock()
		_, running := a.mcpClients[c.Name]
		a.mu.Unlock()
		if running {
			a.mcpApplied = append(a.mcpApplied, c)
		}
	}

	return changes
}

// renderMCPChanges turns reconciler output into a series of tool-call cards
// in the chat. Each change gets its own card so the user sees added/removed/
// restarted/failed distinctly. Failures use FailToolCall, which Zed renders
// with a red status icon — the closest visual to a "red box" that doesn't
// abort the user's turn.
func (a *agent) renderMCPChanges(ctx context.Context, sid string, changes []mcpChange) {
	for _, ch := range changes {
		var title string
		switch ch.action {
		case "parse_error":
			title = "MCP: .codehalter/mcp.toml parse error"
		case "failed":
			title = fmt.Sprintf("MCP: %s failed to start", ch.name)
		case "started":
			title = fmt.Sprintf("MCP: %s started", ch.name)
		case "stopped":
			title = fmt.Sprintf("MCP: %s stopped", ch.name)
		case "restarted":
			title = fmt.Sprintf("MCP: %s restarted (config changed)", ch.name)
		default:
			title = "MCP: " + ch.action
		}

		tcId := a.StartToolCall(ctx, sid, title, "think", nil)
		if ch.err != nil {
			a.FailToolCall(ctx, sid, tcId, ch.err.Error())
			continue
		}
		var body string
		switch ch.action {
		case "started":
			body = "Server is up; its tools are now available as " + ch.name + "__*."
		case "stopped":
			body = "Server has been shut down and its tools unregistered."
		case "restarted":
			body = "Old process replaced; new tools are live as " + ch.name + "__*."
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(body)})
	}
}
