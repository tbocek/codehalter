// Package mcp is codehalter's Model Context Protocol client: it starts a server
// over stdio or Streamable HTTP, negotiates the protocol era, lists the
// server's tools and calls them. What codehalter does with those tools (the
// tool registry, the reconciler for .codehalter/mcp.toml, importing the
// editor's servers) lives in the main package.
package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"math"
	"net/http"
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ---------------------------------------------------------------------------
// MCP settings
// ---------------------------------------------------------------------------

// ServerConfig is one [[server]] entry from .codehalter/mcp.toml. Each
// entry is exactly one of stdio (command/args/env) OR HTTP (url/headers) —
// the reconciler rejects entries that set both. There is no `enabled` field;
// comment an entry out to disable it.
type ServerConfig struct {
	Name    string            `toml:"name"`
	Command string            `toml:"command"`
	Args    []string          `toml:"args"`
	Env     map[string]string `toml:"env"`
	URL     string            `toml:"url"`
	Headers map[string]string `toml:"headers"`
}

// ---------------------------------------------------------------------------
// JSON-RPC envelopes (MCP uses 2.0 over its chosen transport)
// ---------------------------------------------------------------------------

type mcpRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
	// headers are the modern spec's HTTP header mirrors for this request
	// (MCP-Protocol-Version, Mcp-Method, Mcp-Name, Mcp-Param-*). Unexported,
	// so they never appear in the JSON body; stdio ignores them.
	headers map[string]string
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
	Error   *struct {
		Code    int             `json:"code"`
		Message string          `json:"message"`
		Data    json.RawMessage `json:"data,omitempty"`
	} `json:"error,omitempty"`
}

// mcpRPCError is a JSON-RPC error from the server, surfaced as a typed error
// so era detection can branch on the code (see isModernRPCError).
type mcpRPCError struct {
	Code    int
	Message string
	Data    json.RawMessage
}

func (e *mcpRPCError) Error() string { return fmt.Sprintf("mcp error %d: %s", e.Code, e.Message) }

const (
	// mcpModernVersion is the stateless, per-request-metadata revision
	// ("MCP 2"): no initialize handshake, no sessions, _meta on every request.
	mcpModernVersion = "2026-07-28"
	// mcpLegacyVersion is the initialize-handshake revision spoken as the
	// fallback for the servers that dominate the wild today.
	mcpLegacyVersion = "2025-06-18"

	// Error codes only a modern server emits (the spec's reserved range).
	mcpErrHeaderMismatch     = -32020
	mcpErrMissingCapability  = -32021
	mcpErrUnsupportedVersion = -32022
)

// legacyEraVersions are the initialize-based revisions. A modern server that
// rejects our probe but lists one of these as supported is dual-era — fall
// back to the handshake instead of failing.
var legacyEraVersions = []string{"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}

// mcpProbeTimeout bounds the server/discover era probe. Legacy SDK servers
// answer unknown methods with -32601 near-instantly; the timeout is for the
// rare one that ignores pre-initialize traffic. A var so tests can shrink it.
var mcpProbeTimeout = 5 * time.Second

// isModernRPCError reports whether err is a JSON-RPC error only a modern
// (2026-07-28+) server emits. Per the spec's backward-compatibility rule,
// seeing one during the era probe identifies a modern server — the client
// must NOT fall back to the legacy handshake on it.
func isModernRPCError(err error) bool {
	var rpc *mcpRPCError
	if !errors.As(err, &rpc) {
		return false
	}
	return rpc.Code == mcpErrHeaderMismatch || rpc.Code == mcpErrMissingCapability || rpc.Code == mcpErrUnsupportedVersion
}

// modernMeta is the _meta block every modern-era request carries: protocol
// version (mirrored into the MCP-Protocol-Version header on HTTP), client
// identity, and capabilities. Capabilities stay empty on purpose — codehalter
// doesn't service sampling/elicitation/roots, so servers must not solicit
// MRTR input from us.
func modernMeta() map[string]any {
	return map[string]any{
		"io.modelcontextprotocol/protocolVersion":    mcpModernVersion,
		"io.modelcontextprotocol/clientInfo":         map[string]any{"name": "codehalter", "version": "0.1.0"},
		"io.modelcontextprotocol/clientCapabilities": map[string]any{},
	}
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

// Client drives a long-lived MCP server. One instance per [[server]]
// entry, shared across tool calls so initialize cost is paid once. send is
// safe to call concurrently — the underlying transport serialises writes
// and demultiplexes responses by id.
type Client struct {
	Name      string
	transport mcpTransport
	nextID    atomic.Int64
	// modern is true when the server speaks the stateless 2026-07-28 revision:
	// no handshake, per-request _meta, mirrored headers on HTTP. Decided once
	// by the era probe in Start, before the client is published, and
	// cached for the client's lifetime (the era is a property of the server).
	modern bool
	// initialized is set after a successful legacy handshake. It gates the
	// MCP-Protocol-Version header, which legacy servers expect only on
	// post-initialize requests.
	initialized bool
	// tools is what this server advertised at tools/list, retained so callTool
	// can look up a tool's x-mcp-header parameters. Written once by reconcileMCP
	// BEFORE the client is published into a.mcp.clients, so readers never race
	// the write.
	Tools []Tool
}

// StartTimeout bounds a server's bring-up (handshake + tools/list) so a
// stdio server that never answers can't hang the prompt's prepare phase. Long
// enough for a first-run `npx` fetch, short enough that a wedged --bin fails fast.
const StartTimeout = 30 * time.Second

// Start brings up the server described by cfg and runs the MCP
// initialize handshake. The returned client is ready for tools/list and
// tools/call. cwd scopes a stdio child's working directory; HTTP transports
// ignore it.
func Start(ctx context.Context, cfg ServerConfig, cwd string) (*Client, error) {
	if cfg.Command != "" && cfg.URL != "" {
		return nil, fmt.Errorf("mcp config %q sets both command and url — pick one", cfg.Name)
	}
	var t mcpTransport
	switch {
	case cfg.URL != "":
		t = &httpTransport{
			name:    cfg.Name,
			url:     cfg.URL,
			headers: cfg.Headers,
			// Per-server timeout protects against a hung server holding up the
			// agent indefinitely; tools/call for long-running operations should
			// be wrapped in the request ctx for finer control. Same bound as the
			// stdio handshake (StartTimeout) so both transports behave alike.
			client: &http.Client{Timeout: StartTimeout},
		}
	case cfg.Command != "":
		st, err := newStdioTransport(cfg, cwd)
		if err != nil {
			return nil, err
		}
		t = st
	default:
		return nil, fmt.Errorf("mcp config %q has neither command nor url", cfg.Name)
	}

	c := &Client{Name: cfg.Name, transport: t, modern: true}
	if err := c.detectEra(ctx); err != nil {
		c.Close()
		return nil, fmt.Errorf("initialize %s: %w", cfg.Name, err)
	}
	proto := mcpModernVersion
	if !c.modern {
		proto = mcpLegacyVersion
	}
	slog.Info("mcp ready", "name", cfg.Name, "protocol", proto)
	return c, nil
}

// detectEra decides whether the server speaks the stateless 2026-07-28
// revision or needs the legacy initialize handshake, and completes whichever
// bring-up applies. Probe-first, per the spec's backward-compatibility rules:
// server/discover is sent as a modern request; a DiscoverResult (or a
// recognized modern error) identifies a modern server, while anything else —
// -32601 from a legacy SDK, a bare HTTP 400 for a missing session, a probe
// timeout — identifies a legacy one and we fall back to `initialize` +
// `notifications/initialized`.
func (c *Client) detectEra(ctx context.Context) error {
	probeCtx, cancel := context.WithTimeout(ctx, mcpProbeTimeout)
	raw, err := c.send(probeCtx, "server/discover", nil, nil)
	cancel()
	switch {
	case err == nil:
		var disc struct {
			SupportedVersions []string `json:"supportedVersions"`
		}
		if jerr := json.Unmarshal(raw, &disc); jerr == nil && slices.Contains(disc.SupportedVersions, mcpModernVersion) {
			return nil // modern confirmed — stateless, no handshake
		}
		// A DiscoverResult without our modern version: a dual-era server on
		// some other revision. Negotiate the legacy path below.
	case isModernRPCError(err):
		// Definitely a modern server — falling back on these errors would
		// violate the spec. If it lists a legacy revision as supported it is
		// dual-era and the handshake works; otherwise no common version.
		var rpc *mcpRPCError
		errors.As(err, &rpc)
		if rpc.Code != mcpErrUnsupportedVersion || !supportsLegacyEra(rpc.Data) {
			return fmt.Errorf("no protocol version in common (client speaks %s and %s): %w", mcpModernVersion, mcpLegacyVersion, err)
		}
	default:
		// Legacy server — or a broken one, in which case the handshake below
		// fails with the real reason (e.g. the child's captured stderr).
	}

	c.modern = false
	_, err = c.send(ctx, "initialize", map[string]any{
		"protocolVersion": mcpLegacyVersion,
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "codehalter", "version": "0.1.0"},
	}, nil)
	if err == nil {
		err = c.transport.notify(ctx, mcpNotification{
			JSONRPC: "2.0",
			Method:  "notifications/initialized",
			Params:  map[string]any{},
		})
	}
	if err != nil {
		return err
	}
	c.initialized = true
	return nil
}

// supportsLegacyEra reports whether an UnsupportedProtocolVersionError's data
// lists an initialize-based revision, i.e. the server is dual-era.
func supportsLegacyEra(data json.RawMessage) bool {
	var d struct {
		Supported []string `json:"supported"`
	}
	if err := json.Unmarshal(data, &d); err != nil {
		return false
	}
	return slices.ContainsFunc(d.Supported, func(v string) bool {
		return slices.Contains(legacyEraVersions, v)
	})
}

// send issues one request. extraHeaders carries per-call Mcp-Param-* mirrors
// (see headerParamValues); nil for everything else.
func (c *Client) send(ctx context.Context, method string, params map[string]any, extraHeaders map[string]string) (json.RawMessage, error) {
	req := mcpRequest{JSONRPC: "2.0", ID: c.nextID.Add(1), Method: method}
	headers := map[string]string{}
	if c.modern {
		// Every modern request self-describes: version and identity ride in
		// _meta (and, on HTTP, mirrored headers) instead of a session.
		p := make(map[string]any, len(params)+1)
		maps.Copy(p, params)
		p["_meta"] = modernMeta()
		params = p
		headers["MCP-Protocol-Version"] = mcpModernVersion
		headers["Mcp-Method"] = method
		if method == "tools/call" {
			if name, ok := params["name"].(string); ok {
				headers["Mcp-Name"] = encodeMCPHeaderValue(name)
			}
		}
		maps.Copy(headers, extraHeaders)
	} else if c.initialized {
		headers["MCP-Protocol-Version"] = mcpLegacyVersion
	}
	if params != nil {
		req.Params = params
	}
	req.headers = headers
	resp, err := c.transport.send(ctx, req)
	if err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, &mcpRPCError{Code: resp.Error.Code, Message: resp.Error.Message, Data: resp.Error.Data}
	}
	return resp.Result, nil
}

func (c *Client) Close() {
	c.transport.close()
	slog.Info("mcp closed", "name", c.Name)
}

// ---------------------------------------------------------------------------
// stdioTransport — line-delimited JSON-RPC over a child's stdin/stdout
// ---------------------------------------------------------------------------

type stdioTransport struct {
	name   string
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	stderr *ringWriter   // bounded capture of the child's stderr (the failure reason)
	done   chan struct{} // closed when the child process exits

	writeMu sync.Mutex

	pendingMu sync.Mutex
	pending   map[int64]chan mcpResponse
}

// ringWriter keeps only the last `max` bytes written — a bounded stderr capture
// so a crashing MCP server's error survives in the log without letting normal
// liveness chatter grow unbounded.
type ringWriter struct {
	mu  sync.Mutex
	buf []byte
	max int
}

func (w *ringWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf = append(w.buf, p...)
	if len(w.buf) > w.max {
		w.buf = w.buf[len(w.buf)-w.max:]
	}
	return len(p), nil
}

func (w *ringWriter) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return strings.TrimSpace(string(w.buf))
}

func newStdioTransport(cfg ServerConfig, cwd string) (*stdioTransport, error) {
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
	// Capture stderr (bounded) instead of discarding it: it's where a server that
	// can't start prints WHY (e.g. lsmcp's "No such built-in module: node:sqlite"
	// on Node < 22). The ring cap keeps liveness chatter from drowning the log.
	stderr := &ringWriter{max: 8192}
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
		// Close the pipes we opened above; without this each failed start (retried
		// on every mcp.toml mtime bump) leaks two OS file descriptors.
		stdin.Close()
		stdout.Close()
		return nil, fmt.Errorf("starting %s: %w", cfg.Command, err)
	}
	t := &stdioTransport{
		name:    cfg.Name,
		cmd:     cmd,
		stdin:   stdin,
		stdout:  stdout,
		stderr:  stderr,
		done:    make(chan struct{}),
		pending: make(map[int64]chan mcpResponse),
	}
	go t.readLoop()
	return t, nil
}

// readLoop reads one JSON-RPC message per line, forever. Responses with an id
// route to the pending caller; notifications and server-originated requests
// are dropped (we don't register for any).
func (t *stdioTransport) readLoop() {
	// stdout closing means the child exited. Reap it, log the real reason (its
	// captured stderr), and signal done so any pending send() fails fast with that
	// reason instead of blocking on a dead process until the ctx timeout.
	defer func() {
		err := t.cmd.Wait()
		if se := t.stderr.String(); se != "" {
			slog.Warn("mcp server exited", "name", t.name, "err", err, "stderr", clip(se, 800))
		} else {
			slog.Warn("mcp server exited", "name", t.name, "err", err)
		}
		close(t.done)
	}()
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
			slog.Debug("mcp decode error", "name", t.name, "error", err, "line", clip(line, 200))
			continue
		}
		if resp.ID == nil {
			continue
		}
		t.pendingMu.Lock()
		ch, ok := t.pending[*resp.ID]
		t.pendingMu.Unlock()
		if !ok {
			// No waiter: a late response to a cancelled/completed call, or an
			// unknown id. Drop it with a breadcrumb instead of silently.
			slog.Debug("mcp: response for unknown/late id", "name", t.name, "id", *resp.ID)
			continue
		}
		// Non-blocking: ch is buffered (cap 1) for the normal case, but a SECOND
		// response to the same id (a misbehaving server) must not block the read
		// loop here — that would wedge every other in-flight call on this transport.
		select {
		case ch <- resp:
		default:
			slog.Debug("mcp: dropped duplicate response", "name", t.name, "id", *resp.ID)
		}
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
	case <-t.done:
		// The child exited (e.g. crashed on startup) — fail now with its stderr
		// rather than waiting out the ctx timeout on a response that can't come.
		if se := t.stderr.String(); se != "" {
			return mcpResponse{}, fmt.Errorf("server exited: %s", clip(se, 400))
		}
		return mcpResponse{}, fmt.Errorf("server exited before responding")
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
		t.stdin.Close() // EOF → a well-behaved server exits → stdout closes → readLoop reaps
	}
	if t.cmd == nil || t.cmd.Process == nil {
		return
	}
	// readLoop owns cmd.Wait() and closes t.done after reaping. Wait for it; if the
	// server ignores the stdin EOF, Kill forces stdout closed so readLoop unblocks.
	select {
	case <-t.done:
	case <-time.After(500 * time.Millisecond):
		t.cmd.Process.Kill()
		<-t.done
	}
}

// ---------------------------------------------------------------------------
// httpTransport — MCP Streamable HTTP (spec 2025-06-18)
// ---------------------------------------------------------------------------

// httpTransport speaks MCP Streamable HTTP across both eras. Each request is
// a POST whose body is the JSON-RPC envelope; the server replies with either
// a single JSON object or an SSE stream. Era-specific behavior lives in the
// request the client hands us (modern _meta + mirrored headers vs legacy
// bare envelopes); the session-id plumbing below is legacy-only in practice —
// a modern server never mints one, so sessionId stays empty, no session id
// is echoed, and close's DELETE is skipped.
type httpTransport struct {
	name      string
	url       string
	headers   map[string]string
	client    *http.Client
	sessionMu sync.Mutex
	sessionId string
}

func (t *httpTransport) send(ctx context.Context, req mcpRequest) (mcpResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return mcpResponse{}, err
	}
	return t.do(ctx, body, req.headers)
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
	_, err = t.do(ctx, body, nil)
	return err
}

// do issues one HTTP request and decodes the response. Handles both JSON
// and SSE response bodies and captures any Mcp-Session-Id the server emits.
func (t *httpTransport) do(ctx context.Context, body []byte, headers map[string]string) (mcpResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, t.url, bytes.NewReader(body))
	if err != nil {
		return mcpResponse{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	for k, v := range t.headers {
		httpReq.Header.Set(k, v)
	}
	// Per-request mirrors (modern era) go last so they can't be shadowed by a
	// stale user-configured header of the same name.
	for k, v := range headers {
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
		// Modern servers put JSON-RPC errors (UnsupportedProtocolVersion,
		// HeaderMismatch, method-not-found) in 4xx bodies. Surface those as
		// envelopes so the era probe can tell "a modern server said no" from
		// "a legacy server didn't understand the request".
		var resp mcpResponse
		if jerr := json.Unmarshal(buf, &resp); jerr == nil && resp.Error != nil {
			return resp, nil
		}
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
	// Bound the best-effort DELETE so a black-holed endpoint can't hang shutdown
	// (or, when called from reconcile, the user's turn) for the client's full timeout.
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, t.url, nil)
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

// Tool is one entry from the tools/list response.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
	// headerParams are the schema's x-mcp-header annotations (modern HTTP
	// only), precomputed by vetTools so each tools/call can mirror the
	// designated arguments into Mcp-Param-* headers.
	headerParams []mcpHeaderParam
}

// listTools enumerates every tool the server exposes. Some servers paginate
// via `nextCursor`; we follow until empty.
func (c *Client) ListTools(ctx context.Context) ([]Tool, error) {
	var all []Tool
	var cursor string
	for {
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		raw, err := c.send(ctx, "tools/list", params, nil)
		if err != nil {
			return nil, err
		}
		var page struct {
			Tools      []Tool `json:"tools"`
			NextCursor string `json:"nextCursor,omitempty"`
		}
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

// callTool invokes a remote tool and renders the response into a single string
// suitable for surfacing back to the LLM. Only text content blocks are kept;
// image/resource blocks collapse to a "[non-text content of type X]" placeholder.
func (c *Client) CallTool(ctx context.Context, name string, args json.RawMessage) (string, bool, error) {
	argsField := any(map[string]any{})
	if len(args) > 0 && string(args) != "null" {
		var parsed any
		if err := json.Unmarshal(args, &parsed); err == nil {
			argsField = parsed
		}
	}
	// x-mcp-header mirrors: a modern HTTP server may designate arguments to be
	// repeated as Mcp-Param-* headers. vetTools precomputed the paths; nil
	// everywhere else (stdio, legacy, unannotated tools).
	var extra map[string]string
	if argMap, ok := argsField.(map[string]any); ok {
		for _, tl := range c.Tools {
			if tl.Name == name {
				extra = headerParamValues(tl.headerParams, argMap)
				break
			}
		}
	}
	raw, err := c.send(ctx, "tools/call", map[string]any{
		"name":      name,
		"arguments": argsField,
	}, extra)
	if err != nil {
		return "", false, err
	}
	var result struct {
		ResultType string `json:"resultType,omitempty"`
		Content    []struct {
			Type string `json:"type"`
			Text string `json:"text,omitempty"`
		} `json:"content"`
		IsError bool `json:"isError,omitempty"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", false, fmt.Errorf("tools/call decode: %w", err)
	}
	// MRTR (modern era): a server needing client-side input returns an interim
	// "input_required" result instead of a final one. We advertise no client
	// capabilities so this shouldn't happen; if it does, fail loudly rather
	// than hand the LLM an interim result as the tool's output. A missing
	// resultType means "complete" per spec (and covers every legacy server).
	if result.ResultType == "input_required" {
		return "", false, fmt.Errorf("tool %q requires interactive client input (MRTR), which codehalter does not support", name)
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

// ---------------------------------------------------------------------------
// x-mcp-header (Mcp-Param-*) mirrors — modern Streamable HTTP only
// ---------------------------------------------------------------------------

// mcpHeaderParam is one x-mcp-header annotation: mirror the argument at path
// into the Mcp-Param-<header> HTTP header on each tools/call.
type mcpHeaderParam struct {
	header string   // name portion of Mcp-Param-{name}
	path   []string // properties chain from the schema root to the argument
	typ    string   // "string" | "integer" | "boolean"
}

// vetTools enforces the client side of the x-mcp-header contract on modern
// HTTP transports: tools whose annotations violate the spec's constraints are
// excluded (with a warning) so one malformed definition doesn't block the
// rest; valid annotations are precomputed onto the tool. stdio and legacy
// servers pass through untouched — the extension is defined for the modern
// HTTP transport only.
func (c *Client) VetTools(tools []Tool) []Tool {
	if _, isHTTP := c.transport.(*httpTransport); !isHTTP || !c.modern {
		return tools
	}
	kept := make([]Tool, 0, len(tools))
	for _, tl := range tools {
		hp, err := collectHeaderParams(tl.InputSchema)
		if err != nil {
			slog.Warn("mcp: rejecting tool with invalid x-mcp-header annotation", "server", c.Name, "tool", tl.Name, "err", err)
			continue
		}
		tl.headerParams = hp
		kept = append(kept, tl)
	}
	return kept
}

// collectHeaderParams walks a tool's inputSchema for x-mcp-header annotations
// and validates the spec's constraints: header-token names, primitive types
// (number excluded), case-insensitive uniqueness, and static reachability —
// the annotated property must be reachable from the root through `properties`
// keys only. An annotation anywhere else (inside items, oneOf/anyOf/allOf,
// not, if/then/else, $defs, …) invalidates the whole tool definition.
func collectHeaderParams(schema map[string]any) ([]mcpHeaderParam, error) {
	if schema == nil {
		return nil, nil
	}
	var out []mcpHeaderParam
	seen := map[string]bool{} // lowercased header names, for the uniqueness rule
	var walk func(node map[string]any, path []string) error
	walk = func(node map[string]any, path []string) error {
		if hv, ok := node["x-mcp-header"]; ok {
			name, ok := hv.(string)
			if !ok || !headerToken(name) {
				return fmt.Errorf("x-mcp-header %v at %q: not a valid header token", hv, strings.Join(path, "."))
			}
			if len(path) == 0 {
				return fmt.Errorf("x-mcp-header %q on the schema root, not a parameter", name)
			}
			typ, _ := node["type"].(string)
			if typ != "string" && typ != "integer" && typ != "boolean" {
				return fmt.Errorf("x-mcp-header %q at %q: type %q not allowed (string/integer/boolean only)", name, strings.Join(path, "."), typ)
			}
			if lower := strings.ToLower(name); seen[lower] {
				return fmt.Errorf("x-mcp-header %q: duplicate (case-insensitive)", name)
			} else { //nolint:revive // symmetric with the check above
				seen[lower] = true
			}
			out = append(out, mcpHeaderParam{header: name, path: slices.Clone(path), typ: typ})
		}
		for k, v := range node {
			if k == "x-mcp-header" {
				continue
			}
			if k == "properties" {
				props, ok := v.(map[string]any)
				if !ok {
					continue
				}
				for pname, pv := range props {
					if sub, ok := pv.(map[string]any); ok {
						if err := walk(sub, append(path, pname)); err != nil {
							return err
						}
					}
				}
				continue
			}
			// Every other keyword (items, composition, conditionals, $defs,
			// plain values) is off the reachable chain — an annotation inside
			// it poisons the tool.
			if err := noHeaderAnnotations(v); err != nil {
				return err
			}
		}
		return nil
	}
	if err := walk(schema, nil); err != nil {
		return nil, err
	}
	return out, nil
}

// noHeaderAnnotations errors if any map nested under v carries an
// x-mcp-header annotation. Only string values count — a *property* merely
// named "x-mcp-header" maps to a schema object, not a string.
func noHeaderAnnotations(v any) error {
	switch n := v.(type) {
	case map[string]any:
		if s, ok := n["x-mcp-header"].(string); ok {
			return fmt.Errorf("x-mcp-header %q is not reachable via properties keys only", s)
		}
		for _, sub := range n {
			if err := noHeaderAnnotations(sub); err != nil {
				return err
			}
		}
	case []any:
		for _, sub := range n {
			if err := noHeaderAnnotations(sub); err != nil {
				return err
			}
		}
	}
	return nil
}

// headerToken reports whether s is a valid HTTP field-name token
// (RFC 9110 tchar).
func headerToken(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		ok := r == '!' || r == '#' || r == '$' || r == '%' || r == '&' || r == '\'' ||
			r == '*' || r == '+' || r == '-' || r == '.' || r == '^' || r == '_' ||
			r == '`' || r == '|' || r == '~' ||
			(r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
		if !ok {
			return false
		}
	}
	return true
}

// headerParamValues extracts the annotated argument values for one call.
// Absent, null, or type-mismatched values omit the header, per spec.
func headerParamValues(params []mcpHeaderParam, args map[string]any) map[string]string {
	if len(params) == 0 {
		return nil
	}
	out := map[string]string{}
	for _, p := range params {
		node := any(args)
		for _, step := range p.path {
			m, ok := node.(map[string]any)
			if !ok {
				node = nil
				break
			}
			node = m[step]
		}
		var val string
		switch v := node.(type) {
		case string:
			if p.typ != "string" {
				continue
			}
			val = v
		case bool:
			if p.typ != "boolean" {
				continue
			}
			val = strconv.FormatBool(v)
		case float64: // encoding/json's number type; integers only per the schema rule
			if p.typ != "integer" || v != math.Trunc(v) {
				continue
			}
			val = strconv.FormatInt(int64(v), 10)
		default:
			continue
		}
		out["Mcp-Param-"+p.header] = encodeMCPHeaderValue(val)
	}
	return out
}

// encodeMCPHeaderValue renders v as an HTTP header value, applying the spec's
// Base64 sentinel (=?base64?…?=) when v isn't plain printable ASCII, has
// leading/trailing whitespace, or itself matches the sentinel pattern.
func encodeMCPHeaderValue(v string) string {
	safe := v == strings.TrimSpace(v) &&
		!(strings.HasPrefix(v, "=?base64?") && strings.HasSuffix(v, "?="))
	if safe {
		for _, r := range v {
			if r != ' ' && r != '\t' && (r < 0x21 || r > 0x7e) {
				safe = false
				break
			}
		}
	}
	if safe {
		return v
	}
	return "=?base64?" + base64.StdEncoding.EncodeToString([]byte(v)) + "?="
}

// clip shortens s to n bytes for a log line or an error message.
func clip(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}
