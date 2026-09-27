package main

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
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/BurntSushi/toml"
)

// Exactly one of stdio (command/args/env) or HTTP (url/headers); comment an entry out to disable it.
type MCPServerConfig struct {
	Name    string            `toml:"name"`
	Command string            `toml:"command"`
	Args    []string          `toml:"args"`
	Env     map[string]string `toml:"env"`
	URL     string            `toml:"url"`
	Headers map[string]string `toml:"headers"`
}

type mcpRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
	// headers are the modern-era HTTP header mirrors; unexported so they stay out of the JSON body.
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
	Error   *mcpRPCError    `json:"error,omitempty"`
}

type mcpRPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *mcpRPCError) Error() string { return fmt.Sprintf("mcp error %d: %s", e.Code, e.Message) }

const (
	// mcpModernVersion is the stateless revision: no handshake, no sessions, _meta on every request.
	mcpModernVersion = "2026-07-28"
	mcpLegacyVersion = "2025-06-18"

	// Error codes only a modern server emits (the spec's reserved range).
	mcpErrHeaderMismatch     = -32020
	mcpErrMissingCapability  = -32021
	mcpErrUnsupportedVersion = -32022
)

// A modern server listing one of these as supported is dual-era: fall back to the handshake.
var legacyEraVersions = []string{"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}

// For the rare legacy server that ignores pre-initialize traffic instead of answering -32601.
var mcpProbeTimeout = 5 * time.Second

// Per spec, one of these during the era probe means a modern server: never fall back on it.
func isModernRPCError(err error) bool {
	var rpc *mcpRPCError
	if !errors.As(err, &rpc) {
		return false
	}
	return rpc.Code == mcpErrHeaderMismatch || rpc.Code == mcpErrMissingCapability || rpc.Code == mcpErrUnsupportedVersion
}

type mcpTransport interface {
	send(ctx context.Context, req mcpRequest) (mcpResponse, error)
	notify(ctx context.Context, n mcpNotification) error
	close()
}

// One per [[server]]; send is safe for concurrent use (the transport demultiplexes by id).
type MCPClient struct {
	name      string
	cfg       MCPServerConfig
	transport mcpTransport
	nextID    atomic.Int64
	// modern is set once by the era probe, before the client is published.
	modern bool
	// initialized gates MCP-Protocol-Version, which legacy servers expect only after initialize.
	initialized bool
	// tools is written once before the client is published, so readers never race it.
	tools []mcpTool
}

// Long enough for a first-run `npx` fetch; bounds a stdio server that never answers.
const mcpStartTimeout = 30 * time.Second

func StartMCPClient(ctx context.Context, cfg MCPServerConfig, cwd string) (*MCPClient, error) {
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
			client:  &http.Client{Timeout: mcpStartTimeout},
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

	c := &MCPClient{name: cfg.Name, cfg: cfg, transport: t, modern: true}
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

// detectEra probes with server/discover (spec backward-compat rules): a DiscoverResult or a modern
// error means modern; -32601, a bare HTTP 400 or a timeout means legacy, so initialize instead.
func (c *MCPClient) detectEra(ctx context.Context) error {
	probeCtx, cancel := context.WithTimeout(ctx, mcpProbeTimeout)
	raw, err := c.send(probeCtx, "server/discover", nil, nil)
	cancel()
	switch {
	case err == nil:
		var disc struct {
			SupportedVersions []string `json:"supportedVersions"`
		}
		if jerr := json.Unmarshal(raw, &disc); jerr == nil && slices.Contains(disc.SupportedVersions, mcpModernVersion) {
			return nil
		}
		// A DiscoverResult without our version: dual-era on another revision, use the handshake.
	case isModernRPCError(err):
		// Modern server: fall back only if it lists a legacy revision (dual-era).
		var rpc *mcpRPCError
		errors.As(err, &rpc)
		if rpc.Code != mcpErrUnsupportedVersion || !supportsLegacyEra(rpc.Data) {
			return fmt.Errorf("no protocol version in common (client speaks %s and %s): %w", mcpModernVersion, mcpLegacyVersion, err)
		}
	default:
		// Legacy, or broken, in which case the handshake fails with the real reason.
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

func (c *MCPClient) send(ctx context.Context, method string, params map[string]any, extraHeaders map[string]string) (json.RawMessage, error) {
	req := mcpRequest{JSONRPC: "2.0", ID: c.nextID.Add(1), Method: method}
	headers := map[string]string{}
	if c.modern {
		// No session, so version and identity ride in _meta on every modern request.
		p := make(map[string]any, len(params)+1)
		maps.Copy(p, params)
		// Empty capabilities: codehalter serves no sampling, elicitation or roots.
		p["_meta"] = map[string]any{
			"io.modelcontextprotocol/protocolVersion":    mcpModernVersion,
			"io.modelcontextprotocol/clientInfo":         map[string]any{"name": "codehalter", "version": "0.1.0"},
			"io.modelcontextprotocol/clientCapabilities": map[string]any{},
		}
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
		return nil, resp.Error
	}
	return resp.Result, nil
}

func (c *MCPClient) Close() {
	c.transport.close()
	slog.Info("mcp closed", "name", c.name)
}

type stdioTransport struct {
	name   string
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	stderr *ringWriter   // last bytes of stderr: the failure reason
	done   chan struct{} // closed when the child process exits

	writeMu sync.Mutex

	pendingMu sync.Mutex
	pending   map[int64]chan mcpResponse
}

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
	// stderr is where a server that cannot start says why.
	stderr := &ringWriter{max: 8192}
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
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

// Notifications and server-originated requests are dropped: we register for none.
func (t *stdioTransport) readLoop() {
	// stdout closing means the child exited: reap it and close done so pending sends fail fast.
	defer func() {
		err := t.cmd.Wait()
		if se := t.stderr.String(); se != "" {
			slog.Warn("mcp server exited", "name", t.name, "err", err, "stderr", truncate(se, 800))
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
			slog.Debug("mcp: response for unknown/late id", "name", t.name, "id", *resp.ID)
			continue
		}
		// Non-blocking: a duplicate response must not wedge the read loop for every other call.
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
		if se := t.stderr.String(); se != "" {
			return mcpResponse{}, fmt.Errorf("server exited: %s", truncate(se, 400))
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

func (t *stdioTransport) close() {
	if t.stdin != nil {
		t.stdin.Close()
	}
	if t.cmd == nil || t.cmd.Process == nil {
		return
	}
	// readLoop owns cmd.Wait; kill a server that ignores the EOF so readLoop unblocks.
	select {
	case <-t.done:
	case <-time.After(500 * time.Millisecond):
		t.cmd.Process.Kill()
		<-t.done
	}
}

// Session-id handling is legacy-only in practice: a modern server never mints one.
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
	_, err = t.do(ctx, body, nil)
	return err
}

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
	// Per-request mirrors go last so a stale user-configured header cannot shadow them.
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
		// Modern servers send JSON-RPC errors in 4xx bodies; the era probe needs them as envelopes.
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
	var resp mcpResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		if errors.Is(err, io.EOF) {
			return mcpResponse{}, nil
		}
		return mcpResponse{}, fmt.Errorf("decode response: %w", err)
	}
	return resp, nil
}

// The server may send notifications first; the response is the first envelope with an id.
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
	}
}

func (t *httpTransport) close() {
	t.sessionMu.Lock()
	sid := t.sessionId
	t.sessionId = ""
	t.sessionMu.Unlock()
	if sid == "" {
		return
	}
	// Bounded so a black-holed endpoint cannot hang shutdown or a turn's reconcile.
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

type mcpTool struct {
	Name         string         `json:"name"`
	Description  string         `json:"description"`
	InputSchema  map[string]any `json:"inputSchema"`
	headerParams []mcpHeaderParam
}

func (c *MCPClient) listTools(ctx context.Context) ([]mcpTool, error) {
	var all []mcpTool
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
			Tools      []mcpTool `json:"tools"`
			NextCursor string    `json:"nextCursor,omitempty"`
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

func (c *MCPClient) callTool(ctx context.Context, name string, args json.RawMessage) (string, bool, error) {
	argsField := any(map[string]any{})
	if len(args) > 0 && string(args) != "null" {
		var parsed any
		if err := json.Unmarshal(args, &parsed); err == nil {
			argsField = parsed
		}
	}
	var extra map[string]string
	if argMap, ok := argsField.(map[string]any); ok {
		for _, tl := range c.tools {
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
	// "input_required" is an MRTR interim result. We advertise no client capabilities, so fail
	// loudly rather than hand it to the LLM. A missing resultType means complete (all legacy servers).
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

func (a *agent) registerMCPTools(c *MCPClient, tools []mcpTool) {
	for _, t := range tools {
		fullName := c.name + "__" + t.Name
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
		a.tools.add(Tool{
			Def: map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        fullName,
					"description": description,
					"parameters":  params,
				},
			},
			Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
				tcId := a.StartToolCall(ctx, sid, fullName, "search", nil)
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

type mcpHeaderParam struct {
	header string   // name portion of Mcp-Param-{name}
	path   []string // properties chain from the schema root to the argument
	typ    string   // "string" | "integer" | "boolean"
}

// vetTools drops tools with invalid x-mcp-header annotations rather than failing the server.
// The extension exists only on modern HTTP, so other transports pass through.
func (c *MCPClient) vetTools(tools []mcpTool) []mcpTool {
	if _, isHTTP := c.transport.(*httpTransport); !isHTTP || !c.modern {
		return tools
	}
	kept := make([]mcpTool, 0, len(tools))
	for _, tl := range tools {
		hp, err := collectHeaderParams(tl.InputSchema)
		if err != nil {
			slog.Warn("mcp: rejecting tool with invalid x-mcp-header annotation", "server", c.name, "tool", tl.Name, "err", err)
			continue
		}
		tl.headerParams = hp
		kept = append(kept, tl)
	}
	return kept
}

// Per spec, an annotated property must be reachable from the root through `properties` keys
// only; an annotation anywhere else invalidates the whole tool.
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

// Only string values count: a property merely named x-mcp-header is a schema object.
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

// headerToken reports whether s is an RFC 9110 token (tchar).
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

// Absent, null or type-mismatched values omit the header, per spec.
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

type mcpChange struct {
	action string // "started" | "stopped" | "restarted" | "failed" | "parse_error"
	name   string // server name; "" for parse_error
	err    error
	tools  int // tools the server advertised; "started"/"restarted" only
}

// stdio children are started without a context and would outlive codehalter. Closed outside
// the lock, under a deadline, so a wedged server cannot hang exit.
func (a *agent) shutdownMCP() {
	a.mcp.mu.Lock()
	clients := make([]*MCPClient, 0, len(a.mcp.clients))
	for _, c := range a.mcp.clients {
		clients = append(clients, c)
	}
	a.mcp.clients = nil
	a.mcp.mu.Unlock()
	if len(clients) == 0 {
		return
	}
	done := make(chan struct{})
	go func() {
		for _, c := range clients {
			c.Close()
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		slog.Warn("mcp shutdown: timed out closing clients")
	}
}

func mcpConfigPath(cwd string) string { return filepath.Join(cwd, sessionDir, "mcp.toml") }

const elicitMCPKey = "servers"

// offerMCPImport writes every offered editor server to mcp.toml, unchosen ones commented out,
// so the question is asked once and enabling one later is deleting a '#'.
func (a *agent) offerMCPImport(ctx context.Context, cwd, sid string) {
	sess := a.getSession(sid)
	if sess == nil || len(sess.mcpOffer) == 0 {
		return
	}
	path := mcpConfigPath(cwd)
	raw, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		slog.Warn("mcp import: reading config", "path", path, "err", err)
		return
	}
	var fresh []acpMCPServer
	for _, s := range sess.mcpOffer {
		// SSE is the one transport we cannot run (and Initialize does not advertise it).
		if s.Name == "" || s.Type == "sse" || mcpNameInFile(string(raw), s.Name) {
			continue
		}
		fresh = append(fresh, s)
	}
	if len(fresh) == 0 {
		return
	}

	chosen := a.askMCPImport(ctx, sid, fresh)
	var body strings.Builder
	var added []string
	for _, s := range fresh {
		enabled := chosen[s.Name]
		if enabled {
			added = append(added, s.Name)
		}
		body.WriteString(mcpTOMLEntry(s, !enabled))
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		slog.Warn("mcp import: creating config dir", "path", path, "err", err)
		return
	}
	if err := appendFile(path, body.String()); err != nil {
		slog.Warn("mcp import: writing config", "path", path, "err", err)
		return
	}

	msg := fmt.Sprintf("Wrote %d MCP server(s) from your editor's settings into .codehalter/mcp.toml, commented out. Uncomment one to enable it.", len(fresh))
	if len(added) > 0 {
		msg = fmt.Sprintf("Added %s to .codehalter/mcp.toml. Starting with your next message.", strings.Join(added, ", "))
		if rest := len(fresh) - len(added); rest > 0 {
			msg += fmt.Sprintf(" The other %d are in the file commented out.", rest)
		}
	}
	a.say(ctx, sid, msg+"\n\n")
}

func (a *agent) askMCPImport(ctx context.Context, sid string, fresh []acpMCPServer) map[string]bool {
	if a.conn == nil || !a.clientCan("elicitation") {
		return nil
	}
	options := make([]map[string]any, 0, len(fresh))
	for _, s := range fresh {
		summary := "stdio " + strings.TrimSpace(s.Command+" "+strings.Join(s.Args, " "))
		if s.URL != "" {
			summary = "http " + s.URL
		}
		options = append(options, map[string]any{"const": s.Name, "title": s.Name + " — " + summary})
	}
	raw, err := a.conn.sendRequest(ctx, "elicitation/create", map[string]any{
		"sessionId": sid,
		"mode":      "form",
		"message": "Your editor is configured with MCP servers codehalter isn't using yet. " +
			"Pick the ones to add to .codehalter/mcp.toml. The rest are written commented out, so you won't be asked again.",
		"requestedSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				elicitMCPKey: map[string]any{
					"type":  "array",
					"title": "MCP servers to enable",
					"items": map[string]any{"anyOf": options},
				},
			},
		},
	})
	if err != nil {
		slog.Warn("mcp import: elicitation failed", "err", err)
		return nil
	}
	// Content values are a union (string, number, bool, string array), hence map[string]any.
	var resp struct {
		Action  string         `json:"action"`
		Content map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		slog.Warn("mcp import: undecodable elicitation response", "err", err)
		return nil
	}
	if resp.Action != "accept" {
		return nil
	}
	picked := map[string]bool{}
	values, _ := resp.Content[elicitMCPKey].([]any)
	for _, v := range values {
		if name, ok := v.(string); ok {
			picked[name] = true
		}
	}
	return picked
}

// Commented-out entries count too: they are how a declined server is remembered.
func mcpNameInFile(raw, name string) bool {
	quoted := strconv.Quote(name)
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(line), "#"))
		key, value, ok := strings.Cut(line, "=")
		if ok && strings.TrimSpace(key) == "name" && strings.TrimSpace(value) == quoted {
			return true
		}
	}
	return false
}

func mcpTOMLEntry(s acpMCPServer, commented bool) string {
	var b strings.Builder
	b.WriteString("[[server]]\n")
	fmt.Fprintf(&b, "name = %q\n", s.Name)
	if s.URL != "" {
		fmt.Fprintf(&b, "url = %q\n", s.URL)
		if t := tomlInlineTable(s.Headers); t != "" {
			fmt.Fprintf(&b, "headers = %s\n", t)
		}
	} else {
		fmt.Fprintf(&b, "command = %q\n", s.Command)
		if len(s.Args) > 0 {
			quoted := make([]string, len(s.Args))
			for i, arg := range s.Args {
				quoted[i] = strconv.Quote(arg)
			}
			fmt.Fprintf(&b, "args = [%s]\n", strings.Join(quoted, ", "))
		}
		if t := tomlInlineTable(s.Env); t != "" {
			fmt.Fprintf(&b, "env = %s\n", t)
		}
	}
	if !commented {
		return "\n" + b.String()
	}
	var out strings.Builder
	out.WriteString("\n# Offered by the editor, not enabled. Uncomment to use.\n")
	for _, line := range strings.Split(strings.TrimRight(b.String(), "\n"), "\n") {
		fmt.Fprintf(&out, "# %s\n", line)
	}
	return out.String()
}

func tomlInlineTable(kv []acpNameValue) string {
	if len(kv) == 0 {
		return ""
	}
	parts := make([]string, 0, len(kv))
	for _, e := range kv {
		parts = append(parts, tomlKey(e.Name)+" = "+strconv.Quote(e.Value))
	}
	return "{ " + strings.Join(parts, ", ") + " }"
}

func tomlKey(k string) string {
	if k == "" {
		return `""`
	}
	for _, r := range k {
		bare := r == '-' || r == '_' ||
			(r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
		if !bare {
			return strconv.Quote(k)
		}
	}
	return k
}

// Close is checked: it is where a deferred write actually fails.
func appendFile(path, body string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.WriteString(body); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// Never called mid-turn: registering tools rewrites the `tools` array the conversation is
// rendered behind. Restarts are start-then-stop, so a bad config never kills a working server.
func (a *agent) reconcileMCP(ctx context.Context, cwd string) []mcpChange {
	a.mcp.mu.Lock()
	defer a.mcp.mu.Unlock()

	path := mcpConfigPath(cwd)
	var cfgs []MCPServerConfig
	var mtime time.Time
	var err error
	if info, serr := os.Stat(path); serr == nil {
		mtime = info.ModTime()
		var f struct {
			Server []MCPServerConfig `toml:"server"`
		}
		if _, derr := toml.DecodeFile(path, &f); derr != nil {
			err = fmt.Errorf("loading %s: %w", path, derr)
		} else {
			cfgs = f.Server
		}
	} else if !os.IsNotExist(serr) {
		err = serr
	}
	if err != nil {
		// Reported once per mtime, not on every prompt.
		if !mtime.IsZero() && mtime.Equal(a.mcp.appliedMtime) {
			return nil
		}
		a.mcp.appliedMtime = mtime
		return []mcpChange{{action: "parse_error", err: err}}
	}
	// Unchanged file: a failed start is retried only after an edit bumps the mtime.
	if !mtime.IsZero() && mtime.Equal(a.mcp.appliedMtime) {
		return nil
	}
	a.mcp.appliedMtime = mtime

	// Last write wins on duplicate names.
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

	applied := make(map[string]MCPServerConfig, len(a.mcp.applied))
	for _, c := range a.mcp.applied {
		applied[c.Name] = c
	}

	var changes []mcpChange

	for name, want := range desired {
		old, existed := applied[name]
		if existed && old.Command == want.Command && old.URL == want.URL &&
			slices.Equal(old.Args, want.Args) &&
			maps.Equal(old.Env, want.Env) && maps.Equal(old.Headers, want.Headers) {
			continue
		}

		// The stdio transport has no timeout of its own; a hung server is recorded as failed.
		startCtx, cancel := context.WithTimeout(ctx, mcpStartTimeout)
		newClient, err := StartMCPClient(startCtx, want, cwd)
		if err != nil {
			cancel()
			if ctx.Err() == nil { // a start the user stopped is not a failure; the next prompt retries it
				changes = append(changes, mcpChange{action: "failed", name: name, err: err})
			}
			continue
		}
		tools, err := newClient.listTools(startCtx)
		cancel()
		if err != nil {
			newClient.Close()
			if ctx.Err() == nil {
				changes = append(changes, mcpChange{action: "failed", name: name, err: fmt.Errorf("tools/list: %w", err)})
			}
			continue
		}
		tools = newClient.vetTools(tools)

		newClient.tools = tools

		a.mu.Lock()
		if a.mcp.clients == nil {
			a.mcp.clients = make(map[string]*MCPClient)
		}
		oldClient := a.mcp.clients[name]
		a.mcp.clients[name] = newClient
		a.mu.Unlock()

		if oldClient != nil {
			a.tools.removePrefix(name + "__")
		}
		a.registerMCPTools(newClient, tools)
		if oldClient != nil {
			oldClient.Close()
			changes = append(changes, mcpChange{action: "restarted", name: name, tools: len(tools)})
		} else {
			changes = append(changes, mcpChange{action: "started", name: name, tools: len(tools)})
		}
	}

	for name := range applied {
		if _, stillWanted := desired[name]; stillWanted {
			continue
		}
		a.mu.Lock()
		oldClient := a.mcp.clients[name]
		delete(a.mcp.clients, name)
		a.mu.Unlock()

		a.tools.removePrefix(name + "__")
		if oldClient != nil {
			oldClient.Close()
		}
		changes = append(changes, mcpChange{action: "stopped", name: name})
	}

	// Cancelled midway: forget the mtime so the next prompt reconciles again.
	if ctx.Err() != nil {
		a.mcp.appliedMtime = time.Time{}
	}
	// Recorded from what runs, not from the file: a failed or stopped start or
	// restart is retried at the next reconcile.
	a.mcp.applied = a.mcp.applied[:0]
	a.mu.Lock()
	for _, c := range a.mcp.clients {
		a.mcp.applied = append(a.mcp.applied, c.cfg)
	}
	a.mu.Unlock()

	return changes
}
