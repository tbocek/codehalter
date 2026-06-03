package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
)

// Minimal ACP wire types — subset of codehalter/acp.go, just what a client
// needs to drive a turn end-to-end. Field tags match the server's; JSON shape
// stays compatible if the server adds optional fields.

type SessionId string
type StopReason string

const (
	ProtocolVersion              = 1
	StopReasonEndTurn StopReason = "end_turn"
)

type initializeRequest struct {
	ProtocolVersion int `json:"protocolVersion"`
}
type initializeResponse struct {
	ProtocolVersion int `json:"protocolVersion"`
}

type newSessionRequest struct {
	Cwd string `json:"cwd,omitempty"`
}
type newSessionResponse struct {
	SessionId SessionId `json:"sessionId"`
}

type contentBlockText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type promptRequest struct {
	SessionId SessionId          `json:"sessionId"`
	Content   []contentBlockText `json:"prompt"`
}
type promptResponse struct {
	StopReason StopReason `json:"stopReason,omitempty"`
}

// setSessionModeRequest flips the session into "Autopilot" so codehalter
// auto-answers every session/request_permission instead of blocking on us.
// Without this, codehalter would hang waiting for the bench to click
// Execute/Cancel on plan-execution and similar prompts.
type setSessionModeRequest struct {
	SessionId SessionId `json:"sessionId"`
	ModeId    string    `json:"modeId"`
}
type setSessionModeResponse struct{}

// rawRPC is the line-delimited JSON-RPC 2.0 envelope used by codehalter.
// We accept the loose shape: requests, responses, and notifications all share
// it, distinguished by which fields are set.
type rawRPC struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *rpcError) Error() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf("rpc error %d: %s", e.Code, e.Message)
}

// acpClient speaks ACP to a subprocess over its stdin/stdout. Notifications
// (session/update, etc.) are forwarded to onNotify so the caller can log/parse
// agent activity in real time without blocking the request/response path.
type acpClient struct {
	w        io.Writer
	r        *bufio.Reader
	writeMu  sync.Mutex
	id       atomic.Int64
	pendMu   sync.Mutex
	pending  map[string]chan rawRPC
	onNotify func(method string, params json.RawMessage)
	done     chan struct{}
	readErr  atomic.Pointer[error]
}

func newACPClient(w io.Writer, r io.Reader, onNotify func(string, json.RawMessage)) *acpClient {
	c := &acpClient{
		w:        w,
		r:        bufio.NewReaderSize(r, 1<<20),
		pending:  make(map[string]chan rawRPC),
		onNotify: onNotify,
		done:     make(chan struct{}),
	}
	go c.readLoop()
	return c
}

// readLoop demultiplexes incoming lines. Responses (have ID) wake the matching
// pending caller; notifications (have Method but no ID) go to onNotify.
func (c *acpClient) readLoop() {
	defer close(c.done)
	// Match codehalter's framing: 16MB line cap, since tool output dumps can
	// occasionally exceed bufio's 64KB default before truncation kicks in.
	scanner := bufio.NewScanner(c.r)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		var msg rawRPC
		if err := json.Unmarshal(line, &msg); err != nil {
			continue
		}
		if len(msg.ID) > 0 && (msg.Result != nil || msg.Error != nil) {
			key := string(msg.ID)
			c.pendMu.Lock()
			ch, ok := c.pending[key]
			delete(c.pending, key)
			c.pendMu.Unlock()
			if ok {
				ch <- msg
			}
			continue
		}
		if msg.Method != "" && len(msg.ID) == 0 {
			if c.onNotify != nil {
				c.onNotify(msg.Method, msg.Params)
			}
			continue
		}
		// Server-issued request: codehalter delegates filesystem reads/writes
		// to the client via fs/read_text_file and fs/write_text_file (the Zed
		// integration path). Bench has no editor and no unsaved-buffer state,
		// so we just hit disk and return. Dispatching in a goroutine keeps the
		// readLoop free for the response that the request handler will send.
		if msg.Method != "" && len(msg.ID) > 0 {
			go c.handleServerRequest(msg)
		}
	}
	if err := scanner.Err(); err != nil {
		c.readErr.Store(&err)
	}
}

// handleServerRequest answers an inbound JSON-RPC request from codehalter.
// Unknown methods get JSON-RPC -32601 ("method not found") so the agent
// surfaces a clear error instead of blocking on a never-arriving response.
func (c *acpClient) handleServerRequest(msg rawRPC) {
	var result any
	var rpcErr *rpcError
	switch msg.Method {
	case "fs/read_text_file":
		result, rpcErr = handleFSRead(msg.Params)
	case "fs/write_text_file":
		result, rpcErr = handleFSWrite(msg.Params)
	default:
		rpcErr = &rpcError{Code: -32601, Message: "method not found: " + msg.Method}
	}

	resp := rawRPC{JSONRPC: "2.0", ID: msg.ID, Error: rpcErr}
	if rpcErr == nil {
		// Always marshal a result, even for empty struct{} writes — RawMessage
		// with omitempty drops a nil value, which would leave the response with
		// neither result nor error and confuse codehalter's matcher.
		b, err := json.Marshal(result)
		if err != nil {
			resp.Error = &rpcError{Code: -32603, Message: "marshal result: " + err.Error()}
		} else {
			resp.Result = b
		}
	}

	body, err := json.Marshal(resp)
	if err != nil {
		return
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	_, _ = c.w.Write(append(body, '\n'))
}

// handleFSRead reads a file and applies the optional 1-indexed line/limit
// window — mirrors codehalter's directRead so the subagent and main-session
// paths return the same shape for the same args.
func handleFSRead(params json.RawMessage) (any, *rpcError) {
	var p struct {
		Path  string `json:"path"`
		Line  *int   `json:"line,omitempty"`
		Limit *int   `json:"limit,omitempty"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &rpcError{Code: -32602, Message: "invalid params: " + err.Error()}
	}
	data, err := os.ReadFile(p.Path)
	if err != nil {
		return nil, &rpcError{Code: -32603, Message: err.Error()}
	}
	content := string(data)
	if p.Line != nil || p.Limit != nil {
		lines := strings.SplitAfter(content, "\n")
		start := 0
		if p.Line != nil && *p.Line > 0 {
			start = *p.Line - 1
		}
		if start >= len(lines) {
			content = ""
		} else {
			end := len(lines)
			if p.Limit != nil && *p.Limit > 0 && start+*p.Limit < end {
				end = start + *p.Limit
			}
			content = strings.Join(lines[start:end], "")
		}
	}
	return struct {
		Content string `json:"content"`
	}{Content: content}, nil
}

// handleFSWrite writes a file directly. No diff/approval UI exists in bench;
// codehalter's `interactive` vs `autopilot` mode gating runs entirely on the
// server side, and bench has already flipped the session into autopilot.
func handleFSWrite(params json.RawMessage) (any, *rpcError) {
	var p struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &rpcError{Code: -32602, Message: "invalid params: " + err.Error()}
	}
	if err := os.WriteFile(p.Path, []byte(p.Content), 0o644); err != nil {
		return nil, &rpcError{Code: -32603, Message: err.Error()}
	}
	return struct{}{}, nil
}

// call sends a request and blocks until the matching response arrives.
// Cancelling ctx unblocks the caller but does NOT cancel the server-side
// work — for that the caller should send a session/cancel notification.
func (c *acpClient) call(ctx context.Context, method string, params, out any) error {
	id := c.id.Add(1)
	idJSON, _ := json.Marshal(id)
	var paramsJSON json.RawMessage
	if params != nil {
		b, err := json.Marshal(params)
		if err != nil {
			return fmt.Errorf("marshal params: %w", err)
		}
		paramsJSON = b
	}
	req := rawRPC{JSONRPC: "2.0", ID: idJSON, Method: method, Params: paramsJSON}
	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	ch := make(chan rawRPC, 1)
	key := string(idJSON)
	c.pendMu.Lock()
	c.pending[key] = ch
	c.pendMu.Unlock()

	c.writeMu.Lock()
	_, werr := c.w.Write(append(body, '\n'))
	c.writeMu.Unlock()
	if werr != nil {
		c.pendMu.Lock()
		delete(c.pending, key)
		c.pendMu.Unlock()
		return fmt.Errorf("write: %w", werr)
	}

	select {
	case <-ctx.Done():
		c.pendMu.Lock()
		delete(c.pending, key)
		c.pendMu.Unlock()
		return ctx.Err()
	case resp := <-ch:
		if resp.Error != nil {
			return resp.Error
		}
		if out != nil && len(resp.Result) > 0 {
			return json.Unmarshal(resp.Result, out)
		}
		return nil
	}
}

// notify sends a fire-and-forget notification (no ID, no response expected).
func (c *acpClient) notify(method string, params any) error {
	var paramsJSON json.RawMessage
	if params != nil {
		b, err := json.Marshal(params)
		if err != nil {
			return err
		}
		paramsJSON = b
	}
	body, err := json.Marshal(rawRPC{JSONRPC: "2.0", Method: method, Params: paramsJSON})
	if err != nil {
		return err
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	_, werr := c.w.Write(append(body, '\n'))
	return werr
}
