package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
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

// setSessionModeRequest flips the session into "autopilot" so codehalter
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
		}
	}
	if err := scanner.Err(); err != nil {
		c.readErr.Store(&err)
	}
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
