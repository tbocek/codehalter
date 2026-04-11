// Generic JSON-RPC 2.0 transport over line-delimited I/O.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"sync"
)

const (
	initialBufSize = 64 * 1024        // 64KB
	maxBufSize     = 16 * 1024 * 1024 // 16MB
)

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 message types
// ---------------------------------------------------------------------------

type jsonrpcRequest struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Method  string           `json:"method"`
	Params  json.RawMessage  `json:"params,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id"`
	Result  interface{} `json:"result,omitempty"`
	Error   *rpcError   `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// MethodHandler is called for each incoming JSON-RPC message.
type MethodHandler func(ctx context.Context, req *jsonrpcRequest)

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

// Connection is a generic line-delimited JSON-RPC 2.0 transport.
type Connection struct {
	w        io.Writer
	r        io.Reader
	handler  MethodHandler
	done     chan struct{}
	writeMu  sync.Mutex
	log      *slog.Logger
	nextID   uint64
	pendingMu sync.Mutex
	pending  map[string]chan json.RawMessage
}

// NewConnection creates a connection that reads JSON-RPC messages from r and
// writes to w. The handler is called for each incoming message. Processing
// starts in a background goroutine.
func NewConnection(w io.Writer, r io.Reader, handler MethodHandler, log *slog.Logger) *Connection {
	c := &Connection{
		w:       w,
		r:       r,
		handler: handler,
		done:    make(chan struct{}),
		log:     log,
		pending: make(map[string]chan json.RawMessage),
	}
	go c.serve()
	return c
}

// Done returns a channel that is closed when the connection ends.
func (c *Connection) Done() <-chan struct{} {
	return c.done
}

// SendNotification sends a JSON-RPC notification (no id, no response expected).
func (c *Connection) SendNotification(method string, params interface{}) error {
	c.log.Debug("sending notification", "method", method)
	msg := jsonrpcRequest{
		JSONRPC: "2.0",
		Method:  method,
	}
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	msg.Params = raw
	return c.writeMessage(msg)
}

// SendResult sends a successful JSON-RPC response.
func (c *Connection) SendResult(id *json.RawMessage, result interface{}) {
	resp := jsonrpcResponse{JSONRPC: "2.0", ID: id, Result: result}
	_ = c.writeMessage(resp)
}

// SendError sends a JSON-RPC error response.
func (c *Connection) SendError(id *json.RawMessage, code int, message string) {
	resp := jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &rpcError{Code: code, Message: message},
	}
	_ = c.writeMessage(resp)
}

func (c *Connection) writeMessage(msg interface{}) error {
	b, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	c.log.Debug("writing", "msg", string(b))
	b = append(b, '\n')
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	_, err = c.w.Write(b)
	return err
}

// SendRequest sends a JSON-RPC request and waits for the response.
func SendRequest[T any](c *Connection, ctx context.Context, method string, params any) (T, error) {
	var zero T

	c.pendingMu.Lock()
	c.nextID++
	id := fmt.Sprintf("%d", c.nextID)
	ch := make(chan json.RawMessage, 1)
	c.pending[id] = ch
	c.pendingMu.Unlock()

	idRaw := json.RawMessage(`"` + id + `"`)
	msg := jsonrpcRequest{JSONRPC: "2.0", ID: &idRaw, Method: method}
	if params != nil {
		raw, err := json.Marshal(params)
		if err != nil {
			return zero, err
		}
		msg.Params = raw
	}

	c.log.Debug("sending request", "method", method, "id", id)
	if err := c.writeMessage(msg); err != nil {
		return zero, err
	}

	select {
	case <-ctx.Done():
		c.pendingMu.Lock()
		delete(c.pending, id)
		c.pendingMu.Unlock()
		return zero, ctx.Err()
	case raw := <-ch:
		// Check for error response.
		var errResp struct {
			Error *rpcError `json:"error"`
		}
		_ = json.Unmarshal(raw, &errResp)
		if errResp.Error != nil {
			return zero, fmt.Errorf("rpc error %d: %s", errResp.Error.Code, errResp.Error.Message)
		}
		var resp struct {
			Result T `json:"result"`
		}
		if err := json.Unmarshal(raw, &resp); err != nil {
			return zero, err
		}
		return resp.Result, nil
	}
}

func (c *Connection) serve() {
	defer close(c.done)

	scanner := bufio.NewScanner(c.r)
	buf := make([]byte, 0, initialBufSize)
	scanner.Buffer(buf, maxBufSize)

	ctx := context.Background()

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// Check if this is a response (has result or error, no method).
		var probe struct {
			ID     *json.RawMessage `json:"id"`
			Method string           `json:"method"`
			Result *json.RawMessage `json:"result"`
			Error  *json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			c.log.Warn("failed to parse message", "error", err)
			continue
		}

		// Response to our outgoing request.
		if probe.Method == "" && probe.ID != nil {
			id := string(*probe.ID)
			// Strip quotes if present.
			if len(id) >= 2 && id[0] == '"' {
				id = id[1 : len(id)-1]
			}
			c.pendingMu.Lock()
			ch, ok := c.pending[id]
			if ok {
				delete(c.pending, id)
			}
			c.pendingMu.Unlock()
			if ok {
				ch <- line
			}
			continue
		}

		// Incoming request/notification.
		var req jsonrpcRequest
		if err := json.Unmarshal(line, &req); err != nil {
			c.log.Warn("failed to parse message", "error", err)
			continue
		}

		c.log.Debug("received", "method", req.Method, "raw", string(line))
		if req.Method == "session/prompt" {
			go c.handler(ctx, &req)
		} else {
			c.handler(ctx, &req)
		}
	}
}

// Dispatch is a generic helper that unmarshals params, calls the handler, and
// sends the JSON-RPC result or error.
func Dispatch[Req any, Resp any](
	c *Connection,
	ctx context.Context,
	req *jsonrpcRequest,
	handler func(context.Context, Req) (Resp, error),
) {
	var params Req
	if req.Params != nil {
		if err := json.Unmarshal(req.Params, &params); err != nil {
			if req.ID != nil {
				c.SendError(req.ID, -32602, fmt.Sprintf("invalid params: %v", err))
			}
			return
		}
	}
	result, err := handler(ctx, params)
	if err != nil {
		c.log.Error("handler failed", "method", req.Method, "error", err)
		if req.ID != nil {
			c.SendError(req.ID, -32000, err.Error())
		}
		return
	}
	c.log.Debug("handler ok", "method", req.Method)
	if req.ID != nil {
		c.SendResult(req.ID, result)
	}
}
