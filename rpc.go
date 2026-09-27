package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

type jsonrpcRequest struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Method  string           `json:"method"`
	Params  json.RawMessage  `json:"params,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id"`
	Result  any              `json:"result,omitempty"`
	Error   *rpcError        `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    string `json:"data,omitempty"`
}

func (e *rpcError) Error() string {
	if e.Data != "" {
		return fmt.Sprintf("rpc error %d: %s: %s", e.Code, e.Message, e.Data)
	}
	return fmt.Sprintf("rpc error %d: %s", e.Code, e.Message)
}

var errRPCClosed = errors.New("connection closed")

// rpcPeer is one end of a line-delimited JSON-RPC 2.0 stream; acp.go and cli.go each embed one.
type rpcPeer struct {
	w       io.Writer
	writeMu sync.Mutex

	// logPrefix tells the two peers apart in the one log --cli writes.
	logPrefix string
	// trace logs traffic; only the agent sets it, which in cli.log already covers both directions.
	trace bool

	done chan struct{}

	nextID    atomic.Uint64
	pendingMu sync.Mutex
	pending   map[string]chan json.RawMessage

	// Keyed by the raw id bytes: a peer may send 3 or "3", and only echoing the
	// bytes we got matches its $/cancel_request.
	inflightMu sync.Mutex
	inflight   map[string]*inflightRequest
}

// answered lets the agent reply -32800 itself without the handler's late reply becoming a second response.
type inflightRequest struct {
	cancel   context.CancelFunc
	answered atomic.Bool
}

func newRPCPeer(w io.Writer, logPrefix string, trace bool) *rpcPeer {
	return &rpcPeer{
		w:         w,
		logPrefix: logPrefix,
		trace:     trace,
		done:      make(chan struct{}),
		pending:   map[string]chan json.RawMessage{},
		inflight:  map[string]*inflightRequest{},
	}
}

func (p *rpcPeer) Done() <-chan struct{} { return p.done }

func (p *rpcPeer) write(msg any) error {
	b, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	if p.trace {
		slog.Debug("writing", "msg", string(b))
	}
	b = append(b, '\n')
	p.writeMu.Lock()
	defer p.writeMu.Unlock()
	_, err = p.w.Write(b)
	return err
}

// send writes a notification when id is nil; nil params leave the field out.
func (p *rpcPeer) send(id *json.RawMessage, method string, params any) error {
	var raw json.RawMessage
	if params != nil {
		b, err := json.Marshal(params)
		if err != nil {
			return err
		}
		raw = b
	}
	return p.write(jsonrpcRequest{JSONRPC: "2.0", ID: id, Method: method, Params: raw})
}

func (p *rpcPeer) notify(method string, params any) error {
	return p.send(nil, method, params)
}

// sendRequest's errors: *rpcError for an error response, ctx.Err(), or errRPCClosed after EOF.
func (p *rpcPeer) sendRequest(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := strconv.FormatUint(p.nextID.Add(1), 10)
	idRaw := json.RawMessage(`"` + id + `"`)

	ch := make(chan json.RawMessage, 1)
	p.pendingMu.Lock()
	p.pending[id] = ch
	p.pendingMu.Unlock()
	defer func() {
		p.pendingMu.Lock()
		delete(p.pending, id)
		p.pendingMu.Unlock()
	}()

	if err := p.send(&idRaw, method, params); err != nil {
		return nil, err
	}

	var line json.RawMessage
	select {
	case <-ctx.Done():
		// Otherwise a permission or elicitation dialog stays open forever.
		// Best-effort: a peer may ignore $/cancel_request.
		if err := p.notify("$/cancel_request", struct {
			RequestId json.RawMessage `json:"requestId"`
		}{idRaw}); err != nil {
			slog.Debug(p.logPrefix+"$/cancel_request: write failed", "id", id, "err", err)
		}
		return nil, ctx.Err()
	case line = <-ch:
	case <-p.done:
		// serve routes a response before it can see EOF, so one that made it is already in ch.
		select {
		case line = <-ch:
		default:
			return nil, errRPCClosed
		}
	}
	var resp struct {
		Result json.RawMessage `json:"result"`
		Error  *rpcError       `json:"error"`
	}
	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, resp.Error
	}
	return resp.Result, nil
}

// serve closes done at EOF; dispatch runs on the read loop, so it must not block.
// ReadString, not bufio.Scanner: a large message would trip MaxScanTokenSize.
func (p *rpcPeer) serve(r io.Reader, dispatch func(*jsonrpcRequest)) {
	defer close(p.done)
	br := bufio.NewReader(r)
	for {
		s, err := br.ReadString('\n')
		if err != nil {
			if !errors.Is(err, io.EOF) {
				slog.Debug(p.logPrefix+"read error", "err", err)
			}
			return
		}
		s = strings.TrimRight(s, "\r\n")
		if s == "" {
			continue
		}
		line := []byte(s)

		// One decode serves both shapes: a response carries an id and no method.
		var req jsonrpcRequest
		if err := json.Unmarshal(line, &req); err != nil {
			slog.Warn(p.logPrefix+"failed to parse message", "err", err)
			continue
		}

		if req.Method == "" && req.ID != nil {
			id := string(*req.ID)
			if len(id) >= 2 && id[0] == '"' {
				id = id[1 : len(id)-1]
			}
			p.pendingMu.Lock()
			ch, ok := p.pending[id]
			if ok {
				delete(p.pending, id)
			}
			p.pendingMu.Unlock()
			if ok {
				ch <- line
			}
			continue
		}

		if p.trace {
			slog.Debug("received", "method", req.Method, "raw", s)
		}
		dispatch(&req)
	}
}

// track lets $/cancel_request reach an incoming request's ctx until the returned func runs.
func (p *rpcPeer) track(ctx context.Context, id *json.RawMessage) (context.Context, func()) {
	if id == nil {
		return ctx, func() {}
	}
	key := string(*id)
	entry := &inflightRequest{}
	ctx, entry.cancel = context.WithCancel(ctx)
	p.inflightMu.Lock()
	p.inflight[key] = entry
	p.inflightMu.Unlock()
	return ctx, func() {
		p.inflightMu.Lock()
		delete(p.inflight, key)
		p.inflightMu.Unlock()
		entry.cancel()
	}
}

// cancelInflight returns nil for a request that is not running, which the spec says to ignore.
func (p *rpcPeer) cancelInflight(params json.RawMessage) (*inflightRequest, json.RawMessage) {
	var c struct {
		RequestId json.RawMessage `json:"requestId"`
	}
	if params != nil {
		if err := json.Unmarshal(params, &c); err != nil {
			slog.Debug(p.logPrefix+"$/cancel_request: malformed params", "err", err)
			return nil, nil
		}
	}
	if len(c.RequestId) == 0 {
		return nil, nil
	}
	key := string(c.RequestId)
	p.inflightMu.Lock()
	entry := p.inflight[key]
	p.inflightMu.Unlock()
	if entry == nil {
		// Normal race: the handler finished first.
		slog.Debug(p.logPrefix+"$/cancel_request: no such in-flight request", "requestId", key)
		return nil, nil
	}
	entry.cancel()
	return entry, c.RequestId
}

// writeResult and writeError do no claim check: the caller owns the one reply to id.
func (p *rpcPeer) writeResult(id *json.RawMessage, result any) {
	if err := p.write(jsonrpcResponse{JSONRPC: "2.0", ID: id, Result: result}); err != nil {
		slog.Warn(p.logPrefix+"write reply failed", "err", err)
	}
}

func (p *rpcPeer) writeError(id *json.RawMessage, code int, message string) {
	if err := p.write(jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &rpcError{Code: code, Message: message},
	}); err != nil {
		slog.Warn(p.logPrefix+"write error reply failed", "code", code, "err", err)
	}
}
