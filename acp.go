package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
)

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 message envelopes used over the ACP line protocol.
// ---------------------------------------------------------------------------

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
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// ACP wire types
// ---------------------------------------------------------------------------

// Session-update content-chunk kinds.
const (
	protocolVersion  = 1
	KindAgentMessage = "agent_message_chunk"
	// KindAgentThought is reasoning_content from thinking models (Qwen3,
	// DeepSeek-R1, GPT-OSS) — Zed renders it greyed/collapsible so
	// deliberation doesn't blur with visible output.
	KindAgentThought = "agent_thought_chunk"
	KindUserMessage  = "user_message_chunk"
)

type (
	InitializeRequest struct {
		ProtocolVersion int `json:"protocolVersion"`
	}
	InitializeResponse struct {
		ProtocolVersion   int `json:"protocolVersion"`
		AgentCapabilities struct {
			LoadSession        bool `json:"loadSession"`
			PromptCapabilities struct {
				Image bool `json:"image,omitempty"`
			} `json:"promptCapabilities"`
			SessionCapabilities *struct {
				List  *struct{} `json:"list,omitempty"`
				Close *struct{} `json:"close,omitempty"`
			} `json:"sessionCapabilities,omitempty"`
		} `json:"agentCapabilities"`
		AgentInfo   any      `json:"agentInfo,omitempty"`
		AuthMethods []string `json:"authMethods"`
	}

	NewSessionRequest struct {
		Cwd string `json:"cwd,omitempty"`
	}
	NewSessionResponse struct {
		SessionId string            `json:"sessionId"`
		Modes     *SessionModeState `json:"modes,omitempty"`
	}

	SetSessionModeRequest struct {
		SessionId string `json:"sessionId"`
		ModeId    string `json:"modeId"`
	}

	LoadSessionRequest struct {
		SessionId string `json:"sessionId"`
		Cwd       string `json:"cwd"`
	}
	LoadSessionResponse struct {
		SessionId string            `json:"sessionId,omitempty"`
		Modes     *SessionModeState `json:"modes,omitempty"`
	}

	ListSessionsRequest struct {
		Cwd    string `json:"cwd,omitempty"`
		Cursor string `json:"cursor,omitempty"`
	}
	ListSessionsResponse struct {
		Sessions   []SessionInfo `json:"sessions"`
		NextCursor string        `json:"nextCursor,omitempty"`
	}

	CloseSessionRequest struct {
		SessionId string `json:"sessionId"`
	}

	CancelNotification struct {
		SessionId string `json:"sessionId"`
	}

	PromptRequest struct {
		SessionId string         `json:"sessionId"`
		Content   []ContentBlock `json:"prompt"`
	}
	PromptResponse struct {
		StopReason string `json:"stopReason,omitempty"`
	}
)

type SessionModeState struct {
	CurrentModeId  string `json:"currentModeId"`
	AvailableModes []struct {
		Id          string `json:"id"`
		Name        string `json:"name"`
		Description string `json:"description,omitempty"`
	} `json:"availableModes"`
}

// ContentBlock is the ACP wire shape for a prompt/response block: a "type"
// discriminator plus per-variant fields. Flat so encoding/json handles it
// without custom Marshal/Unmarshal.
type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
	Data     string `json:"data,omitempty"` // base64-encoded image bytes
}

type messageChunk struct {
	Kind    string       `json:"sessionUpdate"`
	Content ContentBlock `json:"content"`
}

type PlanEntry struct {
	Content  string `json:"content"`
	Priority string `json:"priority"`
	Status   string `json:"status"`
}

type planUpdate struct {
	Kind    string      `json:"sessionUpdate"`
	Entries []PlanEntry `json:"entries"`
}

// ---------------------------------------------------------------------------
// AgentSideConnection — JSON-RPC dispatch + outgoing-request demux on a
// line-delimited stdio pair. The agent type is concrete (there is only one).
// ---------------------------------------------------------------------------

type AgentSideConnection struct {
	w       io.Writer
	r       io.Reader
	writeMu sync.Mutex
	agent   *agent

	done chan struct{}

	nextID    atomic.Uint64
	pendingMu sync.Mutex
	pending   map[string]chan json.RawMessage
}

func NewAgentSideConnection(a *agent, w io.Writer, r io.Reader) *AgentSideConnection {
	c := &AgentSideConnection{
		w:       w,
		r:       r,
		agent:   a,
		done:    make(chan struct{}),
		pending: make(map[string]chan json.RawMessage),
	}
	go c.serve()
	return c
}

// writeMessage emits one JSON object followed by '\n'. Writes are serialised
// so two concurrent writers can't interleave halves of a JSON object on the
// wire.
func (a *AgentSideConnection) writeMessage(msg any) error {
	b, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	slog.Debug("writing", "msg", string(b))
	b = append(b, '\n')
	a.writeMu.Lock()
	defer a.writeMu.Unlock()
	_, err = a.w.Write(b)
	return err
}

func (a *AgentSideConnection) Done() <-chan struct{} { return a.done }

// --- outbound (codehalter → Zed) ---

// SessionUpdate sends one agent->client JSON-RPC notification (no ID, no
// response expected). `update` is any of the "sessionUpdate"-keyed wire shapes
// (message chunks, plan updates, tool-call cards) — chosen at the call site.
func (a *AgentSideConnection) SessionUpdate(ctx context.Context, sid string, update any) error {
	raw, err := json.Marshal(struct {
		SessionId string `json:"sessionId"`
		Update    any    `json:"update"`
	}{sid, update})
	if err != nil {
		return err
	}
	return a.writeMessage(jsonrpcRequest{
		JSONRPC: "2.0",
		Method:  "session/update",
		Params:  raw,
	})
}

// sendRequest writes a JSON-RPC request and blocks until the matching
// response arrives. Returns the raw `result` bytes — callers unmarshal into
// whatever shape they expect.
func (a *AgentSideConnection) sendRequest(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := a.nextID.Add(1)
	idStr := fmt.Sprintf("%d", id)
	idRaw := json.RawMessage(`"` + idStr + `"`)

	ch := make(chan json.RawMessage, 1)
	a.pendingMu.Lock()
	a.pending[idStr] = ch
	a.pendingMu.Unlock()
	defer func() {
		a.pendingMu.Lock()
		delete(a.pending, idStr)
		a.pendingMu.Unlock()
	}()

	var raw json.RawMessage
	if params != nil {
		b, err := json.Marshal(params)
		if err != nil {
			return nil, err
		}
		raw = b
	}
	if err := a.writeMessage(jsonrpcRequest{
		JSONRPC: "2.0",
		ID:      &idRaw,
		Method:  method,
		Params:  raw,
	}); err != nil {
		return nil, err
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case line := <-ch:
		var resp struct {
			Result json.RawMessage `json:"result"`
			Error  *struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
				Data    string `json:"data,omitempty"`
			} `json:"error"`
		}
		if err := json.Unmarshal(line, &resp); err != nil {
			return nil, err
		}
		if resp.Error != nil {
			if resp.Error.Data != "" {
				return nil, fmt.Errorf("rpc error %d: %s: %s", resp.Error.Code, resp.Error.Message, resp.Error.Data)
			}
			return nil, fmt.Errorf("rpc error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		return resp.Result, nil
	}
}

// --- inbound (Zed → codehalter) ---

// serve reads one JSON object per line forever. Responses to our outgoing
// requests route to the pending map; everything else lands in handle().
// ReadString is used (not bufio.Scanner) so a large MCP tool response can't
// trip the scanner's MaxScanTokenSize cap.
func (a *AgentSideConnection) serve() {
	defer close(a.done)
	ctx := context.Background()
	br := bufio.NewReader(a.r)
	for {
		s, err := br.ReadString('\n')
		if err != nil {
			if !errors.Is(err, io.EOF) {
				slog.Debug("read error", "error", err)
			}
			return
		}
		s = strings.TrimRight(s, "\r\n")
		if s == "" {
			continue
		}
		line := []byte(s)

		var probe struct {
			ID     *json.RawMessage `json:"id"`
			Method string           `json:"method"`
			Result *json.RawMessage `json:"result"`
			Error  *json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			slog.Warn("failed to parse message", "error", err)
			continue
		}

		// Response to one of our outgoing requests.
		if probe.Method == "" && probe.ID != nil {
			id := string(*probe.ID)
			if len(id) >= 2 && id[0] == '"' {
				id = id[1 : len(id)-1]
			}
			a.pendingMu.Lock()
			ch, ok := a.pending[id]
			if ok {
				delete(a.pending, id)
			}
			a.pendingMu.Unlock()
			if ok {
				ch <- line
			}
			continue
		}

		// Incoming request or notification.
		var req jsonrpcRequest
		if err := json.Unmarshal(line, &req); err != nil {
			slog.Warn("failed to parse message", "error", err)
			continue
		}
		slog.Debug("received", "method", req.Method, "raw", string(line))

		// Must be async: handlers issue outbound sendRequest calls whose
		// responses come back through this same read loop. Running handle
		// inline would block the loop and deadlock the response routing.
		go a.handle(ctx, &req)
	}
}

func (a *AgentSideConnection) handle(ctx context.Context, req *jsonrpcRequest) {
	switch req.Method {
	case "initialize":
		var p InitializeRequest
		if !a.decodeParams(req, &p) {
			return
		}
		res, err := a.agent.Initialize(ctx, p)
		a.reply(req, res, err)

	case "authenticate":
		err := a.agent.Authenticate(ctx)
		a.reply(req, struct{}{}, err)

	case "session/new":
		var p NewSessionRequest
		if !a.decodeParams(req, &p) {
			return
		}
		res, err := a.agent.NewSession(ctx, p)
		a.reply(req, res, err)

	case "session/load":
		var p LoadSessionRequest
		if !a.decodeParams(req, &p) {
			return
		}
		res, err := a.agent.LoadSession(ctx, p)
		a.reply(req, res, err)

	case "session/list":
		var p ListSessionsRequest
		if !a.decodeParams(req, &p) {
			return
		}
		res, err := a.agent.ListSessions(ctx, p)
		a.reply(req, res, err)

	case "session/set_mode":
		var p SetSessionModeRequest
		if !a.decodeParams(req, &p) {
			return
		}
		err := a.agent.SetSessionMode(ctx, p)
		a.reply(req, struct{}{}, err)

	case "session/close":
		var p CloseSessionRequest
		if !a.decodeParams(req, &p) {
			return
		}
		err := a.agent.CloseSession(ctx, p)
		a.reply(req, struct{}{}, err)

	case "session/prompt":
		var p PromptRequest
		if !a.decodeParams(req, &p) {
			return
		}
		res, err := a.agent.Prompt(ctx, p)
		a.reply(req, res, err)

	case "session/cancel":
		var p CancelNotification
		if req.Params != nil {
			_ = json.Unmarshal(req.Params, &p)
		}
		a.agent.Cancel(ctx, p)

	default:
		if req.ID != nil {
			a.replyError(req.ID, -32601, fmt.Sprintf("method not found: %s", req.Method))
		}
	}
}

// decodeParams unmarshals req.Params into dst. Returns false (and writes a
// -32602 "invalid params" error) when the bytes won't decode. nil params is
// allowed and treated as an empty object.
func (a *AgentSideConnection) decodeParams(req *jsonrpcRequest, dst any) bool {
	if req.Params == nil {
		return true
	}
	if err := json.Unmarshal(req.Params, dst); err != nil {
		if req.ID != nil {
			a.replyError(req.ID, -32602, fmt.Sprintf("invalid params: %v", err))
		}
		return false
	}
	return true
}

// reply finishes a handler: emit result on success, -32603 error otherwise.
// Skips entirely when req.ID is nil (notification).
func (a *AgentSideConnection) reply(req *jsonrpcRequest, result any, err error) {
	if err != nil {
		slog.Error("handler failed", "method", req.Method, "error", err)
		if req.ID != nil {
			a.replyError(req.ID, -32603, err.Error())
		}
		return
	}
	if req.ID == nil {
		return
	}
	if werr := a.writeMessage(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: result}); werr != nil {
		slog.Warn("write reply failed", "method", req.Method, "error", werr)
	}
}

// replyError emits a JSON-RPC error response. We deliberately avoid -32000:
// ACP reserves it for AUTH_REQUIRED, so using it for generic handler failures
// makes Zed render a misleading "Authentication Required" red box (with an
// Authenticate button) for unrelated problems — e.g. an LLM stream cancelled
// mid-flight by the user.
func (a *AgentSideConnection) replyError(id *json.RawMessage, code int, message string) {
	if err := a.writeMessage(jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error: &struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		}{code, message},
	}); err != nil {
		slog.Warn("write error reply failed", "code", code, "error", err)
	}
}
