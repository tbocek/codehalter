package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
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

const (
	// protocolVersion is the ACP wire version we speak. Bumping breaks Zed
	// clients pinned to the old number — keep at 1 unless Zed's side moves.
	protocolVersion = 1
	// ACP session_update kinds for streamed content. agent_thought_chunk is
	// reasoning_content from thinking models (Qwen3, DeepSeek-R1, GPT-OSS) —
	// clients render it greyed/collapsible so deliberation doesn't blur with
	// visible output.
	KindAgentMessage = "agent_message_chunk"
	KindAgentThought = "agent_thought_chunk"
	KindUserMessage  = "user_message_chunk"
)

type Implementation struct {
	Name    string `json:"name,omitempty"`
	Version string `json:"version,omitempty"`
}

type (
	InitializeRequest struct {
		ProtocolVersion int `json:"protocolVersion"`
	}
	// SessionCapabilities advertises list/close — each non-nil *struct{}
	// serialises as `{}` on the wire, the shape ACP defines for
	// "this capability is supported".
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
		AgentInfo   *Implementation `json:"agentInfo,omitempty"`
		AuthMethods []string        `json:"authMethods"`
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

// Session modes.

type SessionModeState struct {
	CurrentModeId  string `json:"currentModeId"`
	AvailableModes []struct {
		Id          string `json:"id"`
		Name        string `json:"name"`
		Description string `json:"description,omitempty"`
	} `json:"availableModes"`
}

// ContentBlock is the ACP wire shape for one prompt/response block: a flat
// object with a "type" discriminator and per-variant fields. Type is the
// only field always populated; the rest are set per kind. Going flat lets
// encoding/json handle marshal/unmarshal directly — no custom methods.
type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
	Data     string `json:"data,omitempty"` // base64-encoded image bytes
}

func TextBlock(text string) ContentBlock {
	return ContentBlock{Type: "text", Text: text}
}

func ImageBlock(mimeType, data string) ContentBlock {
	return ContentBlock{Type: "image", MimeType: mimeType, Data: data}
}

// Session update (agent -> client notification).

// SessionNotification.Update is `any` because it's a discriminated union on
// the wire (keyed by "sessionUpdate"): message chunks, plan updates, and
// tool-call cards share one envelope and the right concrete struct is
// picked at the call site.
type SessionNotification struct {
	SessionId string `json:"sessionId"`
	Update    any    `json:"update"`
}

type messageChunk struct {
	Kind    string       `json:"sessionUpdate"`
	Content ContentBlock `json:"content"`
}

// Plan update — shown by the client as a checklist.

type PlanEntry struct {
	Content  string `json:"content"`
	Priority string `json:"priority"`
	Status   string `json:"status"`
}

type planUpdate struct {
	Kind    string      `json:"sessionUpdate"`
	Entries []PlanEntry `json:"entries"`
}

func PlanUpdate(entries []PlanEntry) any {
	return planUpdate{Kind: "plan", Entries: entries}
}

// Current-mode update — informs the client UI that the active mode changed.
// Field is "modeId" per ACP spec, distinct from "currentModeId" inside
// SessionModeState advertised at session create/load.

type currentModeUpdate struct {
	Kind   string `json:"sessionUpdate"`
	ModeId string `json:"modeId"`
}

func CurrentModeUpdate(modeId string) any {
	return currentModeUpdate{Kind: "current_mode_update", ModeId: modeId}
}

// ---------------------------------------------------------------------------
// AgentSideConnection — JSON-RPC dispatch + outgoing-request demux. Owns the
// line protocol; the agent type is concrete (there is only one).
// ---------------------------------------------------------------------------

// AgentSideConnection routes ACP methods between the codehalter agent and a
// connected ACP client. There is exactly one agent implementation, so the
// previous Agent interface was abstraction without value and has been dropped.
type AgentSideConnection struct {
	proto *lineProtocol
	agent *agent
	log   *slog.Logger

	done chan struct{}

	nextID    atomic.Uint64
	pendingMu sync.Mutex
	pending   map[string]chan json.RawMessage
}

func NewAgentSideConnection(a *agent, w io.Writer, r io.Reader, log *slog.Logger) *AgentSideConnection {
	c := &AgentSideConnection{
		proto:   newLineProtocol(w, r, log),
		agent:   a,
		log:     log,
		done:    make(chan struct{}),
		pending: make(map[string]chan json.RawMessage),
	}
	go c.serve()
	return c
}

func (a *AgentSideConnection) Done() <-chan struct{} { return a.done }

func (a *AgentSideConnection) SessionUpdate(ctx context.Context, n SessionNotification) error {
	return a.sendNotification("session/update", n)
}

// sendNotification writes a JSON-RPC notification (no ID, no response expected).
func (a *AgentSideConnection) sendNotification(method string, params any) error {
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return a.proto.writeMessage(jsonrpcRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  raw,
	})
}

// sendRequest writes a JSON-RPC request and blocks until the matching
// response arrives. Returns the raw `result` bytes — callers unmarshal into
// whatever shape they expect so this is a single non-generic entry point.
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
	if err := a.proto.writeMessage(jsonrpcRequest{
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
			} `json:"error"`
		}
		if err := json.Unmarshal(line, &resp); err != nil {
			return nil, err
		}
		if resp.Error != nil {
			return nil, fmt.Errorf("rpc error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		return resp.Result, nil
	}
}

func (a *AgentSideConnection) sendResult(id *json.RawMessage, result any) {
	_ = a.proto.writeMessage(jsonrpcResponse{JSONRPC: "2.0", ID: id, Result: result})
}

// sendError emits a JSON-RPC error response. We deliberately avoid -32000:
// ACP reserves it for AUTH_REQUIRED, so using it for generic handler failures
// makes Zed render a misleading "Authentication Required" red box (with an
// Authenticate button) for unrelated problems — e.g. an LLM stream cancelled
// mid-flight by the user.
func (a *AgentSideConnection) sendError(id *json.RawMessage, code int, message string) {
	_ = a.proto.writeMessage(jsonrpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error: &struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		}{Code: code, Message: message},
	})
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
			a.sendError(req.ID, -32602, fmt.Sprintf("invalid params: %v", err))
		}
		return false
	}
	return true
}

// reply finishes a handler: emit result on success, -32603 error otherwise.
// Skips the reply entirely when req.ID is nil (notification).
func (a *AgentSideConnection) reply(req *jsonrpcRequest, result any, err error) {
	if err != nil {
		a.log.Error("handler failed", "method", req.Method, "error", err)
		if req.ID != nil {
			a.sendError(req.ID, -32603, err.Error())
		}
		return
	}
	if req.ID != nil {
		a.sendResult(req.ID, result)
	}
}

// serve reads lines from the underlying transport forever. Responses to our
// outgoing requests route to the pending map; everything else lands in
// handle().
func (a *AgentSideConnection) serve() {
	defer close(a.done)
	ctx := context.Background()
	a.proto.serve(func(line []byte) {
		var probe struct {
			ID     *json.RawMessage `json:"id"`
			Method string           `json:"method"`
			Result *json.RawMessage `json:"result"`
			Error  *json.RawMessage `json:"error"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			a.log.Warn("failed to parse message", "error", err)
			return
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
			return
		}

		// Incoming request or notification.
		var req jsonrpcRequest
		if err := json.Unmarshal(line, &req); err != nil {
			a.log.Warn("failed to parse message", "error", err)
			return
		}
		a.log.Debug("received", "method", req.Method, "raw", string(line))

		// session/prompt runs on its own goroutine so a long turn doesn't
		// block reading the cancel notification that ends it.
		if req.Method == "session/prompt" {
			go a.handle(ctx, &req)
		} else {
			a.handle(ctx, &req)
		}
	})
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
			a.sendError(req.ID, -32601, fmt.Sprintf("method not found: %s", req.Method))
		}
	}
}
