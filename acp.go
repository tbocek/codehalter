package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"runtime/debug"
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

type AuthMethod struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Type        string            `json:"type"`
	Args        []string          `json:"args,omitempty"`
	Env         map[string]string `json:"env,omitempty"`
}

// ClientCapabilities is what the client told us it can do, in the initialize
// request. Only the parts codehalter acts on are modelled; the rest of the
// object is ignored, which is what the spec asks of an agent that doesn't use
// a capability.
//
// fs is the load-bearing one: an agent may only send fs/read_text_file and
// fs/write_text_file to a client that advertised them. Zed does, which is why
// this went unnoticed, but a client that doesn't (and there are several ACP
// clients now) would have failed every read_file with an RPC error. See
// fsRead/fsWrite, which fall back to plain disk I/O.
type ClientCapabilities struct {
	Fs struct {
		ReadTextFile  bool `json:"readTextFile"`
		WriteTextFile bool `json:"writeTextFile"`
	} `json:"fs"`
	Terminal bool `json:"terminal"`

	// Elicitation is the client's structured-input surface. Pointers, not bools,
	// because the spec spells "supported" as the empty object `{}` and
	// "unsupported" as omitted-or-null — a bool can't tell those apart.
	Elicitation *struct {
		Form *struct{} `json:"form"`
		URL  *struct{} `json:"url"`
	} `json:"elicitation"`
}

// acpMCPServer is one entry of session/new's mcpServers. The wire type is a
// union discriminated by "type", which is ABSENT for stdio (the transport every
// agent must support) and "http"/"sse" for the two network ones. Flattening the
// union into one struct keeps the decode trivial: the fields of the other
// variants simply come back zero.
type acpMCPServer struct {
	Type    string         `json:"type"` // "" (stdio), "http", "sse"
	Name    string         `json:"name"`
	Command string         `json:"command"`
	Args    []string       `json:"args"`
	Env     []acpNameValue `json:"env"`
	URL     string         `json:"url"`
	Headers []acpNameValue `json:"headers"`
}

// acpNameValue is the {name, value} pair ACP uses for both env variables and
// HTTP headers, rather than a map.
type acpNameValue struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type (
	InitializeRequest struct {
		ProtocolVersion    int                `json:"protocolVersion"`
		ClientCapabilities ClientCapabilities `json:"clientCapabilities"`
	}
	InitializeResponse struct {
		ProtocolVersion   int `json:"protocolVersion"`
		AgentCapabilities struct {
			LoadSession        bool `json:"loadSession"`
			PromptCapabilities struct {
				Image bool `json:"image,omitempty"`
				// EmbeddedContext tells the client it may attach ContentBlock
				// "resource" blocks (Zed's "@ include context": an editor selection
				// inlined into the prompt). codehalter reads them — see ContentBlock
				// .Resource — so withholding the flag only makes a spec-following
				// client send less context than we can handle.
				EmbeddedContext bool `json:"embeddedContext,omitempty"`
			} `json:"promptCapabilities"`
			// MCPCapabilities tells the client which MCP transports it may put in
			// session/new's mcpServers. stdio needs no flag (every agent must
			// support it); without http here a client silently withholds its HTTP
			// servers, which is exactly the set offerMCPImport wants to see. No
			// sse: codehalter's mcpTransport does stdio and Streamable HTTP only.
			MCPCapabilities struct {
				HTTP bool `json:"http"`
			} `json:"mcpCapabilities"`
			SessionCapabilities *struct {
				List  *struct{} `json:"list,omitempty"`
				Close *struct{} `json:"close,omitempty"`
			} `json:"sessionCapabilities,omitempty"`
		} `json:"agentCapabilities"`
		AgentInfo   any          `json:"agentInfo,omitempty"`
		AuthMethods []AuthMethod `json:"authMethods"`
	}

	NewSessionRequest struct {
		Cwd string `json:"cwd,omitempty"`
		// McpServers is the MCP server list the user configured in their editor.
		// codehalter runs its own MCP clients from .codehalter/mcp.toml rather
		// than adopting these silently, so they're offered for import at
		// bootstrap instead (see offerMCPImport).
		McpServers []acpMCPServer `json:"mcpServers,omitempty"`
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
		SessionId  string         `json:"sessionId"`
		Cwd        string         `json:"cwd"`
		McpServers []acpMCPServer `json:"mcpServers,omitempty"`
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
// discriminator plus per-variant fields. encoding/json handles every variant
// without a custom Unmarshal — the embedded-resource variant nests under
// Resource, the rest are flat.
//
// Variants we read off the wire:
//   - "text"          → Text
//   - "image"         → MimeType + Data (base64)
//   - "resource"      → Resource (embedded snippet, e.g. an editor selection
//     attached via Zed's "@ include context")
//   - "resource_link" → URI + Name (a pointer to a file, no inline content)
type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
	Data     string `json:"data,omitempty"` // base64-encoded image bytes

	// resource_link fields (a bare pointer to a file/resource).
	URI  string `json:"uri,omitempty"`
	Name string `json:"name,omitempty"`

	// Embedded "resource" block — carries the actual attached content inline.
	Resource *EmbeddedResource `json:"resource,omitempty"`
}

// MarshalJSON keeps `text` on a text block even when it is empty. The ACP
// schema requires the field, and Zed rejects the whole notification without it
// ("missing field `text`"), which silently broke the empty separator chunk the
// history replay sends between two same-role messages (LoadSession). Every
// other block type keeps its omitempty shape.
func (b ContentBlock) MarshalJSON() ([]byte, error) {
	type raw ContentBlock
	if b.Type != "text" {
		return json.Marshal(raw(b))
	}
	// The outer Text shadows raw's omitempty one: encoding/json takes the
	// shallower field of two with the same name.
	return json.Marshal(struct {
		raw
		Text string `json:"text"`
	}{raw(b), b.Text})
}

// EmbeddedResource is the nested payload of a "resource" content block. Text is
// set for textual resources (code selections, file excerpts); Blob holds
// base64 bytes for binary ones. URI identifies the source so we can label the
// attachment and the model knows what it's looking at.
type EmbeddedResource struct {
	URI      string `json:"uri,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
	Text     string `json:"text,omitempty"`
	Blob     string `json:"blob,omitempty"`
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

// usageUpdate is the ACP "usage_update" session notification that drives the
// client's context-window ring. Used is the tokens currently in context (the
// last call's prompt_tokens); Size is the total window (per-slot n_ctx). Stable
// since schema v1.17 and these two fields are the whole required shape; the
// spec also allows an optional `cost`, which is meaningless for a local model.
type usageUpdate struct {
	Kind string `json:"sessionUpdate"` // "usage_update"
	Used int    `json:"used"`
	Size int    `json:"size"`
}

// sessionInfoUpdate is the ACP "session_info_update" notification, which is how
// an agent names a thread: without it Zed labels every thread with its id.
// Every field but the kind is optional and independently patchable, so sending
// only Title leaves the client's other metadata alone.
type sessionInfoUpdate struct {
	Kind      string `json:"sessionUpdate"` // "session_info_update"
	Title     string `json:"title,omitempty"`
	UpdatedAt string `json:"updatedAt,omitempty"` // ISO 8601
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

	// inflight tracks incoming requests that carry an id, so a client's
	// $/cancel_request can abort one by id. Keyed by the raw id bytes as they
	// arrived (a client may number requests 3 or "3"; echoing what we were sent
	// is the only way the ids match).
	inflightMu sync.Mutex
	inflight   map[string]*inflightRequest
}

// inflightRequest is one incoming request we may still answer. cancel unblocks
// the handler; answered makes the response single-shot, so a $/cancel_request
// that replies -32800 immediately cannot be followed by the handler's own late
// reply to the same id (two responses for one id is a protocol violation, and
// clients key their pending map by id).
type inflightRequest struct {
	cancel   context.CancelFunc
	answered atomic.Bool
}

func NewAgentSideConnection(a *agent, w io.Writer, r io.Reader) *AgentSideConnection {
	c := &AgentSideConnection{
		w:        w,
		r:        r,
		agent:    a,
		done:     make(chan struct{}),
		pending:  make(map[string]chan json.RawMessage),
		inflight: make(map[string]*inflightRequest),
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
		// Tell the client to drop it. Permission and elicitation requests block
		// on a human, so without this a dialog open when the turn is cancelled
		// stays on screen forever, expecting an answer nobody will read.
		// Best-effort: a client may ignore $/cancel_request.
		if err := a.writeMessage(jsonrpcRequest{
			JSONRPC: "2.0",
			Method:  "$/cancel_request",
			Params:  json.RawMessage(`{"requestId":` + string(idRaw) + `}`),
		}); err != nil {
			slog.Debug("$/cancel_request: write failed", "id", idStr, "err", err)
		}
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

		// $/cancel_request must be handled ON the read loop, not in a goroutine:
		// its whole job is to interrupt a handler that is already running, and
		// the ordering guarantee only holds if we act before reading further.
		if req.Method == "$/cancel_request" {
			a.cancelInflight(&req)
			continue
		}

		// Must be async: handlers issue outbound sendRequest calls whose
		// responses come back through this same read loop. Running handle
		// inline would block the loop and deadlock the response routing.
		go a.runHandler(ctx, &req)
	}
}

// runHandler dispatches one incoming message and guarantees exactly one
// response to an id-carrying request. Requests with an id get their own
// cancellable context registered in a.inflight so $/cancel_request can reach
// them; notifications share the connection context and are never registered
// (there is nothing to cancel and no id to name them by).
func (a *AgentSideConnection) runHandler(ctx context.Context, req *jsonrpcRequest) {
	defer func() {
		if r := recover(); r != nil {
			// A panic in ONE handler must not take down the process and every
			// other session; isolate it and answer the request if it expects one.
			slog.Error("handler panic", "method", req.Method, "panic", r, "stack", string(debug.Stack()))
			a.replyError(req, -32603, fmt.Sprintf("internal error: %v", r))
		}
	}()
	if req.ID == nil {
		a.handle(ctx, req)
		return
	}
	key := string(*req.ID)
	entry := &inflightRequest{}
	ctx, entry.cancel = context.WithCancel(ctx)
	defer entry.cancel()
	a.inflightMu.Lock()
	a.inflight[key] = entry
	a.inflightMu.Unlock()
	defer func() {
		a.inflightMu.Lock()
		delete(a.inflight, key)
		a.inflightMu.Unlock()
	}()
	a.handle(ctx, req)
}

// cancelInflight services an inbound $/cancel_request: abort the named
// request's context and answer it -32800 right away. We do not wait for the
// handler to notice — a handler blocked on something that ignores ctx would
// leave the client hanging on a request it has already given up on, and the
// answered flag makes sure its eventual reply is dropped rather than sent as a
// second response for the same id.
func (a *AgentSideConnection) cancelInflight(req *jsonrpcRequest) {
	var p struct {
		RequestId json.RawMessage `json:"requestId"`
	}
	if req.Params != nil {
		if err := json.Unmarshal(req.Params, &p); err != nil {
			slog.Debug("$/cancel_request: malformed params", "err", err)
			return
		}
	}
	if len(p.RequestId) == 0 {
		return
	}
	key := string(p.RequestId)
	a.inflightMu.Lock()
	entry, ok := a.inflight[key]
	a.inflightMu.Unlock()
	if !ok {
		// Already finished, or an id we never saw. Both are normal races and the
		// spec says a cancel for an unknown request is simply ignored.
		slog.Debug("$/cancel_request: no such in-flight request", "requestId", key)
		return
	}
	entry.cancel()
	id := json.RawMessage(p.RequestId)
	if entry.answered.CompareAndSwap(false, true) {
		a.writeError(&id, -32800, "Request cancelled")
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
		// No auth needed for local llama.cpp — Initialize advertises empty
		// authMethods, so a client should never send this. Ack it anyway
		// rather than reply method-not-found.
		a.reply(req, struct{}{}, nil)

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
			if err := json.Unmarshal(req.Params, &p); err != nil {
				// Cancel anyway — a malformed params object still signals
				// intent to abort the current turn — but don't lose the parse
				// error silently.
				slog.Debug("session/cancel: malformed params", "err", err)
			}
		}
		a.agent.Cancel(ctx, p)
		// Spec'd as a notification (no id), but ack a client that sent one with an
		// id so it doesn't hang on the response (reply no-ops when ID is nil).
		a.reply(req, struct{}{}, nil)

	default:
		a.replyError(req, -32601, fmt.Sprintf("method not found: %s", req.Method))
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
		a.replyError(req, -32602, fmt.Sprintf("invalid params: %v", err))
		return false
	}
	return true
}

// reply finishes a handler: emit result on success, -32603 error otherwise.
// Skips entirely when req.ID is nil (notification) or when the request was
// already answered -32800 by a $/cancel_request.
func (a *AgentSideConnection) reply(req *jsonrpcRequest, result any, err error) {
	if err != nil {
		slog.Error("handler failed", "method", req.Method, "error", err)
		a.replyError(req, -32603, err.Error())
		return
	}
	if req.ID == nil || !a.claimReply(req.ID) {
		return
	}
	if werr := a.writeMessage(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: result}); werr != nil {
		slog.Warn("write reply failed", "method", req.Method, "error", werr)
	}
}

// replyError answers a request with an error, unless it is a notification or
// was already answered. We deliberately avoid -32000: ACP reserves it for
// AUTH_REQUIRED, so using it for generic handler failures makes Zed render a
// misleading "Authentication Required" red box (with an Authenticate button)
// for unrelated problems — e.g. an LLM stream cancelled mid-flight by the user.
func (a *AgentSideConnection) replyError(req *jsonrpcRequest, code int, message string) {
	if req.ID == nil || !a.claimReply(req.ID) {
		return
	}
	a.writeError(req.ID, code, message)
}

// claimReply reserves the single response an id is allowed. Returns false when
// something already answered it — today only cancelInflight, which wins the
// race deliberately so the client stops waiting immediately.
func (a *AgentSideConnection) claimReply(id *json.RawMessage) bool {
	a.inflightMu.Lock()
	entry, ok := a.inflight[string(*id)]
	a.inflightMu.Unlock()
	if !ok {
		// Not registered (a nested or synthetic request); nothing to contend with.
		return true
	}
	return entry.answered.CompareAndSwap(false, true)
}

// writeError puts an error response on the wire with no claim check. Callers
// must already hold the right to answer this id.
func (a *AgentSideConnection) writeError(id *json.RawMessage, code int, message string) {
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
