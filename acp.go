package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"runtime/debug"
)

const (
	protocolVersion  = 1
	KindAgentMessage = "agent_message_chunk"
	// KindAgentThought carries reasoning_content; Zed renders it collapsed.
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

// An agent may only send fs/read_text_file and fs/write_text_file to a client
// that advertised them; fsRead/fsWrite fall back to disk I/O otherwise.
type ClientCapabilities struct {
	Fs struct {
		ReadTextFile  bool `json:"readTextFile"`
		WriteTextFile bool `json:"writeTextFile"`
	} `json:"fs"`
	Terminal bool `json:"terminal"`

	// Pointers, not bools: the spec spells "supported" as `{}` and "unsupported"
	// as omitted or null.
	Elicitation *struct {
		Form *struct{} `json:"form"`
		URL  *struct{} `json:"url"`
	} `json:"elicitation"`
}

type acpMCPServer struct {
	Type    string         `json:"type"` // "" (stdio), "http", "sse"
	Name    string         `json:"name"`
	Command string         `json:"command"`
	Args    []string       `json:"args"`
	Env     []acpNameValue `json:"env"`
	URL     string         `json:"url"`
	Headers []acpNameValue `json:"headers"`
}

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
				// Without it a spec-following client withholds "resource" blocks
				// (Zed's "@ include context"), which ContentBlock.Resource reads.
				EmbeddedContext bool `json:"embeddedContext,omitempty"`
			} `json:"promptCapabilities"`
			// Without http a client silently withholds its HTTP servers from
			// offerMCPImport. No sse: mcpTransport does stdio and Streamable HTTP only.
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
		// Not adopted silently: offered for import into .codehalter/mcp.toml (offerMCPImport).
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

// ContentBlock is a flat union on Type: text (Text), image (MimeType, base64 Data),
// resource (Resource, inline content), resource_link (URI, Name, no content).
type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
	Data     string `json:"data,omitempty"`

	URI  string `json:"uri,omitempty"`
	Name string `json:"name,omitempty"`

	Resource *EmbeddedResource `json:"resource,omitempty"`
}

// MarshalJSON keeps an empty `text` on text blocks: ACP requires it, Zed rejects
// the notification without it, and history replay sends empty separator chunks.
func (b ContentBlock) MarshalJSON() ([]byte, error) {
	type raw ContentBlock
	if b.Type != "text" {
		return json.Marshal(raw(b))
	}
	// The outer Text shadows raw's omitempty one: encoding/json takes the shallower field.
	return json.Marshal(struct {
		raw
		Text string `json:"text"`
	}{raw(b), b.Text})
}

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

// usageUpdate drives the client's context ring: Used is the last call's
// prompt_tokens, Size the per-slot n_ctx.
type usageUpdate struct {
	Kind string `json:"sessionUpdate"` // "usage_update"
	Used int    `json:"used"`
	Size int    `json:"size"`
}

// Without session_info_update Zed labels every thread with its id. Omitted fields
// leave the client's other metadata alone.
type sessionInfoUpdate struct {
	Kind      string `json:"sessionUpdate"` // "session_info_update"
	Title     string `json:"title,omitempty"`
	UpdatedAt string `json:"updatedAt,omitempty"` // ISO 8601
}

type AgentSideConnection struct {
	*rpcPeer
	agent *agent
}

func NewAgentSideConnection(a *agent, w io.Writer, r io.Reader) *AgentSideConnection {
	c := &AgentSideConnection{rpcPeer: newRPCPeer(w, "", true), agent: a}
	go c.serve(r, c.dispatch)
	return c
}

func (a *AgentSideConnection) SessionUpdate(ctx context.Context, sid string, update any) error {
	return a.notify("session/update", struct {
		SessionId string `json:"sessionId"`
		Update    any    `json:"update"`
	}{sid, update})
}

func (a *AgentSideConnection) dispatch(req *jsonrpcRequest) {
	// On the read loop, not in a goroutine: it must act before anything read
	// after it, to interrupt a handler that is already running.
	if req.Method == "$/cancel_request" {
		// Answered here: a handler blocked on something that ignores ctx would leave the client hanging.
		if entry, id := a.cancelInflight(req.Params); entry != nil && entry.answered.CompareAndSwap(false, true) {
			a.writeError(&id, -32800, "Request cancelled")
		}
		return
	}
	// Async: handlers' sendRequest responses arrive through this loop, so
	// running inline would deadlock.
	go a.runHandler(req)
}

func (a *AgentSideConnection) runHandler(req *jsonrpcRequest) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("handler panic", "method", req.Method, "panic", r, "stack", string(debug.Stack()))
			a.replyError(req, -32603, fmt.Sprintf("internal error: %v", r))
		}
	}()
	ctx, untrack := a.track(context.Background(), req.ID)
	defer untrack()
	a.handle(ctx, req)
}

func (a *AgentSideConnection) handle(ctx context.Context, req *jsonrpcRequest) {
	switch req.Method {
	case "initialize":
		serveCall(a, ctx, req, a.agent.Initialize)

	case "authenticate":
		// Initialize's only authMethod is terminal setup (`codehalter --setup`),
		// so this is never needed; ack rather than reply method-not-found.
		a.reply(req, struct{}{}, nil)

	case "session/new":
		serveCall(a, ctx, req, a.agent.NewSession)

	case "session/load":
		serveCall(a, ctx, req, a.agent.LoadSession)

	case "session/list":
		serveCall(a, ctx, req, a.agent.ListSessions)

	case "session/set_mode":
		serveCall(a, ctx, req, func(ctx context.Context, p SetSessionModeRequest) (struct{}, error) {
			return struct{}{}, a.agent.SetSessionMode(ctx, p)
		})

	case "session/close":
		serveCall(a, ctx, req, func(ctx context.Context, p CloseSessionRequest) (struct{}, error) {
			return struct{}{}, a.agent.CloseSession(ctx, p)
		})

	case "session/prompt":
		serveCall(a, ctx, req, a.agent.Prompt)

	case "session/cancel":
		var p CancelNotification
		if req.Params != nil {
			if err := json.Unmarshal(req.Params, &p); err != nil {
				// Cancel anyway: malformed params still signal intent to abort.
				slog.Debug("session/cancel: malformed params", "err", err)
			}
		}
		a.agent.Cancel(ctx, p)
		// A notification per spec; ack a client that sent an id so it doesn't hang.
		a.reply(req, struct{}{}, nil)

	default:
		a.replyError(req, -32601, fmt.Sprintf("method not found: %s", req.Method))
	}
}

// serveCall is a method in all but name: Go methods cannot take type parameters.
func serveCall[P, R any](a *AgentSideConnection, ctx context.Context, req *jsonrpcRequest, fn func(context.Context, P) (R, error)) {
	var p P
	if req.Params != nil {
		if err := json.Unmarshal(req.Params, &p); err != nil {
			a.replyError(req, -32602, fmt.Sprintf("invalid params: %v", err))
			return
		}
	}
	res, err := fn(ctx, p)
	a.reply(req, res, err)
}

func (a *AgentSideConnection) reply(req *jsonrpcRequest, result any, err error) {
	if err != nil {
		slog.Error("handler failed", "method", req.Method, "error", err)
		a.replyError(req, -32603, err.Error())
		return
	}
	if req.ID == nil || !a.claimReply(req.ID) {
		return
	}
	a.writeResult(req.ID, result)
}

// Never pass -32000: ACP reserves it for AUTH_REQUIRED and Zed then shows an
// "Authentication Required" box for an unrelated failure.
func (a *AgentSideConnection) replyError(req *jsonrpcRequest, code int, message string) {
	if req.ID == nil || !a.claimReply(req.ID) {
		return
	}
	a.writeError(req.ID, code, message)
}

// claimReply reserves the single response an id is allowed; false once dispatch answered its $/cancel_request.
func (a *AgentSideConnection) claimReply(id *json.RawMessage) bool {
	a.inflightMu.Lock()
	entry, ok := a.inflight[string(*id)]
	a.inflightMu.Unlock()
	if !ok {
		return true
	}
	return entry.answered.CompareAndSwap(false, true)
}
