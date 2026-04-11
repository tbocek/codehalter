package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
)

const ProtocolVersionNumber ProtocolVersion = 1
const StopReasonEndTurn StopReason = "end_turn"

type ProtocolVersion int
type SessionId string
type StopReason string

type Implementation struct {
	Name    string `json:"name,omitempty"`
	Version string `json:"version,omitempty"`
}

type AgentCapabilities struct {
	LoadSession         bool                 `json:"loadSession"`
	PromptCapabilities  PromptCapabilities   `json:"promptCapabilities"`
	SessionCapabilities *SessionCapabilities `json:"sessionCapabilities,omitempty"`
}

type PromptCapabilities struct{}

type SessionCapabilities struct {
	List *SessionListCapabilities `json:"list,omitempty"`
}

type SessionListCapabilities struct{}

type (
	InitializeRequest  struct{ ProtocolVersion ProtocolVersion `json:"protocolVersion"` }
	InitializeResponse struct {
		ProtocolVersion   ProtocolVersion   `json:"protocolVersion"`
		AgentCapabilities AgentCapabilities `json:"agentCapabilities"`
		AgentInfo         *Implementation   `json:"agentInfo,omitempty"`
		AuthMethods       []string          `json:"authMethods"`
	}

	AuthenticateRequest  struct{}
	AuthenticateResponse struct{}

	NewSessionRequest  struct{ Cwd string `json:"cwd,omitempty"` }
	NewSessionResponse struct {
		SessionId SessionId         `json:"sessionId"`
		Modes     *SessionModeState `json:"modes,omitempty"`
	}

	SetSessionModeRequest struct {
		SessionId SessionId `json:"sessionId"`
		ModeId    string    `json:"modeId"`
	}
	SetSessionModeResponse struct{}

	SetSessionModelRequest  struct{ SessionId SessionId `json:"sessionId"` }
	SetSessionModelResponse struct{}

	LoadSessionRequest  struct {
		SessionId SessionId `json:"sessionId"`
		Cwd       string    `json:"cwd"`
	}
	LoadSessionResponse struct {
		SessionId SessionId        `json:"sessionId,omitempty"`
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

	SetSessionConfigOptionRequest struct {
		SessionId SessionId `json:"sessionId"`
		ConfigId  string    `json:"configId"`
		Value     string    `json:"value"`
	}
	SetSessionConfigOptionResponse struct {
		ConfigOptions []SessionConfigOption `json:"configOptions"`
	}

	CancelNotification struct{ SessionId SessionId `json:"sessionId"` }

	PromptRequest struct {
		SessionId SessionId      `json:"sessionId"`
		Content   []ContentBlock `json:"prompt"`
	}
	PromptResponse struct{ StopReason StopReason `json:"stopReason,omitempty"` }
)

// Session modes.

type SessionModeState struct {
	CurrentModeId string        `json:"currentModeId"`
	AvailableModes []SessionMode `json:"availableModes"`
}

type SessionMode struct {
	Id          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// Session config options (dropdown selectors in client UI).

type SessionConfigOption struct {
	Type         string                     `json:"type"`
	Id           string                     `json:"id"`
	Name         string                     `json:"name"`
	Description  string                     `json:"description,omitempty"`
	CurrentValue string                     `json:"currentValue"`
	Options      []SessionConfigSelectOption `json:"options"`
}

type SessionConfigSelectOption struct {
	Value       string `json:"value"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
}

// Content blocks — union type, serialised flat (not wrapped).

type ContentBlock struct{ Text *ContentBlockText }

type ContentBlockText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func (b ContentBlock) MarshalJSON() ([]byte, error) {
	if b.Text != nil {
		return json.Marshal(b.Text)
	}
	return []byte("null"), nil
}

func (b *ContentBlock) UnmarshalJSON(data []byte) error {
	var probe struct{ Type string `json:"type"` }
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	if probe.Type == "text" {
		b.Text = &ContentBlockText{}
		return json.Unmarshal(data, b.Text)
	}
	return nil
}

func TextBlock(text string) ContentBlock {
	return ContentBlock{Text: &ContentBlockText{Type: "text", Text: text}}
}

// Session update (agent -> client notification).

type SessionNotification struct {
	SessionId SessionId     `json:"sessionId"`
	Update    SessionUpdate `json:"update"`
}

// SessionUpdate is sent as the "update" field in session/update notifications.
// It's a discriminated union keyed by "sessionUpdate". We use any so that
// different update shapes (message chunks, tool calls, etc.) all work.
type SessionUpdate = any

type messageChunk struct {
	Kind    string       `json:"sessionUpdate"`
	Content ContentBlock `json:"content"`
}

func AgentMessageChunk(content ContentBlock) SessionUpdate {
	return messageChunk{Kind: "agent_message_chunk", Content: content}
}

func UserMessageChunk(content ContentBlock) SessionUpdate {
	return messageChunk{Kind: "user_message_chunk", Content: content}
}

// Agent is the interface an ACP agent must implement.

type Agent interface {
	Initialize(context.Context, InitializeRequest) (InitializeResponse, error)
	Authenticate(context.Context, AuthenticateRequest) (AuthenticateResponse, error)
	NewSession(context.Context, NewSessionRequest) (NewSessionResponse, error)
	LoadSession(context.Context, LoadSessionRequest) (LoadSessionResponse, error)
	ListSessions(context.Context, ListSessionsRequest) (ListSessionsResponse, error)
	SetSessionMode(context.Context, SetSessionModeRequest) (SetSessionModeResponse, error)
	SetSessionModel(context.Context, SetSessionModelRequest) (SetSessionModelResponse, error)
	SetSessionConfigOption(context.Context, SetSessionConfigOptionRequest) (SetSessionConfigOptionResponse, error)
	Cancel(context.Context, CancelNotification)
	Prompt(context.Context, PromptRequest) (PromptResponse, error)
}


// AgentSideConnection routes ACP methods over a JSON-RPC connection.

type AgentSideConnection struct {
	conn  *Connection
	agent Agent
}

func NewAgentSideConnection(agent Agent, w io.Writer, r io.Reader, log *slog.Logger) *AgentSideConnection {
	a := &AgentSideConnection{agent: agent}
	a.conn = NewConnection(w, r, a.handle, log)
	return a
}

func (a *AgentSideConnection) Done() <-chan struct{} { return a.conn.Done() }
func (a *AgentSideConnection) RPC() *Connection    { return a.conn }

func (a *AgentSideConnection) SessionUpdate(ctx context.Context, n SessionNotification) error {
	return a.conn.SendNotification("session/update", n)
}

func (a *AgentSideConnection) handle(ctx context.Context, req *jsonrpcRequest) {
	switch req.Method {
	case "initialize":
		Dispatch(a.conn, ctx, req, a.agent.Initialize)
	case "authenticate":
		Dispatch(a.conn, ctx, req, a.agent.Authenticate)
	case "session/new":
		Dispatch(a.conn, ctx, req, a.agent.NewSession)
	case "session/load":
		Dispatch(a.conn, ctx, req, a.agent.LoadSession)
	case "session/list":
		Dispatch(a.conn, ctx, req, a.agent.ListSessions)
	case "session/set_config_option":
		Dispatch(a.conn, ctx, req, a.agent.SetSessionConfigOption)
	case "session/set_mode":
		Dispatch(a.conn, ctx, req, a.agent.SetSessionMode)
	case "session/set_model":
		Dispatch(a.conn, ctx, req, a.agent.SetSessionModel)
	case "session/prompt":
		Dispatch(a.conn, ctx, req, a.agent.Prompt)
	case "session/cancel":
		var params CancelNotification
		if req.Params != nil {
			_ = json.Unmarshal(req.Params, &params)
		}
		a.agent.Cancel(ctx, params)
	default:
		if req.ID != nil {
			a.conn.SendError(req.ID, -32601, fmt.Sprintf("method not found: %s", req.Method))
		}
	}
}
