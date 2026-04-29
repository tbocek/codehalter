package main

import (
	"context"
	"errors"
)

// shouldAutoAnswer reports whether prompts must be auto-answered: either
// global autopilot mode is on, or the session is a subagent (Depth > 0). Zed
// doesn't know subagent session ids, so any permission request sent for one
// would hang forever — auto-answering keeps the subagent moving.
func (a *agent) shouldAutoAnswer(sid SessionId) (bool, string) {
	if a.isAutopilot() {
		return true, "autopilot"
	}
	if sess := a.getSession(sid); sess != nil && sess.Depth > 0 {
		return true, "subagent"
	}
	return false, ""
}

// askYesNoAuto asks the user in interactive mode; in autopilot mode or from a
// subagent it returns defaultYes immediately and sends a chat note so the user
// sees what was auto-answered. Callers are still responsible for completing
// the tool call they opened (typically via CompleteToolCall with a short note).
func (a *agent) askYesNoAuto(ctx context.Context, sid SessionId, tcId, yesLabel, noLabel string, defaultYes bool) (bool, error) {
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		chosen := noLabel
		if defaultYes {
			chosen = yesLabel
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("["+reason+"] "+chosen+"\n\n")))
		return defaultYes, nil
	}
	return a.conn.AskYesNo(ctx, sid, tcId, yesLabel, noLabel)
}

// askChoiceAuto asks the user in interactive mode; in autopilot or from a
// subagent it returns choices[defaultIdx] (or choices[0] if out of range, or
// "abort" if empty).
func (a *agent) askChoiceAuto(ctx context.Context, sid SessionId, tcId string, choices []string, defaultIdx int) (string, error) {
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		if len(choices) == 0 {
			return "abort", nil
		}
		idx := defaultIdx
		if idx < 0 || idx >= len(choices) {
			idx = 0
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("["+reason+"] "+choices[idx]+"\n\n")))
		return choices[idx], nil
	}
	return a.conn.AskChoice(ctx, sid, tcId, choices)
}

func init() {
	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "ask_user",
			"description": "Ask the user a yes/no question",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"question", "yes_label", "no_label"},
				"properties": map[string]any{
					"question":  map[string]any{"type": "string", "description": "The question to display"},
					"yes_label": map[string]any{"type": "string", "description": "Label for the yes button"},
					"no_label":  map[string]any{"type": "string", "description": "Label for the no button"},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
			args := parseArgs(rawArgs)
		question, yesLabel, noLabel := args["question"], args["yes_label"], args["no_label"]
		tcId := a.StartToolCall(ctx, sid, question, "think", nil)
		ok, err := a.askYesNoAuto(ctx, sid, tcId, yesLabel, noLabel, true)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}
		chosen := noLabel
		if ok {
			chosen = yesLabel
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + chosen)})
		if ok {
			return "user said yes"
		}
		return "user said no"
	}})
}

// ---------------------------------------------------------------------------
// Permission RPCs
// ---------------------------------------------------------------------------

type permissionOption struct {
	OptionId string `json:"optionId"`
	Name     string `json:"name"`
	Kind     string `json:"kind"`
}

type permissionRequest struct {
	SessionId SessionId `json:"sessionId"`
	ToolCall  struct {
		ToolCallId string `json:"toolCallId"`
	} `json:"toolCall"`
	Options []permissionOption `json:"options"`
}

type permissionResponse struct {
	Outcome struct {
		Outcome  string `json:"outcome"`
		OptionId string `json:"optionId,omitempty"`
	} `json:"outcome"`
}

func (a *AgentSideConnection) requestPermission(ctx context.Context, sid SessionId, toolCallId string, options []permissionOption) (string, error) {
	r := permissionRequest{SessionId: sid, Options: options}
	r.ToolCall.ToolCallId = toolCallId
	resp, err := SendRequest[permissionResponse](a.conn, ctx, "session/request_permission", r)
	if err != nil {
		return "", err
	}
	// "cancelled" means the dialog was dismissed without a button click
	// (IDE-side cancel, session switch, etc.) — distinct from the user
	// explicitly choosing the "no"/"abort" option.
	if resp.Outcome.Outcome == "cancelled" {
		return "", errPermissionCancelled
	}
	return resp.Outcome.OptionId, nil
}

var errPermissionCancelled = errors.New("permission dialog dismissed")

// AskChoice shows up to 2 choices (green) + abort (red). Returns the chosen optionId.
func (a *AgentSideConnection) AskChoice(ctx context.Context, sid SessionId, toolCallId string, choices []string) (string, error) {
	var options []permissionOption
	for _, c := range choices {
		options = append(options, permissionOption{OptionId: c, Name: c, Kind: "allow_once"})
	}
	options = append(options, permissionOption{OptionId: "abort", Name: "Abort", Kind: "reject_once"})

	choice, err := a.requestPermission(ctx, sid, toolCallId, options)
	if err != nil {
		return "abort", err
	}
	return choice, nil
}

func (a *AgentSideConnection) AskYesNo(ctx context.Context, sid SessionId, toolCallId, yesLabel, noLabel string) (bool, error) {
	choice, err := a.requestPermission(ctx, sid, toolCallId, []permissionOption{
		{OptionId: "yes", Name: yesLabel, Kind: "allow_once"},
		{OptionId: "no", Name: noLabel, Kind: "reject_once"},
	})
	if err != nil {
		return false, err
	}
	return choice == "yes", nil
}
