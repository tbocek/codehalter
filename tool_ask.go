package main

import "context"

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
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, args map[string]string) string {
		question, yesLabel, noLabel := args["question"], args["yes_label"], args["no_label"]
		tcId := a.StartToolCall(ctx, sid, question, "think", nil)
		ok, err := a.conn.AskYesNo(ctx, sid, tcId, yesLabel, noLabel)
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
	return resp.Outcome.OptionId, nil
}

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

func (a *AgentSideConnection) AskWritePermission(ctx context.Context, sid SessionId, toolCallId string) (string, error) {
	choice, err := a.requestPermission(ctx, sid, toolCallId, []permissionOption{
		{OptionId: "allow_once", Name: "Allow once", Kind: "allow_once"},
		{OptionId: "allow_turn", Name: "Allow for this prompt", Kind: "allow_always"},
		{OptionId: "reject", Name: "Reject", Kind: "reject_once"},
	})
	if err != nil {
		return "reject", err
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
