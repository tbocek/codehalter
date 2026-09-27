package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"
)

func (a *agent) autoAnswer(ctx context.Context, sid, answer string) bool {
	if !a.isAutopilot() {
		return false
	}
	a.say(ctx, sid, "[autopilot] "+answer+"\n\n")
	return true
}

// askCard carries the card's title and kind in the request itself, so a client
// that dropped the tool_call update (session-registration race) still shows it.
func (a *agent) askCard(ctx context.Context, sid, title, kind string, options []permissionOption) (choice, tcId string, err error) {
	tcId = a.StartToolCall(ctx, sid, title, kind, nil)
	if len(options) > 0 && a.autoAnswer(ctx, sid, options[0].Name) {
		return options[0].OptionId, tcId, nil
	}
	choice, err = a.doPermissionRequest(ctx, permissionRequest{
		SessionId: sid,
		ToolCall:  permissionToolCall{ToolCallId: tcId, Title: title, Kind: kind, Status: "in_progress"},
		Message:   title,
		Options:   options,
	})
	if err != nil {
		return "abort", tcId, err
	}
	return choice, tcId, nil
}

func (a *agent) askYesNoWithCard(ctx context.Context, sid, title, kind, yesLabel, noLabel string) (bool, string, error) {
	choice, tcId, err := a.askCard(ctx, sid, title, kind, []permissionOption{
		{OptionId: "yes", Name: yesLabel, Kind: "allow_once"},
		{OptionId: "no", Name: noLabel, Kind: "reject_once"},
	})
	return choice == "yes", tcId, err
}

func (a *agent) askChoice(ctx context.Context, sid string, tcId, question string, choices []string) (string, error) {
	choice, err := a.doPermissionRequest(ctx, permissionRequest{
		SessionId: sid,
		ToolCall:  permissionToolCall{ToolCallId: tcId},
		Options:   choiceOptions(choices),
		Message:   question,
	})
	if err != nil {
		return "abort", err
	}
	return choice, nil
}

func choiceOptions(choices []string) []permissionOption {
	var options []permissionOption
	for _, c := range choices {
		options = append(options, permissionOption{OptionId: c, Name: c, Kind: "allow_once"})
	}
	return append(options, permissionOption{OptionId: "abort", Name: "Abort", Kind: "reject_once"})
}

// askFormAuto: request_permission carries only buttons, so a text box needs
// elicitation (else errNoFreeText). Autopilot answers a text-only ask with "".
func (a *agent) askFormAuto(ctx context.Context, sid string, tcId, question string, options []string, allowText bool) (string, error) {
	if len(options) > 0 && a.autoAnswer(ctx, sid, options[0]) {
		return options[0], nil
	}
	if a.autoAnswer(ctx, sid, "no answer available") {
		return "", nil
	}
	if !allowText || !a.clientCan("elicitation") {
		if len(options) == 0 {
			return "", errNoFreeText
		}
		return a.askChoice(ctx, sid, tcId, question, options)
	}

	// Blocks on the user, so the turn's "Done" line must not count it as active.
	start := time.Now()
	defer func() {
		if sess := a.getSession(sid); sess != nil {
			sess.addHumanWait(time.Since(start))
		}
	}()

	props := map[string]any{}
	textTitle := "Your answer"
	var required []string
	if len(options) > 0 {
		values := make([]map[string]any, 0, len(options))
		for _, o := range options {
			values = append(values, map[string]any{"const": o, "title": o})
		}
		props[elicitChoiceKey] = map[string]any{"type": "string", "oneOf": values, "title": "Pick one"}
		textTitle = "Or type your own"
	} else {
		required = []string{elicitTextKey}
	}
	props[elicitTextKey] = map[string]any{"type": "string", "title": textTitle}

	action, content, err := a.elicitForm(ctx, sid, tcId, question, props, required)
	if err != nil {
		return "", err
	}
	if action != "accept" {
		// Unlike the button-only form, "decline" has no reject-option meaning
		// here, so it is a plain dismissal like "cancel".
		return "", errPermissionCancelled
	}
	// A typed answer wins: the user only reaches the text box by declining the
	// buttons.
	if s, _ := content[elicitTextKey].(string); strings.TrimSpace(s) != "" {
		return strings.TrimSpace(s), nil
	}
	if s, _ := content[elicitChoiceKey].(string); s != "" {
		return s, nil
	}
	return "", errPermissionCancelled
}

func (a *agent) elicitForm(ctx context.Context, sid, tcId, message string, props map[string]any, required []string) (string, map[string]any, error) {
	req := map[string]any{
		"sessionId": sid,
		"mode":      "form",
		"message":   message,
		"requestedSchema": map[string]any{
			"type":       "object",
			"properties": props,
			"required":   required,
		},
	}
	if tcId != "" {
		req["toolCallId"] = tcId
	}
	raw, err := a.conn.sendRequest(ctx, "elicitation/create", req)
	if err != nil {
		return "", nil, err
	}
	var resp struct {
		Action  string         `json:"action"`
		Content map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", nil, err
	}
	return resp.Action, resp.Content, nil
}

var askUserTool = Tool{Def: map[string]any{
	"type": "function",
	"function": map[string]any{
		"name":        "ask_user",
		"description": "Ask the user a question. Pass `options` for a pick-one list, set `allow_text` for a typed answer, or both (the options plus an \"or type your own\" box). With neither, the user gets a plain text box.",
		"parameters": map[string]any{
			"type":     "object",
			"required": []string{"question"},
			"properties": map[string]any{
				"question":   map[string]any{"type": "string", "description": "The question to display"},
				"options":    map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Answers to offer as buttons. Each string is both the label and the answer you get back. Two options ([\"Yes\", \"No\"]) is a yes/no question."},
				"allow_text": map[string]any{"type": "boolean", "description": "Let the user type their own answer instead of picking an option."},
			},
		},
	},
}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	var args struct {
		Question  string   `json:"question"`
		Options   []string `json:"options"`
		AllowText bool     `json:"allow_text"`
	}
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		return "error: invalid JSON: " + err.Error(), false
	}
	if args.Question == "" {
		return "error: question is required", false
	}
	allowText := args.AllowText || len(args.Options) == 0

	tcId := a.StartToolCall(ctx, sid, args.Question, "think", nil)
	answer, err := a.askFormAuto(ctx, sid, tcId, args.Question, args.Options, allowText)
	switch {
	case errors.Is(err, errNoFreeText):
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("This editor has no free-text prompt")})
		return "error: this editor cannot show a free-text prompt — ask again with `options`", false
	case errors.Is(err, errPermissionCancelled):
		// Not a tool failure: the user just closed the form.
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("No answer")})
		return "user dismissed the question without answering", false
	case err != nil:
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error: " + err.Error(), false
	}
	if answer == "" || answer == "abort" {
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("No answer")})
		return "no answer given — use your own judgement and continue", false
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User answered: " + answer)})
	return "user answered: " + answer, false
}}

type permissionOption struct {
	OptionId string `json:"optionId"`
	Name     string `json:"name"`
	Kind     string `json:"kind"`
}

// Title/Kind/Status are set only by askCard, so Zed can register the card inline
// when the tool_call update was lost in the session-registration race.
type permissionToolCall struct {
	ToolCallId string `json:"toolCallId"`
	Title      string `json:"title,omitempty"`
	Kind       string `json:"kind,omitempty"`
	Status     string `json:"status,omitempty"`
}

type permissionRequest struct {
	SessionId string             `json:"sessionId"`
	ToolCall  permissionToolCall `json:"toolCall"`
	Options   []permissionOption `json:"options"`

	// Not sent: request_permission has no such field and a strict client rejects
	// unknown properties. elicitation/create requires it.
	Message string `json:"-"`
}

type permissionResponse struct {
	Outcome struct {
		Outcome  string `json:"outcome"`
		OptionId string `json:"optionId,omitempty"`
	} `json:"outcome"`
}

// unknownSessionBackoffs: Zed answers request_permission with -32603 "unknown
// session" until it has registered the id from our session/new response.
var unknownSessionBackoffs = []time.Duration{
	16 * time.Millisecond,
	32 * time.Millisecond,
	64 * time.Millisecond,
	128 * time.Millisecond,
	128 * time.Millisecond,
	128 * time.Millisecond,
	128 * time.Millisecond,
	128 * time.Millisecond,
}

func (a *agent) doPermissionRequest(ctx context.Context, r permissionRequest) (string, error) {
	// Blocks on the user, so the turn's "Done" line must not count it as active.
	start := time.Now()
	defer func() {
		if sess := a.getSession(r.SessionId); sess != nil {
			sess.addHumanWait(time.Since(start))
		}
	}()
	// Every caller is asking a question, which is what elicitation is for;
	// request_permission is the fallback for clients without it.
	if a.clientCan("elicitation") {
		return a.doElicitation(ctx, r)
	}
	var raw json.RawMessage
	var err error
	for attempt := 0; attempt <= len(unknownSessionBackoffs); attempt++ {
		raw, err = a.conn.sendRequest(ctx, "session/request_permission", r)
		if err == nil || !strings.Contains(err.Error(), "unknown session") || attempt == len(unknownSessionBackoffs) {
			break
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(unknownSessionBackoffs[attempt]):
		}
	}
	if err != nil {
		return "", err
	}
	var resp permissionResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", err
	}
	// "cancelled": dismissed without a button, unlike choosing "no"/"abort".
	if resp.Outcome.Outcome == "cancelled" {
		return "", errPermissionCancelled
	}
	return resp.Outcome.OptionId, nil
}

const elicitChoiceKey = "choice"

// doElicitation uses option ids as oneOf consts, so the answer is exactly the
// optionId the permission path would return.
func (a *agent) doElicitation(ctx context.Context, r permissionRequest) (string, error) {
	values := make([]map[string]any, 0, len(r.Options))
	for _, o := range r.Options {
		values = append(values, map[string]any{"const": o.OptionId, "title": o.Name})
	}
	message := r.Message
	if message == "" {
		message = r.ToolCall.Title
	}
	action, content, err := a.elicitForm(ctx, r.SessionId, r.ToolCall.ToolCallId, message,
		map[string]any{elicitChoiceKey: map[string]any{"type": "string", "oneOf": values}},
		[]string{elicitChoiceKey})
	if err != nil {
		return "", err
	}
	switch action {
	case "accept":
		if choice, ok := content[elicitChoiceKey].(string); ok {
			return choice, nil
		}
		return "", errPermissionCancelled
	case "decline":
		// An explicit "no": the last option is the reject one in every caller.
		if n := len(r.Options); n > 0 {
			return r.Options[n-1].OptionId, nil
		}
		return "", errPermissionCancelled
	default:
		return "", errPermissionCancelled
	}
}

const elicitTextKey = "text"

var errNoFreeText = errors.New("client cannot show a free-text prompt")

var errPermissionCancelled = errors.New("permission dialog dismissed")
