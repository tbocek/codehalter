package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"
)

// autoAnswer is what autopilot does with a question: take the offered answer
// and say so in the transcript. It reports false in interactive mode, where
// the caller goes on to ask.
func (a *agent) autoAnswer(ctx context.Context, sid, answer string) bool {
	if !a.isAutopilot() {
		return false
	}
	a.say(ctx, sid, "[autopilot] "+answer+"\n\n")
	return true
}

// askChoiceAuto asks the user to pick one of choices on an existing tool card;
// autopilot takes the first (or "abort" when there is none).
func (a *agent) askChoiceAuto(ctx context.Context, sid string, tcId, question string, choices []string) (string, error) {
	if len(choices) == 0 && a.isAutopilot() {
		return "abort", nil
	}
	if len(choices) > 0 && a.autoAnswer(ctx, sid, choices[0]) {
		return choices[0], nil
	}
	return a.askChoice(ctx, sid, tcId, question, choices)
}

// askCard opens a tool card AND asks its question in one flow, returning the
// chosen option id and the card for the caller to Complete or Fail. The
// request carries the card's title and kind, so a client that dropped the
// tool_call update (the session-registration race) can still show the card
// from the request alone. Autopilot takes the first option; an error reads as
// "abort".
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

// askYesNoWithCard is askCard with two buttons.
func (a *agent) askYesNoWithCard(ctx context.Context, sid, title, kind, yesLabel, noLabel string) (bool, string, error) {
	choice, tcId, err := a.askCard(ctx, sid, title, kind, []permissionOption{
		{OptionId: "yes", Name: yesLabel, Kind: "allow_once"},
		{OptionId: "no", Name: noLabel, Kind: "reject_once"},
	})
	return choice == "yes", tcId, err
}

// askChoice shows N green choices + a red Abort and returns the chosen option.
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

// choiceOptions is choices as allow buttons plus the Abort that ends every
// choice card.
func choiceOptions(choices []string) []permissionOption {
	var options []permissionOption
	for _, c := range choices {
		options = append(options, permissionOption{OptionId: c, Name: c, Kind: "allow_once"})
	}
	return append(options, permissionOption{OptionId: "abort", Name: "Abort", Kind: "reject_once"})
}

// askFormAuto asks one question as N options, a free-text box, or both. Options
// alone go out as a choice card, so clients with no elicitation still work via
// session/request_permission. A text box does NOT: request_permission can only
// carry buttons, so a typed answer needs elicitation, and without it the caller
// gets errNoFreeText. Returns the chosen option or the trimmed typed text.
//
// Under autopilot nobody is there to type, so a free-text-only ask has no
// answer to give: it returns "" and the caller tells the model to decide for
// itself rather than inventing a reply on the user's behalf.
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

	// Same human-wait accounting as doPermissionRequest: this blocks on the user,
	// so the turn's "✅ Done" line must not count it as active time.
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
		// Nothing to pick, so the box is the whole form and has to be filled.
		required = []string{elicitTextKey}
	}
	props[elicitTextKey] = map[string]any{"type": "string", "title": textTitle}

	req := map[string]any{
		"sessionId": sid,
		"mode":      "form",
		"message":   question,
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
		return "", err
	}
	// Content values are a union (string/int/number/bool/string-array), so they
	// decode into any; only the string case is ever asked for here.
	var resp struct {
		Action  string         `json:"action"`
		Content map[string]any `json:"content"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", err
	}
	if resp.Action != "accept" {
		// "decline" has no reject-option meaning here (unlike the button-only
		// form, where the last option IS the no), so it joins "cancel" and any
		// vendor action as a plain dismissal.
		return "", errPermissionCancelled
	}
	// A typed answer beats a selected option: the user only reaches the text box
	// by declining to use the buttons.
	if s, _ := resp.Content[elicitTextKey].(string); strings.TrimSpace(s) != "" {
		return strings.TrimSpace(s), nil
	}
	if s, _ := resp.Content[elicitChoiceKey].(string); s != "" {
		return s, nil
	}
	return "", errPermissionCancelled
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
		// Models trained on the older yes/no shape still emit these. Accept
		// them as a two-option list rather than dropping the question; not
		// advertised in the schema, since new calls should use `options`.
		YesLabel string `json:"yes_label"`
		NoLabel  string `json:"no_label"`
	}
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		return "error: invalid JSON: " + err.Error(), false
	}
	if args.Question == "" {
		return "error: question is required", false
	}
	options := args.Options
	if len(options) == 0 && args.YesLabel != "" && args.NoLabel != "" {
		options = []string{args.YesLabel, args.NoLabel}
	}
	// Nothing to pick means a typed answer is the only one possible.
	allowText := args.AllowText || len(options) == 0

	tcId := a.StartToolCall(ctx, sid, args.Question, "think", nil)
	answer, err := a.askFormAuto(ctx, sid, tcId, args.Question, options, allowText)
	switch {
	case errors.Is(err, errNoFreeText):
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("This editor has no free-text prompt")})
		return "error: this editor cannot show a free-text prompt — ask again with `options`", false
	case errors.Is(err, errPermissionCancelled):
		// Far likelier with a text box than with two buttons, and not a tool
		// failure: the user just closed the form.
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

// ---------------------------------------------------------------------------
// Permission RPCs
// ---------------------------------------------------------------------------

type permissionOption struct {
	OptionId string `json:"optionId"`
	Name     string `json:"name"`
	Kind     string `json:"kind"`
}

// permissionToolCall is the ACP toolCall block carried in a
// session/request_permission. ToolCallId is always set; Title/Kind/Status are
// populated only by the WithCard variants, which need Zed to register the card
// inline (the prior tool_call SessionUpdate may have been dropped during the
// session-registration race — see unknownSessionBackoffs).
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

	// Message is the question in prose. It is NOT sent as part of
	// session/request_permission (whose wire shape has no such field, and a
	// strict client rejects unknown properties) — it exists because
	// elicitation/create requires a human-readable message, and only the call
	// site knows it. The permission path carries the same text as the card title.
	Message string `json:"-"`
}

type permissionResponse struct {
	Outcome struct {
		Outcome  string `json:"outcome"`
		OptionId string `json:"optionId,omitempty"`
	} `json:"outcome"`
}

// unknownSessionBackoffs covers the race between Zed acknowledging our
// session/new response and registering the sessionId in its session map:
// a request_permission sent immediately after returns -32603 "unknown
// session" until that registration lands. ~750ms total has been enough
// in practice; a real failure still surfaces within a second.
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
	// Every interactive card blocks here waiting on the user; record that span
	// so the turn's "✅ Done" line can exclude it from active time. (Auto-answer
	// paths never reach here: they return before asking.)
	start := time.Now()
	defer func() {
		if sess := a.getSession(r.SessionId); sess != nil {
			sess.addHumanWait(time.Since(start))
		}
	}()
	// session/request_permission means "may I do this dangerous thing"; every
	// caller here is really asking the user a question, which is what
	// elicitation is for. Prefer it when the client has one, and keep the
	// permission dialog as the fallback for clients that don't (which is every
	// client that predates the feature).
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
	// "cancelled" means the dialog was dismissed without a button click
	// (IDE-side cancel, session switch, etc.) — distinct from the user
	// explicitly choosing the "no"/"abort" option.
	if resp.Outcome.Outcome == "cancelled" {
		return "", errPermissionCancelled
	}
	return resp.Outcome.OptionId, nil
}

// elicitChoiceKey is the single form field codehalter asks for. The name is
// arbitrary but must match between the requested schema and the lookup in the
// response content.
const elicitChoiceKey = "choice"

// doElicitation asks the same question as doPermissionRequest, as an
// elicitation/create form with one single-select enum. The option ids become
// the enum values and the labels become oneOf titles, so the answer maps back
// to exactly the optionId the permission path would have returned and every
// caller is unchanged.
func (a *agent) doElicitation(ctx context.Context, r permissionRequest) (string, error) {
	values := make([]map[string]any, 0, len(r.Options))
	for _, o := range r.Options {
		values = append(values, map[string]any{"const": o.OptionId, "title": o.Name})
	}
	message := r.Message
	if message == "" {
		message = r.ToolCall.Title
	}
	req := map[string]any{
		"sessionId": r.SessionId,
		"mode":      "form",
		"message":   message,
		"requestedSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				elicitChoiceKey: map[string]any{"type": "string", "oneOf": values},
			},
			"required": []string{elicitChoiceKey},
		},
	}
	if r.ToolCall.ToolCallId != "" {
		req["toolCallId"] = r.ToolCall.ToolCallId
	}
	raw, err := a.conn.sendRequest(ctx, "elicitation/create", req)
	if err != nil {
		return "", err
	}
	var resp struct {
		Action  string            `json:"action"`
		Content map[string]string `json:"content"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", err
	}
	switch resp.Action {
	case "accept":
		if choice, ok := resp.Content[elicitChoiceKey]; ok {
			return choice, nil
		}
		// Accepted with nothing in it. Treat as a dismissal rather than silently
		// returning "" and letting the caller read it as some unnamed option.
		return "", errPermissionCancelled
	case "decline":
		// An explicit "no". The last option is the reject one by construction in
		// every Ask* builder below, so that is the answer the caller expects.
		if n := len(r.Options); n > 0 {
			return r.Options[n-1].OptionId, nil
		}
		return "", errPermissionCancelled
	default:
		// "cancel", plus any future or vendor action we don't know: dismissed.
		return "", errPermissionCancelled
	}
}

// elicitTextKey is the free-text field of an ask_user form, paired with
// elicitChoiceKey when the model offers options too ("or type your own").
const elicitTextKey = "text"

// errNoFreeText is returned when the model wanted a typed answer but the client
// has no elicitation form and no options to fall back to.
var errNoFreeText = errors.New("client cannot show a free-text prompt")

var errPermissionCancelled = errors.New("permission dialog dismissed")
