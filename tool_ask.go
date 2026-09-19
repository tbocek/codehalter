package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/tbocek/codehalter/acp"
)

// shouldAutoAnswer reports whether prompts must be auto-answered (autopilot
// mode), and the label to show in front of the automatic answer.
func (a *agent) shouldAutoAnswer(_ string) (bool, string) {
	return a.isAutopilot(), "autopilot"
}

// askChoiceAuto asks the user in interactive mode; in autopilot it returns
// choices[0] (or "abort" if empty).
func (a *agent) askChoiceAuto(ctx context.Context, sid string, tcId, question string, choices []string) (string, error) {
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		if len(choices) == 0 {
			return "abort", nil
		}
		a.say(ctx, sid, "["+reason+"] "+choices[0]+"\n\n")
		return choices[0], nil
	}
	return a.askChoice(ctx, sid, tcId, question, choices)
}

// askChoice shows N green choices + a red Abort and returns the chosen option.
func (a *agent) askChoice(ctx context.Context, sid string, tcId, question string, choices []string) (string, error) {
	choice, err := a.doPermissionRequest(ctx, acp.PermissionRequest{
		SessionId: sid,
		ToolCall:  acp.PermissionToolCall{ToolCallId: tcId},
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
func choiceOptions(choices []string) []acp.PermissionOption {
	var options []acp.PermissionOption
	for _, c := range choices {
		options = append(options, acp.PermissionOption{OptionId: c, Name: c, Kind: "allow_once"})
	}
	return append(options, acp.PermissionOption{OptionId: "abort", Name: "Abort", Kind: "reject_once"})
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
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		answer, note := "", "no answer available"
		if len(options) > 0 {
			answer, note = options[0], options[0]
		}
		a.say(ctx, sid, "["+reason+"] "+note+"\n\n")
		return answer, nil
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
	raw, err := a.conn.SendRequest(ctx, "elicitation/create", req)
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

// askChoiceWithCard opens a tool card AND asks for permission in a single
// flow, returning the new tcId so the caller can Complete/Fail it. The
// request_permission payload carries the card title/kind, so if the prior
// tool_call SessionUpdate was dropped (the session-registration race), Zed
// can still register the card from the permission request alone.
//
// Use this in the bootstrap phase (ensureDevcontainer, ensureGitignore); the
// execute-phase tools open a card first and only sometimes ask for permission,
// so they keep the split API.
func (a *agent) askChoiceWithCard(ctx context.Context, sid, title, kind string, choices []string) (string, string, error) {
	tcId := a.StartToolCall(ctx, sid, title, kind, nil)
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		if len(choices) == 0 {
			return "abort", tcId, nil
		}
		a.say(ctx, sid, "["+reason+"] "+choices[0]+"\n\n")
		return choices[0], tcId, nil
	}
	choice, err := a.doPermissionRequest(ctx, acp.PermissionRequest{
		SessionId: sid,
		ToolCall:  acp.PermissionToolCall{ToolCallId: tcId, Title: title, Kind: kind, Status: "in_progress"},
		Message:   title,
		Options:   choiceOptions(choices),
	})
	if err != nil {
		return "abort", tcId, err
	}
	return choice, tcId, nil
}

// askYesNoWithCard is askChoiceWithCard's two-button cousin.
func (a *agent) askYesNoWithCard(ctx context.Context, sid, title, kind, yesLabel, noLabel string) (bool, string, error) {
	tcId := a.StartToolCall(ctx, sid, title, kind, nil)
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		a.say(ctx, sid, "["+reason+"] "+yesLabel+"\n\n")
		return true, tcId, nil
	}
	choice, err := a.doPermissionRequest(ctx, acp.PermissionRequest{
		SessionId: sid,
		ToolCall:  acp.PermissionToolCall{ToolCallId: tcId, Title: title, Kind: kind, Status: "in_progress"},
		Message:   title,
		Options: []acp.PermissionOption{
			{OptionId: "yes", Name: yesLabel, Kind: "allow_once"},
			{OptionId: "no", Name: noLabel, Kind: "reject_once"},
		},
	})
	if err != nil {
		return false, tcId, err
	}
	return choice == "yes", tcId, nil
}

// askAcknowledgeWithCard is a single-button card — the user clicks `label` to
// acknowledge, no decline path. Used by the Prepare phase's LLM-unreachable
// Retry loop: codehalter can't proceed without an LLM, so the only useful
// option is "I've edited the file, try again". In auto-answer modes the card
// completes immediately (callers cap retries themselves). Returns the tcId so
// the caller can Complete/Fail it after acting on the acknowledgement.
func (a *agent) askAcknowledgeWithCard(ctx context.Context, sid, title, kind, label string) (string, error) {
	tcId := a.StartToolCall(ctx, sid, title, kind, nil)
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		a.say(ctx, sid, "["+reason+"] "+label+"\n\n")
		return tcId, nil
	}
	_, err := a.doPermissionRequest(ctx, acp.PermissionRequest{
		SessionId: sid,
		ToolCall:  acp.PermissionToolCall{ToolCallId: tcId, Title: title, Kind: kind, Status: "in_progress"},
		Message:   title,
		Options:   []acp.PermissionOption{{OptionId: "ack", Name: label, Kind: "allow_once"}},
	})
	return tcId, err
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
		a.CompleteToolCall(ctx, sid, tcId, []acp.ToolCallContent{acp.TextContent("This editor has no free-text prompt")})
		return "error: this editor cannot show a free-text prompt — ask again with `options`", false
	case errors.Is(err, errPermissionCancelled):
		// Far likelier with a text box than with two buttons, and not a tool
		// failure: the user just closed the form.
		a.CompleteToolCall(ctx, sid, tcId, []acp.ToolCallContent{acp.TextContent("No answer")})
		return "user dismissed the question without answering", false
	case err != nil:
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error: " + err.Error(), false
	}
	if answer == "" || answer == "abort" {
		a.CompleteToolCall(ctx, sid, tcId, []acp.ToolCallContent{acp.TextContent("No answer")})
		return "no answer given — use your own judgement and continue", false
	}
	a.CompleteToolCall(ctx, sid, tcId, []acp.ToolCallContent{acp.TextContent("User answered: " + answer)})
	return "user answered: " + answer, false
}}

// ---------------------------------------------------------------------------
// Permission RPCs
// ---------------------------------------------------------------------------

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

func (a *agent) doPermissionRequest(ctx context.Context, r acp.PermissionRequest) (string, error) {
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
		raw, err = a.conn.SendRequest(ctx, "session/request_permission", r)
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
	var resp acp.PermissionResponse
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
func (a *agent) doElicitation(ctx context.Context, r acp.PermissionRequest) (string, error) {
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
	raw, err := a.conn.SendRequest(ctx, "elicitation/create", req)
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
