package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// shouldAutoAnswer reports whether prompts must be auto-answered: either
// global autopilot mode is on, or the session is a subagent (Depth > 0). Zed
// doesn't know subagent session ids, so any permission request sent for one
// would hang forever — auto-answering keeps the subagent moving.
func (a *agent) shouldAutoAnswer(sid string) (bool, string) {
	if a.isAutopilot() {
		return true, "autopilot"
	}
	if sess := a.getSession(sid); sess != nil && sess.Depth > 0 {
		return true, "subagent"
	}
	return false, ""
}

// askYesNoAuto asks the user in interactive mode; in autopilot mode or from a
// subagent it returns yes immediately and sends a chat note so the user sees
// what was auto-answered. Callers are still responsible for completing the
// tool call they opened (typically via CompleteToolCall with a short note).
func (a *agent) askYesNoAuto(ctx context.Context, sid string, tcId, yesLabel, noLabel string) (bool, error) {
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "[" + reason + "] " + yesLabel + "\n\n"}})
		return true, nil
	}
	return a.conn.AskYesNo(ctx, sid, tcId, yesLabel, noLabel)
}

// askChoiceAuto asks the user in interactive mode; in autopilot or from a
// subagent it returns choices[0] (or "abort" if empty).
func (a *agent) askChoiceAuto(ctx context.Context, sid string, tcId string, choices []string) (string, error) {
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		if len(choices) == 0 {
			return "abort", nil
		}
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "[" + reason + "] " + choices[0] + "\n\n"}})
		return choices[0], nil
	}
	return a.conn.AskChoice(ctx, sid, tcId, choices)
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
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "[" + reason + "] " + choices[0] + "\n\n"}})
		return choices[0], tcId, nil
	}
	choice, err := a.conn.AskChoiceWithCard(ctx, sid, tcId, title, kind, choices)
	return choice, tcId, err
}

// askYesNoWithCard is askChoiceWithCard's two-button cousin.
func (a *agent) askYesNoWithCard(ctx context.Context, sid, title, kind, yesLabel, noLabel string) (bool, string, error) {
	tcId := a.StartToolCall(ctx, sid, title, kind, nil)
	if auto, reason := a.shouldAutoAnswer(sid); auto {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "[" + reason + "] " + yesLabel + "\n\n"}})
		return true, tcId, nil
	}
	ok, err := a.conn.AskYesNoWithCard(ctx, sid, tcId, title, kind, yesLabel, noLabel)
	return ok, tcId, err
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
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "[" + reason + "] " + label + "\n\n"}})
		return tcId, nil
	}
	err := a.conn.AskAcknowledgeWithCard(ctx, sid, tcId, title, kind, label)
	return tcId, err
}

func init() {
	RegisterTool(Tool{Def: map[string]any{
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
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		question, yesLabel, noLabel := args["question"], args["yes_label"], args["no_label"]
		// /improve fans out one ask per proposed change; hold the flow to the top
		// improveAskCap in code so a chatty model can't loop through dozens. Only
		// the Apply/Skip improvement prompts count — the final Yes/No submit
		// prompt is exempt (different labels).
		if strings.Contains(strings.ToLower(yesLabel), "apply") || strings.Contains(strings.ToLower(noLabel), "skip") {
			if sess := a.getSession(sid); sess != nil && sess.improveAskBlocked() {
				return fmt.Sprintf("[improve cap: you have already presented the top %d improvements (the maximum). Do NOT call ask_user for more improvements. Apply or skip what is shown, then go straight to the submit and verify steps.]", improveAskCap), false
			}
		}
		tcId := a.StartToolCall(ctx, sid, question, "think", nil)
		ok, err := a.askYesNoAuto(ctx, sid, tcId, yesLabel, noLabel)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error(), false
		}
		chosen := noLabel
		if ok {
			chosen = yesLabel
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + chosen)})
		if ok {
			return "user said yes", false
		}
		return "user said no", false
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

func isUnknownSessionErr(err error) bool {
	return err != nil && strings.Contains(err.Error(), "unknown session")
}

func (a *AgentSideConnection) doPermissionRequest(ctx context.Context, r permissionRequest) (string, error) {
	// Every interactive card blocks here waiting on the user; record that span
	// so the turn's "✅ Done" line can exclude it from active time. (Auto-answer
	// paths never reach here — they return before calling conn.Ask*.)
	start := time.Now()
	defer func() {
		if sess := a.agent.getSession(r.SessionId); sess != nil {
			sess.addHumanWait(time.Since(start))
		}
	}()
	var raw json.RawMessage
	var err error
	for attempt := 0; attempt <= len(unknownSessionBackoffs); attempt++ {
		raw, err = a.sendRequest(ctx, "session/request_permission", r)
		if err == nil || !isUnknownSessionErr(err) || attempt == len(unknownSessionBackoffs) {
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

func (a *AgentSideConnection) requestPermission(ctx context.Context, sid string, toolCallId string, options []permissionOption) (string, error) {
	return a.doPermissionRequest(ctx, permissionRequest{
		SessionId: sid,
		ToolCall:  permissionToolCall{ToolCallId: toolCallId},
		Options:   options,
	})
}

var errPermissionCancelled = errors.New("permission dialog dismissed")

// AskChoice shows N green choices + a red Abort. Returns the chosen optionId.
func (a *AgentSideConnection) AskChoice(ctx context.Context, sid string, toolCallId string, choices []string) (string, error) {
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

func (a *AgentSideConnection) AskYesNo(ctx context.Context, sid string, toolCallId, yesLabel, noLabel string) (bool, error) {
	choice, err := a.requestPermission(ctx, sid, toolCallId, []permissionOption{
		{OptionId: "yes", Name: yesLabel, Kind: "allow_once"},
		{OptionId: "no", Name: noLabel, Kind: "reject_once"},
	})
	if err != nil {
		return false, err
	}
	return choice == "yes", nil
}

// AskChoiceWithCard is AskChoice that also carries title/kind so Zed can
// register the tool card inline if the prior tool_call SessionUpdate was
// dropped (the session-registration race). Use for bootstrap-phase prompts
// where the session has only just been created.
func (a *AgentSideConnection) AskChoiceWithCard(ctx context.Context, sid, toolCallId, title, kind string, choices []string) (string, error) {
	var options []permissionOption
	for _, c := range choices {
		options = append(options, permissionOption{OptionId: c, Name: c, Kind: "allow_once"})
	}
	options = append(options, permissionOption{OptionId: "abort", Name: "Abort", Kind: "reject_once"})

	choice, err := a.doPermissionRequest(ctx, permissionRequest{
		SessionId: sid,
		ToolCall:  permissionToolCall{ToolCallId: toolCallId, Title: title, Kind: kind, Status: "in_progress"},
		Options:   options,
	})
	if err != nil {
		return "abort", err
	}
	return choice, nil
}

// AskYesNoWithCard is AskYesNo that also carries title/kind, for the same
// reason as AskChoiceWithCard.
func (a *AgentSideConnection) AskYesNoWithCard(ctx context.Context, sid, toolCallId, title, kind, yesLabel, noLabel string) (bool, error) {
	choice, err := a.doPermissionRequest(ctx, permissionRequest{
		SessionId: sid,
		ToolCall:  permissionToolCall{ToolCallId: toolCallId, Title: title, Kind: kind, Status: "in_progress"},
		Options: []permissionOption{
			{OptionId: "yes", Name: yesLabel, Kind: "allow_once"},
			{OptionId: "no", Name: noLabel, Kind: "reject_once"},
		},
	})
	if err != nil {
		return false, err
	}
	return choice == "yes", nil
}

// AskAcknowledgeWithCard shows a single-button card and waits for the user to
// click it. There's no decline path — the discarded outcomeId is implicit
// (only one option exists). Used for unrecoverable-but-fixable states where
// the only useful user action is "I fixed it, retry".
func (a *AgentSideConnection) AskAcknowledgeWithCard(ctx context.Context, sid, toolCallId, title, kind, label string) error {
	_, err := a.doPermissionRequest(ctx, permissionRequest{
		SessionId: sid,
		ToolCall:  permissionToolCall{ToolCallId: toolCallId, Title: title, Kind: kind, Status: "in_progress"},
		Options: []permissionOption{
			{OptionId: "ack", Name: label, Kind: "allow_once"},
		},
	})
	return err
}
