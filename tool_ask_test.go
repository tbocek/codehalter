package main

import (
	"bufio"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

type elicitationRequest struct {
	ID     *json.RawMessage `json:"id"`
	Method string           `json:"method"`
	Params struct {
		SessionId       string `json:"sessionId"`
		Mode            string `json:"mode"`
		Message         string `json:"message"`
		ToolCallId      string `json:"toolCallId"`
		RequestedSchema struct {
			Properties map[string]struct {
				Type  string `json:"type"`
				Title string `json:"title"`
				OneOf []struct {
					Const string `json:"const"`
					Title string `json:"title"`
				} `json:"oneOf"`
			} `json:"properties"`
			Required []string `json:"required"`
		} `json:"requestedSchema"`
	} `json:"params"`
}

// readElicitation loops because a call through runToolCall sends its tool card
// first.
func readElicitation(t *testing.T, br *bufio.Reader) elicitationRequest {
	t.Helper()
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		var req elicitationRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			t.Fatalf("parse %s: %v", line, err)
		}
		if req.Method == "elicitation/create" {
			return req
		}
		if req.Method != "session/update" {
			t.Fatalf("method = %q, want elicitation/create", req.Method)
		}
	}
}

// Option ids survive the round trip, so every caller's return value is unchanged.
func TestAskUsesElicitationWhenAdvertised(t *testing.T) {
	a, s, br, peerW := elicitingAgent(t)

	type result struct {
		choice string
		err    error
	}
	res := make(chan result, 1)
	go func() {
		choice, err := a.askChoice(context.Background(), s.ID, "tc1", "Mount your .git?", []string{"Yes, mount", "No"})
		res <- result{choice, err}
	}()

	req := readElicitation(t, br)
	if req.Params.Mode != "form" {
		t.Errorf("mode = %q, want form", req.Params.Mode)
	}
	if req.Params.Message != "Mount your .git?" {
		t.Errorf("message = %q, want the question text", req.Params.Message)
	}
	if req.Params.ToolCallId != "tc1" {
		t.Errorf("toolCallId = %q, want tc1", req.Params.ToolCallId)
	}
	prop, ok := req.Params.RequestedSchema.Properties[elicitChoiceKey]
	if !ok {
		t.Fatalf("schema has no %q property", elicitChoiceKey)
	}
	if prop.Type != "string" {
		t.Errorf("property type = %q, want string", prop.Type)
	}
	// Two choices plus the Abort option AskChoice appends.
	if len(prop.OneOf) != 3 || prop.OneOf[0].Const != "Yes, mount" || prop.OneOf[0].Title != "Yes, mount" {
		t.Errorf("oneOf = %+v, want the option ids with their labels", prop.OneOf)
	}
	if prop.OneOf[2].Const != "abort" {
		t.Errorf("last option = %+v, want the abort option", prop.OneOf[2])
	}

	reply := jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{
		"action":  "accept",
		"content": map[string]string{elicitChoiceKey: "Yes, mount"},
	}}
	b, _ := json.Marshal(reply)
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}

	select {
	case got := <-res:
		if got.err != nil {
			t.Fatalf("AskChoice: %v", got.err)
		}
		if got.choice != "Yes, mount" {
			t.Errorf("AskChoice = %q, want the accepted option id back", got.choice)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AskChoice did not return")
	}
}

// Neither field is required, and a typed answer wins over a selected option.
func TestAskUserOptionsPlusFreeText(t *testing.T) {
	a, s, br, peerW := elicitingAgent(t)

	res := make(chan string, 1)
	go func() {
		var tc toolCall
		tc.Function.Name = "ask_user"
		tc.Function.Arguments = `{"question":"Which port?","options":["8080","3000"],"allow_text":true}`
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		out := tu.Output
		res <- out
	}()

	req := readElicitation(t, br)
	if req.Params.Message != "Which port?" {
		t.Errorf("message = %q, want the question", req.Params.Message)
	}
	choice, ok := req.Params.RequestedSchema.Properties[elicitChoiceKey]
	if !ok || len(choice.OneOf) != 2 || choice.OneOf[0].Const != "8080" {
		t.Errorf("choice property = %+v, want the two options", choice)
	}
	text, ok := req.Params.RequestedSchema.Properties[elicitTextKey]
	if !ok || text.Type != "string" || len(text.OneOf) != 0 {
		t.Errorf("text property = %+v, want a free-text string", text)
	}
	if len(req.Params.RequestedSchema.Required) != 0 {
		t.Errorf("required = %v, want neither field required when both are offered", req.Params.RequestedSchema.Required)
	}

	reply := jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{
		"action":  "accept",
		"content": map[string]any{elicitChoiceKey: "8080", elicitTextKey: " 9999 "},
	}}
	b, _ := json.Marshal(reply)
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}

	select {
	case out := <-res:
		if out != "user answered: 9999" {
			t.Errorf("tool result = %q, want the trimmed typed answer", out)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ask_user did not return")
	}
}

// allow_text is implied and the field is required: an empty form carries no answer.
func TestAskUserTextOnlyIsRequired(t *testing.T) {
	a, s, br, peerW := elicitingAgent(t)

	res := make(chan string, 1)
	go func() {
		var tc toolCall
		tc.Function.Name = "ask_user"
		tc.Function.Arguments = `{"question":"What should I name it?"}`
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		out := tu.Output
		res <- out
	}()

	req := readElicitation(t, br)
	if _, ok := req.Params.RequestedSchema.Properties[elicitChoiceKey]; ok {
		t.Error("no options were given, so the form must not carry a choice field")
	}
	if got := req.Params.RequestedSchema.Required; len(got) != 1 || got[0] != elicitTextKey {
		t.Errorf("required = %v, want the text field", got)
	}

	// Dismissing a text box is routine: "no answer", not a failed tool call.
	b, _ := json.Marshal(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{"action": "cancel"}})
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}

	select {
	case out := <-res:
		if !strings.Contains(out, "dismissed") {
			t.Errorf("tool result = %q, want a dismissal note", out)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ask_user did not return")
	}
}

// request_permission carries buttons only, so the model is told to re-ask with
// options.
func TestAskUserFreeTextNeedsElicitation(t *testing.T) {
	a, s, _, _ := elicitingAgent(t)
	a.clientCaps.Elicitation = nil // client without form support

	var tc toolCall
	tc.Function.Name = "ask_user"
	tc.Function.Arguments = `{"question":"What should I name it?"}`
	tu, _ := a.runToolCall(context.Background(), s.ID, tc)
	out := tu.Output
	if !strings.Contains(out, "options") {
		t.Errorf("tool result = %q, want a steer back to options", out)
	}
}

// "decline" is the last (reject) option; "cancel" and unknown actions are
// errPermissionCancelled, not a silent default choice.
func TestElicitationActionsMapToOutcomes(t *testing.T) {
	for _, tc := range []struct {
		action  string
		want    string
		wantErr bool
	}{
		{action: "decline", want: "no"},
		{action: "cancel", wantErr: true},
		{action: "_vendor_thing", wantErr: true},
		{action: "accept", wantErr: true}, // accepted with no content
	} {
		t.Run(tc.action, func(t *testing.T) {
			a, s, br, peerW := elicitingAgent(t)
			type result struct {
				choice string
				err    error
			}
			res := make(chan result, 1)
			go func() {
				choice, err := a.doElicitation(context.Background(), permissionRequest{
					SessionId: s.ID,
					Message:   "pick",
					Options: []permissionOption{
						{OptionId: "yes", Name: "Yes"},
						{OptionId: "no", Name: "No"},
					},
				})
				res <- result{choice, err}
			}()

			line, err := br.ReadString('\n')
			if err != nil {
				t.Fatalf("read: %v", err)
			}
			var req struct {
				ID *json.RawMessage `json:"id"`
			}
			if err := json.Unmarshal([]byte(line), &req); err != nil {
				t.Fatalf("parse: %v", err)
			}
			b, _ := json.Marshal(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{"action": tc.action}})
			if _, err := peerW.Write(append(b, '\n')); err != nil {
				t.Fatal(err)
			}

			select {
			case got := <-res:
				if tc.wantErr {
					if got.err != errPermissionCancelled {
						t.Errorf("err = %v, want errPermissionCancelled", got.err)
					}
					return
				}
				if got.err != nil {
					t.Fatalf("err = %v, want nil", got.err)
				}
				if got.choice != tc.want {
					t.Errorf("choice = %q, want %q", got.choice, tc.want)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("doElicitation did not return")
			}
		})
	}
}
