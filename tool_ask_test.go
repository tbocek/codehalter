package main

import (
	"bufio"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// elicitationRequest is the wire shape of an elicitation/create as a test reads
// it back: enough of the requested schema to assert on the fields, their types,
// their enum options and which ones are required.
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

// readElicitation reads from the fake client until an elicitation/create
// arrives, returning it decoded so the caller can reply with the matching id.
// A call routed through executeTool opens its tool card first, so the form is
// never the first line on the wire.
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

// TestAskUsesElicitationWhenAdvertised pins that a client advertising
// elicitation.form gets an elicitation/create form instead of a
// session/request_permission — permission means "may I do this dangerous
// thing", and none of codehalter's Ask* call sites are that. The option ids
// must survive the round trip so every caller's return value is unchanged.
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

// TestAskUserOptionsPlusFreeText pins the ask_user form the model can build:
// `options` become a single-select and `allow_text` adds a text box beside
// them, neither one required (the user answers with either). A typed answer
// wins over a selected option, since the only way to reach the box is to ignore
// the buttons.
func TestAskUserOptionsPlusFreeText(t *testing.T) {
	a, s, br, peerW := elicitingAgent(t)

	res := make(chan string, 1)
	go func() {
		var tc toolCall
		tc.Function.Name = "ask_user"
		tc.Function.Arguments = `{"question":"Which port?","options":["8080","3000"],"allow_text":true}`
		out, _ := a.executeTool(context.Background(), s.ID, tc)
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

	// Answer BOTH: the typed text must win.
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

// TestAskUserTextOnlyIsRequired pins that a question with no options is a plain
// text box: allow_text is implied (there would be nothing to answer with
// otherwise) and the field is required, since an empty form carries no answer.
func TestAskUserTextOnlyIsRequired(t *testing.T) {
	a, s, br, peerW := elicitingAgent(t)

	res := make(chan string, 1)
	go func() {
		var tc toolCall
		tc.Function.Name = "ask_user"
		tc.Function.Arguments = `{"question":"What should I name it?"}`
		out, _ := a.executeTool(context.Background(), s.ID, tc)
		res <- out
	}()

	req := readElicitation(t, br)
	if _, ok := req.Params.RequestedSchema.Properties[elicitChoiceKey]; ok {
		t.Error("no options were given, so the form must not carry a choice field")
	}
	if got := req.Params.RequestedSchema.Required; len(got) != 1 || got[0] != elicitTextKey {
		t.Errorf("required = %v, want the text field", got)
	}

	// Dismissing a text box is routine, so it must read as "no answer" rather
	// than failing the tool call.
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

// TestAskUserFreeTextNeedsElicitation pins the one thing free text cannot do:
// session/request_permission carries buttons only, so a client with no
// elicitation form and a question with no options has nowhere to put the
// answer. The model is told to re-ask with options instead of the turn dying.
func TestAskUserFreeTextNeedsElicitation(t *testing.T) {
	a, s, _, _ := elicitingAgent(t)
	a.clientCaps.Elicitation = nil // client without form support

	var tc toolCall
	tc.Function.Name = "ask_user"
	tc.Function.Arguments = `{"question":"What should I name it?"}`
	out, _ := a.executeTool(context.Background(), s.ID, tc)
	if !strings.Contains(out, "options") {
		t.Errorf("tool result = %q, want a steer back to options", out)
	}
}

// TestAskUserLegacyYesNoLabels pins that the pre-form argument shape still
// works: a model that emits yes_label/no_label (not in the schema any more, but
// a strong trained prior) gets a two-option question rather than a dropped one.
func TestAskUserLegacyYesNoLabels(t *testing.T) {
	a, s := newTestAgent(t)
	a.mode = "Autopilot" // auto-answer picks options[0], no editor conn needed

	var tc toolCall
	tc.Function.Name = "ask_user"
	tc.Function.Arguments = `{"question":"Deploy?","yes_label":"Ship it","no_label":"Hold"}`
	out, _ := a.executeTool(context.Background(), s.ID, tc)
	if out != "user answered: Ship it" {
		t.Errorf("tool result = %q, want the yes label as the first option", out)
	}
}

// TestElicitationActionsMapToOutcomes pins the three response actions:
// "decline" is an explicit no (the last option, which every Ask* builder makes
// the reject one), while "cancel" and an unrecognised action are dismissals and
// must surface as errPermissionCancelled rather than a silent default choice.
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
