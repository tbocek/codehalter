package acp

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

// pipePair returns two os.Pipe halves wired so writes on agentW arrive on
// peerR, and writes on peerW arrive on agentR. Kernel-buffered, so small
// writes don't deadlock the writer when no reader is yet waiting.
func pipePair(t *testing.T) (agentW *os.File, agentR *os.File, peerW *os.File, peerR *os.File) {
	t.Helper()
	var err error
	agentR, peerW, err = os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	peerR, agentW, err = os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		agentW.Close()
		agentR.Close()
		peerW.Close()
		peerR.Close()
	})
	return
}

func readLine(t *testing.T, r io.Reader) []byte {
	t.Helper()
	br := bufio.NewReader(r)
	line, err := br.ReadString('\n')
	if err != nil {
		t.Fatalf("read line: %v", err)
	}
	return []byte(strings.TrimRight(line, "\r\n"))
}

func TestJSONRPCRequestEncoding(t *testing.T) {
	id := json.RawMessage(`"7"`)
	req := JSONRPCRequest{JSONRPC: "2.0", ID: &id, Method: "session/prompt", Params: json.RawMessage(`{"x":1}`)}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	want := `{"jsonrpc":"2.0","id":"7","method":"session/prompt","params":{"x":1}}`
	if got != want {
		t.Fatalf("got %s\nwant %s", got, want)
	}

	// Notification: no id field on the wire.
	notif := JSONRPCRequest{JSONRPC: "2.0", Method: "session/update", Params: json.RawMessage(`{}`)}
	b, _ = json.Marshal(notif)
	if strings.Contains(string(b), `"id"`) {
		t.Fatalf("notification leaked id field: %s", b)
	}
}

func TestContentBlockOmitsEmptyFields(t *testing.T) {
	b, _ := json.Marshal(ContentBlock{Type: "text", Text: "hello"})
	got := string(b)
	if got != `{"type":"text","text":"hello"}` {
		t.Fatalf("text block leaked optional fields: %s", got)
	}
}

func TestSessionUpdateWrapsPayload(t *testing.T) {
	agentW, agentR, _, peerR := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)

	chunk := MessageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "hi"}}
	if err := c.SessionUpdate(context.Background(), "sid-42", chunk); err != nil {
		t.Fatal(err)
	}

	line := readLine(t, peerR)
	var env struct {
		Method string `json:"method"`
		Params struct {
			SessionId string       `json:"sessionId"`
			Update    MessageChunk `json:"update"`
		} `json:"params"`
	}
	if err := json.Unmarshal(line, &env); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if env.Method != "session/update" || env.Params.SessionId != "sid-42" {
		t.Fatalf("bad envelope: %s", line)
	}
	if env.Params.Update.Kind != KindAgentMessage || env.Params.Update.Content.Text != "hi" {
		t.Fatalf("bad update: %+v", env.Params.Update)
	}
}

func TestSendRequestRoundtrip(t *testing.T) {
	agentW, agentR, peerW, peerR := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)

	// Fake peer: read the request, echo back a result keyed to its id.
	go func() {
		line := readLine(t, peerR)
		var probe struct {
			ID *json.RawMessage `json:"id"`
		}
		_ = json.Unmarshal(line, &probe)
		resp := JSONRPCResponse{JSONRPC: "2.0", ID: probe.ID, Result: map[string]string{"ok": "yes"}}
		b, _ := json.Marshal(resp)
		_, _ = peerW.Write(append(b, '\n'))
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	raw, err := c.SendRequest(ctx, "fs/read_text_file", map[string]string{"path": "/tmp/x"})
	if err != nil {
		t.Fatalf("sendRequest: %v", err)
	}
	var got map[string]string
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if got["ok"] != "yes" {
		t.Fatalf("unexpected result: %v", got)
	}
}

func TestSendRequestContextCancel(t *testing.T) {
	agentW, agentR, _, _ := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := c.SendRequest(ctx, "fs/read_text_file", nil); err == nil {
		t.Fatal("expected context error, got nil")
	}
}

func TestUnknownMethodRepliesMethodNotFound(t *testing.T) {
	agentW, agentR, peerW, peerR := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)
	_ = c

	// Send a request whose method doesn't match any case in handle().
	req := JSONRPCRequest{JSONRPC: "2.0", ID: ptrRaw(`"99"`), Method: "no/such/method"}
	b, _ := json.Marshal(req)
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}

	line := readLine(t, peerR)
	var resp JSONRPCResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.Error == nil || resp.Error.Code != -32601 {
		t.Fatalf("expected -32601, got %+v", resp.Error)
	}
	_ = c
}

func ptrRaw(s string) *json.RawMessage {
	m := json.RawMessage(s)
	return &m
}

// TestCancelRequestAnswersImmediately pins $/cancel_request: the named request
// gets -32800 right away (not whenever the handler notices), its context is
// cancelled, and the handler's own late reply is then suppressed — two
// responses for one id would corrupt the client's pending map.
func TestCancelRequestAnswersImmediately(t *testing.T) {
	agentW, agentR, peerW, peerR := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)

	// Stand in for a handler that has registered itself and is still running.
	entry := &inflightRequest{}
	ctx, cancel := context.WithCancel(context.Background())
	entry.cancel = cancel
	c.inflightMu.Lock()
	c.inflight[`"7"`] = entry
	c.inflightMu.Unlock()

	cancelReq := JSONRPCRequest{JSONRPC: "2.0", Method: "$/cancel_request", Params: json.RawMessage(`{"requestId":"7"}`)}
	b, _ := json.Marshal(cancelReq)
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}

	line := readLine(t, peerR)
	var resp JSONRPCResponse
	if err := json.Unmarshal(line, &resp); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.Error == nil || resp.Error.Code != -32800 {
		t.Fatalf("got %+v, want error -32800", resp.Error)
	}
	if resp.ID == nil || string(*resp.ID) != `"7"` {
		t.Errorf("answered id %v, want \"7\"", resp.ID)
	}

	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Error("handler context was not cancelled")
	}
	if c.claimReply(ptrRaw(`"7"`)) {
		t.Error("handler could still reply after -32800 — that would be a second response for one id")
	}
}

// TestCancelRequestUnknownIdIsIgnored: a cancel that races a finished handler
// must not answer an id nobody is waiting on.
func TestCancelRequestUnknownIdIsIgnored(t *testing.T) {
	agentW, agentR, peerW, peerR := pipePair(t)
	NewAgentSideConnection(nil, agentW, agentR)

	b, _ := json.Marshal(JSONRPCRequest{JSONRPC: "2.0", Method: "$/cancel_request", Params: json.RawMessage(`{"requestId":"404"}`)})
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}
	// Then a request we know produces a reply — if the cancel wrote anything,
	// this read returns that instead of the -32601.
	b, _ = json.Marshal(JSONRPCRequest{JSONRPC: "2.0", ID: ptrRaw(`"1"`), Method: "no/such/method"})
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}
	var resp JSONRPCResponse
	if err := json.Unmarshal(readLine(t, peerR), &resp); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.Error == nil || resp.Error.Code != -32601 {
		t.Fatalf("got %+v, want the -32601 for no/such/method", resp.Error)
	}
}

// TestSendRequestCancelsOutboundOnCtxDone pins the other direction: when we
// abandon a request we told the client to drop it. Without this an open
// permission/elicitation dialog stays on screen after the turn is cancelled.
func TestSendRequestCancelsOutboundOnCtxDone(t *testing.T) {
	agentW, agentR, _, peerR := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)
	br := bufio.NewReader(peerR)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		if _, err := c.SendRequest(ctx, "session/request_permission", map[string]string{"sessionId": "s1"}); err == nil {
			t.Error("sendRequest returned nil error after cancel")
		}
	}()

	first, err := br.ReadString('\n')
	if err != nil {
		t.Fatalf("read request: %v", err)
	}
	var out JSONRPCRequest
	if err := json.Unmarshal([]byte(first), &out); err != nil {
		t.Fatalf("parse request: %v", err)
	}
	cancel()

	second, err := br.ReadString('\n')
	if err != nil {
		t.Fatalf("read cancel: %v", err)
	}
	var got JSONRPCRequest
	if err := json.Unmarshal([]byte(second), &got); err != nil {
		t.Fatalf("parse cancel: %v", err)
	}
	if got.Method != "$/cancel_request" {
		t.Fatalf("second message was %q, want $/cancel_request", got.Method)
	}
	if got.ID != nil {
		t.Error("$/cancel_request carried an id; it is a notification")
	}
	var p struct {
		RequestId json.RawMessage `json:"requestId"`
	}
	if err := json.Unmarshal(got.Params, &p); err != nil {
		t.Fatalf("parse cancel params: %v", err)
	}
	if string(p.RequestId) != string(*out.ID) {
		t.Errorf("cancelled requestId %s, want %s", p.RequestId, *out.ID)
	}
	<-done
}
