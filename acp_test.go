package main

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

// ACP requires `text` on a text block (Zed rejects it otherwise), and history
// replay uses an empty text chunk to separate two same-role messages.
func TestContentBlockJSON(t *testing.T) {
	for _, c := range []struct {
		name  string
		block ContentBlock
		want  string
	}{
		{"text leaks no optional fields", ContentBlock{Type: "text", Text: "hello"}, `{"type":"text","text":"hello"}`},
		{"empty text keeps its text field", ContentBlock{Type: "text"}, `{"type":"text","text":""}`},
		{"an image grows no text field", ContentBlock{Type: "image", MimeType: "image/png", Data: "AA=="}, `{"type":"image","mimeType":"image/png","data":"AA=="}`},
	} {
		b, err := json.Marshal(c.block)
		if err != nil {
			t.Fatal(err)
		}
		if string(b) != c.want {
			t.Errorf("%s: got %s, want %s", c.name, b, c.want)
		}
	}
}

func TestSessionUpdateWrapsPayload(t *testing.T) {
	agentW, agentR, _, peerR := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)

	chunk := messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "hi"}}
	if err := c.SessionUpdate(context.Background(), "sid-42", chunk); err != nil {
		t.Fatal(err)
	}

	line := readLine(t, peerR)
	var env struct {
		Method string `json:"method"`
		Params struct {
			SessionId string       `json:"sessionId"`
			Update    messageChunk `json:"update"`
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

	go func() {
		line := readLine(t, peerR)
		var probe struct {
			ID *json.RawMessage `json:"id"`
		}
		_ = json.Unmarshal(line, &probe)
		resp := jsonrpcResponse{JSONRPC: "2.0", ID: probe.ID, Result: map[string]string{"ok": "yes"}}
		b, _ := json.Marshal(resp)
		_, _ = peerW.Write(append(b, '\n'))
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	raw, err := c.sendRequest(ctx, "fs/read_text_file", map[string]string{"path": "/tmp/x"})
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
	if _, err := c.sendRequest(ctx, "fs/read_text_file", nil); err == nil {
		t.Fatal("expected context error, got nil")
	}
}

func TestUnknownMethodRepliesMethodNotFound(t *testing.T) {
	agentW, agentR, peerW, peerR := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)
	_ = c

	req := jsonrpcRequest{JSONRPC: "2.0", ID: ptrRaw(`"99"`), Method: "no/such/method"}
	b, _ := json.Marshal(req)
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}

	line := readLine(t, peerR)
	var resp jsonrpcResponse
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

// The handler's late reply must be suppressed: two responses for one id would
// corrupt the client's pending map.
func TestCancelRequestAnswersImmediately(t *testing.T) {
	agentW, agentR, peerW, peerR := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)

	ctx, untrack := c.track(context.Background(), ptrRaw(`"7"`))
	defer untrack()

	cancelReq := jsonrpcRequest{JSONRPC: "2.0", Method: "$/cancel_request", Params: json.RawMessage(`{"requestId":"7"}`)}
	b, _ := json.Marshal(cancelReq)
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}

	line := readLine(t, peerR)
	var resp jsonrpcResponse
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
		t.Error("handler could still reply after -32800: that would be a second response for one id")
	}
}

func TestCancelRequestUnknownIdIsIgnored(t *testing.T) {
	agentW, agentR, peerW, peerR := pipePair(t)
	NewAgentSideConnection(nil, agentW, agentR)

	b, _ := json.Marshal(jsonrpcRequest{JSONRPC: "2.0", Method: "$/cancel_request", Params: json.RawMessage(`{"requestId":"404"}`)})
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}
	// If the cancel wrote anything, this read returns it instead of the -32601.
	b, _ = json.Marshal(jsonrpcRequest{JSONRPC: "2.0", ID: ptrRaw(`"1"`), Method: "no/such/method"})
	if _, err := peerW.Write(append(b, '\n')); err != nil {
		t.Fatal(err)
	}
	var resp jsonrpcResponse
	if err := json.Unmarshal(readLine(t, peerR), &resp); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.Error == nil || resp.Error.Code != -32601 {
		t.Fatalf("got %+v, want the -32601 for no/such/method", resp.Error)
	}
}

// Without the outbound cancel an open permission dialog outlives the cancelled turn.
func TestSendRequestCancelsOutboundOnCtxDone(t *testing.T) {
	agentW, agentR, _, peerR := pipePair(t)
	c := NewAgentSideConnection(nil, agentW, agentR)
	br := bufio.NewReader(peerR)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		if _, err := c.sendRequest(ctx, "session/request_permission", map[string]string{"sessionId": "s1"}); err == nil {
			t.Error("sendRequest returned nil error after cancel")
		}
	}()

	first, err := br.ReadString('\n')
	if err != nil {
		t.Fatalf("read request: %v", err)
	}
	var out jsonrpcRequest
	if err := json.Unmarshal([]byte(first), &out); err != nil {
		t.Fatalf("parse request: %v", err)
	}
	cancel()

	second, err := br.ReadString('\n')
	if err != nil {
		t.Fatalf("read cancel: %v", err)
	}
	var got jsonrpcRequest
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
