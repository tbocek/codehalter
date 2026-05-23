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

func TestJsonrpcRequestEncoding(t *testing.T) {
	id := json.RawMessage(`"7"`)
	req := jsonrpcRequest{JSONRPC: "2.0", ID: &id, Method: "session/prompt", Params: json.RawMessage(`{"x":1}`)}
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
	notif := jsonrpcRequest{JSONRPC: "2.0", Method: "session/update", Params: json.RawMessage(`{}`)}
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

	// Fake peer: read the request, echo back a result keyed to its id.
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

	// Send a request whose method doesn't match any case in handle().
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
