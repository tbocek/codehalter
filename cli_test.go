package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"slices"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

type syncBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuf) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuf) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// peer stands in for the agent, so the client is tested over real line-delimited JSON-RPC.
type peer struct {
	t  *testing.T
	w  io.Writer
	br *bufio.Reader
}

func newCLIHarness(t *testing.T, stdin string) (*cliClient, *peer, *syncBuf) {
	t.Helper()
	clientR, peerW := io.Pipe()
	peerR, clientW := io.Pipe()

	buf := &syncBuf{}
	ui := newCLIUI()
	ui.out = bufio.NewWriter(buf)
	ui.tty = false
	ui.echoInput = false
	ui.cols, ui.rows = 120, 40

	c := &cliClient{
		ui:    ui,
		in:    bufio.NewReader(strings.NewReader(stdin)),
		cwd:   t.TempDir(),
		terms: map[string]*cliTerminal{},
	}
	c.conn = newCLIConn(clientW, clientR, c)

	t.Cleanup(func() {
		c.releaseAll()
		peerW.Close()
		clientW.Close()
	})
	return c, &peer{t: t, w: peerW, br: bufio.NewReader(peerR)}, buf
}

func (p *peer) send(msg any) {
	p.t.Helper()
	b, err := json.Marshal(msg)
	if err != nil {
		p.t.Fatal(err)
	}
	if _, err := p.w.Write(append(b, '\n')); err != nil {
		p.t.Fatal(err)
	}
}

func (p *peer) sendRaw(s string) {
	p.t.Helper()
	if _, err := p.w.Write([]byte(s + "\n")); err != nil {
		p.t.Fatal(err)
	}
}

func (p *peer) reply(req map[string]any, result any) {
	p.t.Helper()
	p.send(map[string]any{"jsonrpc": "2.0", "id": req["id"], "result": result})
}

func (p *peer) recv() map[string]any {
	p.t.Helper()
	line, err := p.br.ReadString('\n')
	if err != nil {
		p.t.Fatalf("reading from client: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		p.t.Fatalf("client sent unparseable JSON %q: %v", line, err)
	}
	return m
}

func TestCLIConnRequestUsesACPWireShape(t *testing.T) {
	c, p, _ := newCLIHarness(t, "")
	done := make(chan json.RawMessage, 1)
	go func() {
		raw, err := c.conn.request("session/new", NewSessionRequest{Cwd: "/tmp/x"})
		if err != nil {
			t.Errorf("request: %v", err)
		}
		done <- raw
	}()

	msg := p.recv()
	if msg["jsonrpc"] != "2.0" {
		t.Errorf("jsonrpc = %v, want 2.0", msg["jsonrpc"])
	}
	if msg["method"] != "session/new" {
		t.Errorf("method = %v", msg["method"])
	}
	// String ids, mirroring the agent's convention.
	id, ok := msg["id"].(string)
	if !ok {
		t.Fatalf("id = %#v, want a quoted string", msg["id"])
	}
	p.send(map[string]any{"jsonrpc": "2.0", "id": id, "result": map[string]any{"sessionId": "s1"}})

	select {
	case raw := <-done:
		var res NewSessionResponse
		if err := json.Unmarshal(raw, &res); err != nil || res.SessionId != "s1" {
			t.Errorf("result = %s (%v)", raw, err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("response never routed back to the caller")
	}
}

func TestCLIConnSurfacesRPCErrors(t *testing.T) {
	c, p, _ := newCLIHarness(t, "")
	errc := make(chan error, 1)
	go func() {
		_, err := c.conn.request("session/load", nil)
		errc <- err
	}()
	msg := p.recv()
	p.send(map[string]any{"jsonrpc": "2.0", "id": msg["id"], "error": map[string]any{
		"code": -32603, "message": "no such session", "data": "s9",
	}})
	select {
	case err := <-errc:
		if err == nil || !strings.Contains(err.Error(), "no such session") || !strings.Contains(err.Error(), "s9") {
			t.Errorf("error = %v, want the message and its data", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("error never routed back")
	}
}

func TestCLIConnFailsPendingOnClose(t *testing.T) {
	clientR, peerW := io.Pipe()
	peerR, clientW := io.Pipe()
	defer clientW.Close()
	defer peerR.Close()

	c := &cliClient{ui: newCLIUI(), terms: map[string]*cliTerminal{}}
	c.ui.out = bufio.NewWriter(&syncBuf{})
	c.conn = newCLIConn(clientW, clientR, c)

	errc := make(chan error, 1)
	go func() {
		_, err := c.conn.request("initialize", nil)
		errc <- err
	}()
	if _, err := bufio.NewReader(peerR).ReadString('\n'); err != nil {
		t.Fatal(err)
	}
	peerW.Close()
	select {
	case err := <-errc:
		if err == nil {
			t.Error("request succeeded after the connection closed")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("pending request hung after close")
	}
}

func TestCLIConnRejectsUnknownMethod(t *testing.T) {
	_, p, _ := newCLIHarness(t, "")
	p.send(map[string]any{"jsonrpc": "2.0", "id": "7", "method": "fs/read_text_file"})
	reply := p.recv()
	if reply["id"] != "7" {
		t.Errorf("reply id = %v, want 7", reply["id"])
	}
	e, ok := reply["error"].(map[string]any)
	if !ok {
		t.Fatalf("want an error reply, got %#v", reply)
	}
	if e["code"].(float64) != -32601 {
		t.Errorf("code = %v, want -32601", e["code"])
	}
}

func TestSessionUpdatesRenderInOrder(t *testing.T) {
	_, p, buf := newCLIHarness(t, "")
	const n = 300
	for i := 0; i < n; i++ {
		p.send(map[string]any{"jsonrpc": "2.0", "method": "session/update", "params": map[string]any{
			"sessionId": "s1",
			"update": map[string]any{
				"sessionUpdate": KindAgentMessage,
				"content":       map[string]any{"type": "text", "text": fmt.Sprintf("%d\n", i)},
			},
		}})
	}
	deadline := time.Now().Add(5 * time.Second)
	var got []string
	for time.Now().Before(deadline) {
		got = strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
		if len(got) >= n {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(got) != n {
		t.Fatalf("got %d rows, want %d", len(got), n)
	}
	for i, row := range got {
		if row != fmt.Sprintf("%d", i) {
			t.Fatalf("row %d = %q, chunks were reordered", i, row)
		}
	}
}

func TestSessionUpdateIgnoresGarbage(t *testing.T) {
	_, p, buf := newCLIHarness(t, "")
	p.sendRaw(`{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s","update":{"sessionUpdate":"who_knows"}}}`)
	p.sendRaw(`not json at all`)
	p.send(map[string]any{"jsonrpc": "2.0", "method": "session/update", "params": map[string]any{
		"sessionId": "s",
		"update":    map[string]any{"sessionUpdate": KindAgentMessage, "content": map[string]any{"type": "text", "text": "alive\n"}},
	}})
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && !strings.Contains(buf.String(), "alive") {
		time.Sleep(10 * time.Millisecond)
	}
	if !strings.Contains(buf.String(), "alive") {
		t.Errorf("a malformed message killed the read loop: %q", buf.String())
	}
}

func requireSh(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skip("no sh on PATH")
	}
}

func (c *cliClient) mustCreate(t *testing.T, args ...string) string {
	t.Helper()
	params, _ := json.Marshal(map[string]any{
		"command": "sh", "args": args, "cwd": c.cwd, "outputByteLimit": terminalOutputLimit,
	})
	res, err := c.terminalCreate(params)
	if err != nil {
		t.Fatalf("terminal/create: %v", err)
	}
	return res.(map[string]any)["terminalId"].(string)
}

func idParams(id string) json.RawMessage {
	b, _ := json.Marshal(map[string]any{"terminalId": id})
	return b
}

func TestCLITerminalRunsAndReportsExit(t *testing.T) {
	requireSh(t)
	c, _, _ := newCLIHarness(t, "")
	id := c.mustCreate(t, "-c", "echo out; echo err 1>&2; exit 3")

	res, err := c.terminalWait(t.Context(), idParams(id))
	if err != nil {
		t.Fatalf("wait_for_exit: %v", err)
	}
	exit := res.(*terminalExit)
	if exit.ExitCode == nil || *exit.ExitCode != 3 {
		t.Errorf("exit = %#v, want code 3", exit)
	}

	out, err := c.terminalOutput(idParams(id))
	if err != nil {
		t.Fatal(err)
	}
	m := out.(map[string]any)
	if got := m["output"].(string); !strings.Contains(got, "out") || !strings.Contains(got, "err") {
		t.Errorf("output = %q, want both streams", got)
	}
	if m["truncated"].(bool) {
		t.Errorf("short output reported as truncated")
	}
}

func TestCLITerminalKillReportsSignal(t *testing.T) {
	requireSh(t)
	c, _, _ := newCLIHarness(t, "")
	id := c.mustCreate(t, "-c", "sleep 30")
	if _, err := c.terminalKill(idParams(id)); err != nil {
		t.Fatalf("kill: %v", err)
	}
	res, err := c.terminalWait(t.Context(), idParams(id))
	if err != nil {
		t.Fatalf("wait_for_exit: %v", err)
	}
	exit := res.(*terminalExit)
	if exit.Signal == "" {
		t.Errorf("exit = %#v, want a signal name", exit)
	}
	// The id stays valid after a kill: run_command's idle watchdog kills first and reads after.
	if _, err := c.terminalOutput(idParams(id)); err != nil {
		t.Errorf("output after kill: %v", err)
	}
}

func TestCLITerminalKeepsTailAtLimit(t *testing.T) {
	requireSh(t)
	c, _, _ := newCLIHarness(t, "")
	params, _ := json.Marshal(map[string]any{
		"command": "sh", "args": []string{"-c", "printf 'aaaaaaaaaabbbbbbbbbb'"},
		"cwd": c.cwd, "outputByteLimit": 10,
	})
	res, err := c.terminalCreate(params)
	if err != nil {
		t.Fatal(err)
	}
	id := res.(map[string]any)["terminalId"].(string)
	if _, err := c.terminalWait(t.Context(), idParams(id)); err != nil {
		t.Fatal(err)
	}
	out, _ := c.terminalOutput(idParams(id))
	m := out.(map[string]any)
	if got := m["output"].(string); got != "bbbbbbbbbb" {
		t.Errorf("output = %q, want the last 10 bytes", got)
	}
	if !m["truncated"].(bool) {
		t.Errorf("truncated flag not set")
	}
}

func TestCLITerminalStdinIsEmpty(t *testing.T) {
	requireSh(t)
	c, _, _ := newCLIHarness(t, "this line belongs to the prompt\n")
	id := c.mustCreate(t, "-c", "cat")
	done := make(chan struct{})
	go func() {
		defer close(done)
		if _, err := c.terminalWait(t.Context(), idParams(id)); err != nil {
			t.Errorf("wait: %v", err)
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("cat never saw EOF, so its stdin was not /dev/null")
	}
}

func TestCLITerminalCreateFailsLoudly(t *testing.T) {
	c, _, _ := newCLIHarness(t, "")
	params, _ := json.Marshal(map[string]any{"command": "codehalter-no-such-binary", "cwd": c.cwd})
	if _, err := c.terminalCreate(params); err == nil {
		t.Error("starting a missing binary reported success")
	}
}

func TestCLITerminalReleaseIsIdempotent(t *testing.T) {
	requireSh(t)
	c, _, _ := newCLIHarness(t, "")
	id := c.mustCreate(t, "-c", "sleep 30")
	if _, err := c.terminalRelease(idParams(id)); err != nil {
		t.Fatalf("release: %v", err)
	}
	// The agent releases in a defer, which can fire twice on a cancelled turn.
	if _, err := c.terminalRelease(idParams(id)); err != nil {
		t.Errorf("second release: %v", err)
	}
	if _, err := c.terminalOutput(idParams(id)); err == nil {
		t.Error("a released terminal is still addressable")
	}
}

func TestCancelRequestUnblocksWaitForExit(t *testing.T) {
	requireSh(t)
	c, p, _ := newCLIHarness(t, "")
	id := c.mustCreate(t, "-c", "sleep 30")

	p.send(map[string]any{"jsonrpc": "2.0", "id": "42", "method": "terminal/wait_for_exit",
		"params": map[string]any{"terminalId": id}})
	// Give the handler a moment to register itself before cancelling it.
	time.Sleep(50 * time.Millisecond)
	p.sendRaw(`{"jsonrpc":"2.0","method":"$/cancel_request","params":{"requestId":"42"}}`)

	reply := make(chan map[string]any, 1)
	go func() { reply <- p.recv() }()
	select {
	case m := <-reply:
		if m["id"] != "42" {
			t.Errorf("reply id = %v, want 42", m["id"])
		}
		if _, ok := m["error"]; !ok {
			t.Errorf("cancelled wait replied %#v, want an error", m)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("wait_for_exit ignored $/cancel_request")
	}
}

func TestRequestPermissionSelectsByNumber(t *testing.T) {
	c, _, _ := newCLIHarness(t, "2\n")
	params, _ := json.Marshal(permissionRequest{
		SessionId: "s1",
		ToolCall:  permissionToolCall{ToolCallId: "tc1", Title: "Run `rm -rf build`?"},
		Options: []permissionOption{
			{OptionId: "yes", Name: "Allow", Kind: "allow_once"},
			{OptionId: "no", Name: "Deny", Kind: "reject_once"},
		},
	})
	res, err := c.requestPermission(params)
	if err != nil {
		t.Fatal(err)
	}
	got := res.(permissionResponse)
	if got.Outcome.Outcome != "selected" || got.Outcome.OptionId != "no" {
		t.Errorf("outcome = %#v, want selected/no", got.Outcome)
	}
}

func TestRequestPermissionEmptyLineDismisses(t *testing.T) {
	c, _, _ := newCLIHarness(t, "\n")
	c.interactive = true
	params, _ := json.Marshal(permissionRequest{
		SessionId: "s1",
		Options:   []permissionOption{{OptionId: "yes", Name: "Allow"}},
	})
	res, err := c.requestPermission(params)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.(permissionResponse); got.Outcome.Outcome != "cancelled" {
		t.Errorf("outcome = %#v, want cancelled", got.Outcome)
	}
}

func TestElicitReturnsTheChosenConst(t *testing.T) {
	c, _, _ := newCLIHarness(t, "3\n")
	params, _ := json.Marshal(map[string]any{
		"sessionId": "s1",
		"mode":      "form",
		"message":   "Which base image?",
		"requestedSchema": map[string]any{
			"type": "object",
			"properties": map[string]any{
				elicitChoiceKey: map[string]any{"type": "string", "oneOf": []map[string]any{
					{"const": "alpine", "title": "Alpine"},
					{"const": "arch", "title": "Arch"},
					{"const": "debian", "title": "Debian"},
				}},
			},
		},
	})
	res, err := c.elicit(params)
	if err != nil {
		t.Fatal(err)
	}
	m := res.(map[string]any)
	if m["action"] != "accept" {
		t.Fatalf("action = %v, want accept", m["action"])
	}
	if got := m["content"].(map[string]string)[elicitChoiceKey]; got != "debian" {
		t.Errorf("choice = %q, want debian", got)
	}
}

func TestElicitAcceptsTypedText(t *testing.T) {
	c, _, _ := newCLIHarness(t, "use the staging bucket\n")
	params, _ := json.Marshal(map[string]any{
		"sessionId": "s1",
		"message":   "Which bucket?",
		"requestedSchema": map[string]any{
			"properties": map[string]any{elicitTextKey: map[string]any{"type": "string"}},
			"required":   []string{elicitTextKey},
		},
	})
	res, err := c.elicit(params)
	if err != nil {
		t.Fatal(err)
	}
	m := res.(map[string]any)
	if m["action"] != "accept" {
		t.Fatalf("action = %v", m["action"])
	}
	if got := m["content"].(map[string]string)[elicitTextKey]; got != "use the staging bucket" {
		t.Errorf("text = %q", got)
	}
}

func TestElicitDeclinesSchemasItCannotRender(t *testing.T) {
	c, _, _ := newCLIHarness(t, "")
	params, _ := json.Marshal(map[string]any{
		"sessionId":       "s1",
		"requestedSchema": map[string]any{"properties": map[string]any{"colour": map[string]any{"type": "string"}}},
	})
	res, err := c.elicit(params)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.(map[string]any)["action"]; got != "decline" {
		t.Errorf("action = %v, want decline", got)
	}
}

func TestAskRejectsOutOfRangeThenGivesUp(t *testing.T) {
	c, _, buf := newCLIHarness(t, "9\n0\nnope\n")
	c.interactive = true
	idx, typed := c.askChoiceOrText("Pick", []string{"one", "two"}, false)
	if idx != -1 || typed != "" {
		t.Errorf("got (%d, %q), want a dismissal", idx, typed)
	}
	if !strings.Contains(buf.String(), "pick a number between 1 and 2") {
		t.Errorf("no complaint about the bad input: %q", buf.String())
	}
}

func TestCommandRouting(t *testing.T) {
	c, _, _ := newCLIHarness(t, "")
	cases := []struct {
		line        string
		wantHandled bool
		wantQuit    bool
	}{
		{"/quit", true, true},
		{"/exit", true, true},
		{"/help", true, false},
		{"/cwd", true, false},
		// Not ours: the agent parses these itself.
		{"/clean", false, false},
		{"/settings", false, false},
		{"/some-macro with args", false, false},
		{"just a prompt", false, false},
	}
	for _, tc := range cases {
		handled, quit := c.command(tc.line)
		if handled != tc.wantHandled || quit != tc.wantQuit {
			t.Errorf("command(%q) = (%v, %v), want (%v, %v)", tc.line, handled, quit, tc.wantHandled, tc.wantQuit)
		}
	}
}

func TestHelpListsAgentCommands(t *testing.T) {
	c, p, buf := newCLIHarness(t, "")
	p.send(map[string]any{"jsonrpc": "2.0", "method": "session/update", "params": map[string]any{
		"sessionId": "s1",
		"update": map[string]any{
			"sessionUpdate":     "available_commands_update",
			"availableCommands": []map[string]any{{"name": "clean", "description": "delete session logs"}},
		},
	}})
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		c.cmdMu.Lock()
		n := len(c.commands)
		c.cmdMu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	c.help()
	out := buf.String()
	if !strings.Contains(out, "/quit") {
		t.Errorf("client commands missing: %q", out)
	}
	if !strings.Contains(out, "clean") || !strings.Contains(out, "delete session logs") {
		t.Errorf("agent commands missing: %q", out)
	}
}

func TestBlockText(t *testing.T) {
	cases := []struct {
		in   ContentBlock
		want string
	}{
		{ContentBlock{Type: "text", Text: "hello"}, "hello"},
		{ContentBlock{Type: "image", MimeType: "image/png"}, "[image image/png]"},
		{ContentBlock{Type: "resource_link", URI: "file:///a.go"}, "[file:///a.go]"},
		{ContentBlock{Type: "resource", Resource: &EmbeddedResource{Text: "inline"}}, "inline"},
	}
	for _, c := range cases {
		if got := blockText(c.in); got != c.want {
			t.Errorf("blockText(%+v) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestAdoptSwitchesSession(t *testing.T) {
	c, _, _ := newCLIHarness(t, "")
	c.ui.Usage(4000, 8000)
	c.adopt("s2")
	if c.sessionID() != "s2" {
		t.Errorf("session id = %q", c.sessionID())
	}
	if c.ui.used != 0 || c.ui.size != 0 {
		t.Errorf("usage carried over: %d/%d", c.ui.used, c.ui.size)
	}
}

// replFor runs the repl on piped input the test writes; it returns the writer and a
// channel closed when the repl returns.
func replFor(t *testing.T) (*cliClient, *peer, io.Writer, chan struct{}) {
	t.Helper()
	pr, pw := io.Pipe()
	c, p, _ := newCLIHarness(t, "")
	c.in = bufio.NewReader(pr)
	c.setSessionID("s1")
	done := make(chan struct{})
	go func() {
		c.repl()
		close(done)
	}()
	t.Cleanup(func() { pw.Close() })
	// The repl's reader runs before anything is asked: a question that came first would read
	// stdin itself, as in a one-shot -p run.
	for deadline := time.Now().Add(3 * time.Second); ; time.Sleep(time.Millisecond) {
		c.waitMu.Lock()
		up := c.inputs != nil
		c.waitMu.Unlock()
		if up {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the repl never started reading")
		}
	}
	return c, p, pw, done
}

// within receives the client's next message, failing the test rather than hanging it.
func within(t *testing.T, p *peer) map[string]any {
	t.Helper()
	got := make(chan map[string]any, 1)
	go func() { got <- p.recv() }()
	select {
	case m := <-got:
		return m
	case <-time.After(3 * time.Second):
		t.Fatal("the client sent nothing")
		return nil
	}
}

// A /spec loop is one turn that runs for hours: a line typed while it runs reaches the
// agent at once, as a second prompt it folds into the turn, not after the turn ends.
func TestTypingDuringATurnSteersIt(t *testing.T) {
	_, p, in, done := replFor(t)
	fmt.Fprintln(in, "/spec")
	turn := p.recv()
	if turn["method"] != "session/prompt" {
		t.Fatalf("first request = %v", turn["method"])
	}
	fmt.Fprintln(in, "/spec stop")
	steer := within(t, p) // the turn is still unanswered
	text, _ := json.Marshal(steer["params"])
	if steer["method"] != "session/prompt" || !strings.Contains(string(text), "/spec stop") {
		t.Fatalf("typed during the turn: %v %s", steer["method"], text)
	}
	p.reply(steer, map[string]any{"stopReason": "end_turn"})
	p.reply(turn, map[string]any{"stopReason": "end_turn"})
	fmt.Fprintln(in, "/quit")
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the repl did not leave on /quit")
	}
}

// With the repl's reader owning stdin, the line typed while a question is open answers it
// and starts no turn.
func TestALineAnswersTheOpenQuestion(t *testing.T) {
	c, p, in, done := replFor(t)
	p.send(map[string]any{"jsonrpc": "2.0", "id": 7, "method": "session/request_permission", "params": map[string]any{
		"sessionId": "s1", "toolCall": map[string]any{"toolCallId": "c1", "title": "Start the spec loop anyway?"},
		"options": []map[string]any{{"optionId": "go", "name": "Start anyway", "kind": "allow_once"}, {"optionId": "no", "name": "Stop", "kind": "reject_once"}}}})
	for deadline := time.Now().Add(3 * time.Second); ; time.Sleep(5 * time.Millisecond) {
		c.waitMu.Lock()
		up := c.waiting != nil
		c.waitMu.Unlock()
		if up {
			break // the question is up before the answer is typed
		}
		if time.Now().After(deadline) {
			t.Fatal("the question never opened")
		}
	}
	fmt.Fprintln(in, "2")
	resp := within(t, p)
	if out, _ := json.Marshal(resp["result"]); resp["id"] != float64(7) || !strings.Contains(string(out), `"optionId":"no"`) {
		t.Fatalf("answer = %v %s", resp["id"], out)
	}
	fmt.Fprintln(in, "/quit")
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the repl did not leave on /quit")
	}
}

func TestPasteIsOnePrompt(t *testing.T) {
	c, _, _ := newCLIHarness(t, "why does this fail\npanic: nil map\n\tmain.go:12\n")
	c.interactive = true
	got, err := c.readPrompt()
	if err != nil {
		t.Fatalf("readPrompt: %v", err)
	}
	want := "why does this fail\npanic: nil map\n\tmain.go:12"
	if got != want {
		t.Errorf("readPrompt() = %q, want %q", got, want)
	}
}

func TestPipedLinesStaySeparate(t *testing.T) {
	c, _, _ := newCLIHarness(t, "/mode\n/quit\n")
	c.interactive = false
	for _, want := range []string{"/mode", "/quit"} {
		got, err := c.readPrompt()
		if err != nil {
			t.Fatalf("readPrompt: %v", err)
		}
		if got != want {
			t.Errorf("readPrompt() = %q, want %q", got, want)
		}
	}
}

func TestSecondInterruptLeaves(t *testing.T) {
	c, _, buf := newCLIHarness(t, "")
	quit := make(chan struct{})
	c.hardQuit = func() { close(quit) }
	c.installSignals()

	self, err := os.FindProcess(os.Getpid())
	if err != nil {
		t.Fatalf("FindProcess: %v", err)
	}
	if err := self.Signal(os.Interrupt); err != nil {
		t.Fatalf("signal: %v", err)
	}
	select {
	case <-quit:
		t.Fatal("one Ctrl+C left the CLI")
	case <-time.After(300 * time.Millisecond):
	}
	if !strings.Contains(buf.String(), "Ctrl+C again") {
		t.Errorf("no hint about how to leave: %q", buf.String())
	}

	if err := self.Signal(os.Interrupt); err != nil {
		t.Fatalf("signal: %v", err)
	}
	select {
	case <-quit:
	case <-time.After(3 * time.Second):
		t.Fatal("a second Ctrl+C did not leave")
	}
}

func TestOneShotExitsOnTheStopReason(t *testing.T) {
	for _, tc := range []struct {
		stop string
		want int
	}{{"end_turn", 0}, {"cancelled", 1}, {"refusal", 1}} {
		c, p, buf := newCLIHarness(t, "")
		code := make(chan int, 1)
		go func() { code <- c.run(false, "", "do the thing", true) }()

		p.reply(p.recv(), map[string]any{"protocolVersion": protocolVersion})
		p.reply(p.recv(), map[string]any{"sessionId": "s1"})
		req := p.recv()
		if req["method"] != "session/prompt" {
			t.Fatalf("expected session/prompt, got %v", req["method"])
		}
		p.reply(req, map[string]any{"stopReason": tc.stop})

		select {
		case got := <-code:
			if got != tc.want {
				t.Errorf("stopReason %q exited %d, want %d", tc.stop, got, tc.want)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("stopReason %q never returned", tc.stop)
		}
		if strings.Contains(buf.String(), "codehalter cli") || strings.Contains(buf.String(), "❯") {
			t.Errorf("one-shot printed session chrome: %q", buf.String())
		}
	}
}

func TestNewSessionSwapsTheSessionID(t *testing.T) {
	c, p, _ := newCLIHarness(t, "")
	c.adopt("old")
	done := make(chan struct{})
	go func() {
		c.newSession()
		close(done)
	}()
	req := p.recv()
	if req["method"] != "session/new" {
		t.Fatalf("expected session/new, got %v", req["method"])
	}
	p.reply(req, map[string]any{"sessionId": "fresh"})
	<-done
	if c.sessionID() != "fresh" {
		t.Errorf("session id = %q, want fresh", c.sessionID())
	}
}

func TestResumeByIDLoadsThatSession(t *testing.T) {
	c, p, _ := newCLIHarness(t, "")
	c.adopt("old")
	done := make(chan struct{})
	go func() {
		c.resumeSession("20260821_085603")
		close(done)
	}()
	req := p.recv()
	if req["method"] != "session/load" {
		t.Fatalf("expected session/load, got %v", req["method"])
	}
	params := req["params"].(map[string]any)
	if params["sessionId"] != "20260821_085603" {
		t.Errorf("loaded %v", params["sessionId"])
	}
	// A response that omits the id means "the one you asked for".
	p.reply(req, map[string]any{})
	<-done
	if c.sessionID() != "20260821_085603" {
		t.Errorf("session id = %q", c.sessionID())
	}
}

func TestResumeWithoutIDPicksFromTheList(t *testing.T) {
	c, p, _ := newCLIHarness(t, "2\n")
	done := make(chan struct{})
	go func() {
		c.resumeSession("")
		close(done)
	}()
	req := p.recv()
	if req["method"] != "session/list" {
		t.Fatalf("expected session/list, got %v", req["method"])
	}
	p.reply(req, map[string]any{"sessions": []map[string]any{
		{"sessionId": "older", "updatedAt": "2026-08-01T00:00:00Z"},
		{"sessionId": "newest", "updatedAt": "2026-08-22T00:00:00Z"},
	}})
	req = p.recv()
	if req["method"] != "session/load" {
		t.Fatalf("expected session/load, got %v", req["method"])
	}
	// Newest first, so the second entry offered is the older one.
	if params := req["params"].(map[string]any); params["sessionId"] != "older" {
		t.Errorf("loaded %v, want older", params["sessionId"])
	}
	p.reply(req, map[string]any{})
	<-done
}

// exit and quit leave, as in a shell, with or without the slash; a sentence that starts with
// the word is a prompt.
// The typed line is edited at a cursor: arrows (CSI and SS3), Home and End, Delete, a word
// at a time, and Up and Down through the lines sent, the unsent line kept while browsing.
func TestCLILineEditing(t *testing.T) {
	const (
		left, right, home, end = "\x1b[D", "\x1b[C", "\x1b[H", "\x1b[F"
		up, down, del          = "\x1b[A", "\x1b[B", "\x1b[3~"
	)
	keys := strings.Join([]string{
		"helo" + left + "l\r",                                // insert in the middle
		"world" + home + "hello " + end + "!\r",              // Home and End, CSI
		"b\x1bOHa\x1bOF\x1bOD\x1bOC" + "\r",                  // Home, End, Left and Right, SS3
		"abc" + home + del + "\r",                            // Delete at the cursor
		"alpha beta\x1b[1;5D" + "X\r",                        // a word back
		"one two three" + strings.Repeat(left, 6) + "\x17\r", // Ctrl+W before the cursor
		"\x1bOPf1 ignored\r",                                 // an unknown key is ignored
		"dra" + up + up + down + down + "ft\r",               // browsing keeps the unsent line
		up + "\r",                                            // Up sends the last line again
	}, "")
	c, _, _ := newCLIHarness(t, keys)
	c.ui.inputOn = true
	c.inputs = make(chan cliLine, 32)
	c.readInput()
	var got []string
	for len(c.inputs) > 0 {
		if l := <-c.inputs; l.err == nil {
			got = append(got, l.text)
		}
	}
	want := []string{"hello", "hello world!", "ab", "bc", "alpha Xbeta", "one  three", "f1 ignored", "draft", "draft"}
	if !slices.Equal(got, want) {
		t.Errorf("lines = %q\nwant    %q", got, want)
	}
}

func TestCLIExitWordsLeave(t *testing.T) {
	c := &cliClient{}
	for _, line := range []string{"exit", "quit", " Exit ", "/quit", "/exit"} {
		if handled, quit := c.command(line); !handled || !quit {
			t.Errorf("%q: handled=%v quit=%v, want it to leave", line, handled, quit)
		}
	}
	if handled, _ := c.command("exit the loop when the suite is green"); handled {
		t.Error("a prompt starting with exit was taken as a command")
	}
}

// With no terminal (a script running -p), a question with options takes its first whenever
// the script gave no usable answer: input ended, an empty line, or a line that is no choice,
// which is not asked again so the script's later lines stay theirs. At a terminal an empty
// line or Ctrl+D dismisses it, and a question without options has nothing to take.
func TestCLIQuestionWithoutAnyoneTakesTheFirstOption(t *testing.T) {
	for _, tc := range []struct{ name, in, note string }{
		{"ended input", "", "no one to answer (input ended): took 1) Start anyway"},
		{"an empty line", "\n", "no one to answer (empty line): took 1) Start anyway"},
		{"no choice", "9\nkept\n", `no one to answer ("9" is not one of them): took 1) Start anyway`},
	} {
		c, _, out := newCLIHarness(t, tc.in)
		if idx, _ := c.askChoiceOrText("Start the spec loop anyway?", []string{"Start anyway", "Stop"}, false); idx != 0 ||
			!strings.Contains(out.String(), tc.note) {
			t.Errorf("%s: idx %d, output %q", tc.name, idx, out.String())
		}
		if tc.name == "no choice" {
			if rest, _ := c.in.ReadString('\n'); rest != "kept\n" {
				t.Errorf("no choice: the next line went too, left %q", rest)
			}
		}
	}
	c, _, _ := newCLIHarness(t, "2")
	if idx, _ := c.askChoiceOrText("Which?", []string{"a", "b"}, false); idx != 1 {
		t.Errorf("a last answer without its newline: idx %d, want 1", idx)
	}
	c, _, _ = newCLIHarness(t, "")
	if idx, typed := c.askChoiceOrText("Name it", nil, true); idx != -1 || typed != "" {
		t.Errorf("a text question at ended input: (%d, %q), want a dismissal", idx, typed)
	}
	for _, in := range []string{"", "\n"} {
		c, _, _ = newCLIHarness(t, in)
		c.interactive = true // Ctrl+D or Enter at a terminal is the user dismissing it
		if idx, _ := c.askChoiceOrText("Which?", []string{"a", "b"}, false); idx != -1 {
			t.Errorf("at a terminal, input %q: idx %d, want a dismissal", in, idx)
		}
	}
}

// `</dev/null` is a character device, not a terminal: a script run with it got the dismissals
// meant for a person at a keyboard.
func TestIsTerminalLeavesOutTheNullDevice(t *testing.T) {
	null, err := os.Open(os.DevNull)
	if err != nil {
		t.Fatal(err)
	}
	defer null.Close()
	if isTerminal(null) {
		t.Error("the null device counts as a terminal")
	}
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()
	if isTerminal(r) {
		t.Error("a pipe counts as a terminal")
	}
}

// Piped into the repl, the script's input may end while its turn runs: the turn finishes,
// its questions take their first option instead of waiting for a line that cannot come, and
// only then does the CLI leave.
func TestScriptedReplFinishesTheTurnAfterInputEnds(t *testing.T) {
	c, p, in, done := replFor(t)
	fmt.Fprintln(in, "build it")
	prompt := within(t, p)
	if prompt["method"] != "session/prompt" {
		t.Fatalf("expected session/prompt, got %v", prompt["method"])
	}
	in.(io.Closer).Close()
	for deadline := time.Now().Add(3 * time.Second); ; time.Sleep(5 * time.Millisecond) {
		c.waitMu.Lock()
		ended := c.inputEnded
		c.waitMu.Unlock()
		select {
		case <-ended:
		default:
			if time.Now().After(deadline) {
				t.Fatal("the end of input was never noticed")
			}
			continue
		}
		break
	}
	select {
	case <-done:
		t.Fatal("the repl left with the turn still running")
	case <-time.After(50 * time.Millisecond):
	}
	p.send(map[string]any{"jsonrpc": "2.0", "id": 7, "method": "session/request_permission", "params": map[string]any{
		"sessionId": "s1", "toolCall": map[string]any{"toolCallId": "c1", "title": "Run /spec on spec/ now?"},
		"options": []map[string]any{{"optionId": "yes", "name": "Run /spec", "kind": "allow_once"}, {"optionId": "no", "name": "Let me read it first", "kind": "reject_once"}}}})
	resp := within(t, p)
	if out, _ := json.Marshal(resp["result"]); resp["id"] != float64(7) || !strings.Contains(string(out), `"optionId":"yes"`) {
		t.Fatalf("answer = %v %s", resp["id"], out)
	}
	p.reply(prompt, map[string]any{"stopReason": "end_turn"})
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the repl did not leave once the turn ended")
	}
}

// The /resume picker runs inside the repl's own loop, which is what hands lines to a
// question: it waited for a line nobody would pass on.
func TestResumePickerInsideTheRepl(t *testing.T) {
	c, p, in, done := replFor(t)
	fmt.Fprintln(in, "/resume")
	req := within(t, p)
	if req["method"] != "session/list" {
		t.Fatalf("expected session/list, got %v", req["method"])
	}
	p.reply(req, map[string]any{"sessions": []map[string]any{{"sessionId": "older", "updatedAt": "2026-08-01T00:00:00Z"}}})
	fmt.Fprintln(in, "1")
	req = within(t, p)
	if req["method"] != "session/load" {
		t.Fatalf("expected session/load, got %v", req["method"])
	}
	p.reply(req, map[string]any{})
	fmt.Fprintln(in, "/quit")
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the repl did not leave on /quit")
	}
	if c.sessionID() != "older" {
		t.Errorf("session id = %q, want older", c.sessionID())
	}
}

// In character mode the keys are edited here: backspace, Ctrl+W and Ctrl+U take text back,
// an arrow key is ignored, and a bracketed paste keeps its newlines in one prompt.
func TestKeysAreEditedInCharacterMode(t *testing.T) {
	pr, pw := io.Pipe()
	c, p, _ := newCLIHarness(t, "")
	c.in = bufio.NewReader(pr)
	c.setSessionID("s1")
	c.ui.inputOn = true // what startInput sets on a terminal
	go c.repl()
	t.Cleanup(func() { pw.Close() })
	for _, k := range []struct{ keys, want string }{
		{"ab\x7fc\x1b[Ad\r", "acd"},
		{"one two\x17three\r", "one three"},
		{"gone\x15kept\r", "kept"},
		{"\x1b[200~line one\nline two\x1b[201~\r", "line one\nline two"},
	} {
		if _, err := pw.Write([]byte(k.keys)); err != nil {
			t.Fatal(err)
		}
		req := within(t, p)
		params, _ := json.Marshal(req["params"])
		want, _ := json.Marshal(k.want)
		if !strings.Contains(string(params), `"text":`+string(want)) {
			t.Errorf("keys %q sent %s, want the text %s", k.keys, params, want)
		}
		p.reply(req, map[string]any{"stopReason": "end_turn"})
	}
}

// Ctrl+V attaches the clipboard's picture: a marker in the line, an image block in the prompt beside the
// text. A marker deleted before Enter leaves its picture out, and a model that takes no images gets none.
func TestCtrlVAttachesTheClipboardImage(t *testing.T) {
	pr, pw := io.Pipe()
	c, p, out := newCLIHarness(t, "")
	c.in = bufio.NewReader(pr)
	c.setSessionID("s1")
	c.ui.inputOn = true
	c.imagesAccepted.Store(true)
	png := []byte("\x89PNG\r\n\x1a\nfake")
	c.clipboard = func() ([]byte, string, error) { return png, "image/png", nil }
	go c.repl()
	t.Cleanup(func() { pw.Close() })

	send := func(keys string) []ContentBlock {
		t.Helper()
		if _, err := pw.Write([]byte(keys)); err != nil {
			t.Fatal(err)
		}
		req := within(t, p)
		raw, _ := json.Marshal(req["params"])
		var got PromptRequest
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		p.reply(req, map[string]any{"stopReason": "end_turn"})
		return got.Content
	}

	content := send("what is wrong here? \x16\r")
	if len(content) != 2 || content[0].Text != "what is wrong here? [image 1]" {
		t.Fatalf("content = %+v, want the text with its marker and one image", content)
	}
	if img := content[1]; img.Type != "image" || img.MimeType != "image/png" ||
		img.Data != base64.StdEncoding.EncodeToString(png) {
		t.Errorf("image block = %+v, want the clipboard's PNG", img)
	}

	// The marker typed away (Ctrl+U), so the picture is not sent.
	if content := send("\x16\x15no picture\r"); len(content) != 1 {
		t.Errorf("a deleted marker still sent %d blocks", len(content))
	}

	// A model that cannot see: Ctrl+V says so and attaches nothing.
	c.imagesAccepted.Store(false)
	if content := send("\x16text only\r"); len(content) != 1 || strings.Contains(content[0].Text, "[image") {
		t.Errorf("content = %+v, want the text alone", content)
	}
	if !strings.Contains(out.String(), "this model takes no images") {
		t.Errorf("the refusal was not said: %q", out.String())
	}
}

// A terminal's command leads a process group of its own, as a pty gives one in Zed: in
// codehalter's group a job's group kill reached only its wrapper, and the Ctrl+C meant for
// the turn reached every job.
func TestCLITerminalLeadsItsOwnGroup(t *testing.T) {
	requireSh(t)
	c, _, _ := newCLIHarness(t, "")
	id := c.mustCreate(t, "-c", "sleep 5")
	term, err := c.terminal(idParams(id))
	if err != nil {
		t.Fatal(err)
	}
	pid := term.cmd.Process.Pid
	if pgid, err := syscall.Getpgid(pid); err != nil || pgid != pid {
		t.Errorf("terminal command %d is in group %d (%v), want its own", pid, pgid, err)
	}
	if _, err := c.terminalRelease(idParams(id)); err != nil {
		t.Error(err)
	}
}
