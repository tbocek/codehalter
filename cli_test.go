package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"
)

// syncBuf is the UI's output while a serve goroutine is writing to it and the
// test is reading it.
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

// peer is the other end of the wire: it stands in for the agent, so the client
// is exercised over real line-delimited JSON-RPC rather than by calling its
// methods directly.
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

// reply answers a request the client sent us, echoing its id back.
func (p *peer) reply(req map[string]any, result any) {
	p.t.Helper()
	p.send(map[string]any{"jsonrpc": "2.0", "id": req["id"], "result": result})
}

// recv reads one message the client sent us.
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

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

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
	// The agent quotes its ids and matches replies by the string inside the
	// quotes; a client numbering them as JSON numbers would still work, but
	// mirroring the agent keeps one convention on the wire.
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

// A closed connection must fail every waiting request rather than hanging: at a
// prompt that is the difference between "the agent died" and a frozen terminal.
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

// Streamed chunks must reach the screen in the order they were sent. Handling
// them in goroutines (as the agent does for its own inbound requests) would
// shuffle a sentence into nonsense, which is why session/update is handled on
// the read loop.
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

// ---------------------------------------------------------------------------
// Terminals
// ---------------------------------------------------------------------------

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
	// stdout and stderr are one stream here, exactly as in a real terminal.
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
	// The id stays valid after a kill: run_command's idle watchdog kills first
	// and reads the output afterwards.
	if _, err := c.terminalOutput(idParams(id)); err != nil {
		t.Errorf("output after kill: %v", err)
	}
}

// The agent asks for a byte cap and expects the client to enforce it by keeping
// the TAIL (see terminalOutputLimit), because it does its own head+tail elision
// on top.
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

// A command must not inherit the user's terminal: one that reads stdin would
// swallow the keystrokes meant for the prompt.
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

// $/cancel_request has to reach a handler that blocks on something other than
// the user, or an interrupted turn leaves the client waiting on a command the
// agent has already given up on.
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

// ---------------------------------------------------------------------------
// Asking the user
// ---------------------------------------------------------------------------

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

// The option ids, not the labels, are what the agent maps back to its own
// choices (see doElicitation).
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

// ask_user's free-text form is the reason the CLI advertises elicitation at
// all: without it the agent fails those with errNoFreeText.
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
	// Declining is the point: an unanswerable form must not leave the agent
	// blocked on a dialog the CLI never showed.
	if got := res.(map[string]any)["action"]; got != "decline" {
		t.Errorf("action = %v, want decline", got)
	}
}

func TestAskRejectsOutOfRangeThenGivesUp(t *testing.T) {
	c, _, buf := newCLIHarness(t, "9\n0\nnope\n")
	idx, typed := c.askChoiceOrText("Pick", []string{"one", "two"}, false)
	if idx != -1 || typed != "" {
		t.Errorf("got (%d, %q), want a dismissal", idx, typed)
	}
	if !strings.Contains(buf.String(), "pick a number between 1 and 2") {
		t.Errorf("no complaint about the bad input: %q", buf.String())
	}
}

// ---------------------------------------------------------------------------
// Client-side commands and content
// ---------------------------------------------------------------------------

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
		// Not ours: the agent parses "/name args" itself, so swallowing an
		// unknown slash command would break every TEMPLATE-*.md macro.
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
		// An image the terminal cannot draw is named, not dropped: a screenshot
		// that renders as nothing looks like a tool that did nothing.
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

func TestAdoptRecordsAvailableModes(t *testing.T) {
	c, _, _ := newCLIHarness(t, "")
	c.ui.Usage(4000, 8000)
	c.adopt("s2", &SessionModeState{
		CurrentModeId: "Interactive",
		AvailableModes: []struct {
			Id          string `json:"id"`
			Name        string `json:"name"`
			Description string `json:"description,omitempty"`
		}{{Id: "Interactive"}, {Id: "Autopilot"}},
	})
	if len(c.modes) != 2 || c.modes[1] != "Autopilot" {
		t.Errorf("modes = %v", c.modes)
	}
	if c.ui.mode != "Interactive" {
		t.Errorf("ui mode = %q", c.ui.mode)
	}
	if c.sessionID() != "s2" {
		t.Errorf("session id = %q", c.sessionID())
	}
	// The meter described the conversation we just left.
	if c.ui.used != 0 || c.ui.size != 0 {
		t.Errorf("usage carried over: %d/%d", c.ui.used, c.ui.size)
	}
}

// ---------------------------------------------------------------------------
// Input ownership and prompts
// ---------------------------------------------------------------------------

// A cancelled turn returns while its question is still parked in a read on
// stdin. Without the token, the input loop starts a second read on the same
// bufio.Reader, which is a data race and hands the user's line to whichever
// goroutine wins.
func TestReplWaitsForAnOpenQuestion(t *testing.T) {
	pr, pw := io.Pipe()
	defer pw.Close()
	c, _, _ := newCLIHarness(t, "")
	c.in = bufio.NewReader(pr)

	c.askMu.Lock() // stand in for the question still on screen
	done := make(chan struct{})
	go func() {
		c.repl()
		close(done)
	}()
	go pw.Write([]byte("/quit\n"))

	select {
	case <-done:
		t.Fatal("the input loop read stdin while a question owned it")
	case <-time.After(200 * time.Millisecond):
	}

	c.askMu.Unlock()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the input loop never took the token back")
	}
}

// A paste arrives as one chunk, so every line of it is one prompt. Without
// this, pasting a ten-line stack trace runs ten turns.
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

// Piped input is buffered exactly the same way, but there each line is its own
// command: joining would turn a two-command script into one nonsense prompt.
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

// The first Ctrl+C at an idle prompt says how to leave, because a blocked read
// on stdin cannot be unblocked; the second one within the window means it.
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

// ---------------------------------------------------------------------------
// Sessions and one-shot
// ---------------------------------------------------------------------------

// -p is what a script runs, so the exit status has to mean something: 0 only
// when the turn actually finished.
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
		// No banner and no prompt row: the output is the turn and nothing else.
		if strings.Contains(buf.String(), "codehalter cli") || strings.Contains(buf.String(), "❯") {
			t.Errorf("one-shot printed session chrome: %q", buf.String())
		}
	}
}

func TestNewSessionSwapsTheSessionID(t *testing.T) {
	c, p, _ := newCLIHarness(t, "")
	c.adopt("old", nil)
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
	c.adopt("old", nil)
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

// /resume with no id offers the list, newest first, and loads the pick.
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
