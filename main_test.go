package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

func ptr[T any](v T) *T { return &v }

// One second apart: a healthy tool-loop cadence, never read as an idle eviction.
func lineageClock() func() time.Time {
	at := time.Date(2026, 8, 21, 22, 0, 0, 0, time.UTC)
	return func() time.Time {
		at = at.Add(time.Second)
		return at
	}
}

// a.conn is left nil, so sendUpdate is a no-op.
func newTestAgent(t *testing.T) (*agent, *Session) {
	t.Helper()
	s, err := newSession(t.TempDir())
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	return &agent{sessions: map[string]*Session{s.ID: s}}, s
}

// elicitingAgent advertises form elicitation; the test reads and answers on the returned peer end.
func elicitingAgent(t *testing.T) (*agent, *Session, *bufio.Reader, *os.File) {
	t.Helper()
	a, s := newTestAgent(t)
	agentW, agentR, peerW, peerR := pipePair(t)
	a.conn = NewAgentSideConnection(a, agentW, agentR)
	a.clientCaps.Elicitation = &struct {
		Form *struct{} `json:"form"`
		URL  *struct{} `json:"url"`
	}{Form: &struct{}{}}
	return a, s, bufio.NewReader(peerR), peerW
}

// codehalter runs no processes of its own, so this stands in for the editor: it
// serves terminal/* with os/exec and records the wire traffic.
type terminalHarness struct {
	agent *agent
	sess  *Session

	mu      sync.Mutex
	terms   map[string]*fakeTerminal
	seq     int
	methods []string
	updates []map[string]any
}

// exit is valid once done is closed.
type fakeTerminal struct {
	cmd  *exec.Cmd
	out  bytes.Buffer // guarded by terminalHarness.mu
	done chan struct{}
	exit terminalExit
}

type termWriter struct {
	h *terminalHarness
	t *fakeTerminal
}

func (w termWriter) Write(p []byte) (int, error) {
	w.h.mu.Lock()
	defer w.h.mu.Unlock()
	return w.t.out.Write(p)
}

func newTerminalHarness(t *testing.T) *terminalHarness {
	t.Helper()
	a, s := newTestAgent(t)
	agentW, agentR, peerW, peerR := pipePair(t)
	a.conn = NewAgentSideConnection(a, agentW, agentR)
	a.clientCaps.Terminal = true
	h := &terminalHarness{agent: a, sess: s, terms: map[string]*fakeTerminal{}}
	t.Cleanup(h.killAll)

	var wmu sync.Mutex
	br := bufio.NewReader(peerR)
	go func() {
		for {
			line, err := br.ReadString('\n')
			if err != nil {
				return
			}
			var msg struct {
				ID     *json.RawMessage `json:"id"`
				Method string           `json:"method"`
				Params json.RawMessage  `json:"params"`
			}
			if json.Unmarshal([]byte(line), &msg) != nil {
				continue
			}
			if msg.ID == nil {
				if msg.Method == "session/update" {
					var p struct {
						Update map[string]any `json:"update"`
					}
					_ = json.Unmarshal(msg.Params, &p)
					h.mu.Lock()
					h.updates = append(h.updates, p.Update)
					h.mu.Unlock()
				}
				continue
			}
			h.mu.Lock()
			h.methods = append(h.methods, msg.Method)
			h.mu.Unlock()
			// Off the read loop: wait_for_exit blocks until a later terminal/kill
			// may finish the command, so inline would deadlock.
			go func(id json.RawMessage, method string, params json.RawMessage) {
				body, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": id, "result": h.serve(method, params)})
				if err != nil {
					return
				}
				wmu.Lock()
				defer wmu.Unlock()
				_, _ = peerW.Write(append(body, '\n'))
			}(*msg.ID, msg.Method, msg.Params)
		}
	}()
	return h
}

func (h *terminalHarness) serve(method string, params json.RawMessage) any {
	var p struct {
		Command    string   `json:"command"`
		Args       []string `json:"args"`
		Cwd        string   `json:"cwd"`
		TerminalId string   `json:"terminalId"`
	}
	_ = json.Unmarshal(params, &p)

	switch method {
	case "terminal/create":
		cmd := exec.Command(p.Command, p.Args...)
		cmd.Dir = p.Cwd
		// Own process group, as under a pty: the job wrapper's group kill needs a leader.
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		term := &fakeTerminal{cmd: cmd, done: make(chan struct{})}
		cmd.Stdout = termWriter{h, term}
		cmd.Stderr = termWriter{h, term}
		if err := cmd.Start(); err != nil {
			return map[string]any{}
		}
		go func() {
			err := cmd.Wait()
			h.mu.Lock()
			term.exit = exitOf(err)
			h.mu.Unlock()
			close(term.done)
		}()
		h.mu.Lock()
		h.seq++
		id := "term-" + strconv.Itoa(h.seq)
		h.terms[id] = term
		h.mu.Unlock()
		return map[string]any{"terminalId": id}

	case "terminal/output":
		term := h.term(p.TerminalId)
		if term == nil {
			return map[string]any{"output": ""}
		}
		res := map[string]any{"output": h.outputOf(term)}
		select {
		case <-term.done:
			h.mu.Lock()
			res["exitStatus"] = term.exit
			h.mu.Unlock()
		default:
		}
		return res

	case "terminal/wait_for_exit":
		term := h.term(p.TerminalId)
		if term == nil {
			return map[string]any{}
		}
		<-term.done
		h.mu.Lock()
		defer h.mu.Unlock()
		return term.exit

	case "terminal/kill":
		// SIGHUP to the leader only, as a pty delivers: the wrapper's trap must take
		// the group down, or a child holding the pipe blocks cmd.Wait.
		if term := h.term(p.TerminalId); term != nil && term.cmd.Process != nil {
			_ = term.cmd.Process.Signal(syscall.SIGHUP)
		}
		return map[string]any{}

	case "terminal/release":
		term := h.term(p.TerminalId)
		if term == nil {
			return map[string]any{}
		}
		h.mu.Lock()
		delete(h.terms, p.TerminalId)
		h.mu.Unlock()
		if term.cmd.Process != nil {
			_ = term.cmd.Process.Kill() // release kills a live process, per the spec
		}
		return map[string]any{}
	}
	return map[string]any{}
}

func exitOf(err error) terminalExit {
	if err == nil {
		zero := 0
		return terminalExit{ExitCode: &zero}
	}
	if ee, ok := err.(*exec.ExitError); ok {
		if code := ee.ExitCode(); code >= 0 {
			return terminalExit{ExitCode: &code}
		}
		return terminalExit{Signal: "SIGKILL"}
	}
	return terminalExit{Signal: "SIGKILL"}
}

func (h *terminalHarness) term(id string) *fakeTerminal {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.terms[id]
}

func (h *terminalHarness) outputOf(t *fakeTerminal) string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return t.out.String()
}

func (h *terminalHarness) killAll() {
	h.mu.Lock()
	terms := make([]*fakeTerminal, 0, len(h.terms))
	for _, t := range h.terms {
		terms = append(terms, t)
	}
	h.terms = map[string]*fakeTerminal{}
	h.mu.Unlock()
	for _, t := range terms {
		if t.cmd.Process != nil {
			_ = t.cmd.Process.Kill()
		}
	}
}

func (h *terminalHarness) sentMethods() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.methods...)
}

func (h *terminalHarness) sentUpdates() []map[string]any {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]map[string]any(nil), h.updates...)
}

// Polls: a call's last message is often still in the pipe when the call returns.
func (h *terminalHarness) sawMethod(method string) bool {
	return h.waitFor(func() bool {
		for _, m := range h.sentMethods() {
			if m == method {
				return true
			}
		}
		return false
	})
}

func (h *terminalHarness) waitForStatus(status string) map[string]any {
	var found map[string]any
	h.waitFor(func() bool {
		for _, u := range h.sentUpdates() {
			if u["status"] == status {
				found = u
				return true
			}
		}
		return false
	})
	return found
}

func (h *terminalHarness) updatesOfKind(kind string) []map[string]any {
	var out []map[string]any
	for _, u := range h.sentUpdates() {
		if u["sessionUpdate"] == kind {
			out = append(out, u)
		}
	}
	return out
}

func (h *terminalHarness) waitForKind(kind string) map[string]any {
	h.waitFor(func() bool { return len(h.updatesOfKind(kind)) > 0 })
	if got := h.updatesOfKind(kind); len(got) > 0 {
		return got[0]
	}
	return nil
}

func (h *terminalHarness) embeddedTerminal() bool {
	for _, u := range h.sentUpdates() {
		content, _ := u["content"].([]any)
		for _, c := range content {
			if cm, _ := c.(map[string]any); cm["type"] == "terminal" && cm["terminalId"] != "" {
				return true
			}
		}
	}
	return false
}

func (h *terminalHarness) waitFor(cond func() bool) bool {
	for i := 0; i < 400; i++ {
		if cond() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return false
}

func withTools(a *agent, ts ...Tool) {
	a.tools.tools = map[string]Tool{}
	a.tools.add(ts...)
}

// mockLLM serves one queued SSE response per call; an unexpected call fails the test.
type mockLLM struct {
	ts    *httptest.Server
	resps []string

	mu   sync.Mutex
	reqs []map[string]any
	idx  atomic.Int32
	t    *testing.T
}

func newMockLLM(t *testing.T, responses ...string) *mockLLM {
	t.Helper()
	m := &mockLLM{resps: responses, t: t}
	m.ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A 404 on /slots makes connFor treat the server as available.
		if r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("mockLLM: decode request body: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		m.mu.Lock()
		m.reqs = append(m.reqs, body)
		m.mu.Unlock()

		i := int(m.idx.Add(1)) - 1
		if i >= len(m.resps) {
			t.Errorf("mockLLM: unexpected call %d (only %d responses queued)", i+1, len(m.resps))
			http.Error(w, "no response queued", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(m.resps[i]))
	}))
	return m
}

func (m *mockLLM) Close() { m.ts.Close() }

func (m *mockLLM) conn(name string) *LLMConnection {
	return &LLMConnection{Tag: name, Server: m.ts.URL, Model: "test-model"}
}

func (m *mockLLM) callCount() int { return int(m.idx.Load()) }

func (m *mockLLM) request(i int) map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	if i < 0 || i >= len(m.reqs) {
		return nil
	}
	return m.reqs[i]
}

func sseText(text string) string {
	chunk := map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{"content": text},
		}},
	}
	data, _ := json.Marshal(chunk)
	return fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", data)
}

func sseTruncated(reasoning string, promptTokens, completionTokens int) string {
	var b strings.Builder
	c1, _ := json.Marshal(map[string]any{"choices": []map[string]any{{
		"delta":         map[string]any{"reasoning_content": reasoning},
		"finish_reason": "length",
	}}})
	fmt.Fprintf(&b, "data: %s\n\n", c1)
	c2, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{},
		"usage":   map[string]any{"prompt_tokens": promptTokens, "completion_tokens": completionTokens},
	})
	fmt.Fprintf(&b, "data: %s\n\n", c2)
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

// Content, not reasoning: classifies as a genuine max_tokens cap, not a recoverable stall.
func sseTruncatedContent(content string, promptTokens, completionTokens int) string {
	var b strings.Builder
	c1, _ := json.Marshal(map[string]any{"choices": []map[string]any{{
		"delta":         map[string]any{"content": content},
		"finish_reason": "length",
	}}})
	fmt.Fprintf(&b, "data: %s\n\n", c1)
	c2, _ := json.Marshal(map[string]any{
		"choices": []map[string]any{},
		"usage":   map[string]any{"prompt_tokens": promptTokens, "completion_tokens": completionTokens},
	})
	fmt.Fprintf(&b, "data: %s\n\n", c2)
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func sseToolCall(id, name, args string) string {
	var b strings.Builder
	first := map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{
				"tool_calls": []map[string]any{{
					"id":   id,
					"type": "function",
					"function": map[string]any{
						"name":      name,
						"arguments": args,
					},
				}},
			},
		}},
	}
	d, _ := json.Marshal(first)
	fmt.Fprintf(&b, "data: %s\n\n", d)
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func sseContentThenToolCall(text, id, name, args string) string {
	var b strings.Builder
	c1, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]any{"content": text}}}})
	fmt.Fprintf(&b, "data: %s\n\n", c1)
	c2, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]any{"tool_calls": []map[string]any{{
		"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": args},
	}}}}}})
	fmt.Fprintf(&b, "data: %s\n\n", c2)
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

func TestCwdOrDefaultAbsolutes(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	for _, in := range []string{".", "", "./"} {
		got := cwdOrDefault(in)
		if !filepath.IsAbs(got) {
			t.Errorf("cwdOrDefault(%q) = %q, want absolute path", in, got)
		}
		if got != cwd {
			t.Errorf("cwdOrDefault(%q) = %q, want %q", in, got, cwd)
		}
	}
}

func TestCwdAvailable(t *testing.T) {
	dir := t.TempDir()
	if err := cwdAvailable(dir); err != nil {
		t.Errorf("cwdAvailable(existing dir) = %v, want nil", err)
	}

	missing := filepath.Join(dir, "does-not-exist")
	if err := cwdAvailable(missing); err == nil {
		t.Errorf("cwdAvailable(missing) = nil, want error")
	}

	file := filepath.Join(dir, "afile")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if err := cwdAvailable(file); err == nil {
		t.Errorf("cwdAvailable(file) = nil, want error")
	}
}

func TestHeartbeatDotsThenClosesLine(t *testing.T) {
	h := newTerminalHarness(t)
	old := heartbeatEvery
	heartbeatEvery = 5 * time.Millisecond
	t.Cleanup(func() { heartbeatEvery = old })

	chunks := func() []string {
		var out []string
		for _, u := range h.updatesOfKind("agent_message_chunk") {
			content, _ := u["content"].(map[string]any)
			text, _ := content["text"].(string)
			out = append(out, text)
		}
		return out
	}

	stop := h.agent.heartbeat(context.Background(), h.sess.ID)
	if !h.waitFor(func() bool { return len(chunks()) >= 2 }) {
		t.Fatalf("no heartbeat dots arrived, got %q", chunks())
	}
	stop()

	// stop() joins the goroutine, but its notifications may still be in the pipe.
	last := func() string {
		c := chunks()
		if len(c) == 0 {
			return ""
		}
		return c[len(c)-1]
	}
	if !h.waitFor(func() bool { return last() == "\n" }) {
		t.Fatalf("dot line was never closed, got %q", chunks())
	}

	got := chunks()
	if len(got) < 3 {
		t.Fatalf("want dots plus a closing newline, got %q", got)
	}
	for i, c := range got[:len(got)-1] {
		if c != "." {
			t.Errorf("chunk %d = %q, want a dot", i, c)
		}
	}

	n := len(got)
	time.Sleep(20 * time.Millisecond)
	if after := len(chunks()); after != n {
		t.Errorf("%d chunk(s) arrived after stop", after-n)
	}
}

func TestHeartbeatSilentWhenFast(t *testing.T) {
	h := newTerminalHarness(t)
	old := heartbeatEvery
	heartbeatEvery = time.Hour
	t.Cleanup(func() { heartbeatEvery = old })

	stop := h.agent.heartbeat(context.Background(), h.sess.ID)
	stop()
	if got := h.updatesOfKind("agent_message_chunk"); len(got) != 0 {
		t.Errorf("heartbeat that never ticked emitted %d chunk(s)", len(got))
	}
}

func TestSayLandsInSessionLog(t *testing.T) {
	h := newTerminalHarness(t)
	h.agent.say(context.Background(), h.sess.ID, "🧪 `just test` passed in the round, after its last change; not run again\n")
	h.agent.say(context.Background(), h.sess.ID, "  \n")
	data, err := os.ReadFile(sessionPath(h.sess.Cwd, h.sess.ID, "log"))
	if err != nil {
		t.Fatalf("no session log: %v", err)
	}
	if !strings.Contains(string(data), "[SAY] ===\n🧪 `just test` passed in the round") {
		t.Errorf("the line is not in the log:\n%s", data)
	}
	if strings.Count(string(data), "[SAY]") != 1 {
		t.Errorf("a blank line was logged:\n%s", data)
	}
}

// callTool runs one tool call the way the loop does and returns what it recorded.
func callTool(t *testing.T, a *agent, sid, name, args string) ToolUse {
	t.Helper()
	var tc toolCall
	tc.ID, tc.Function.Name, tc.Function.Arguments = "c-"+name, name, args
	tu, _ := a.runToolCall(context.Background(), sid, tc)
	return tu
}

// writeTree writes files under dir, making their directories.
func writeTree(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	for rel, body := range files {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// writeFiles creates empty files.
func writeFiles(t *testing.T, dir string, names ...string) {
	t.Helper()
	files := map[string]string{}
	for _, n := range names {
		files[n] = ""
	}
	writeTree(t, dir, files)
}

// gitInit makes dir a repository with an identity for later commits, and commits what it holds.
func gitInit(t *testing.T, dir string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git")
	}
	for _, args := range [][]string{{"init", "-q"}, {"config", "user.email", "t@t"}, {"config", "user.name", "t"}, {"add", "-A"}, {"commit", "-q", "-m", "base"}} {
		if out, err := exec.Command("git", append([]string{"-C", dir}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v %s", args, err, out)
		}
	}
}

// gitRepo is a new repository holding files, committed.
func gitRepo(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	writeTree(t, dir, files)
	gitInit(t, dir)
	return dir
}

// sseWriteFile is a scripted write_file call.
func sseWriteFile(id, path, content string) string {
	b, err := json.Marshal(map[string]string{"path": path, "content": content})
	if err != nil {
		panic(err)
	}
	return sseToolCall(id, "write_file", string(b))
}

func lastUserMessage(s *Session) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.Messages) - 1; i >= 0; i-- {
		if s.Messages[i].Role == "user" {
			return s.Messages[i].Content
		}
	}
	return ""
}
