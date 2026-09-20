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
	"testing"
	"time"
)

// ptr returns a pointer to v, for struct literals with *int/*bool fields
// (Settings.Parallel and friends, which distinguish unset from zero).
func ptr[T any](v T) *T { return &v }

// lineageClock hands out call times one second apart. Most of the lineage
// tests do not care when their calls happened, but noteCacheLineage records the
// gap between them now, and feeding it one frozen instant everywhere would
// leave that arithmetic exercised nowhere. A second is a healthy tool-loop
// cadence, so every call these tests make reads as "too soon to be an idle
// eviction" unless a test says otherwise.
func lineageClock() func() time.Time {
	at := time.Date(2026, 8, 21, 22, 0, 0, 0, time.UTC)
	return func() time.Time {
		at = at.Add(time.Second)
		return at
	}
}

// newTestAgent returns an agent with one session rooted at a fresh tempdir.
// a.conn is left nil so sendUpdate becomes a no-op (covered by the nil-check).
func newTestAgent(t *testing.T) (*agent, *Session) {
	t.Helper()
	s, err := newSession(t.TempDir())
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	return &agent{sessions: map[string]*Session{s.ID: s}}, s
}

// elicitingAgent wires an agent to a pipe-backed connection and advertises
// form elicitation, so a caller that prefers elicitation (Ask*, the MCP import)
// takes the elicitation/create path. Returns the peer end for the test to read
// the request off and answer it.
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

// ---------------------------------------------------------------------------
// Fake ACP client with terminals
// ---------------------------------------------------------------------------

// terminalHarness wires an agent to a pipe-backed connection, advertises the
// terminal capability, and serves terminal/* by actually running the command
// with os/exec. codehalter runs no processes of its own any more (see
// terminal.go), so without a client that can, nothing in a test executes at all
// — this stands in for the editor. It also records every method the agent sent
// and every session/update it emitted, so a test can assert on the wire traffic
// and on what the tool-call card ended up holding.
type terminalHarness struct {
	agent *agent
	sess  *Session

	mu      sync.Mutex
	terms   map[string]*fakeTerminal
	seq     int
	methods []string
	updates []map[string]any
}

// fakeTerminal is one running command. exit is valid once done is closed.
type fakeTerminal struct {
	cmd  *exec.Cmd
	out  bytes.Buffer // guarded by terminalHarness.mu
	done chan struct{}
	exit terminalExit
}

// termWriter funnels a command's output into the harness buffer under the
// harness lock, so terminal/output can read it while the command is writing.
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
			// Serve off the read loop: terminal/wait_for_exit blocks until the
			// command finishes, and a later terminal/kill is what finishes some of
			// them, so handling inline would deadlock the harness against itself.
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
		if term := h.term(p.TerminalId); term != nil && term.cmd.Process != nil {
			_ = term.cmd.Process.Kill()
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

// exitOf converts a cmd.Wait error into the ACP exit status: a code for a
// normal exit, a signal name (and no code) for a kill.
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

// killAll reaps anything a test left running (a background job, a command
// killed mid-flight) so a failing assertion can't leak a process.
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

// sawMethod waits briefly for the agent to send method. Polling rather than a
// straight read because the harness records off its own goroutine: the last
// message of a tool call is often still in the pipe when the call has returned.
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

// waitForStatus waits for a tool-call update with the given status and returns
// it, or nil if none arrived.
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

// updatesOfKind returns every session/update with the given sessionUpdate kind.
func (h *terminalHarness) updatesOfKind(kind string) []map[string]any {
	var out []map[string]any
	for _, u := range h.sentUpdates() {
		if u["sessionUpdate"] == kind {
			out = append(out, u)
		}
	}
	return out
}

// waitForKind waits for an update of the given kind and returns the first one,
// or nil if none arrived.
func (h *terminalHarness) waitForKind(kind string) map[string]any {
	h.waitFor(func() bool { return len(h.updatesOfKind(kind)) > 0 })
	if got := h.updatesOfKind(kind); len(got) > 0 {
		return got[0]
	}
	return nil
}

// embeddedTerminal reports whether any update put a terminal in the card.
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

// withTools gives a exactly the tools ts, for tests that must not depend on the
// real ones.
func withTools(a *agent, ts ...Tool) {
	a.tools.tools = map[string]Tool{}
	a.tools.add(ts...)
}

// ---------------------------------------------------------------------------
// Fake LLM server
// ---------------------------------------------------------------------------

// mockLLM stands up an httptest server that accepts OpenAI chat-completions
// requests and returns a queued SSE response for each call. Tests queue one
// response per LLM call they expect; an unexpected call fails the test.
type mockLLM struct {
	ts    *httptest.Server
	resps []string

	mu   sync.Mutex
	reqs []map[string]any // captured request bodies, in order
	idx  atomic.Int32
	t    *testing.T
}

func newMockLLM(t *testing.T, responses ...string) *mockLLM {
	t.Helper()
	m := &mockLLM{resps: responses, t: t}
	m.ts = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Runtime callers probe /slots before each LLM call. Mock doesn't
		// implement it — 404 lets connFor treat the server as "unknown,
		// assume available" so the chat-completions path still runs.
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

// sseText builds an SSE body with a single text-delta chunk followed by [DONE].
func sseText(text string) string {
	chunk := map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{"content": text},
		}},
	}
	data, _ := json.Marshal(chunk)
	return fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", data)
}

// sseTruncated emits a reasoning delta with finish_reason="length", then a usage
// chunk, then [DONE] — a generation that truncated at a length limit.
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

// sseTruncatedContent is sseTruncated's cousin where the truncated output is
// message content (not reasoning) — a verbose/looping generation rather than a
// <think> stall, so it classifies as a genuine max_tokens cap (not recoverable).
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

// sseToolCall builds an SSE body that emits a single tool call with the given
// name + JSON args, then [DONE]. First chunk carries the tool-call id (triggers
// append); the second delta extends arguments (per the llmStream protocol).
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

// sseContentThenToolCall emits a content delta (assistant prose) followed by a
// single tool call, then [DONE] — the shape a planner produces when it writes a
// direct answer AND calls submit_plan in the same turn.
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

// sseReasoning emits a reasoning_content delta with NO visible content and no
// tool call — a thinking model that dumped everything into its (never-shown)
// reasoning channel.
func sseReasoning(reasoning string) string {
	chunk := map[string]any{
		"choices": []map[string]any{{
			"delta": map[string]any{"reasoning_content": reasoning},
		}},
	}
	data, _ := json.Marshal(chunk)
	return fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", data)
}

// TestCwdOrDefaultAbsolutes pins the contract that sess.Cwd is always an
// absolute path. A client may send Cwd: "." over ACP, and an unresolved
// "." breaks resolvePath's prefix check (filepath.Clean drops the leading
// "./" so "go.mod" matches neither "./" nor "."). resolvePath then rejects
// every project-relative path with "outside project directory".
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

// TestCwdAvailable pins the guard that stops session/new and session/load from
// scaffolding .codehalter under a workspace root that isn't mounted here — the
// case where Zed restores an agent thread pinned to another project's cwd.
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
	// Real pacing is seconds; the goroutine is the same either way.
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

	// stop() joins the ticker goroutine, but the notifications it wrote are
	// still in flight over the pipe, so wait for the closing newline to land.
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

	// stop() joins the ticker goroutine, so nothing follows the newline.
	n := len(got)
	time.Sleep(20 * time.Millisecond)
	if after := len(chunks()); after != n {
		t.Errorf("%d chunk(s) arrived after stop", after-n)
	}
}

// TestHeartbeatSilentWhenFast pins the no-op case: work that finishes inside
// one interval must not leave a stray newline in the thread.
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
