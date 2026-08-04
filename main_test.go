package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"
)

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

// withFreshToolRegistry saves and restores the package-global registeredTools
// around a test body, so registering fake tools doesn't leak into other tests.
func withFreshToolRegistry(t *testing.T) {
	t.Helper()
	saved := registeredTools
	registeredTools = nil
	t.Cleanup(func() { registeredTools = saved })
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
