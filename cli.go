package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ---------------------------------------------------------------------------
// Standalone CLI client.
//
// codehalter is an ACP agent, so it normally sits behind an editor that speaks
// the client half of the protocol. --cli supplies that half itself: it starts
// the same agent in-process, wires the two halves together with a pair of
// pipes, and drives it from a terminal. Nothing about the agent is special-
// cased for it. It goes through initialize / session/new / session/prompt and
// answers session/request_permission, elicitation/create and terminal/* exactly
// as Zed does, which is the point: if the CLI works, the protocol works.
//
// The pipes look redundant when both ends are in one process, and a direct call
// would be faster. They are what keeps this honest. Every byte the agent emits
// is real ACP on a real wire, so a bug in framing, ordering or capability
// gating shows up here rather than being papered over by an in-process
// shortcut, and pointing the client at a codehalter SUBPROCESS later is a
// change of two lines (the io.Pipe pair becomes cmd.StdinPipe/StdoutPipe).
//
// Two capabilities are not optional here:
//
//   - terminal, because ensureTerminals aborts any session whose client didn't
//     advertise it. There is no in-process exec fallback in the agent, by
//     design, so the CLI implements real terminals over os/exec.
//   - elicitation.form, because ask_user's free-text form has no
//     session/request_permission equivalent (buttons can't carry a typed
//     answer) and fails with errNoFreeText without it.
//
// fs is deliberately NOT advertised. fs/read_text_file exists so an editor can
// serve unsaved buffer contents; the CLI has no buffers, and the agent's own
// fallback reads the same file off the same disk. Diffs still arrive as
// ToolCallContent, because the agent computes those itself either way.
//
// Like every other way of running codehalter, this must run INSIDE the
// devcontainer (ensureDevcontainer aborts otherwise), which also means the
// terminals below spawn processes in the container, matching Zed exactly.
// ---------------------------------------------------------------------------

// cliUsage is printed for --help and for a flag we don't know.
const cliUsage = `usage: codehalter --cli [--cwd DIR] [--resume [SESSION_ID]] [-p PROMPT]

  --cli               run the standalone terminal client instead of an ACP server
  --cwd DIR           project directory (default: current directory)
  --resume [ID]       continue a stored session; without an ID, the most recent
                      one for this directory
  -p PROMPT           run one turn and exit, instead of opening a prompt. The
                      exit status is 0 only when the turn finished normally,
                      so a script can branch on it.
`

func runCLI(argv []string) int {
	cwd, _ := os.Getwd()
	resumeID, prompt := "", ""
	resume, oneshot := false, false
	for i := 0; i < len(argv); i++ {
		switch argv[i] {
		case "--cwd":
			if i+1 >= len(argv) {
				fmt.Fprintln(os.Stderr, "--cwd needs a directory")
				return 2
			}
			i++
			cwd = argv[i]
		case "--resume":
			resume = true
			if i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "-") {
				i++
				resumeID = argv[i]
			}
		case "-p", "--prompt":
			if i+1 >= len(argv) {
				fmt.Fprintln(os.Stderr, "-p needs a prompt")
				return 2
			}
			i++
			prompt, oneshot = argv[i], true
		case "--help", "-h":
			fmt.Print(cliUsage)
			return 0
		default:
			fmt.Fprintf(os.Stderr, "unknown flag %q\n\n%s", argv[i], cliUsage)
			return 2
		}
	}
	abs, err := filepath.Abs(cwd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolving %s: %v\n", cwd, err)
		return 2
	}
	cwd = abs

	// The agent logs at debug on every turn. On stderr that is a wall of text
	// straight through the TUI, so both slog AND os.Stderr are redirected to a
	// file for the whole run: os.Stderr as well because tool_web hands firefox's
	// output to it directly, and a child process writing over the live region
	// would desync the cursor arithmetic. Runtime panics still reach fd 2 and
	// the real terminal, which is where a panic belongs.
	if logf := openCLILog(cwd); logf != nil {
		defer logf.Close()
		slog.SetDefault(slog.New(slog.NewTextHandler(logf, &slog.HandlerOptions{Level: slog.LevelDebug})))
		os.Stderr = logf
	} else {
		slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	}

	// Two pipes, one per direction: the agent reads what the client writes and
	// the client reads what the agent writes.
	agentIn, clientOut := io.Pipe()
	clientIn, agentOut := io.Pipe()

	a := &agent{sessions: make(map[string]*Session), mode: "Interactive", standalone: true}
	acp := NewAgentSideConnection(a, agentOut, agentIn)
	a.conn = acp

	c := &cliClient{
		ui:          newCLIUI(),
		in:          bufio.NewReader(os.Stdin),
		cwd:         cwd,
		interactive: stdinIsTTY(),
		terms:       map[string]*cliTerminal{},
	}
	c.conn = newCLIConn(clientOut, clientIn, c)

	// One shutdown for both ways out: the normal return below, and the hard exit
	// a second Ctrl+C takes. That one cannot unwind the stack, because the repl
	// is parked in a read on stdin that no signal interrupts, so it has to run
	// the same teardown from the signal goroutine instead of returning.
	shutdown := func() {
		// Close our end first: that EOFs the agent's reader and ends its serve
		// loop, which is the same shutdown path a departing editor triggers.
		clientOut.Close()
		select {
		case <-acp.Done():
		case <-time.After(2 * time.Second):
		}
		agentOut.Close()
		a.shutdownMCP()
		a.shutdownBackground()
		c.releaseAll()
	}
	c.hardQuit = func() {
		shutdown()
		os.Exit(130)
	}

	code := c.run(resume, resumeID, prompt, oneshot)
	shutdown()
	return code
}

// stdinIsTTY reports whether input comes from a terminal rather than a pipe or
// a file. Two things turn on it: the UI echoes submitted lines itself when
// nothing else will, and only a terminal's input is joined on paste, because
// piped lines arrive in one chunk too and are meant to stay separate prompts.
func stdinIsTTY() bool {
	fi, err := os.Stdin.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// openCLILog opens .codehalter/cli.log, creating the directory if the project
// hasn't been used yet. Returns nil (and the caller discards logs) rather than
// failing the run: a read-only project directory is a reason to lose the log,
// not a reason to refuse to work.
//
// Truncated per run rather than appended: this is debug-level output for the
// session you are in, it grows by megabytes an hour, and nothing collects it.
func openCLILog(cwd string) *os.File {
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil
	}
	f, err := os.OpenFile(filepath.Join(dir, "cli.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND|os.O_TRUNC, 0o644)
	if err != nil {
		return nil
	}
	return f
}

// ---------------------------------------------------------------------------
// cliClient — the ACP client half
// ---------------------------------------------------------------------------

type cliClient struct {
	ui   *cliUI
	conn *cliConn
	in   *bufio.Reader
	cwd  string

	// interactive is stdin-is-a-terminal, which is what decides whether a
	// second line arriving with the first is a paste or just the next command.
	interactive bool

	// hardQuit tears down the agent and exits, for the one exit that cannot
	// return through runCLI. Tests leave it nil and get the hint instead.
	hardQuit func()

	// sid is swapped by /new and /resume on the input goroutine and read by the
	// signal goroutine, so it goes through sessionID/setSessionID.
	sidMu sync.Mutex
	sid   string

	modes []string

	// commands is what the agent advertised through available_commands_update,
	// shown by /help next to the client's own commands.
	cmdMu    sync.Mutex
	commands []availableCommand

	// turnActive is read by the signal handler to decide whether Ctrl+C cancels
	// a turn or just prints a hint, so it can't take the UI lock.
	turnActive atomic.Bool

	// askMu is the stdin token. Exactly one goroutine may read at a time: the
	// input loop when idle, or one inline question during a turn. The agent can
	// have two tool calls waiting on the user at once (parallel tool dispatch),
	// and a cancelled turn returns while its question is still parked in a read,
	// so "no turn is running" is not enough to make the loop safe on its own.
	askMu sync.Mutex

	termMu   sync.Mutex
	terms    map[string]*cliTerminal
	termSeq  int
	released bool
}

func (c *cliClient) sessionID() string {
	c.sidMu.Lock()
	defer c.sidMu.Unlock()
	return c.sid
}

func (c *cliClient) setSessionID(id string) {
	c.sidMu.Lock()
	c.sid = id
	c.sidMu.Unlock()
}

func (c *cliClient) run(resume bool, resumeID, prompt string, oneshot bool) int {
	if _, err := c.conn.request("initialize", map[string]any{
		"protocolVersion": protocolVersion,
		"clientCapabilities": map[string]any{
			"terminal":    true,
			"elicitation": map[string]any{"form": map[string]any{}},
			"fs":          map[string]any{"readTextFile": false, "writeTextFile": false},
		},
	}); err != nil {
		fmt.Fprintf(os.Stdout, "initialize failed: %v\n", err)
		return 1
	}

	if err := c.openSession(resume, resumeID); err != nil {
		fmt.Fprintf(os.Stdout, "%v\n", err)
		return 1
	}
	c.installSignals()

	// One-shot prints no banner and opens no prompt row: the output is the turn
	// and nothing else, so the run reads as a command rather than as a session.
	if oneshot {
		if c.turn(prompt) {
			return 0
		}
		return 1
	}

	c.ui.Note(ansiBold, "codehalter cli")
	c.ui.Note(ansiDim, "  "+c.cwd)
	c.ui.Note(ansiDim, "  session "+c.sessionID())
	c.ui.Note(ansiDim, "  /help for commands, Ctrl+C to interrupt a turn, Ctrl+D to quit")
	return c.repl()
}

// openSession picks up an existing session or starts a new one. A --resume with
// no id takes the most recently updated session for this directory, which is
// what "carry on where I left off" means at a shell prompt.
func (c *cliClient) openSession(resume bool, resumeID string) error {
	if resume && resumeID == "" {
		sessions, err := c.storedSessions()
		if err != nil {
			return err
		}
		if len(sessions) == 0 {
			return errors.New("no stored session for this directory")
		}
		resumeID = sessions[0].SessionId
	}
	if resumeID != "" {
		return c.loadSession(resumeID)
	}

	raw, err := c.conn.request("session/new", NewSessionRequest{Cwd: c.cwd})
	if err != nil {
		return fmt.Errorf("session/new failed: %w", err)
	}
	var res NewSessionResponse
	if err := json.Unmarshal(raw, &res); err != nil {
		return fmt.Errorf("session/new returned nonsense: %w", err)
	}
	if res.SessionId == "" {
		return errors.New("session/new returned no session id")
	}
	c.adopt(res.SessionId, res.Modes)
	return nil
}

// loadSession resumes a stored session by id.
func (c *cliClient) loadSession(id string) error {
	raw, err := c.conn.request("session/load", LoadSessionRequest{SessionId: id, Cwd: c.cwd})
	if err != nil {
		return fmt.Errorf("session/load failed: %w", err)
	}
	var res LoadSessionResponse
	if err := json.Unmarshal(raw, &res); err != nil {
		return fmt.Errorf("session/load returned nonsense: %w", err)
	}
	// The response may echo the id back or leave it out, meaning "the one you
	// asked for".
	if res.SessionId != "" {
		id = res.SessionId
	}
	c.adopt(id, res.Modes)
	return nil
}

// storedSessions lists this directory's sessions, newest first. UpdatedAt is
// ISO 8601, so lexical order is chronological order. The agent already sorts,
// but a client that relies on that is a client that breaks quietly the day it
// stops being true.
func (c *cliClient) storedSessions() ([]SessionInfo, error) {
	raw, err := c.conn.request("session/list", ListSessionsRequest{Cwd: c.cwd})
	if err != nil {
		return nil, fmt.Errorf("session/list failed: %w", err)
	}
	var list ListSessionsResponse
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("session/list returned nonsense: %w", err)
	}
	sort.Slice(list.Sessions, func(i, j int) bool {
		return list.Sessions[i].UpdatedAt > list.Sessions[j].UpdatedAt
	})
	return list.Sessions, nil
}

// adopt switches to a session the agent just handed us. The context meter is
// zeroed because it describes the conversation we just left; the agent sends a
// usage_update for the new one on its first turn.
func (c *cliClient) adopt(sid string, modes *SessionModeState) {
	c.setSessionID(sid)
	c.modes = nil
	if modes != nil {
		for _, m := range modes.AvailableModes {
			c.modes = append(c.modes, m.Id)
		}
		c.ui.Mode(modes.CurrentModeId)
	}
	c.ui.Usage(0, 0)
}

// installSignals makes Ctrl+C interrupt the turn rather than the process. A
// blocking read on stdin is restarted by the Go runtime after a signal, so
// Ctrl+C at an idle prompt cannot unblock the reader: the first one says how to
// leave, and a second one within two seconds is taken as meaning it. That exit
// can't unwind the stack (the read is still parked), so it goes through
// hardQuit, which runs the same teardown the normal path does.
func (c *cliClient) installSignals() {
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt)
	go func() {
		var last time.Time
		for range sigs {
			if c.turnActive.Load() {
				c.ui.Note(ansiYell, "  interrupting…")
				if err := c.conn.notify("session/cancel", CancelNotification{SessionId: c.sessionID()}); err != nil {
					slog.Debug("cli: cancel failed", "err", err)
				}
				continue
			}
			if time.Since(last) < 2*time.Second && c.hardQuit != nil {
				c.ui.Note(ansiDim, "")
				c.hardQuit()
			}
			last = time.Now()
			c.ui.Note(ansiDim, "  (Ctrl+C again, or Ctrl+D, to leave)")
		}
	}()
}

// repl is the input loop.
func (c *cliClient) repl() int {
	for {
		// askMu, not just "no turn is running": a cancelled turn returns while
		// its question is still parked in a read on the same bufio.Reader, and
		// two readers on one of those is a data race, not merely a confusing
		// prompt. Held across the echo so the prompt row has one owner too.
		c.askMu.Lock()
		c.ui.Prompt("\n❯ ")
		text, err := c.readPrompt()
		c.ui.Submitted(text)
		c.askMu.Unlock()

		if err != nil {
			if !errors.Is(err, io.EOF) {
				c.ui.Note(ansiRed, "input: "+err.Error())
			}
			c.ui.Note(ansiDim, "")
			return 0
		}
		if text == "" {
			continue
		}
		if handled, quit := c.command(text); handled {
			if quit {
				return 0
			}
			continue
		}
		c.turn(text)
	}
}

// readPrompt reads one prompt. A paste is several lines that arrive in a single
// chunk, so anything still buffered the instant the first line ends came in with
// it and belongs to the same prompt: without this, pasting a stack trace runs
// one turn per line. Typing can't trip it, since a human leaves the reader idle
// between lines. Only for a terminal: piped input is buffered the same way, but
// there each line is meant to be its own prompt. A paste larger than bufio's
// buffer splits at that boundary, which is what every line did before.
func (c *cliClient) readPrompt() (string, error) {
	line, err := c.in.ReadString('\n')
	for err == nil && c.interactive && c.in.Buffered() > 0 {
		var next string
		next, err = c.in.ReadString('\n')
		line += next
	}
	return strings.TrimSpace(line), err
}

// command handles the commands the CLIENT owns. Everything else beginning with
// a slash is passed through untouched: the agent parses "/name args" itself
// (see splitMacro) for /clean, /settings and every TEMPLATE-*.md macro, so the
// client must not swallow names it doesn't recognise.
func (c *cliClient) command(text string) (handled, quit bool) {
	name, rest, _ := strings.Cut(text, " ")
	rest = strings.TrimSpace(rest)
	switch name {
	case "/quit", "/exit":
		return true, true
	case "/help":
		c.help()
	case "/mode":
		c.setMode(rest)
	case "/sessions":
		c.printSessions()
	case "/new":
		c.newSession()
	case "/resume":
		c.resumeSession(rest)
	case "/cwd":
		c.ui.Note(ansiDim, "  "+c.cwd+"  (session "+c.sessionID()+")")
	default:
		return false, false
	}
	return true, false
}

func (c *cliClient) help() {
	c.ui.Note(ansiBold, "\nclient commands")
	for _, l := range []string{
		"  /help              this list",
		"  /mode [name]       show or switch the session mode",
		"  /sessions          stored sessions for this directory",
		"  /new               start a fresh session here",
		"  /resume [id]       switch to a stored session, or pick one from a list",
		"  /cwd               project directory and session id",
		"  /quit              leave (same as Ctrl+D)",
	} {
		c.ui.Note("", l)
	}
	c.cmdMu.Lock()
	cmds := append([]availableCommand(nil), c.commands...)
	c.cmdMu.Unlock()
	if len(cmds) == 0 {
		return
	}
	c.ui.Note(ansiBold, "\nagent commands")
	for _, cmd := range cmds {
		c.ui.Note("", fmt.Sprintf("  /%-17s %s", cmd.Name, cmd.Description))
	}
}

func (c *cliClient) setMode(mode string) {
	if mode == "" {
		c.ui.Note(ansiDim, "  modes: "+strings.Join(c.modes, ", "))
		return
	}
	for _, m := range c.modes {
		if strings.EqualFold(m, mode) {
			if _, err := c.conn.request("session/set_mode", SetSessionModeRequest{SessionId: c.sessionID(), ModeId: m}); err != nil {
				c.ui.Note(ansiRed, "  set_mode: "+err.Error())
				return
			}
			c.ui.Mode(m)
			c.ui.Note(ansiDim, "  mode "+m)
			return
		}
	}
	c.ui.Note(ansiRed, "  no such mode: "+mode+" (have: "+strings.Join(c.modes, ", ")+")")
}

func (c *cliClient) printSessions() {
	sessions, err := c.storedSessions()
	if err != nil {
		c.ui.Note(ansiRed, "  "+err.Error())
		return
	}
	if len(sessions) == 0 {
		c.ui.Note(ansiDim, "  no stored sessions")
		return
	}
	current := c.sessionID()
	for _, s := range sessions {
		mark := "  "
		if s.SessionId == current {
			mark = "→ "
		}
		c.ui.Note(ansiDim, fmt.Sprintf("%s%s  %s", mark, s.SessionId, s.UpdatedAt))
	}
	c.ui.Note(ansiDim, "  switch with /resume <id>")
}

// newSession starts a fresh conversation without leaving the CLI. The old one
// is not closed: it stays on disk and /resume brings it back, exactly like
// opening a second thread in an editor.
func (c *cliClient) newSession() {
	raw, err := c.conn.request("session/new", NewSessionRequest{Cwd: c.cwd})
	if err != nil {
		c.ui.Note(ansiRed, "  session/new: "+err.Error())
		return
	}
	var res NewSessionResponse
	if err := json.Unmarshal(raw, &res); err != nil || res.SessionId == "" {
		c.ui.Note(ansiRed, "  session/new returned no session id")
		return
	}
	c.adopt(res.SessionId, res.Modes)
	c.ui.Note(ansiDim, "  session "+res.SessionId)
}

// resumeSession switches to a stored session. With no id it offers the list,
// which is the only way to pick one without first running /sessions and copying
// a timestamp by hand.
func (c *cliClient) resumeSession(id string) {
	if id == "" {
		sessions, err := c.storedSessions()
		if err != nil {
			c.ui.Note(ansiRed, "  "+err.Error())
			return
		}
		if len(sessions) == 0 {
			c.ui.Note(ansiDim, "  no stored sessions")
			return
		}
		labels := make([]string, len(sessions))
		for i, s := range sessions {
			labels[i] = s.SessionId + "  " + s.UpdatedAt
		}
		idx, _ := c.askChoiceOrText("Resume which session?", labels, false)
		if idx < 0 {
			return
		}
		id = sessions[idx].SessionId
	}
	if err := c.loadSession(id); err != nil {
		c.ui.Note(ansiRed, "  "+err.Error())
		return
	}
	c.ui.Note(ansiDim, "  session "+c.sessionID())
}

// turn sends one prompt and blocks until the agent reports a stop reason. It
// reports whether the turn finished normally, which is what -p exits on. The
// render ticker runs only for the duration of the turn: with nothing in flight
// there is nothing to animate, so an idle CLI wakes up zero times a second.
func (c *cliClient) turn(text string) bool {
	c.turnActive.Store(true)
	c.ui.Begin()
	stop := c.startTicker()

	raw, err := c.conn.request("session/prompt", PromptRequest{
		SessionId: c.sessionID(),
		Content:   []ContentBlock{{Type: "text", Text: text}},
	})

	stop()
	c.turnActive.Store(false)
	c.ui.End()

	if err != nil {
		c.ui.Note(ansiRed, "  turn failed: "+err.Error())
		return false
	}
	var res PromptResponse
	if err := json.Unmarshal(raw, &res); err != nil {
		c.ui.Note(ansiRed, "  turn returned nonsense: "+err.Error())
		return false
	}
	// end_turn is the normal case and needs no announcement; the others explain
	// why the agent stopped short.
	switch res.StopReason {
	case "", "end_turn":
		return true
	case "cancelled":
		c.ui.Note(ansiYell, "  interrupted")
	default:
		c.ui.Note(ansiYell, "  stopped: "+res.StopReason)
	}
	return false
}

func (c *cliClient) startTicker() func() {
	stop := make(chan struct{})
	go func() {
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for {
			select {
			case <-stop:
				return
			case <-t.C:
				c.pumpTails()
				c.ui.Tick()
			}
		}
	}()
	return func() { close(stop) }
}

// pumpTails publishes the tail of every running command to the UI. It is a pull
// on the render tick rather than a push from the pipe reader on purpose: a
// chatty build writes thousands of times a second and each push would be a full
// redraw.
func (c *cliClient) pumpTails() {
	c.termMu.Lock()
	terms := make([]*cliTerminal, 0, len(c.terms))
	for _, t := range c.terms {
		terms = append(terms, t)
	}
	c.termMu.Unlock()
	for _, t := range terms {
		c.ui.TerminalTail(t.id, t.tail(cliTermTail))
	}
}

// ---------------------------------------------------------------------------
// Inbound: session/update
// ---------------------------------------------------------------------------

func (c *cliClient) sessionUpdate(params json.RawMessage) {
	var p struct {
		SessionId string          `json:"sessionId"`
		Update    json.RawMessage `json:"update"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		slog.Debug("cli: bad session/update", "err", err)
		return
	}
	var kind struct {
		Kind string `json:"sessionUpdate"`
	}
	if err := json.Unmarshal(p.Update, &kind); err != nil {
		return
	}
	switch kind.Kind {
	case KindAgentMessage, KindAgentThought, KindUserMessage:
		var m messageChunk
		if json.Unmarshal(p.Update, &m) != nil {
			return
		}
		sty, pre := "", ""
		switch kind.Kind {
		case KindAgentThought:
			sty, pre = ansiDim, "  "
		case KindUserMessage:
			sty, pre = ansiDim, "❯ "
		}
		c.ui.Stream(sty, pre, blockText(m.Content))

	case "tool_call", "tool_call_update":
		var up toolCallUpdate
		if json.Unmarshal(p.Update, &up) != nil {
			return
		}
		c.ui.Card(up)

	case "plan":
		var pl planUpdate
		if json.Unmarshal(p.Update, &pl) != nil {
			return
		}
		c.ui.Plan(pl.Entries)

	case "usage_update":
		var u usageUpdate
		if json.Unmarshal(p.Update, &u) != nil {
			return
		}
		c.ui.Usage(u.Used, u.Size)

	case "current_mode_update":
		var m struct {
			CurrentModeId string `json:"currentModeId"`
		}
		if json.Unmarshal(p.Update, &m) != nil {
			return
		}
		c.ui.Mode(m.CurrentModeId)

	case "available_commands_update":
		var a availableCommandsUpdate
		if json.Unmarshal(p.Update, &a) != nil {
			return
		}
		c.cmdMu.Lock()
		c.commands = a.Commands
		c.cmdMu.Unlock()

	case "session_info_update":
		var s sessionInfoUpdate
		if json.Unmarshal(p.Update, &s) != nil || s.Title == "" {
			return
		}
		c.ui.Note(ansiDim, "  ["+s.Title+"]")
	}
}

// blockText renders one content block as terminal text. Images are named rather
// than drawn: the CLI has no way to show one, and silently dropping the block
// would make a screenshot look like it produced nothing.
func blockText(b ContentBlock) string {
	switch b.Type {
	case "text":
		return b.Text
	case "image":
		return "[image " + b.MimeType + "]"
	case "resource_link":
		return "[" + b.URI + "]"
	case "resource":
		if b.Resource != nil && b.Resource.Text != "" {
			return b.Resource.Text
		}
	}
	return b.Text
}

// ---------------------------------------------------------------------------
// Inbound: asking the user
// ---------------------------------------------------------------------------

// requestPermission renders the button-only form: N options, pick one by
// number. Empty input dismisses, which the agent reads as "cancelled" and is
// how you back out without choosing.
func (c *cliClient) requestPermission(params json.RawMessage) (any, error) {
	var p permissionRequest
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, err
	}
	labels := make([]string, len(p.Options))
	for i, o := range p.Options {
		labels[i] = o.Name
	}
	title := p.ToolCall.Title
	if title == "" {
		title = "Permission required"
	}
	idx, _ := c.askChoiceOrText(title, labels, false)
	var resp permissionResponse
	if idx < 0 {
		resp.Outcome.Outcome = "cancelled"
		return resp, nil
	}
	resp.Outcome.Outcome = "selected"
	resp.Outcome.OptionId = p.Options[idx].OptionId
	return resp, nil
}

// elicit renders elicitation/create. Only "form" mode is implemented, with the
// two fields codehalter actually asks for: a single-select enum (choice) and a
// free-text box (text). Anything else in the schema is ignored rather than
// guessed at, and an unfillable form is declined so the agent gets an answer
// instead of a hang.
func (c *cliClient) elicit(params json.RawMessage) (any, error) {
	var p struct {
		Message         string `json:"message"`
		RequestedSchema struct {
			Properties map[string]struct {
				Title string `json:"title"`
				OneOf []struct {
					Const string `json:"const"`
					Title string `json:"title"`
				} `json:"oneOf"`
			} `json:"properties"`
		} `json:"requestedSchema"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, err
	}
	choice, hasChoice := p.RequestedSchema.Properties[elicitChoiceKey]
	_, hasText := p.RequestedSchema.Properties[elicitTextKey]
	if !hasChoice && !hasText {
		return map[string]any{"action": "decline"}, nil
	}

	labels := make([]string, len(choice.OneOf))
	for i, o := range choice.OneOf {
		labels[i] = o.Title
		if labels[i] == "" {
			labels[i] = o.Const
		}
	}
	idx, typed := c.askChoiceOrText(p.Message, labels, hasText)
	content := map[string]string{}
	switch {
	case idx >= 0:
		content[elicitChoiceKey] = choice.OneOf[idx].Const
	case typed != "":
		content[elicitTextKey] = typed
	default:
		return map[string]any{"action": "cancel"}, nil
	}
	return map[string]any{"action": "accept", "content": content}, nil
}

// askChoiceOrText is the one place that reads stdin during a turn. It suspends
// the live region first: the region is rewritten by the render ticker, and
// anything the user types into rows that get redrawn is erased under them.
//
// It does not take a context. The agent cancels an open dialog with
// $/cancel_request when a turn is interrupted, but a read already blocked on
// stdin cannot be abandoned safely: the line the user is halfway through typing
// would be delivered to whoever reads next, i.e. it would become their next
// prompt. So a cancelled question stays on screen and Enter dismisses it, which
// loses nothing (the agent has already stopped waiting for the answer).
func (c *cliClient) askChoiceOrText(question string, labels []string, allowText bool) (int, string) {
	c.askMu.Lock()
	defer c.askMu.Unlock()

	c.ui.Suspend()
	defer c.ui.Resume()

	c.ui.Note(ansiBold, "\n? "+strings.TrimSpace(question))
	for i, l := range labels {
		c.ui.Note("", fmt.Sprintf("  %d) %s", i+1, l))
	}
	hint := "  [1-" + strconv.Itoa(len(labels)) + ", Enter to dismiss] "
	switch {
	case len(labels) == 0:
		hint = "  [type an answer, Enter to dismiss] "
	case allowText:
		hint = "  [1-" + strconv.Itoa(len(labels)) + ", or type an answer, Enter to dismiss] "
	}

	for attempt := 0; attempt < 3; attempt++ {
		c.ui.Prompt(hint)
		line, err := c.in.ReadString('\n')
		ans := strings.TrimSpace(line)
		c.ui.Submitted(ans)
		if err != nil {
			return -1, ""
		}
		if ans == "" {
			return -1, ""
		}
		if n, err := strconv.Atoi(ans); err == nil && n >= 1 && n <= len(labels) {
			return n - 1, ""
		}
		if allowText || len(labels) == 0 {
			return -1, ans
		}
		c.ui.Note(ansiRed, "  pick a number between 1 and "+strconv.Itoa(len(labels)))
	}
	return -1, ""
}

// ---------------------------------------------------------------------------
// Inbound: terminals
// ---------------------------------------------------------------------------

// cliTerminal is one command the agent asked us to run. The client owns the
// process; the agent only ever names it by id.
type cliTerminal struct {
	id    string
	cmd   *exec.Cmd
	limit int

	mu        sync.Mutex
	buf       []byte
	truncated bool
	exit      *terminalExit

	done chan struct{}
}

// Write is stdout and stderr both, interleaved in arrival order the way a real
// terminal shows them. Over the limit, the FRONT is dropped: the agent asks for
// 4 MiB and does its own head+tail elision (see terminalOutputLimit), so the
// tail is the half worth keeping here.
func (t *cliTerminal) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	if t.limit > 0 && len(t.buf) > t.limit {
		t.buf = append(t.buf[:0], t.buf[len(t.buf)-t.limit:]...)
		t.truncated = true
	}
	return len(p), nil
}

func (t *cliTerminal) snapshot() (string, bool, *terminalExit) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf), t.truncated, t.exit
}

// tail returns the last n non-empty lines, for the live region under the card.
func (t *cliTerminal) tail(n int) []string {
	t.mu.Lock()
	out := string(t.buf)
	done := t.exit != nil
	t.mu.Unlock()
	if done {
		return nil
	}
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	var keep []string
	for i := len(lines) - 1; i >= 0 && len(keep) < n; i-- {
		if strings.TrimSpace(lines[i]) == "" {
			continue
		}
		keep = append([]string{strings.TrimRight(lines[i], "\r")}, keep...)
	}
	return keep
}

func (c *cliClient) terminalCreate(params json.RawMessage) (any, error) {
	var p struct {
		Command         string   `json:"command"`
		Args            []string `json:"args"`
		Cwd             string   `json:"cwd"`
		OutputByteLimit int      `json:"outputByteLimit"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, err
	}
	if p.Command == "" {
		return nil, errors.New("terminal/create: no command")
	}
	cmd := exec.Command(p.Command, p.Args...)
	cmd.Dir = p.Cwd
	if cmd.Dir == "" {
		cmd.Dir = c.cwd
	}
	// nil stdin is /dev/null. It must NOT be os.Stdin: a command that reads
	// would eat the keystrokes meant for the prompt, and the agent's contract
	// is non-interactive commands anyway.
	cmd.Stdin = nil

	c.termMu.Lock()
	c.termSeq++
	id := "term_" + strconv.Itoa(c.termSeq)
	c.termMu.Unlock()

	t := &cliTerminal{id: id, cmd: cmd, limit: p.OutputByteLimit, done: make(chan struct{})}
	cmd.Stdout, cmd.Stderr = t, t
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("terminal/create: %w", err)
	}

	c.termMu.Lock()
	c.terms[id] = t
	c.termMu.Unlock()

	go func() {
		err := cmd.Wait()
		t.mu.Lock()
		code := cmd.ProcessState.ExitCode()
		if code < 0 {
			// Killed by a signal. Without importing syscall for WaitStatus the
			// name has to come from the formatted state ("signal: killed"),
			// which is exactly what an ACP client reports here anyway.
			sig := "unknown"
			if err != nil {
				sig = strings.TrimPrefix(err.Error(), "signal: ")
			}
			t.exit = &terminalExit{Signal: sig}
		} else {
			t.exit = &terminalExit{ExitCode: &code}
		}
		t.mu.Unlock()
		close(t.done)
	}()
	return map[string]any{"terminalId": id}, nil
}

func (c *cliClient) terminal(params json.RawMessage) (*cliTerminal, error) {
	var p struct {
		TerminalId string `json:"terminalId"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, err
	}
	c.termMu.Lock()
	t := c.terms[p.TerminalId]
	c.termMu.Unlock()
	if t == nil {
		return nil, fmt.Errorf("no such terminal: %s", p.TerminalId)
	}
	return t, nil
}

func (c *cliClient) terminalOutput(params json.RawMessage) (any, error) {
	t, err := c.terminal(params)
	if err != nil {
		return nil, err
	}
	out, truncated, exit := t.snapshot()
	return map[string]any{"output": out, "truncated": truncated, "exitStatus": exit}, nil
}

func (c *cliClient) terminalWait(ctx context.Context, params json.RawMessage) (any, error) {
	t, err := c.terminal(params)
	if err != nil {
		return nil, err
	}
	select {
	case <-t.done:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	_, _, exit := t.snapshot()
	return exit, nil
}

// terminalKill stops the command but keeps its output readable, which is the
// contract run_command's idle watchdog relies on. Only the direct child is
// killed: putting it in its own process group would mean importing syscall for
// SysProcAttr, and a shell that spawned background children leaves them behind
// exactly as it does under any other ACP client.
func (c *cliClient) terminalKill(params json.RawMessage) (any, error) {
	t, err := c.terminal(params)
	if err != nil {
		return nil, err
	}
	t.kill()
	return struct{}{}, nil
}

func (t *cliTerminal) kill() {
	if t.cmd.Process != nil {
		if err := t.cmd.Process.Kill(); err != nil {
			slog.Debug("cli: kill failed", "terminal", t.id, "err", err)
		}
	}
}

func (c *cliClient) terminalRelease(params json.RawMessage) (any, error) {
	t, err := c.terminal(params)
	if err != nil {
		// Releasing something already gone is not an error worth failing a turn
		// over: release is a deferred cleanup on the agent side.
		return struct{}{}, nil
	}
	t.kill()
	c.termMu.Lock()
	delete(c.terms, t.id)
	c.termMu.Unlock()
	return struct{}{}, nil
}

// releaseAll kills anything still running at exit. The agent releases its own
// terminals, so this only catches commands abandoned by a crash or a hard quit.
func (c *cliClient) releaseAll() {
	c.termMu.Lock()
	terms := c.terms
	c.terms = map[string]*cliTerminal{}
	c.termMu.Unlock()
	for _, t := range terms {
		t.kill()
	}
}

// ---------------------------------------------------------------------------
// cliConn — the client half of the JSON-RPC framing.
//
// A deliberate mirror of AgentSideConnection rather than a refactor of it into
// a shared peer type: that code is load-bearing and stable, and a client whose
// framing is written independently is a second implementation that can disagree
// with the first, which is how framing bugs get caught instead of shared.
// ---------------------------------------------------------------------------

type cliConn struct {
	w       io.Writer
	writeMu sync.Mutex
	client  *cliClient

	nextID    atomic.Uint64
	pendingMu sync.Mutex
	pending   map[string]chan json.RawMessage

	inflightMu sync.Mutex
	inflight   map[string]context.CancelFunc

	done chan struct{}
}

func newCLIConn(w io.Writer, r io.Reader, client *cliClient) *cliConn {
	c := &cliConn{
		w:        w,
		client:   client,
		pending:  map[string]chan json.RawMessage{},
		inflight: map[string]context.CancelFunc{},
		done:     make(chan struct{}),
	}
	go c.serve(r)
	return c
}

func (c *cliConn) write(msg any) error {
	b, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	_, err = c.w.Write(b)
	return err
}

func (c *cliConn) notify(method string, params any) error {
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return c.write(jsonrpcRequest{JSONRPC: "2.0", Method: method, Params: raw})
}

// request sends one client->agent call and waits for its reply. It takes no
// context: nothing on this side ever cancels an outbound request (an
// interrupted turn travels as a session/cancel notification, so the agent can
// finish the turn properly and answer with a stop reason), and the only other
// way out is the connection closing, which c.done already covers.
func (c *cliConn) request(method string, params any) (json.RawMessage, error) {
	id := strconv.FormatUint(c.nextID.Add(1), 10)
	idRaw := json.RawMessage(`"` + id + `"`)

	ch := make(chan json.RawMessage, 1)
	c.pendingMu.Lock()
	c.pending[id] = ch
	c.pendingMu.Unlock()
	defer func() {
		c.pendingMu.Lock()
		delete(c.pending, id)
		c.pendingMu.Unlock()
	}()

	var raw json.RawMessage
	if params != nil {
		b, err := json.Marshal(params)
		if err != nil {
			return nil, err
		}
		raw = b
	}
	if err := c.write(jsonrpcRequest{JSONRPC: "2.0", ID: &idRaw, Method: method, Params: raw}); err != nil {
		return nil, err
	}

	select {
	case <-c.done:
		return nil, errors.New("agent connection closed")
	case line := <-ch:
		var resp struct {
			Result json.RawMessage `json:"result"`
			Error  *struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
				Data    string `json:"data,omitempty"`
			} `json:"error"`
		}
		if err := json.Unmarshal(line, &resp); err != nil {
			return nil, err
		}
		if resp.Error != nil {
			if resp.Error.Data != "" {
				return nil, fmt.Errorf("%s: %s", resp.Error.Message, resp.Error.Data)
			}
			return nil, errors.New(resp.Error.Message)
		}
		return resp.Result, nil
	}
}

func (c *cliConn) serve(r io.Reader) {
	defer close(c.done)
	br := bufio.NewReader(r)
	for {
		s, err := br.ReadString('\n')
		if err != nil {
			if !errors.Is(err, io.EOF) {
				slog.Debug("cli: read error", "err", err)
			}
			return
		}
		s = strings.TrimRight(s, "\r\n")
		if s == "" {
			continue
		}
		line := []byte(s)

		var probe struct {
			ID     *json.RawMessage `json:"id"`
			Method string           `json:"method"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			slog.Warn("cli: unparseable message", "err", err)
			continue
		}

		if probe.Method == "" && probe.ID != nil {
			id := strings.Trim(string(*probe.ID), `"`)
			c.pendingMu.Lock()
			ch, ok := c.pending[id]
			if ok {
				delete(c.pending, id)
			}
			c.pendingMu.Unlock()
			if ok {
				ch <- line
			}
			continue
		}

		var req jsonrpcRequest
		if err := json.Unmarshal(line, &req); err != nil {
			slog.Warn("cli: unparseable request", "err", err)
			continue
		}

		// session/update is handled ON the read loop, not in a goroutine: it
		// carries the streamed message in chunks, and dispatching those
		// concurrently would render them in whatever order the scheduler felt
		// like. It never blocks (the UI writes to a buffered stdout), so it
		// cannot stall the loop the way a request handler would.
		switch req.Method {
		case "session/update":
			c.client.sessionUpdate(req.Params)
			continue
		case "$/cancel_request":
			c.cancelInflight(req.Params)
			continue
		}
		go c.handle(&req)
	}
}

func (c *cliConn) cancelInflight(params json.RawMessage) {
	var p struct {
		RequestId json.RawMessage `json:"requestId"`
	}
	if json.Unmarshal(params, &p) != nil || len(p.RequestId) == 0 {
		return
	}
	c.inflightMu.Lock()
	cancel := c.inflight[string(p.RequestId)]
	c.inflightMu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// handle runs one agent->client request or notification. Handlers that can
// block on something other than the user take a context so $/cancel_request can
// reach them; see askChoiceOrText for why the ones blocked on stdin don't.
func (c *cliConn) handle(req *jsonrpcRequest) {
	ctx := context.Background()
	if req.ID != nil {
		key := string(*req.ID)
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(ctx)
		defer cancel()
		c.inflightMu.Lock()
		c.inflight[key] = cancel
		c.inflightMu.Unlock()
		defer func() {
			c.inflightMu.Lock()
			delete(c.inflight, key)
			c.inflightMu.Unlock()
		}()
	}

	var res any
	var err error
	switch req.Method {
	case "session/request_permission":
		res, err = c.client.requestPermission(req.Params)
	case "elicitation/create":
		res, err = c.client.elicit(req.Params)
	case "terminal/create":
		res, err = c.client.terminalCreate(req.Params)
	case "terminal/output":
		res, err = c.client.terminalOutput(req.Params)
	case "terminal/wait_for_exit":
		res, err = c.client.terminalWait(ctx, req.Params)
	case "terminal/kill":
		res, err = c.client.terminalKill(req.Params)
	case "terminal/release":
		res, err = c.client.terminalRelease(req.Params)
	default:
		if req.ID != nil {
			c.replyError(req.ID, -32601, "method not found: "+req.Method)
		}
		return
	}
	if req.ID == nil {
		return
	}
	if err != nil {
		c.replyError(req.ID, -32603, err.Error())
		return
	}
	if writeErr := c.write(jsonrpcResponse{JSONRPC: "2.0", ID: req.ID, Result: res}); writeErr != nil {
		slog.Debug("cli: reply failed", "method", req.Method, "err", writeErr)
	}
}

func (c *cliConn) replyError(id *json.RawMessage, code int, msg string) {
	resp := jsonrpcResponse{JSONRPC: "2.0", ID: id}
	resp.Error = &struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	}{code, msg}
	if err := c.write(resp); err != nil {
		slog.Debug("cli: error reply failed", "err", err)
	}
}
