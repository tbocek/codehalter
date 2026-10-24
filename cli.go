package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// The standalone CLI talks real ACP to the in-process agent over pipes, so framing and
// ordering bugs show up here too. fs is not advertised: the CLI has no unsaved buffers.

const cliUsage = `usage: codehalter --cli [--cwd DIR] [--resume [SESSION_ID]] [-p PROMPT]

  --cli               run the standalone terminal client instead of an ACP server
  --cwd DIR           project directory (default: current directory)
  --resume [ID]       continue a stored session; without an ID, the most recent
                      one for this directory
  -p PROMPT           run one turn and exit, instead of opening a prompt. The
                      exit status is 0 only when the turn finished normally,
                      so a script can branch on it.
  --rebuild           rebuild the devcontainer image before starting it

Run outside a container in a project that has .devcontainer/devcontainer.json,
this starts that container with docker (or podman) compose and runs the CLI
inside it. Inside a container it just starts.

Other flags: --version prints the release tag, --build its build date and
hash, --update installs the newest release over this binary, --setup
reconfigures the LLM connection.
`

func runCLI(argv []string) int {
	cwd, _ := os.Getwd()
	resumeID, prompt := "", ""
	resume, oneshot, rebuild := false, false, false
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
		case "--rebuild":
			rebuild = true
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

	// Before anything is opened: an update re-execs this binary, and nothing exists yet to lose.
	offerUpdate(context.Background(), cwd, stdinIsTTY() && !oneshot)

	// On the host this process is only a launcher. Without a devcontainer.json it falls
	// through and the agent offers to scaffold one.
	if containerKind() == "" {
		var inner []string
		if resume {
			inner = append(inner, "--resume")
			if resumeID != "" {
				inner = append(inner, resumeID)
			}
		}
		if oneshot {
			inner = append(inner, "-p", prompt)
		}
		if code, launched := launchInDevcontainer(cwd, inner, rebuild); launched {
			return code
		}
	}

	// Debug logs and child stderr (tool_web hands firefox's output to os.Stderr) would write
	// through the TUI and desync the live region, so both go to a file. Panics still reach fd 2.
	if logf := openCLILog(cwd); logf != nil {
		defer logf.Close()
		slog.SetDefault(slog.New(slog.NewTextHandler(logf, &slog.HandlerOptions{Level: slog.LevelDebug})))
		os.Stderr = logf
	} else {
		slog.SetDefault(slog.New(slog.NewTextHandler(io.Discard, nil)))
	}

	agentIn, clientOut := io.Pipe()
	clientIn, agentOut := io.Pipe()

	a := &agent{sessions: make(map[string]*Session), standalone: true}
	go killOrphanedJobs()
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

	// Shared with hardQuit: a second Ctrl+C cannot unwind the stack while the repl is parked
	// in a stdin read, so it runs this teardown from the signal goroutine.
	shutdown := func() {
		// EOF on the agent's reader ends its serve loop, the same path a departing editor takes.
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
		if restore := c.restoreTTY.Load(); restore != nil {
			(*restore)()
		}
		shutdown()
		os.Exit(130)
	}

	code := c.run(resume, resumeID, prompt, oneshot)
	shutdown()
	return code
}

func stdinIsTTY() bool {
	fi, err := os.Stdin.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// openCLILog truncates: the log covers one run. nil on failure: a read-only project
// should lose the log, not the run.
func openCLILog(cwd string) *os.File {
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil
	}
	f, err := os.OpenFile(filepath.Join(dir, "cli.log"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil
	}
	return f
}

type cliClient struct {
	ui   *cliUI
	conn *cliConn
	in   *bufio.Reader
	cwd  string

	interactive bool

	// hardQuit is nil in tests, which then only get the hint.
	hardQuit func()

	// sid is written by the input goroutine and read by the signal goroutine.
	sidMu sync.Mutex
	sid   string

	cmdMu    sync.Mutex
	commands []availableCommand

	// turnActive is read by the signal handler, which must not take the UI lock.
	turnActive atomic.Bool

	// imagesAccepted is the agent's promptCapabilities.image: a model that cannot see takes no picture,
	// so Ctrl+V says so rather than attaching one the turn would drop.
	imagesAccepted atomic.Bool
	// clipboard reads the image Ctrl+V attaches; nil is the system's (clipboardImage; tests set their own).
	clipboard func() ([]byte, string, error)

	// askMu serialises questions: parallel tool calls can both ask the user.
	askMu sync.Mutex

	// inputs carries every line the repl's reader reads, once the repl runs; until then (a
	// one-shot -p run) a question reads stdin itself. waiting is the open question's channel:
	// the next line answers it rather than becoming a prompt. waitMu guards both: a question
	// and the repl's start race.
	inputs  chan cliLine
	waitMu  sync.Mutex
	waiting chan cliLine
	// restoreTTY undoes the terminal's character mode; hardQuit calls it before exiting.
	restoreTTY atomic.Pointer[func()]

	termMu  sync.Mutex
	terms   map[string]*cliTerminal
	termSeq int
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
	raw, err := c.conn.request("initialize", map[string]any{
		"protocolVersion": protocolVersion,
		"clientCapabilities": map[string]any{
			"terminal":    true,
			"elicitation": map[string]any{"form": map[string]any{}},
			"fs":          map[string]any{"readTextFile": false, "writeTextFile": false},
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stdout, "initialize failed: %v\n", err)
		return 1
	}
	var init InitializeResponse
	if json.Unmarshal(raw, &init) == nil {
		c.imagesAccepted.Store(init.AgentCapabilities.PromptCapabilities.Image)
	}

	if err := c.openSession(resume, resumeID); err != nil {
		fmt.Fprintf(os.Stdout, "%v\n", err)
		return 1
	}
	c.installSignals()

	if oneshot {
		if c.turn(prompt, nil) {
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
	return c.startSession()
}

func (c *cliClient) startSession() error {
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
	c.adopt(res.SessionId)
	return nil
}

func (c *cliClient) loadSession(id string) error {
	raw, err := c.conn.request("session/load", LoadSessionRequest{SessionId: id, Cwd: c.cwd})
	if err != nil {
		return fmt.Errorf("session/load failed: %w", err)
	}
	var res LoadSessionResponse
	if err := json.Unmarshal(raw, &res); err != nil {
		return fmt.Errorf("session/load returned nonsense: %w", err)
	}
	c.adopt(id)
	return nil
}

// UpdatedAt is ISO 8601, so string order is time order. Re-sorted rather than trusting the agent.
func (c *cliClient) storedSessions() ([]SessionInfo, error) {
	raw, err := c.conn.request("session/list", ListSessionsRequest{Cwd: c.cwd})
	if err != nil {
		return nil, fmt.Errorf("session/list failed: %w", err)
	}
	var list ListSessionsResponse
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("session/list returned nonsense: %w", err)
	}
	slices.SortFunc(list.Sessions, func(x, y SessionInfo) int { return strings.Compare(y.UpdatedAt, x.UpdatedAt) })
	return list.Sessions, nil
}

func (c *cliClient) adopt(sid string) {
	c.setSessionID(sid)
	c.ui.Usage(0, 0)
}

// A stdin read is not interrupted by a signal, so Ctrl+C at an idle prompt only hints; a
// second one within two seconds exits through hardQuit.
func (c *cliClient) installSignals() {
	c.ui.watchResize()
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

// cliLine is one line of input, or the error that ended input.
type cliLine struct {
	text string
	err  error
	// images are the pictures Ctrl+V attached to this line, each still named by its marker in text.
	images []ContentBlock
}

// repl keeps reading while a turn runs: a /spec loop is one turn that runs for hours, and
// with the read parked behind it nothing could be typed, not even `/spec stop`. A line
// typed during a turn goes to the agent at once, which takes it as steering.
func (c *cliClient) repl() int {
	restore := c.ui.startInput()
	c.restoreTTY.Store(&restore)
	defer restore()
	c.waitMu.Lock()
	c.inputs = make(chan cliLine)
	c.waitMu.Unlock()
	go c.readInput()
	turnDone := make(chan struct{})
	busy := false
	for {
		if !busy {
			c.ui.Prompt("\n❯ ")
		}
		var ln cliLine
		select {
		case <-turnDone:
			busy = false
			continue
		case ln = <-c.inputs:
		}
		c.waitMu.Lock()
		waiting := c.waiting
		c.waitMu.Unlock()
		if waiting != nil {
			waiting <- ln // the open question's answer
			continue
		}
		c.ui.Submitted(ln.text)
		if ln.err != nil {
			if !errors.Is(ln.err, io.EOF) {
				c.ui.Note(ansiRed, "input: "+ln.err.Error())
			}
			c.ui.Note(ansiDim, "")
			return 0
		}
		text := ln.text
		if text == "" {
			continue
		}
		if name, _, _ := strings.Cut(text, " "); busy && (name == "/new" || name == "/resume") {
			c.ui.Note(ansiYell, "  "+name+" waits until the turn has ended (Ctrl+C interrupts it)")
			continue
		}
		if handled, quit := c.command(text); handled {
			if quit {
				return 0
			}
			continue
		}
		if busy {
			go c.steer(text, ln.images)
			continue
		}
		busy = true
		go func() {
			c.turn(text, ln.images)
			turnDone <- struct{}{}
		}()
	}
}

// steer sends a line typed during a turn; the agent folds it into the running turn and
// answers at once.
func (c *cliClient) steer(text string, images []ContentBlock) {
	if _, err := c.conn.request("session/prompt", PromptRequest{
		SessionId: c.sessionID(),
		Content:   promptBlocks(text, images),
	}); err != nil {
		c.ui.Note(ansiRed, "  not delivered: "+err.Error())
	}
}

// readInput is the one reader of stdin while the repl runs: keys on a terminal in
// character mode, lines otherwise.
func (c *cliClient) readInput() {
	if !c.ui.inputActive() {
		for {
			text, err := c.readPrompt()
			c.inputs <- cliLine{text: text, err: err}
			if err != nil {
				return
			}
		}
	}
	// A line editor: buf is the line, cur the cursor in it, history the lines sent so far with
	// hist the one shown (len(history) is the line being typed, kept in draft while browsing).
	var buf, draft []rune
	var history [][]rune
	cur, hist, paste := 0, 0, false
	// attached holds the pictures Ctrl+V put on the line being typed; each is named in the line by its
	// marker, and deleting the marker leaves the picture out of the prompt.
	var attached []ContentBlock
	insert := func(rs ...rune) {
		buf = slices.Insert(buf, cur, rs...)
		cur += len(rs)
	}
	recall := func(i int) {
		if i < 0 || i > len(history) || i == hist {
			return
		}
		if hist == len(history) {
			draft = slices.Clone(buf)
		}
		hist = i
		if i == len(history) {
			buf = slices.Clone(draft)
		} else {
			buf = slices.Clone(history[i])
		}
		cur = len(buf)
	}
	// wordStart and wordEnd skip the spaces next to the cursor, then the word.
	wordStart := func() int {
		i := cur
		for i > 0 && buf[i-1] == ' ' {
			i--
		}
		for i > 0 && buf[i-1] != ' ' {
			i--
		}
		return i
	}
	wordEnd := func() int {
		i := cur
		for i < len(buf) && buf[i] == ' ' {
			i++
		}
		for i < len(buf) && buf[i] != ' ' {
			i++
		}
		return i
	}
	for {
		r, _, err := c.in.ReadRune()
		if err != nil {
			c.inputs <- cliLine{err: err}
			return
		}
		switch {
		case r == 0x1b:
			// A key's escape sequence arrives in one read; a lone ESC has nothing behind it.
			if c.in.Buffered() == 0 {
				continue
			}
			next, _, _ := c.in.ReadRune()
			var seq []rune
			switch {
			case next == '[': // CSI: parameters, then one final byte
				for c.in.Buffered() > 0 {
					q, _, _ := c.in.ReadRune()
					seq = append(seq, q)
					if q >= 0x40 && q <= 0x7e {
						break
					}
				}
			case next == 'O' && c.in.Buffered() > 0: // SS3: what some terminals send for arrows, Home and End
				q, _, _ := c.in.ReadRune()
				seq = []rune{q}
			}
			switch string(seq) {
			case "200~": // bracketed paste: its newlines are text, not Enter
				paste = true
			case "201~":
				paste = false
			case "D":
				cur = max(cur-1, 0)
			case "C":
				cur = min(cur+1, len(buf))
			case "1;5D", "1;3D": // Ctrl or Alt with Left: a word back
				cur = wordStart()
			case "1;5C", "1;3C":
				cur = wordEnd()
			case "H", "1~", "7~":
				cur = 0
			case "F", "4~", "8~":
				cur = len(buf)
			case "3~": // Delete
				if cur < len(buf) {
					buf = slices.Delete(buf, cur, cur+1)
				}
			case "A":
				recall(hist - 1)
			case "B":
				recall(hist + 1)
			default:
				continue
			}
		case (r == '\r' || r == '\n') && paste:
			insert('\n')
		case r == '\r' || r == '\n':
			line := strings.TrimSpace(string(buf))
			if line != "" && (len(history) == 0 || string(history[len(history)-1]) != line) {
				history = append(history, []rune(line))
			}
			var images []ContentBlock
			for i, img := range attached {
				if strings.Contains(line, imageMarker(i)) {
					images = append(images, img)
				}
			}
			c.inputs <- cliLine{text: line, images: images}
			buf, draft, cur, hist, attached = buf[:0], nil, 0, len(history), nil
		case r == 0x16: // Ctrl+V: a picture from the clipboard
			if img, ok := c.pasteImage(len(attached)); ok {
				attached = append(attached, img)
				insert([]rune(imageMarker(len(attached)-1) + " ")...)
			}
		case r == 0x7f || r == 0x08: // backspace
			if cur > 0 {
				buf = slices.Delete(buf, cur-1, cur)
				cur--
			}
		case r == 0x04: // Ctrl+D leaves on an empty line
			if len(buf) == 0 {
				c.inputs <- cliLine{err: io.EOF}
				return
			}
		case r == 0x01: // Ctrl+A
			cur = 0
		case r == 0x05: // Ctrl+E
			cur = len(buf)
		case r == 0x0b: // Ctrl+K: the rest of the line
			buf = buf[:cur]
		case r == 0x15: // Ctrl+U: the line up to the cursor
			buf = slices.Delete(buf, 0, cur)
			cur = 0
		case r == 0x17: // Ctrl+W: the word before the cursor
			start := wordStart()
			buf = slices.Delete(buf, start, cur)
			cur = start
		case r == '\t':
			insert(' ')
		case r < 0x20:
		default:
			insert(r)
		}
		c.ui.SetInput(slices.Clone(buf), cur)
	}
}

// imageMarker names the i-th picture attached to a line, in the line itself.
func imageMarker(i int) string {
	return fmt.Sprintf("[image %d]", i+1)
}

// pasteImage reads the clipboard's picture for Ctrl+V, saying why when there is none to attach.
func (c *cliClient) pasteImage(n int) (ContentBlock, bool) {
	if !c.imagesAccepted.Load() {
		c.ui.Note(ansiYell, "  this model takes no images")
		return ContentBlock{}, false
	}
	read := c.clipboard
	if read == nil {
		read = clipboardImage
	}
	data, mime, err := read()
	if err != nil {
		c.ui.Note(ansiYell, "  Ctrl+V: "+err.Error())
		return ContentBlock{}, false
	}
	c.ui.Note(ansiDim, fmt.Sprintf("  %s: %s, %d KB", imageMarker(n), mime, (len(data)+1023)/1024))
	return ContentBlock{Type: "image", MimeType: mime, Data: base64.StdEncoding.EncodeToString(data)}, true
}

// On a terminal, input already buffered when the first line ends is a paste and joins the
// same prompt. Piped input is buffered too, but there each line is its own prompt.
func (c *cliClient) readPrompt() (string, error) {
	line, err := c.in.ReadString('\n')
	for err == nil && c.interactive && c.in.Buffered() > 0 {
		var next string
		next, err = c.in.ReadString('\n')
		line += next
	}
	return strings.TrimSpace(line), err
}

// command handles client-owned commands only. Other slash commands pass through: the agent
// parses "/name args" itself (splitMacro).
func (c *cliClient) command(text string) (handled, quit bool) {
	// A bare `exit` or `quit` is how a shell is left; sent as a prompt it only started a turn about it.
	if t := strings.ToLower(strings.TrimSpace(text)); t == "exit" || t == "quit" {
		return true, true
	}
	name, rest, _ := strings.Cut(text, " ")
	rest = strings.TrimSpace(rest)
	switch name {
	case "/quit", "/exit":
		return true, true
	case "/help":
		c.help()
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
		"  /sessions          stored sessions for this directory",
		"  /new               start a fresh session here",
		"  /resume [id]       switch to a stored session, or pick one from a list",
		"  /cwd               project directory and session id",
		"  /quit, exit        leave (same as Ctrl+D)",
	} {
		c.ui.Note("", l)
	}
	c.cmdMu.Lock()
	cmds := slices.Clone(c.commands)
	c.cmdMu.Unlock()
	if len(cmds) == 0 {
		return
	}
	c.ui.Note(ansiBold, "\nagent commands")
	for _, cmd := range cmds {
		c.ui.Note("", fmt.Sprintf("  /%-17s %s", cmd.Name, cmd.Description))
	}
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

func (c *cliClient) newSession() {
	if err := c.startSession(); err != nil {
		c.ui.Note(ansiRed, "  "+err.Error())
		return
	}
	c.ui.Note(ansiDim, "  session "+c.sessionID())
}

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

// promptBlocks is a prompt's content: the typed text, then the pictures attached to it.
func promptBlocks(text string, images []ContentBlock) []ContentBlock {
	return append([]ContentBlock{{Type: "text", Text: text}}, images...)
}

func (c *cliClient) turn(text string, images []ContentBlock) bool {
	c.turnActive.Store(true)
	c.ui.Begin()
	stop := c.startTicker()

	raw, err := c.conn.request("session/prompt", PromptRequest{
		SessionId: c.sessionID(),
		Content:   promptBlocks(text, images),
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

// Pulled on the render tick, not pushed per write: a chatty build would force thousands of redraws a second.
func (c *cliClient) pumpTails() {
	c.termMu.Lock()
	terms := slices.Collect(maps.Values(c.terms))
	c.termMu.Unlock()
	for _, t := range terms {
		c.ui.TerminalTail(t.id, t.tail(cliTermTail))
	}
}

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
		if u.Meta != nil {
			c.ui.Speeds(u.Meta.PromptPerSec, u.Meta.GenPerSec)
		}

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

// Only the choice and text fields codehalter sends are supported; any other form is declined
// so the agent gets an answer instead of a hang.
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

// No context: a blocked stdin read cannot be abandoned without handing the half-typed line to
// the next reader, so a cancelled question stays until Enter (the agent has stopped waiting).
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

	// With the repl running, its reader owns stdin and hands this question the next line.
	next := func() (string, error) { return c.in.ReadString('\n') }
	c.waitMu.Lock()
	repl := c.inputs != nil
	ch := make(chan cliLine, 1)
	if repl {
		c.waiting = ch
	}
	c.waitMu.Unlock()
	if repl {
		defer func() {
			c.waitMu.Lock()
			c.waiting = nil
			c.waitMu.Unlock()
		}()
		next = func() (string, error) {
			ln := <-ch
			return ln.text, ln.err
		}
	}
	for attempt := 0; attempt < 3; attempt++ {
		c.ui.Ask(hint)
		line, err := next()
		ans := strings.TrimSpace(line)
		c.ui.Submitted(ans)
		// Run from a script (-p, no terminal) whose input has run out, nobody can answer:
		// the first option, said aloud, rather than a dismissal that ends the work.
		if err != nil && ans == "" && !c.interactive && len(labels) > 0 {
			c.ui.Note(ansiDim, "  no one to answer (input ended): took 1) "+labels[0])
			return 0, ""
		}
		if err != nil || ans == "" {
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

// Over the limit the front is dropped: the agent does its own head+tail elision
// (terminalOutputLimit), so the tail is what matters here.
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
	// Stdin stays nil (/dev/null): os.Stdin would steal the prompt's keystrokes.
	cmd := exec.Command(p.Command, p.Args...)
	cmd.Dir = orElse(p.Cwd, c.cwd)
	// A group of its own, as Zed's pty gives one: in codehalter's group a job's group kill
	// reached only its wrapper, and the Ctrl+C meant for the turn reached every job.
	newProcessGroup(cmd)

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
			// Killed by a signal: the name comes from the formatted state ("signal: killed").
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

// terminalKill keeps the output readable (run_command's idle watchdog relies on that). Only
// the direct child is killed, as under any other ACP client.
func (c *cliClient) terminalKill(params json.RawMessage) (any, error) {
	t, err := c.terminal(params)
	if err != nil {
		return nil, err
	}
	t.kill()
	return struct{}{}, nil
}

func (t *cliTerminal) kill() {
	if err := t.cmd.Process.Kill(); err != nil {
		slog.Debug("cli: kill failed", "terminal", t.id, "err", err)
	}
}

func (c *cliClient) terminalRelease(params json.RawMessage) (any, error) {
	t, err := c.terminal(params)
	if err != nil {
		// Release is a deferred cleanup on the agent side; an unknown id must not fail a turn.
		return struct{}{}, nil
	}
	t.kill()
	c.termMu.Lock()
	delete(c.terms, t.id)
	c.termMu.Unlock()
	return struct{}{}, nil
}

func (c *cliClient) releaseAll() {
	c.termMu.Lock()
	terms := c.terms
	c.terms = map[string]*cliTerminal{}
	c.termMu.Unlock()
	for _, t := range terms {
		t.kill()
	}
}

// cliConn shares rpcPeer with the agent: wire-shape tests on both ends, not a second copy, catch framing bugs.
type cliConn struct {
	*rpcPeer
	client *cliClient
}

func newCLIConn(w io.Writer, r io.Reader, client *cliClient) *cliConn {
	c := &cliConn{rpcPeer: newRPCPeer(w, "cli: ", false), client: client}
	go c.serve(r, c.dispatch)
	return c
}

// No ctx: a turn is interrupted by session/cancel. A person reads the errors, so the RPC code is dropped.
func (c *cliConn) request(method string, params any) (json.RawMessage, error) {
	raw, err := c.sendRequest(context.Background(), method, params)
	var rerr *rpcError
	switch {
	case errors.As(err, &rerr):
		if rerr.Data != "" {
			return nil, fmt.Errorf("%s: %s", rerr.Message, rerr.Data)
		}
		return nil, errors.New(rerr.Message)
	case errors.Is(err, errRPCClosed):
		return nil, errors.New("agent connection closed")
	}
	return raw, err
}

func (c *cliConn) dispatch(req *jsonrpcRequest) {
	// session/update runs on the read loop so streamed chunks render in order; it never
	// blocks (buffered stdout).
	switch req.Method {
	case "session/update":
		c.client.sessionUpdate(req.Params)
	case "$/cancel_request":
		// Only the ctx: the handler's own error reply answers the request.
		c.cancelInflight(req.Params)
	default:
		go c.handle(req)
	}
}

func (c *cliConn) handle(req *jsonrpcRequest) {
	ctx, untrack := c.track(context.Background(), req.ID)
	defer untrack()

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
			c.writeError(req.ID, -32601, "method not found: "+req.Method)
		}
		return
	}
	if req.ID == nil {
		return
	}
	if err != nil {
		c.writeError(req.ID, -32603, err.Error())
		return
	}
	c.writeResult(req.ID, res)
}
