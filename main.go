package main

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// resMD is every shipped prompt, skill and template macro; read it through builtin.
//
//go:embed res/*.md
var resMD embed.FS

//go:embed res/Dockerfile.devcontainer.*
var devcontainerDockerfiles embed.FS

// builtin returns .codehalter/<name> when it exists (even empty), else the shipped res/<name>.
func builtin(cwd, name string) (string, bool) {
	if cwd != "" {
		if data, err := os.ReadFile(filepath.Join(cwd, sessionDir, name)); err == nil {
			return string(data), true
		}
	}
	data, err := resMD.ReadFile("res/" + name)
	return string(data), err == nil
}

func shipped(name string) bool {
	_, err := fs.Stat(resMD, "res/"+name)
	return err == nil
}

//go:embed res/devcontainer.json
var defaultDevcontainerJSON string

//go:embed res/settings.toml
var defaultSettingsTOML string

//go:embed res/mcp.toml
var defaultMCPToml string

type agent struct {
	// mu guards cancel, sessions, mode, abortReason, clientCaps, emptyProject, indexDone
	// and mcp.clients.
	mu sync.Mutex
	// cfgMu guards settings, connSems and connProbe, which prepare reassigns while a prior turn's
	// goroutine reads them. A leaf: never held across a blocking call or while taking a.mu or sess.mu.
	cfgMu        sync.RWMutex
	conn         *AgentSideConnection
	cancel       context.CancelFunc
	sessions     map[string]*Session
	settings     Settings
	emptyProject bool
	// indexDone is the latest session's startup; read it through startupDone.
	indexDone chan struct{}
	mode      string // "Interactive" | "Autopilot"
	// asking counts questions waiting on the user (a card or a form): a prompt
	// during startup is refused only while startup asks, and waits otherwise.
	asking atomic.Int32

	// Keyed by Server+"\x00"+Model; nil before the first prepare reads as the zero result.
	connProbe map[string]probeResult

	// Consecutive summariser-connection failures and the latest one's unix nanos.
	// Atomic, not under a.mu: written by the summarise goroutine, read on the turn path.
	summaryStrikes  atomic.Int32
	summaryStruckAt atomic.Int64

	// ctkIgnored: Server+Model already warned for ignoring chat_template_kwargs. A
	// sync.Map so background llmStream calls don't queue behind a.mu.
	ctkIgnored sync.Map

	// 0 means unknown, which ensureLLM treats like "below minSlotTokens". Atomic: a
	// background llmStream reads it while the next turn's probe rewrites it.
	mainSlotTokens atomic.Int64

	imagesSupported atomic.Bool

	clientCaps ClientCapabilities

	// connSems caps concurrent calls per [[llm]] entry; a nil entry means no limit (test mocks).
	connSems []chan struct{}

	mcp   mcpState
	tools toolRegistry

	// abortReason, when set, makes Prompt refuse every turn (e.g. not in a devcontainer).
	abortReason string

	// standalone (--cli) only changes hint wording. Written once by runCLI before the
	// connection exists, so it needs no lock.
	standalone bool

	// bgMu guards bgJobs and bgSeq.
	bgMu   sync.Mutex
	bgJobs map[int]*backgroundJob
	// logPrev is the last logged request body per session and connection (requestLogDelta).
	logPrevMu sync.Mutex
	logPrev   map[string][]byte
	bgSeq     int
}

type mcpState struct {
	// mu serializes reconcileMCP and shutdownMCP. clients is under agent.mu instead,
	// so notifyCapabilities need not wait out a slow server start.
	mu      sync.Mutex
	clients map[string]*MCPClient
	// applied holds only the servers that run, so the next diff retries a failed
	// start. The diff runs only when the mtime changed, so a broken server does not
	// re-emit its failed card on every prompt.
	applied      []MCPServerConfig
	appliedMtime time.Time
}

func main() {
	// Exactly one line: installed updaters compare the whole output of the
	// downloaded binary, and a second line leaves them unable to update.
	if len(os.Args) > 1 && os.Args[1] == "--version" {
		fmt.Println(versionLine(version))
		os.Exit(0)
	}
	if len(os.Args) > 1 && os.Args[1] == "--build" {
		fmt.Println(versionBanner())
		os.Exit(0)
	}

	if len(os.Args) > 1 && os.Args[1] == "--update" {
		os.Exit(runUpdate())
	}

	if len(os.Args) > 1 && os.Args[1] == "--setup" {
		runSetup()
		os.Exit(0)
	}

	if len(os.Args) > 1 && os.Args[1] == "--cli" {
		os.Exit(runCLI(os.Args[2:]))
	}

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))

	a := &agent{sessions: make(map[string]*Session), mode: "Interactive"}
	go killOrphanedJobs()
	conn := NewAgentSideConnection(a, os.Stdout, os.Stdin)
	a.conn = conn

	slog.Info("waiting for connection")
	<-conn.Done()
	slog.Info("connection closed")
	a.shutdownMCP()
	a.shutdownBackground()
}

func (a *agent) Initialize(ctx context.Context, req InitializeRequest) (InitializeResponse, error) {
	// Never fail negotiation: the spec says answer with our version and let the client decide.
	if req.ProtocolVersion != protocolVersion {
		slog.Info("initialize: client speaks a different protocol version, answering with ours",
			"client", req.ProtocolVersion, "agent", protocolVersion)
	}
	a.mu.Lock()
	a.clientCaps = req.ClientCapabilities
	a.mu.Unlock()
	// No cwd yet, so global settings only; the first prepare re-probes with project settings.
	if gs, err := loadGlobalSettings(); err == nil {
		a.cfgMu.Lock()
		a.setSettings(gs)
		a.cfgMu.Unlock()
		if conn := a.connFor("execute"); conn != nil {
			// The setting wins: a server without /props (Halogen) cannot report
			// vision, and the client learns image support only here.
			if conn.ImageSupport != nil {
				a.imagesSupported.Store(*conn.ImageSupport)
			} else {
				a.imagesSupported.Store(probeLLM(ctx, conn).ImageSupport)
			}
		}
	}
	var res InitializeResponse
	res.ProtocolVersion = protocolVersion
	res.AgentCapabilities.LoadSession = true
	res.AgentCapabilities.PromptCapabilities.Image = a.imagesSupported.Load()
	res.AgentCapabilities.PromptCapabilities.EmbeddedContext = true
	res.AgentCapabilities.MCPCapabilities.HTTP = true
	res.AgentCapabilities.SessionCapabilities = &struct {
		List  *struct{} `json:"list,omitempty"`
		Close *struct{} `json:"close,omitempty"`
	}{List: &struct{}{}, Close: &struct{}{}}
	res.AgentInfo = struct {
		Name    string `json:"name,omitempty"`
		Version string `json:"version,omitempty"`
	}{"codehalter", version}
	res.AuthMethods = []AuthMethod{{
		ID:          "terminal-setup",
		Name:        "Terminal Setup",
		Description: "Interactive LLM configuration",
		Type:        "terminal",
		Args:        []string{"--setup"},
	}}
	return res, nil
}

// A method the client did not claim must never be sent. An unknown which reads as "read".
func (a *agent) clientCan(which string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch which {
	case "write":
		return a.clientCaps.Fs.WriteTextFile
	case "elicitation":
		return a.clientCaps.Elicitation != nil && a.clientCaps.Elicitation.Form != nil
	case "terminal":
		return a.clientCaps.Terminal
	}
	return a.clientCaps.Fs.ReadTextFile
}

func (a *agent) NewSession(_ context.Context, req NewSessionRequest) (NewSessionResponse, error) {
	cwd, _, err := usableCwd(req.Cwd)
	if err != nil {
		return NewSessionResponse{}, err
	}
	slog.Debug("NewSession: enter", "cwd", cwd)
	s, err := newSession(cwd)
	if err != nil {
		slog.Debug("NewSession: newSession err", "err", err)
		return NewSessionResponse{}, err
	}
	// Ids are second-granular, and an unsaved session exists only in a.sessions,
	// where putSession would silently evict it.
	for n, base := 2, s.ID; a.getSession(s.ID) != nil; n++ {
		s = newSessionWithID(cwd, fmt.Sprintf("%s_%d", base, n))
	}
	if err := a.initSession(cwd, s, req.McpServers); err != nil {
		slog.Debug("NewSession: initSession err", "err", err, "sid", s.ID)
		return NewSessionResponse{}, err
	}
	a.startIndexing(s.ID, cwd)
	slog.Debug("NewSession: returning", "sid", s.ID)
	return NewSessionResponse{SessionId: s.ID, Modes: a.sessionModes()}, nil
}

func (a *agent) LoadSession(ctx context.Context, req LoadSessionRequest) (LoadSessionResponse, error) {
	cwd, substituted, err := usableCwd(req.Cwd)
	if err != nil {
		return LoadSessionResponse{}, err
	}
	slog.Debug("LoadSession: enter", "cwd", cwd, "sid", req.SessionId)
	if substituted {
		a.say(ctx, req.SessionId, fmt.Sprintf("Started a new session: the workspace this thread was created in (%s) isn't available here, so there was nothing to restore.\n\n", req.Cwd))
	}
	s, err := a.restoreSession(ctx, cwd, req.SessionId, req.McpServers)
	if err != nil {
		return LoadSessionResponse{}, err
	}
	a.startIndexing(s.ID, cwd)
	return LoadSessionResponse{Modes: a.sessionModes()}, nil
}

func (a *agent) ListSessions(_ context.Context, req ListSessionsRequest) (ListSessionsResponse, error) {
	sessions, err := listSessions(cwdOrDefault(req.Cwd))
	if err != nil {
		return ListSessionsResponse{Sessions: []SessionInfo{}}, err
	}
	if sessions == nil {
		sessions = []SessionInfo{}
	}
	return ListSessionsResponse{Sessions: sessions}, nil
}

func (a *agent) SetSessionMode(ctx context.Context, req SetSessionModeRequest) error {
	if req.ModeId != "Interactive" && req.ModeId != "Autopilot" {
		return nil
	}
	a.mu.Lock()
	a.mode = req.ModeId
	a.mu.Unlock()
	// "currentModeId", not the request's "modeId": the client rejects that with -32602.
	a.sendUpdate(ctx, req.SessionId, struct {
		Kind          string `json:"sessionUpdate"`
		CurrentModeId string `json:"currentModeId"`
	}{Kind: "current_mode_update", CurrentModeId: req.ModeId})
	a.say(ctx, req.SessionId, "Mode: "+req.ModeId+"\n\n")
	return nil
}

// CloseSession keeps the on-disk TOML: close means the client stopped watching, not delete.
func (a *agent) CloseSession(ctx context.Context, req CloseSessionRequest) error {
	a.Cancel(ctx, CancelNotification(req))
	a.deleteSession(req.SessionId)
	return nil
}

func (a *agent) Cancel(_ context.Context, n CancelNotification) {
	if sess := a.getSession(n.SessionId); sess != nil {
		sess.cancelTurn()
	}
	a.mu.Lock() // global slot: the pre-turn bootstrap
	if a.cancel != nil {
		a.cancel()
	}
	a.mu.Unlock()
}

func (a *agent) getSession(id string) *Session {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.sessions[id]
}

func (a *agent) putSession(s *Session) {
	a.mu.Lock()
	defer a.mu.Unlock()
	slog.Info("putSession", "sid", s.ID)
	a.sessions[s.ID] = s
}

func (a *agent) deleteSession(id string) {
	a.mu.Lock()
	sess := a.sessions[id]
	delete(a.sessions, id)
	a.mu.Unlock()
	if sess != nil {
		sess.ctl.mu.Lock()
		stop := sess.ctl.warmStop
		sess.ctl.warmStop = nil
		sess.ctl.mu.Unlock()
		if stop != nil {
			stop()
		}
	}
}

// mcpOffer is only held here: importing needs an elicitation, which the client
// accepts only after session/new has returned the id (see offerMCPImport).
func (a *agent) initSession(cwd string, s *Session, mcpOffer []acpMCPServer) (err error) {
	s.mcpOffer = mcpOffer
	a.putSession(s)
	defer func() {
		if err != nil {
			a.deleteSession(s.ID)
		}
	}()

	// Prompts, skills and macros are never copied out, or a project would pin the
	// release that created it; a same-named .codehalter file only overrides. mcp.toml
	// is written once: its header comments are the only schema documentation.
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating %s: %w", dir, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "mcp.toml")); os.IsNotExist(err) {
		if err := os.WriteFile(filepath.Join(dir, "mcp.toml"), []byte(defaultMCPToml), 0o644); err != nil {
			return fmt.Errorf("writing the mcp.toml placeholder: %w", err)
		}
	}
	// systemPrompt below picks skills from knownStacks before checkEnv has run.
	s.knownStacks = detectStacks(cwd)

	// Always rebuild: a loaded session may carry one from another host, and a fix
	// card must not send an LLM call with it.
	if sp, err := a.systemPrompt(s.ID); err != nil {
		slog.Warn("initSession: systemPrompt build failed", "sid", s.ID, "err", err)
	} else {
		s.SystemPrompt = sp
	}
	settings, err := loadSettings(cwd)
	if err != nil {
		return err
	}
	// Empty settings are fine: the first prepare scaffolds them and blocks on a Retry card.
	a.cfgMu.Lock()
	a.setSettings(settings)
	a.cfgMu.Unlock()
	a.mu.Lock()
	a.emptyProject = isEmptyProject(cwd)
	a.mu.Unlock()
	a.discoverSandbox()
	return nil
}

// Devcontainer first: the gitignore prompt assumes a sandbox.
func (a *agent) startIndexing(sid string, cwd string) {
	// The goroutine closes its own channel: a second session may replace a.indexDone meanwhile.
	done := make(chan struct{})
	// Lets Cancel reach a fix card prepare dispatches. One slot, last writer wins.
	ctx, cancel := context.WithCancel(context.Background())
	a.mu.Lock()
	a.indexDone = done
	a.cancel = cancel
	a.mu.Unlock()
	slog.Debug("startIndexing: spawning bootstrap goroutine", "sid", sid, "cwd", cwd)
	go func() {
		defer close(done)
		defer slog.Debug("startIndexing: bootstrap goroutine done", "sid", sid)
		defer cancel()

		// Zed registers the session only after reading our session/new response
		// and drops an earlier update as "unknown session".
		select {
		case <-ctx.Done():
			return
		case <-time.After(100 * time.Millisecond):
		}

		if !a.ensureDevcontainer(ctx, cwd, sid) {
			slog.Debug("startIndexing: ensureDevcontainer false, aborting", "sid", sid)
			return
		}
		if !a.ensureTerminals(ctx, sid) {
			slog.Debug("startIndexing: client has no terminal capability, aborting", "sid", sid)
			return
		}
		slog.Debug("startIndexing: devcontainer ok, about to ensureGitignore", "sid", sid)
		a.ensureGitignore(ctx, cwd, sid)
		// An imported server must be in the file before the first reconcileMCP reads it.
		a.offerMCPImport(ctx, cwd, sid)

		sess := a.getSession(sid)
		if sess != nil {
			slog.Debug("startIndexing: gitignore done, about to prepare", "sid", sid)
			// Otherwise a slow probe is indistinguishable from a hang.
			a.say(ctx, sid, "Setting up: probing the LLM, reading the project, checking its tooling. The first probe can take a while if your server still has to load the model.\n\n")
			fixes := a.prepareChecks(ctx, sess, sid)
			// After the checks, so the warmed bytes match turn one; before the fix
			// cards, whose accepted turn the warm must beat to its first call.
			go a.prewarm(sess)
			// An accepted card runs a whole turn, so it holds the turn like a prompt.
			if len(fixes) > 0 {
				turnCtx, release, _ := a.holdTurn(ctx, sess, true)
				a.drainFixes(turnCtx, sid, fixes)
				release()
			}
		}
		slog.Debug("startIndexing: bootstrap done", "sid", sid)
	}()
}

// startupDone is closed once the latest session's startup has finished; nil before any.
func (a *agent) startupDone() chan struct{} {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.indexDone
}

func (a *agent) sessionModes() *SessionModeState {
	a.mu.Lock()
	current := a.mode
	a.mu.Unlock()
	return &SessionModeState{
		CurrentModeId: current,
		AvailableModes: []struct {
			Id          string `json:"id"`
			Name        string `json:"name"`
			Description string `json:"description,omitempty"`
		}{
			{Id: "Interactive", Name: "Interactive", Description: "Ask before setup and anything outside the container"},
			{Id: "Autopilot", Name: "Autopilot", Description: "Auto-answer prompts: no user interruption"},
		},
	}
}

func (a *agent) isAutopilot() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.mode == "Autopilot"
}

func (a *agent) sendUpdate(ctx context.Context, sid string, u any) {
	if a.conn == nil {
		return
	}
	if err := a.conn.SessionUpdate(ctx, sid, u); err != nil {
		// A broken transport surfaces on the next read; debug keeps the stream quiet.
		slog.Debug("sendUpdate: SessionUpdate write failed", "sid", sid, "err", err)
	}
}

// say also logs the text, so a run reads back from the session log alone. Callers
// own trailing newlines: chunks may be streamed fragments.
func (a *agent) say(ctx context.Context, sid, text string) {
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: text}})
	if t := strings.TrimSpace(text); t != "" {
		a.logSession(sid, "SAY", "%s", t)
	}
}

// A var so a test can shorten it.
var heartbeatEvery = 2 * time.Second

// Wrap only work that asks nothing: around a card it would tick while the card waits.
func (a *agent) heartbeat(ctx context.Context, sid string) func() {
	tickCtx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	ticks := 0
	go func() {
		defer close(done)
		t := time.NewTicker(heartbeatEvery)
		defer t.Stop()
		for {
			select {
			case <-tickCtx.Done():
				return
			case <-t.C:
				ticks++
				a.say(tickCtx, sid, ".")
			}
		}
	}()
	// ticks is read only after <-done, so the close orders it; no lock needed.
	return func() {
		cancel()
		<-done
		if ticks > 0 {
			a.say(ctx, sid, "\n")
		}
	}
}

func (a *agent) sendUpdateAndAbort(ctx context.Context, sid, reason string) {
	a.mu.Lock()
	a.abortReason = reason
	a.mu.Unlock()
	a.say(ctx, sid, reason+"\n")
}

func (a *agent) logSession(sid string, tag, format string, args ...any) {
	if sid == "" {
		return
	}
	sess := a.getSession(sid)
	if sess == nil {
		return
	}
	path := sessionPath(sess.Cwd, sid, "log")
	logF, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer logF.Close()
	fmt.Fprintf(logF, "\n=== %s [%s] ===\n", time.Now().Format(time.RFC3339), tag)
	fmt.Fprintf(logF, format, args...)
	if !strings.HasSuffix(format, "\n") {
		fmt.Fprintln(logF)
	}
}
