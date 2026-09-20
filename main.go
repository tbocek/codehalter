package main

import (
	"context"
	_ "embed"
	"encoding/base64"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

//go:embed res/PLAN.md
var defaultPlanMD string

//go:embed res/EXECUTE.md
var defaultExecuteMD string

//go:embed res/DOCUMENT.md
var defaultDocumentMD string

//go:embed res/SUMMARISE.md
var defaultSummariseMD string

//go:embed res/RESUMMARISE.md
var defaultResummariseMD string

//go:embed res/SPEC.md
var defaultSpecMD string

//go:embed res/SPEC-SETUP.md
var defaultSpecSetupMD string

//go:embed res/SPEC-REMOVE.md
var defaultSpecRemoveMD string

//go:embed res/Dockerfile.devcontainer.alpine
var defaultDevcontainerDockerfileAlpine string

//go:embed res/Dockerfile.devcontainer.arch
var defaultDevcontainerDockerfileArch string

//go:embed res/Dockerfile.devcontainer.debian
var defaultDevcontainerDockerfileDebian string

//go:embed res/Dockerfile.devcontainer.fedora
var defaultDevcontainerDockerfileFedora string

//go:embed res/Dockerfile.devcontainer.ubuntu
var defaultDevcontainerDockerfileUbuntu string

//go:embed res/devcontainer.json
var defaultDevcontainerJSON string

//go:embed res/settings.toml
var defaultSettingsTOML string

//go:embed res/mcp.toml
var defaultMCPToml string

// agent implements acp.Agent.
type agent struct {
	// mu guards the mutable top-level fields: cancel, sessions, mode,
	// abortReason and the probe-derived LLM fields. MCP state has its own mutex,
	// and settings + connSems are reassigned under cfgMu.
	mu sync.Mutex
	// cfgMu guards settings + connSems, which a turn's prepare phase reassigns
	// while a PRIOR turn's background goroutine may still read them. Writers
	// Lock; readers RLock, copy what they need and release before any blocking
	// call. A strict leaf: never held while acquiring a.mu or sess.mu.
	cfgMu        sync.RWMutex
	conn         *AgentSideConnection
	cancel       context.CancelFunc
	sessions     map[string]*Session
	settings     Settings
	emptyProject bool // set once at startup: cwd held no files of its own (projectIsEmpty)
	indexDone    chan struct{}
	mode         string // "Interactive" | "Autopilot"

	// connProbe holds the probe result per connection, keyed by
	// Server+"\x00"+Model, so the banner can warn when a reachable server does
	// not list the configured model (the silent cause of empty completions).
	// nil before the first prepare, which reads as the zero result.
	connProbe map[string]probeResult

	// summaryStrikes counts consecutive failures of the dedicated summariser
	// connection (purpose = "summary"), and summaryStruckAt is when the latest
	// one happened (unix nanos). At summaryMaxStrikes, connForBackgroundLLM
	// routes the notes to llm[0] until summaryCooldown has passed since that
	// strike. Atomic, not under a.mu: written from the summarise goroutine, read
	// from the turn path.
	summaryStrikes  atomic.Int32
	summaryStruckAt atomic.Int64

	// ctkIgnored remembers which Server+Model has already been reported for
	// accepting chat_template_kwargs and ignoring it, so the warning fires once
	// per deployment instead of once per call. Its own map, not under a.mu:
	// llmStream reaches it from every background goroutine and must not queue
	// behind a foreground handler for a bool.
	ctkIgnored sync.Map

	// mainSlotTokens is LLM[0]'s per-slot context window in tokens, from the
	// startup probe: llama.cpp reports it per slot, otherwise a known total is
	// divided by the slot count. 0 means unknown, which ensureLLM treats like
	// "below minSlotTokens": a Retry card, so any turn that runs can rely on it.
	mainSlotTokens int

	// imagesSupported is whether LLM[0] accepts inline images — the agent-wide
	// image capability advertised over ACP. Derived from LLM[0]'s config
	// image_support (else the probe) by probeAllLLMs; gates view_image and the
	// history image encoding.
	imagesSupported bool

	// clientCaps is what the editor told us it supports in the initialize
	// request. Set once by Initialize (before any session exists) and read by
	// fsRead/fsWrite to decide between the ACP filesystem and plain disk I/O.
	// Zero value = "the client advertised nothing", which is the safe reading:
	// every capability gate falls back to doing the work ourselves.
	clientCaps ClientCapabilities

	// connSems caps concurrent calls per [[llm]] entry: a buffered channel of
	// capacity parallelCap() that llmStream acquires and releases, so a busy
	// conn queues excess calls. Rebuilt by buildConnSems on every settings
	// reload. A nil entry means no semaphore (test mocks).
	connSems []chan struct{}

	// streamRules is the compiled stream-rule set (see rules.go): patterns that
	// abort a generation mid-token when the reply goes off the rails. Loaded per
	// project from .codehalter/rules.toml (else the built-in defaults) by
	// initSession, and reassigned wholesale like the rest of the config — so it
	// lives under cfgMu with settings and connSems, not under mu.
	streamRules []streamRule

	// mcp owns the MCP server children and the bookkeeping reconcileMCP
	// needs; its mutex guards the whole group (see mcpState).
	mcp mcpState
	// tools is every tool the model can call (toolRegistry): the built-ins,
	// the project's run_command/run_background, and MCP tools.
	tools toolRegistry

	// abortReason is set by the bootstrap goroutine when codehalter must not
	// run in this environment (today: started outside a devcontainer). Empty
	// means proceed; non-empty causes Prompt to refuse with this message.
	// Read under mu.
	abortReason string

	// standalone is true when our own terminal client is driving us (--cli,
	// see cli.go) rather than an editor. It only picks the wording of the
	// "you are not in a container" hints, which otherwise tell a terminal user
	// to press a Zed keyboard shortcut. Written once by runCLI before the
	// connection exists, so it needs no lock.
	standalone bool

	// bgJobs tracks run_background jobs (long-lived processes the model starts:
	// dev servers, watchers) so shutdownBackground can release their terminals on
	// exit — which kills them — instead of orphaning them with a held port. Keyed
	// by the sequential id shown to the model; bgSeq hands out ids. Guarded by bgMu.
	bgMu   sync.Mutex
	bgJobs map[int]*backgroundJob
	bgSeq  int
}

// mcpState owns the spawned MCP server children plus the bookkeeping
// reconcileMCP uses to diff the desired set against what's running. mu guards
// the whole group, so a reconcile's diff/start/stop sequence is atomic against
// a concurrent pass (startIndexing's banner pass vs Prompt's per-turn pass).
type mcpState struct {
	// mu serialises reconcileMCP across concurrent callers; its scope covers
	// the full diff/start/stop sequence so two reconciles can't race on
	// clients or the global tool registry.
	mu sync.Mutex
	// clients holds the spawned MCP server children, keyed by their configured
	// `name`. Tools they advertise are registered into the global tool registry
	// as `<name>__<tool>` so multiple servers can ship a tool called e.g.
	// "search" without colliding. Populated and mutated by reconcileMCP; nil on
	// projects without an mcp.toml.
	clients map[string]*MCPClient
	// applied is the set of [[server]] entries the last successful reconcile
	// actually brought up. The next pass diffs the new file against this to
	// decide what to start, stop, or restart. Entries that failed to start are
	// NOT included, so the next reconcile retries them with a fresh
	// StartMCPClient call once the file changes again.
	applied []MCPServerConfig
	// appliedMtime is the mtime of .codehalter/mcp.toml at the time of the last
	// reconcile. Unchanged mtime → skip the diff entirely, which also keeps a
	// persistently-broken server from re-emitting the same failed card on every
	// prompt. Zero value means "never reconciled yet".
	appliedMtime time.Time
	// Deferred-reconcile scheduler (see mcpState.schedule). An mcp.toml written
	// mid-turn must not be applied mid-turn: registering tools rewrites the
	// `tools` array, which renders ahead of the whole conversation, so the
	// prompt would move under a turn in flight. flushMu guards this group only
	// and is a leaf, never held across a reconcile.
	flushMu      sync.Mutex
	flushing     bool
	flushPending bool
	// flushDone is closed when the running flush finishes; nil while idle, so a
	// starting turn can wait one out (mcpState.wait).
	flushDone chan struct{}
	// flushNotes / flushFixes hold what a between-turns flush produced. Said
	// there, a card would have no turn to dispatch from and a notice might not
	// render, so both are parked and the next checkMCP replays them in a turn.
	flushNotes []string
	flushFixes []fixProblem
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

func main() {
	// --version: the release tag this binary was built from. selfUpdate runs the
	// binary it has just downloaded with this flag and compares the whole line
	// before replacing anything, so the format is load-bearing rather than
	// cosmetic (see versionLine).
	if len(os.Args) > 1 && os.Args[1] == "--version" {
		fmt.Println(versionLine(version))
		os.Exit(0)
	}

	// --update: install the newest release over this binary and exit. The
	// terminal path asks first (offerUpdate); this is the form for a run with
	// nobody watching, and the one the capabilities banner points an editor
	// user at, since replacing the binary an editor is currently talking to is
	// something to do between sessions rather than during one.
	if len(os.Args) > 1 && os.Args[1] == "--update" {
		os.Exit(runUpdate())
	}

	// --setup flag: interactive LLM configuration, skips the ACP server.
	if len(os.Args) > 1 && os.Args[1] == "--setup" {
		runSetup()
		os.Exit(0)
	}

	// --cli flag: be our own ACP client and drive the agent from a terminal,
	// instead of speaking the protocol over stdio to an editor. Same agent,
	// same protocol; see cli.go. It sets up its own logging, because stderr is
	// the user's screen here rather than an editor's log pane.
	if len(os.Args) > 1 && os.Args[1] == "--cli" {
		os.Exit(runCLI(os.Args[2:]))
	}

	// Global slog → stderr at debug (Zed captures it live); per-session detail
	// (LLM req/reply, errors) goes to .codehalter/session_<id>.log via logSession.
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))

	a := &agent{sessions: make(map[string]*Session), mode: "Interactive"}
	conn := NewAgentSideConnection(a, os.Stdout, os.Stdin)
	a.conn = conn

	slog.Info("waiting for connection")
	<-conn.Done()
	slog.Info("connection closed")
	a.shutdownMCP()        // reap MCP children (spawned with no context) so they don't orphan
	a.shutdownBackground() // reap detached run_background daemons (dev servers etc.)
}

// ---------------------------------------------------------------------------
// ACP protocol handlers
// ---------------------------------------------------------------------------

func (a *agent) Initialize(ctx context.Context, req InitializeRequest) (InitializeResponse, error) {
	// Version negotiation must not fail: the spec says an agent that does not
	// speak the requested version answers with the latest it supports and lets
	// the client decide. res.ProtocolVersion is always ours; this only logs.
	if req.ProtocolVersion != protocolVersion {
		slog.Info("initialize: client speaks a different protocol version, answering with ours",
			"client", req.ProtocolVersion, "agent", protocolVersion)
	}
	// Remember what the client can do before anything tries to use it. Written
	// once, here, before any session exists — but handlers run concurrently, so
	// it takes the same lock as the other agent-wide capability fields.
	a.mu.Lock()
	a.clientCaps = req.ClientCapabilities
	a.mu.Unlock()
	// Probe the LLM's metadata endpoints to advertise image support. Only the
	// global settings are loadable here (there is no cwd yet); if project-local
	// settings override the LLM, the first prepare re-probes and corrects it.
	if gs, err := loadGlobalSettings(); err == nil {
		a.cfgMu.Lock()
		a.settings = gs
		a.buildConnSems()
		a.cfgMu.Unlock()
		if conn := a.settings.MainLLM("execute"); conn != nil {
			a.imagesSupported = probeLLM(ctx, conn).ImageSupport
		}
	}
	var res InitializeResponse
	res.ProtocolVersion = protocolVersion
	res.AgentCapabilities.LoadSession = true
	res.AgentCapabilities.PromptCapabilities.Image = a.imagesSupported
	res.AgentCapabilities.PromptCapabilities.EmbeddedContext = true
	res.AgentCapabilities.MCPCapabilities.HTTP = true
	res.AgentCapabilities.SessionCapabilities = &struct {
		List  *struct{} `json:"list,omitempty"`
		Close *struct{} `json:"close,omitempty"`
	}{List: &struct{}{}, Close: &struct{}{}}
	// Static implementation block advertised in the initialize response —
	// name and version don't change at runtime. The version is the release tag
	// this binary was built from ("dev" for a local build), so an editor
	// reporting a protocol problem names the build that produced it.
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

// clientCan reports whether the editor advertised "read", "write",
// "elicitation" or "terminal". A method the client did not claim must never
// be sent, so false means a fallback (fsRead/fsWrite do the I/O, asks go out
// as permission requests), except "terminal", which has none.
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
	// newSession steps over the ids already on disk. A session that has not
	// saved yet (no turn taken) exists only in a.sessions, and putSession would
	// evict it without a word: ids are second-granular, so a client that opens a
	// second session right away (the CLI's /new) lands there routinely.
	for n, base := 2, s.ID; a.getSession(s.ID) != nil; n++ {
		s = newSessionWithID(cwd, fmt.Sprintf("%s_%d", base, n))
	}
	// Hold the editor's MCP list for offerMCPImport, which runs in the bootstrap
	// goroutine below: importing needs an elicitation, and session/new must
	// return the id before the client will accept one.
	s.mcpOffer = req.McpServers
	if err := a.initSession(cwd, s); err != nil {
		slog.Debug("NewSession: initSession err", "err", err, "sid", s.ID)
		a.deleteSession(s.ID)
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
	// The requested workspace isn't mounted here (Zed restored a thread from a
	// project that no longer exists in this environment), so there was nothing
	// to restore — tell the user this is a fresh session rather than silently
	// dropping their old thread's history.
	if substituted {
		a.say(ctx, req.SessionId, fmt.Sprintf("Started a new session: the workspace this thread was created in (%s) isn't available here, so there was nothing to restore.\n\n", req.Cwd))
	}
	s, err := loadSession(cwd, req.SessionId)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Debug("LoadSession: not found, treating as new", "sid", req.SessionId)
			// Zed cached an id from a session/new that never wrote a file, and now
			// loads it and prompts under it. It does NOT honour a sessionId we send
			// back, so the cached id is accepted as-is or prompts would not route.
			s = newSessionWithID(cwd, req.SessionId)
			s.mcpOffer = req.McpServers
			if err := a.initSession(cwd, s); err != nil {
				a.deleteSession(s.ID)
				return LoadSessionResponse{}, err
			}
			a.startIndexing(s.ID, cwd)
			return LoadSessionResponse{Modes: a.sessionModes()}, nil
		}
		return LoadSessionResponse{}, fmt.Errorf("loading session: %w", err)
	}
	s.mcpOffer = req.McpServers
	if err := a.initSession(cwd, s); err != nil {
		a.deleteSession(s.ID)
		return LoadSessionResponse{}, err
	}
	// Re-announce the thread's name: the client asked us to restore this
	// session, so it's ours to label, and a reload otherwise drops back to the
	// session id.
	if s.Title != "" {
		a.sendUpdate(ctx, req.SessionId, sessionInfoUpdate{Kind: "session_info_update", Title: s.Title})
	}
	// Replay the restored thread's messages to the client so the UI shows the
	// prior conversation. An empty chunk of the opposite role is emitted before
	// two consecutive same-role messages so Zed renders them as distinct turns;
	// images are re-read from disk and re-sent inline (missing files degrade to
	// a placeholder line).
	lastRole := ""
	for _, m := range s.Messages {
		if m.Role == lastRole {
			if m.Role == "user" {
				a.say(ctx, req.SessionId, "")
			} else {
				a.sendUpdate(ctx, req.SessionId, messageChunk{Kind: KindUserMessage, Content: ContentBlock{Type: "text", Text: ""}})
			}
		}
		if m.Role == "user" {
			a.sendUpdate(ctx, req.SessionId, messageChunk{Kind: KindUserMessage, Content: ContentBlock{Type: "text", Text: m.Content}})
			for _, img := range m.Images {
				data, mime, err := readImageFile(s.Cwd, img.ID)
				if err != nil {
					a.sendUpdate(ctx, req.SessionId, messageChunk{Kind: KindUserMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("[image %s missing on disk]", img.ID)}})
					continue
				}
				if mime == "" {
					mime = img.MimeType
				}
				a.sendUpdate(ctx, req.SessionId, messageChunk{Kind: KindUserMessage, Content: ContentBlock{Type: "image", MimeType: mime, Data: base64.StdEncoding.EncodeToString(data)}})
			}
		} else {
			a.say(ctx, req.SessionId, m.Content)
		}
		lastRole = m.Role
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
	// The current_mode_update notification carries "currentModeId" (matching
	// SessionModeState.CurrentModeId), not the "modeId" of the inbound
	// SetSessionModeRequest. Sending "modeId" makes the client reject the update
	// with -32602 "missing field currentModeId".
	a.sendUpdate(ctx, req.SessionId, struct {
		Kind          string `json:"sessionUpdate"`
		CurrentModeId string `json:"currentModeId"`
	}{Kind: "current_mode_update", CurrentModeId: req.ModeId})
	a.say(ctx, req.SessionId, "Mode: "+req.ModeId+"\n\n")
	return nil
}

// CloseSession cancels any in-flight turn for sid and drops the session from
// the live map. The on-disk TOML is preserved so /session resume still works
// — close is a "this client is done watching" signal, not a delete.
func (a *agent) CloseSession(_ context.Context, req CloseSessionRequest) error {
	if sess := a.getSession(req.SessionId); sess != nil {
		sess.cancelTurn()
	}
	a.mu.Lock() // global slot: the pre-turn bootstrap (ensureDevcontainer)
	if a.cancel != nil {
		a.cancel()
	}
	a.mu.Unlock()
	a.deleteSession(req.SessionId)
	return nil
}

func (a *agent) Cancel(_ context.Context, n CancelNotification) {
	if sess := a.getSession(n.SessionId); sess != nil {
		sess.cancelTurn() // the in-flight turn for THIS session
	}
	a.mu.Lock() // global slot: the pre-turn bootstrap
	if a.cancel != nil {
		a.cancel()
	}
	a.mu.Unlock()
}

// ---------------------------------------------------------------------------
// Session registry
// ---------------------------------------------------------------------------

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
	// A closed session has no next turn to keep a prefix warm for.
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

// ---------------------------------------------------------------------------
// Session bootstrap
// ---------------------------------------------------------------------------

func (a *agent) initSession(cwd string, s *Session) error {
	a.putSession(s)

	// Seed .codehalter/ defaults when absent. Phase prompts
	// (PLAN/EXECUTE/DOCUMENT/SUMMARISE/RESUMMARISE) are user-owned templates seeded once;
	// every SKILL-*.md (including the always-on base skill) is owned by
	// ensureSkills (skills.go), which seeds it once and otherwise leaves it.
	dir := filepath.Join(cwd, ".codehalter")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating %s: %w", dir, err)
	}
	for _, f := range []struct{ name, content string }{
		{"PLAN.md", defaultPlanMD},
		{"EXECUTE.md", defaultExecuteMD},
		{"DOCUMENT.md", defaultDocumentMD},
		{"SUMMARISE.md", defaultSummariseMD},
		{"RESUMMARISE.md", defaultResummariseMD},
		{"SPEC.md", defaultSpecMD},
		{"SPEC-SETUP.md", defaultSpecSetupMD},
		{"SPEC-REMOVE.md", defaultSpecRemoveMD},
	} {
		if err := seedFile(dir, f.name, f.content); err != nil {
			return err
		}
	}
	if err := ensureSkills(cwd, detectStacks(cwd), readOSInfo()); err != nil {
		return err
	}
	// Template macros (TEMPLATE-*.md): seed editable copies once, same as the
	// phase prompts. After this the on-disk copy wins, so users can edit them.
	if err := seedTemplates(cwd); err != nil {
		return err
	}
	// mcp.toml — only seeded on first run with the minimal placeholder (a header
	// and ONE generic commented example). Once this file exists we never touch
	// it again — the user owns it.
	if err := seedFile(dir, "mcp.toml", defaultMCPToml); err != nil {
		return err
	}

	// Always rebuild SystemPrompt from the freshly-seeded directory: a loaded
	// session may carry one from another host (a different OS skill set), and it
	// must be replaced BEFORE a fix card can send an LLM call with the stale one.
	if sp, err := a.systemPrompt(s.ID); err != nil {
		slog.Warn("initSession: systemPrompt build failed", "sid", s.ID, "err", err)
	} else {
		s.SystemPrompt = sp
	}
	settings, err := loadSettings(cwd)
	if err != nil {
		return err
	}
	// Empty settings are tolerated (path == "") — the first Prompt's prepare
	// phase scaffolds the skeleton and blocks on a Retry card until the user
	// fills it in. Running with no LLM until then is graceful (renderLLMStatus
	// prints a warning instead of crashing).
	a.cfgMu.Lock()
	a.settings = settings
	a.buildConnSems()
	// Stream rules are project config like the rest: reloaded here so editing
	// .codehalter/rules.toml takes effect on the next session without a rebuild.
	a.streamRules = loadStreamRules(cwd)
	a.cfgMu.Unlock()
	// An empty project gets no skeleton: the first turn carries a hint telling
	// the model to ask which language and runner to use (emptyProjectHint).
	a.mu.Lock()
	a.emptyProject = isEmptyProject(cwd)
	a.mu.Unlock()
	a.discoverSandbox()
	return nil
}

// startIndexing runs the once-per-session bootstrap in a goroutine: the
// devcontainer and gitignore prompts, then the first prepare, so the banner
// shows at session open. Devcontainer first, since the gitignore prompt
// assumes a sandbox; a failure there sets abortReason and skips the rest.
func (a *agent) startIndexing(sid string, cwd string) {
	a.indexDone = make(chan struct{})
	slog.Debug("startIndexing: spawning bootstrap goroutine", "sid", sid, "cwd", cwd)
	go func() {
		defer close(a.indexDone)
		defer slog.Debug("startIndexing: bootstrap goroutine done", "sid", sid)
		// Install a.cancel so Zed's Cancel button can interrupt a fix-install
		// orchestrate that prepare may dispatch via proposeFix. Same pattern
		// as Prompt(): one cancel slot, last-writer-wins.
		ctx, cancel := context.WithCancel(context.Background())
		a.mu.Lock()
		a.cancel = cancel
		a.mu.Unlock()
		defer cancel()

		// Brief pause before the first session/update. Zed registers the session
		// asynchronously AFTER reading our session/new response, and an update
		// landing inside that window is dropped as "unknown session", which is why
		// the devcontainer notice never showed until the first prompt. 100ms lets
		// registration win the race.
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
		// After the gitignore question and before any LLM work: this is the
		// last interactive gate, and an imported server has to be in the file
		// before the first turn's reconcileMCP reads it.
		a.offerMCPImport(ctx, cwd, sid)

		sess := a.getSession(sid)
		if sess != nil {
			slog.Debug("startIndexing: gitignore done, about to prepare", "sid", sid)
			// prepareChecks is the longest silent stretch of a session open: it
			// probes every LLM (a local server that still has to load a 27B
			// model answers in tens of seconds), seeds skills, and reconciles
			// MCP servers. Without this line the thread sits empty after the
			// gitignore card and a slow probe is indistinguishable from a hang.
			a.say(ctx, sid, "Setting up: probing the LLM, seeding skills, checking project tooling. The first probe can take a while if your server still has to load the model.\n\n")
			fixes := a.prepareChecks(ctx, sess, sid)
			// Prewarm AFTER the checks (SystemPrompt is final, so the warmed bytes
			// match turn one) but BEFORE the fix cards, whose accepted turn the warm
			// must beat to its first call. Backgrounded, so bootstrap never waits.
			go a.prewarm(sess)
			// An accepted card runs a whole turn, so it holds the turn like a
			// typed prompt does: the Cancel button reaches it, and it closes
			// its phase row when done.
			if len(fixes) > 0 {
				turnCtx, release, _ := a.holdTurn(ctx, sess, true)
				a.drainFixes(turnCtx, sid, fixes)
				release()
			}
		}
		slog.Debug("startIndexing: bootstrap done", "sid", sid)
	}()
}

// ---------------------------------------------------------------------------
// Session mode
// ---------------------------------------------------------------------------

// sessionModes is the mode state advertised to the client on session
// create/load. The client uses this to render the mode selector. The mode id
// IS the display name — there's no separate identifier to keep in sync.
func (a *agent) sessionModes() *SessionModeState {
	a.mu.Lock()
	current := a.mode
	a.mu.Unlock()
	if current == "" {
		current = "Interactive"
	}
	return &SessionModeState{
		CurrentModeId: current,
		AvailableModes: []struct {
			Id          string `json:"id"`
			Name        string `json:"name"`
			Description string `json:"description,omitempty"`
		}{
			{Id: "Interactive", Name: "Interactive", Description: "Ask before setup and anything outside the container"},
			{Id: "Autopilot", Name: "Autopilot", Description: "Auto-answer prompts — no user interruption"},
		},
	}
}

// isAutopilot reports whether questions should be auto-answered.
func (a *agent) isAutopilot() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.mode == "Autopilot"
}

// get/setMainSlotTokens guard a.mainSlotTokens for its ONE cross-goroutine access:
// a background llmStream (summariser / git-commit drafter) reads it on its
// finish=length path (llm.go) while the next turn's probeAllLLMs rewrites it. The
// foreground prepare/prompt reads run on the same goroutine as the writes, so they
// read the field directly; only the writer and the background reader need the lock.
func (a *agent) getMainSlotTokens() int  { a.mu.Lock(); defer a.mu.Unlock(); return a.mainSlotTokens }
func (a *agent) setMainSlotTokens(n int) { a.mu.Lock(); a.mainSlotTokens = n; a.mu.Unlock() }

// ---------------------------------------------------------------------------
// Agent → client output
// ---------------------------------------------------------------------------

func (a *agent) sendUpdate(ctx context.Context, sid string, u any) {
	if a.conn == nil {
		return
	}
	if err := a.conn.SessionUpdate(ctx, sid, u); err != nil {
		// Best-effort UI sync: a write failure here means the client
		// transport is broken, which surfaces on the next request read. Log
		// at debug so it's visible during diagnosis without flooding the
		// steady-state token stream.
		slog.Debug("sendUpdate: SessionUpdate write failed", "sid", sid, "err", err)
	}
}

// say emits `text` to the session's chat transcript. This is the overwhelmingly
// common sendUpdate shape, so it gets a name: everything routed through here is
// prose the user reads, as opposed to a tool card, a plan entry, or a status
// line. Callers own their own trailing newlines, because some of these chunks
// are streamed fragments that must concatenate seamlessly.
func (a *agent) say(ctx context.Context, sid, text string) {
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: text}})
}

// heartbeatEvery paces the "I am still here" dots. A var, not a const, so a
// test can shorten it instead of sleeping for real seconds.
var heartbeatEvery = 2 * time.Second

// heartbeat streams a dot into the chat every heartbeatEvery until the
// returned stop is called. Bootstrap blocks for tens of seconds on work the
// user cannot see, and a silent thread looks like a hang. Wrap ONLY work that
// asks nothing: around a card it would tick for as long as the card waits.
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
	// ticks is written only by the goroutine above and read only after <-done,
	// so the channel close orders the two — no lock needed.
	return func() {
		cancel()
		<-done
		if ticks > 0 {
			a.say(ctx, sid, "\n")
		}
	}
}

// sendUpdateAndAbort marks the session as do-not-run and emits the reason to
// chat. Prompt reads a.abortReason under mu and fails every turn until the
// process is restarted (inside a container).
func (a *agent) sendUpdateAndAbort(ctx context.Context, sid, reason string) {
	a.mu.Lock()
	a.abortReason = reason
	a.mu.Unlock()
	a.say(ctx, sid, reason+"\n")
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

// logSession appends a tagged, timestamped block to the session's debug log,
// opening the file per call (it is diagnostic, not time-critical). A no-op for
// an empty sid. The body is written verbatim; keep tags short and greppable.
func (a *agent) logSession(sid string, tag, format string, args ...any) {
	if sid == "" {
		return
	}
	sess := a.getSession(sid)
	if sess == nil {
		return
	}
	path := sess.sessionFilePath(fmt.Sprintf("session_%s.log", sid))
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
