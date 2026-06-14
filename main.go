package main

import (
	"context"
	_ "embed"
	"encoding/base64"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

// dumpGoroutinesOnSignal writes every goroutine's stack to
// .codehalter/goroutines.txt on SIGUSR1, so a wedged turn can be diagnosed
// without ptrace/pprof: `kill -USR1 <pid>` while it's stuck, then read the file.
func dumpGoroutinesOnSignal() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGUSR1)
	go func() {
		for range ch {
			buf := make([]byte, 1<<20)
			for {
				if n := runtime.Stack(buf, true); n < len(buf) {
					buf = buf[:n]
					break
				}
				buf = make([]byte, 2*len(buf))
			}
			path := ".codehalter-goroutines.txt"
			if cwd, err := os.Getwd(); err == nil {
				path = filepath.Join(cwd, ".codehalter", "goroutines.txt")
			}
			if err := os.WriteFile(path, buf, 0o644); err != nil {
				slog.Error("goroutine dump failed", "err", err)
			} else {
				slog.Info("goroutine dump written", "path", path, "bytes", len(buf))
			}
		}
	}()
}

//go:embed res/PLAN.md
var defaultPlanMD string

//go:embed res/EXECUTE.md
var defaultExecuteMD string

//go:embed res/DOCUMENT.md
var defaultDocumentMD string

//go:embed res/SUMMARISE.md
var defaultSummariseMD string

//go:embed res/COMMIT.md
var defaultCommitMD string

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
	// mu guards the mutable top-level fields touched from concurrent ACP
	// handlers and the bootstrap goroutine: cancel, sessions, mode,
	// abortReason, and the probe-derived LLM fields (connReachable,
	// mainSlotTokens, detectedSlots, imagesSupported). MCP state has its own mutex (see
	// mcp mcpState); the per-conn semaphores in connSems lock themselves.
	mu           sync.Mutex
	conn         *AgentSideConnection
	cancel       context.CancelFunc
	sessions     map[string]*Session
	settings     Settings
	runners      []taskRunner
	capabilities capabilities
	emptyProject bool // true on first session if cwd had no source/manifest files
	indexDone    chan struct{}
	mode         string // "Interactive" | "Autopilot"

	// connReachable records whether each configured LLMConnection answered
	// the prepare-phase probe. Keyed by Server+"\x00"+Model. connForBackgroundLLM
	// filters candidates against this so a dead extra slot doesn't burn a
	// timeout on every background summarise call. Populated by probeAllLLMs;
	// nil before the first prepare runs.
	connReachable map[string]bool

	// connProbe holds the full prepare-phase probe result per configured
	// LLMConnection, keyed the same as connReachable (Server+"\x00"+Model).
	// renderLLMStatus reads ModelKnown/ModelLoaded/AvailableModels from it to
	// warn when a server is reachable but the configured model id isn't in its
	// /v1/models list (the silent cause of empty completions). Populated by
	// probeAllLLMs; nil before the first prepare — a nil-map read is the zero
	// probeResult, so renderLLMStatus stays safe.
	connProbe map[string]probeResult

	// mainSlotTokens is the per-slot context window for LLM[0] in tokens.
	// Discovered by the startup probe: llama.cpp /props reports it per-slot
	// directly (default_generation_settings.n_ctx); otherwise a known total
	// (config context_size or /v1/models --ctx-size) is divided by the slot
	// count. 0 means unknown (probe failed or server didn't report it);
	// ensureLLM treats both that and "below minSlotTokens" as a hard failure and
	// loops on a Retry card until the gate passes, so any turn that runs can
	// assume this is ≥ minSlotTokens. Read by the input-size guard (prompt.go) and
	// the startup banner (prepare.go).
	mainSlotTokens int

	// detectedSlots is the total slot count summed across every [[llm]] entry as
	// of the last probeAllLLMs (each conn's parallelCap, after auto-detected
	// total_slots is back-filled). Stored rather than recomputed live because
	// ensureLLM resets settings.Parallel via loadSettings every pass — a live
	// sum would read the post-reset default before the next probe, defeating the
	// "nothing changed, skip the probe" short-circuit. totalSlots() returns this.
	detectedSlots int

	// imagesSupported is whether LLM[0] accepts inline images — the agent-wide
	// image capability advertised over ACP. Derived from LLM[0]'s config
	// image_support (else the probe) by probeAllLLMs; gates view_image and the
	// history image encoding.
	imagesSupported bool

	// connSems caps concurrent LLM calls per configured [[llm]] entry —
	// settings.LLM[i] has a buffered channel at connSems[i] of capacity
	// LLM[i].parallelCap(). llmStream acquires on entry and releases on exit,
	// so a busy conn naturally queues excess calls instead of over-dispatching
	// to its server. Sized by buildConnSems on startup and after any settings
	// reload. nil entry → no semaphore (test mocks).
	connSems []chan struct{}

	// mcp owns the MCP server children and the bookkeeping reconcileMCP
	// needs; its mutex guards the whole group (see mcpState).
	mcp mcpState

	// abortReason is set by the bootstrap goroutine when codehalter must not
	// run in this environment (today: started outside a devcontainer). Empty
	// means proceed; non-empty causes Prompt to refuse with this message.
	// Read under mu.
	abortReason string

	// subagentMeter folds the live status meter of each running subagent into
	// its parent's in-progress phase row. Subagent sessions have no UI of their
	// own (Zed only knows the parent sid), so their "↑sent ↓tokens… Ns" meter
	// would otherwise be dropped. Keyed parentSid -> subSid -> latest meter
	// fragment; setSubagentStatus re-renders the parent row as a compact join of
	// every active subagent. Guarded by subagentMeterMu (concurrent subagents).
	subagentMeterMu sync.Mutex
	subagentMeter   map[string]map[string]string
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
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

func main() {
	// Global slog → stderr at debug (Zed captures it live); per-session detail
	// (LLM req/reply, errors) goes to .codehalter/session_<id>.log via logSession.
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))
	dumpGoroutinesOnSignal()

	a := &agent{sessions: make(map[string]*Session), mode: "Interactive"}
	conn := NewAgentSideConnection(a, os.Stdout, os.Stdin)
	a.conn = conn

	slog.Info("waiting for connection")
	<-conn.Done()
	slog.Info("connection closed")
}

// ---------------------------------------------------------------------------
// ACP protocol handlers
// ---------------------------------------------------------------------------

func (a *agent) Initialize(ctx context.Context, req InitializeRequest) (InitializeResponse, error) {
	if req.ProtocolVersion != protocolVersion {
		return InitializeResponse{}, fmt.Errorf("unsupported protocol version %d (this agent speaks %d)", req.ProtocolVersion, protocolVersion)
	}
	// Probe the execute/thinking LLM cheaply (metadata endpoints, no
	// inference) to advertise image support in capabilities. We load the
	// global settings only here — project-local settings live under a cwd
	// we do not yet have. If project-local settings override the LLM later,
	// the first Prompt's prepare phase re-probes and updates the flag plus
	// the LLM banner.
	if gs, err := loadGlobalSettings(); err == nil {
		a.settings = gs
		a.buildConnSems()
		if conn := a.settings.MainLLM("execute"); conn != nil {
			a.imagesSupported = a.probeLLM(ctx, conn).ImageSupport
		}
	}
	var res InitializeResponse
	res.ProtocolVersion = protocolVersion
	res.AgentCapabilities.LoadSession = true
	res.AgentCapabilities.PromptCapabilities.Image = a.imagesSupported
	res.AgentCapabilities.SessionCapabilities = &struct {
		List  *struct{} `json:"list,omitempty"`
		Close *struct{} `json:"close,omitempty"`
	}{List: &struct{}{}, Close: &struct{}{}}
	// Static implementation block advertised in the initialize response —
	// name and version don't change at runtime.
	res.AgentInfo = struct {
		Name    string `json:"name,omitempty"`
		Version string `json:"version,omitempty"`
	}{"codehalter", "0.1.0"}
	res.AuthMethods = []string{}
	return res, nil
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
		a.sendUpdate(ctx, req.SessionId, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("Started a new session: the workspace this thread was created in (%s) isn't available here, so there was nothing to restore.\n\n", req.Cwd)}})
	}
	s, err := loadSession(cwd, req.SessionId)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Debug("LoadSession: not found, treating as new", "sid", req.SessionId)
			// Zed cached an ID from an earlier session/new that never
			// wrote a file (no prompt). It then sends session/load with
			// that ID, and subsequently session/prompt under the same
			// ID — LoadSessionResponse.sessionId is NOT honored by Zed,
			// so we must accept the cached id as-is or prompts won't
			// route. The filename inherits the cached id's stale
			// timestamp, which is a known cosmetic wart.
			s = newSessionWithID(cwd, req.SessionId)
			if err := a.initSession(cwd, s); err != nil {
				a.deleteSession(s.ID)
				return LoadSessionResponse{}, err
			}
			a.startIndexing(s.ID, cwd)
			return LoadSessionResponse{Modes: a.sessionModes()}, nil
		}
		return LoadSessionResponse{}, fmt.Errorf("loading session: %w", err)
	}
	if err := a.initSession(cwd, s); err != nil {
		a.deleteSession(s.ID)
		return LoadSessionResponse{}, err
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
				a.sendUpdate(ctx, req.SessionId, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: ""}})
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
			a.sendUpdate(ctx, req.SessionId, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: m.Content}})
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
	// Field is "modeId" per ACP spec — distinct from "currentModeId" in SessionModeState.
	a.sendUpdate(ctx, req.SessionId, struct {
		Kind   string `json:"sessionUpdate"`
		ModeId string `json:"modeId"`
	}{Kind: "current_mode_update", ModeId: req.ModeId})
	a.sendUpdate(ctx, req.SessionId, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Mode: " + req.ModeId + "\n\n"}})
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
	defer a.mu.Unlock()
	delete(a.sessions, id)
}

// ---------------------------------------------------------------------------
// Session bootstrap
// ---------------------------------------------------------------------------

func (a *agent) initSession(cwd string, s *Session) error {
	a.putSession(s)

	// Seed .codehalter/ defaults when absent. Phase prompts
	// (PLAN/EXECUTE/DOCUMENT/SUMMARISE) are user-owned templates seeded once;
	// every SKILL-*.md (including the always-on container skill) is owned by
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
		{"COMMIT.md", defaultCommitMD},
	} {
		path := filepath.Join(dir, f.name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			if err := os.WriteFile(path, []byte(f.content), 0o644); err != nil {
				return fmt.Errorf("seeding %s: %w", path, err)
			}
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
	// mcp.toml — only seeded on first run with the bare placeholder. Per-stack
	// MCP wiring (e.g. gopls for Go) is the prepare phase's job: it asks the
	// user before installing tools, and the same flow appends the matching
	// [[server]] entry to this file. Once it exists we never touch it again —
	// the user owns it.
	mcpPath := filepath.Join(dir, "mcp.toml")
	if _, err := os.Stat(mcpPath); os.IsNotExist(err) {
		if err := os.WriteFile(mcpPath, []byte(defaultMCPToml), 0o644); err != nil {
			return fmt.Errorf("seeding %s: %w", mcpPath, err)
		}
	}

	// Always rebuild SystemPrompt from the freshly-seeded .codehalter/
	// directory. NewSession starts with SystemPrompt == "" so this is the
	// first-and-only build; LoadSession has a possibly-stale SystemPrompt
	// from a prior run on a different host (different OS skill set), and
	// we must overwrite it BEFORE prepare's proposeFix can dispatch an
	// LLM call carrying the stale prefix.
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
	a.settings = settings
	a.buildConnSems()
	a.discoverRunners(cwd)
	a.discoverSandbox()
	a.registerSubagentTool()
	return nil
}

// startIndexing runs the once-per-session bootstrap in a goroutine: the
// interactive devcontainer/gitignore prompts plus the first prepare(), so the
// capabilities banner shows at session open rather than only after the first
// turn (prepare also re-runs every turn from Prompt). Devcontainer goes first
// because the gitignore prompt assumes a sandbox; if ensureDevcontainer fails
// it sets abortReason and the rest is skipped, so Prompt then refuses every turn.
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

		// Brief pause before the first user-visible session/update. Zed
		// registers the session (builds its AcpThread, inserts it into the
		// session map) asynchronously AFTER it reads our session/new response;
		// an update that lands inside that window is dropped as "Received
		// session notification for unknown session" — which is exactly why the
		// devcontainer notice never shows until the first prompt. We already
		// write the response before the update, so this is purely Zed-side
		// registration latency. 100ms lets registration win the race.
		// Experimental: testing whether session-open notices then render.
		select {
		case <-ctx.Done():
			return
		case <-time.After(100 * time.Millisecond):
		}

		if !a.ensureDevcontainer(ctx, cwd, sid) {
			slog.Debug("startIndexing: ensureDevcontainer false, aborting", "sid", sid)
			return
		}
		slog.Debug("startIndexing: devcontainer ok, about to ensureGitignore", "sid", sid)
		a.ensureGitignore(ctx, cwd, sid)

		sess := a.getSession(sid)
		if sess != nil {
			slog.Debug("startIndexing: gitignore done, about to prepare", "sid", sid)
			a.prepare(ctx, sess, sid)
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
			{Id: "Interactive", Name: "Interactive", Description: "Ask the user before non-trivial actions"},
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

// ---------------------------------------------------------------------------
// Agent → client output
// ---------------------------------------------------------------------------

func (a *agent) sendUpdate(ctx context.Context, sid string, u any) {
	if a.conn == nil {
		return
	}
	// Zed only knows the parent session id; SessionUpdate for a "sub_*"
	// sid produces a "Received session notification for unknown session"
	// warning on every emit. Subagent activity is intentionally not
	// surfaced to the parent UI (only the final tool result is), so drop
	// the notification entirely here. The per-session TOML log still
	// captures the subagent's full message history.
	if strings.HasPrefix(sid, "sub_") {
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

// sendUpdateAndAbort marks the session as do-not-run and emits the reason to
// chat. Prompt reads a.abortReason under mu and fails every turn until the
// process is restarted (inside a container).
func (a *agent) sendUpdateAndAbort(ctx context.Context, sid, reason string) {
	a.mu.Lock()
	a.abortReason = reason
	a.mu.Unlock()
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: reason + "\n"}})
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

// logSession appends a tagged, timestamped block to the per-session debug log
// at .codehalter/session_<sid>.log, opening and closing the file per call (the
// log is strictly diagnostic and not time-critical, so a long-lived handle
// isn't worth it). No-op when sid is empty/unknown or the file can't be opened.
// The body is written verbatim — caller decides whether to truncate. Use a
// short tag like "WEB" or "TOOL" so the log stays grep-friendly.
func (a *agent) logSession(sid string, tag, format string, args ...any) {
	if sid == "" {
		return
	}
	sess := a.getSession(sid)
	if sess == nil {
		return
	}
	path := filepath.Join(sess.Cwd, sessionDir, fmt.Sprintf("session_%s.log", sid))
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
