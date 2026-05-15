package main

import (
	"context"
	_ "embed"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

//go:embed docs/PLAN.md.example
var defaultPlanMD string

//go:embed docs/EXECUTE.md.example
var defaultExecuteMD string

//go:embed docs/VERIFY.md.example
var defaultVerifyMD string

//go:embed docs/DOCUMENT.md.example
var defaultDocumentMD string

//go:embed docs/SKILL-buildfile.md.example
var skillBuildfile string

//go:embed docs/SKILL-go.md.example
var skillGo string

//go:embed docs/SKILL-ts.md.example
var skillTS string

//go:embed docs/SKILL-js.md.example
var skillJS string

//go:embed docs/SKILL-java.md.example
var skillJava string

//go:embed docs/SKILL-bash.md.example
var skillBash string

//go:embed docs/Dockerfile.devcontainer.example
var defaultDevcontainerDockerfile string

//go:embed docs/devcontainer.json.example
var defaultDevcontainerJSON string

//go:embed docs/settings.toml.example
var defaultSettingsTOML string

var defaultSkills = map[string]string{
	"go":   skillGo,
	"ts":   skillTS,
	"js":   skillJS,
	"java": skillJava,
	"bash": skillBash,
}

// ensureDefaults copies embedded default files into .codehalter/ if they don't exist.
// Phase prompts (PLAN/EXECUTE/VERIFY) are always seeded; SKILL files are seeded
// only for stacks detected in cwd, so polyglot projects get the relevant ones
// and single-language projects don't accumulate noise.
func ensureDefaults(cwd string) {
	dir := filepath.Join(cwd, ".codehalter")
	os.MkdirAll(dir, 0o755)
	for _, f := range []struct{ name, content string }{
		{"PLAN.md", defaultPlanMD},
		{"EXECUTE.md", defaultExecuteMD},
		{"VERIFY.md", defaultVerifyMD},
		{"DOCUMENT.md", defaultDocumentMD},
		{"SKILL-buildfile.md", skillBuildfile},
	} {
		path := filepath.Join(dir, f.name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			os.WriteFile(path, []byte(f.content), 0o644)
		}
	}
	for _, stack := range detectStacks(cwd) {
		body, ok := defaultSkills[stack]
		if !ok {
			continue
		}
		path := filepath.Join(dir, "SKILL-"+stack+".md")
		if _, err := os.Stat(path); os.IsNotExist(err) {
			os.WriteFile(path, []byte(body), 0o644)
		}
	}
}

// agent implements acp.Agent.
type agent struct {
	conn            *AgentSideConnection
	mu              sync.Mutex
	cancel          context.CancelFunc
	sessions        map[SessionId]*Session
	settings        Settings
	runners         []taskRunner
	capabilities    capabilities
	emptyProject    bool // true on first session if cwd had no source/manifest files
	indexDone       chan struct{}
	imagesSupported bool
	mode            string // "interactive" | "autopilot"

	// connReachable records whether each configured LLMConnection answered
	// the startup probe. Keyed by connKey(URL+model). pickAvailable filters
	// candidates against this so a dead server doesn't burn a 2s /slots
	// timeout on every LLM call. Populated by checkLLM; nil before that.
	connReachable map[string]bool

	// gopls is a lazily-started LSP client shared across tool calls so the
	// workspace index isn't rebuilt on every query. ensureGopls handles the
	// once-per-process startup; goplsErr records a fatal start failure so we
	// don't retry on every call.
	gopls     *Gopls
	goplsOnce sync.Once
	goplsErr  error
}

// connKey returns a stable map key for a connection.
func connKey(c *LLMConnection) string { return c.URL + "\x00" + c.Model }

const (
	modeInteractive = "interactive"
	modeAutopilot   = "autopilot"
)

// sessionModes is the mode state advertised to the client on session
// create/load. The client uses this to render the mode selector.
func (a *agent) sessionModes() *SessionModeState {
	current := a.mode
	if current == "" {
		current = modeInteractive
	}
	return &SessionModeState{
		CurrentModeId: current,
		AvailableModes: []SessionMode{
			{Id: modeInteractive, Name: "Interactive", Description: "Ask the user before non-trivial actions"},
			{Id: modeAutopilot, Name: "Autopilot", Description: "Auto-answer prompts — no user interruption"},
		},
	}
}

// isAutopilot reports whether questions should be auto-answered.
func (a *agent) isAutopilot() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.mode == modeAutopilot
}

var _ Agent = (*agent)(nil)

func cwdOrDefault(cwd string) string {
	if cwd != "" {
		return cwd
	}
	d, _ := os.Getwd()
	return d
}

func (a *agent) getSession(id SessionId) *Session {
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

func (a *agent) deleteSession(id SessionId) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.sessions, id)
}

// llmTier returns the connection tier for the given session: "subagent" if
// the session's depth is non-zero (it was spawned by another session), else
// "main". Empty sid (pre-session probes, tests) is treated as main.
func (a *agent) llmTier(sid SessionId) string {
	if sid == "" {
		return "main"
	}
	if sess := a.getSession(sid); sess != nil && sess.Depth > 0 {
		return "subagent"
	}
	return "main"
}

func (a *agent) sendUpdate(ctx context.Context, sid SessionId, u SessionUpdate) {
	if a.conn == nil {
		return
	}
	_ = a.conn.SessionUpdate(ctx, SessionNotification{SessionId: sid, Update: u})
}

func (a *agent) Initialize(ctx context.Context, req InitializeRequest) (InitializeResponse, error) {
	// Probe the execute/thinking LLM cheaply (metadata endpoints, no
	// inference) to advertise image support in capabilities. We load the
	// global settings only here — project-local settings live under a cwd
	// we do not yet have. If project-local settings override the LLM later,
	// startIndexing → checkLLM re-probes and updates the flag plus the
	// startup banner.
	if gs, err := loadGlobalSettings(); err == nil {
		a.settings = gs
		if conn := a.settings.LLMFor("execute", "main"); conn != nil {
			a.imagesSupported = a.probeLLM(ctx, conn).ImageSupport
		}
	}
	return InitializeResponse{
		ProtocolVersion: ProtocolVersionNumber,
		AgentCapabilities: AgentCapabilities{
			LoadSession:        true,
			PromptCapabilities: PromptCapabilities{Image: a.imagesSupported},
			SessionCapabilities: &SessionCapabilities{
				List: &SessionListCapabilities{},
			},
		},
		AgentInfo: &Implementation{
			Name:    "llama-acp",
			Version: "0.1.0",
		},
		AuthMethods: []string{},
	}, nil
}

func (a *agent) Authenticate(_ context.Context, _ AuthenticateRequest) (AuthenticateResponse, error) {
	// No auth needed for local llama.cpp.
	return AuthenticateResponse{}, nil
}

func (a *agent) initSession(cwd string, s *Session) error {
	a.putSession(s)
	ensureDefaults(cwd)
	settings, err := loadSettings(cwd)
	if err != nil {
		return err
	}
	// Empty settings are tolerated (path == "") — startIndexing prompts the
	// user to write a skeleton on first run; running with no LLM until then
	// is graceful (checkLLM prints a warning instead of crashing).
	a.settings = settings
	a.discoverRunners(cwd)
	a.registerSubagentTool()
	return nil
}

func (a *agent) startIndexing(sid SessionId, cwd string) {
	a.indexDone = make(chan struct{})
	go func() {
		defer close(a.indexDone)
		ctx := context.Background()

		// settings prompt before any LLM probe — if the user creates a
		// skeleton here, checkLLM picks up the new path on the same startup.
		a.ensureSettings(ctx, cwd, sid)

		if a.settings.path != "" {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Using "+a.settings.path+"\n\n")))
		}

		// Order: config → project → environment → prompts. Keeping the
		// container/firefox banner adjacent to the devcontainer prompt avoids
		// "container, tooling, container again" interleaving.
		a.checkLLM(ctx, sid)
		a.notifyCapabilities(ctx, sid)
		a.checkEnvironment(ctx, sid)

		a.ensureGitignore(ctx, cwd, sid)
		a.ensureDevcontainer(ctx, cwd, sid)
	}()
}

// notifyCapabilities emits a summary of discovered build/test/lint/format
// runners at session start, flagging any category where nothing was found so
// the user knows to either configure their runner or accept the gap.
func (a *agent) notifyCapabilities(ctx context.Context, sid SessionId) {
	a.mu.Lock()
	caps := a.capabilities
	empty := a.emptyProject
	a.mu.Unlock()

	if empty {
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
			"Empty project — I'll ask about language and runner on your first message.\n\n")))
		return
	}
	if len(caps.runners) == 0 {
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
			"🟡 No task runner detected (just, make, npm, go, cargo). Add one so I can build/test/lint.\n\n")))
		return
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Project tooling (%s):\n", strings.Join(caps.runners, ", "))
	row := func(label string, entries []string, hint string) {
		if len(entries) > 0 {
			fmt.Fprintf(&b, "  %-7s %s\n", label+":", strings.Join(entries, ", "))
		} else {
			fmt.Fprintf(&b, "  %-7s (none — %s)\n", label+":", hint)
		}
	}
	row("build", caps.build, "consider adding a `build` target")
	row("test", caps.test, "consider adding a `test` target")
	row("lint", caps.lint, "consider adding a `lint`/`vet`/`check` target")
	row("format", caps.format, "consider adding a `fmt`/`format` target")
	b.WriteString("\n")
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(b.String())))
}

// checkLLM probes every configured LLMConnection in parallel, records which
// ones answered, and reports each line to the user. Runtime selection
// (pickAvailable) then skips the unreachable ones instead of eating a /slots
// timeout on every call. Image support is taken from the first reachable
// connection — that's the one execute/main calls land on by default.
func (a *agent) checkLLM(ctx context.Context, sid SessionId) {
	if len(a.settings.LLMConnections) == 0 {
		a.imagesSupported = false
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("🟡 LLM: no [[llmconnections]] in settings.toml — codehalter cannot run until you add one.\n\n")))
		return
	}
	if settingsLooksPlaceholder(a.settings) {
		a.imagesSupported = false
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
			"🟡 LLM: "+a.settings.path+" still has the placeholder model \"your-model-id\". Edit it with your real url and model, then restart this Zed session.\n\n")))
		return
	}

	results := make([]probeResult, len(a.settings.LLMConnections))
	parallel(len(a.settings.LLMConnections), func(i int) {
		c := a.settings.LLMConnections[i]
		results[i] = a.probeLLM(ctx, &c)
	})

	a.connReachable = make(map[string]bool, len(a.settings.LLMConnections))
	var b strings.Builder
	firstReachable := -1
	// Each LLM gets its own paragraph (\n\n) — markdown collapses single
	// newlines to spaces, so "LLM[0]: ...\nLLM[1]: ..." would render on one
	// wrapped line and obscure that there are two separate connections.
	for i, r := range results {
		c := a.settings.LLMConnections[i]
		a.connReachable[connKey(&c)] = r.Reachable
		switch {
		case !r.Reachable:
			fmt.Fprintf(&b, "🟡 LLM[%d]: unreachable at %s — start your server or fix the url.\n\n", i, c.URL)
		case r.ModelKnown && !r.ModelLoaded:
			fmt.Fprintf(&b, "🟡 LLM[%d]: %s reachable but model %q not loaded.\n\n", i, c.URL, c.Model)
		default:
			fmt.Fprintf(&b, "✅ LLM[%d]: %s @ %s\n\n", i, c.Model, c.URL)
		}
		if r.Reachable && firstReachable < 0 {
			firstReachable = i
		}
	}

	if firstReachable < 0 {
		a.imagesSupported = false
		b.WriteString("🟡 No LLM reachable — every connection above failed. Codehalter cannot run any prompt until at least one comes back.\n\n")
	} else {
		a.imagesSupported = results[firstReachable].ImageSupport
		if a.imagesSupported {
			b.WriteString("✅ Image support: enabled\n\n")
		} else {
			b.WriteString("Image support: disabled\n\n")
		}
	}
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(b.String())))
}

// checkEnvironment reports whether codehalter is running inside a container
// and whether Firefox (required for web_search/web_read) is available, so
// devcontainer users see at startup whether the toolchain is wired up.
func (a *agent) checkEnvironment(ctx context.Context, sid SessionId) {
	var b strings.Builder
	if kind := containerKind(); kind != "" {
		fmt.Fprintf(&b, "✅ Container: %s\n\n", kind)
	} else {
		b.WriteString("🟡 Container: none (running on host — file edits and tasks hit your real filesystem)\n\n")
	}
	if _, err := findFirefox(); err == nil {
		b.WriteString("✅ Firefox: found (web_search/web_read enabled)\n\n")
	} else {
		b.WriteString("🟡 Firefox: not found — web_search/web_read disabled. Install firefox or set FIREFOX_PATH.\n\n")
	}
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(b.String())))
}

// containerKind returns a short label identifying the container runtime, or
// "" if we appear to be on the host. Cheap file/env probes only — no shelling
// out — since this runs on the indexing goroutine at session start.
func containerKind() string {
	if os.Getenv("REMOTE_CONTAINERS") == "true" || os.Getenv("DEVCONTAINER") == "true" {
		return "devcontainer"
	}
	if _, err := os.Stat("/.dockerenv"); err == nil {
		return "docker"
	}
	if _, err := os.Stat("/run/.containerenv"); err == nil {
		return "podman"
	}
	if v := os.Getenv("container"); v != "" {
		return v
	}
	return ""
}

func (a *agent) waitForIndex() {
	if a.indexDone != nil {
		<-a.indexDone
	}
}

func (a *agent) NewSession(_ context.Context, req NewSessionRequest) (NewSessionResponse, error) {
	cwd := cwdOrDefault(req.Cwd)
	s, err := newSession(cwd)
	if err != nil {
		return NewSessionResponse{}, err
	}
	if err := a.initSession(cwd, s); err != nil {
		a.deleteSession(s.ID)
		return NewSessionResponse{}, err
	}
	a.startIndexing(s.ID, cwd)
	return NewSessionResponse{SessionId: s.ID, Modes: a.sessionModes()}, nil
}

func (a *agent) LoadSession(ctx context.Context, req LoadSessionRequest) (LoadSessionResponse, error) {
	cwd := cwdOrDefault(req.Cwd)
	s, err := loadSession(cwd, req.SessionId)
	if err != nil {
		if os.IsNotExist(err) {
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
	a.replayHistory(ctx, req.SessionId, s)
	a.startIndexing(s.ID, cwd)
	return LoadSessionResponse{Modes: a.sessionModes()}, nil
}

func (a *agent) replayHistory(ctx context.Context, sid SessionId, s *Session) {
	lastRole := ""
	for _, m := range s.Messages {
		if m.Role == lastRole {
			if m.Role == "user" {
				a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("")))
			} else {
				a.sendUpdate(ctx, sid, UserMessageChunk(TextBlock("")))
			}
		}
		if m.Role == "user" {
			a.sendUpdate(ctx, sid, UserMessageChunk(TextBlock(m.Content)))
			for _, img := range m.Images {
				a.sendUpdate(ctx, sid, UserMessageChunk(ImageBlock(img.MimeType, img.Data)))
			}
		} else {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(m.Content)))
		}
		lastRole = m.Role
	}
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

func (a *agent) SetSessionConfigOption(_ context.Context, req SetSessionConfigOptionRequest) (SetSessionConfigOptionResponse, error) {
	return SetSessionConfigOptionResponse{ConfigOptions: []SessionConfigOption{}}, nil
}

func (a *agent) SetSessionMode(ctx context.Context, req SetSessionModeRequest) (SetSessionModeResponse, error) {
	if req.ModeId != modeInteractive && req.ModeId != modeAutopilot {
		return SetSessionModeResponse{}, nil
	}
	a.mu.Lock()
	a.mode = req.ModeId
	a.mu.Unlock()
	a.sendUpdate(ctx, req.SessionId, CurrentModeUpdate(req.ModeId))
	a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock("Mode: "+req.ModeId+"\n\n")))
	return SetSessionModeResponse{}, nil
}

func (a *agent) SetSessionModel(_ context.Context, req SetSessionModelRequest) (SetSessionModelResponse, error) {
	return SetSessionModelResponse{}, nil
}

func (a *agent) Cancel(_ context.Context, _ CancelNotification) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.cancel != nil {
		a.cancel()
	}
}

func (a *agent) systemPrompt(sid SessionId) (string, error) {
	sess := a.getSession(sid)
	if sess == nil {
		return "", fmt.Errorf("no session found")
	}

	var b strings.Builder
	if skills := loadSkills(sess.Cwd); skills != "" {
		b.WriteString(skills)
	}
	b.WriteString("Project directory: " + sess.Cwd + "\n")
	// Local/older models often have training data ending well before today —
	// without this they reason about "future" releases as unreleased and give
	// stale conclusions. Captured at session start; resumed sessions inherit
	// that day's date, which is good enough for typical use.
	b.WriteString("Today's date: " + time.Now().Format("2006-01-02") + "\n")
	b.WriteString("When the user asks about a technology, language, library, or version X, " +
		"first check how X and its adjacent tooling are configured in this project " +
		"(e.g. go → also tinygo; npm → also pnpm/yarn; python → also uv/poetry/pypy; " +
		"a framework → also the build/runtime variant) before searching the web or " +
		"making assumptions. Read manifest/build files (go.mod, package.json, Cargo.toml, " +
		"Makefile, justfile, Dockerfile, README) and grep for related terms — the right " +
		"answer often depends on which variant the project actually uses.\n")

	return b.String(), nil
}

// sessionLog opens (append-only) the per-session debug log at
// .codehalter/session_<sid>.log. Returns nil if sid is empty, the session is
// unknown, or the file can't be opened — callers must nil-check. The log is
// strictly diagnostic: codehalter never reads it back.
func (a *agent) sessionLog(sid SessionId) *os.File {
	if sid == "" {
		return nil
	}
	sess := a.getSession(sid)
	if sess == nil {
		return nil
	}
	path := filepath.Join(sess.Cwd, sessionDir, fmt.Sprintf("session_%s.log", sid))
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil
	}
	return f
}

// logSession writes a tagged, timestamped block to the per-session debug log.
// No-op when sid is unknown or the log can't be opened. The body is written
// verbatim — caller decides whether to truncate. Use a short tag like "WEB"
// or "TOOL" so the log stays grep-friendly.
func (a *agent) logSession(sid SessionId, tag, format string, args ...any) {
	logF := a.sessionLog(sid)
	if logF == nil {
		return
	}
	defer logF.Close()
	fmt.Fprintf(logF, "\n=== %s [%s] ===\n", time.Now().Format(time.RFC3339), tag)
	fmt.Fprintf(logF, format, args...)
	if !strings.HasSuffix(format, "\n") {
		fmt.Fprintln(logF)
	}
}

func (a *agent) loadPromptFile(sid SessionId, filename string) string {
	sess := a.getSession(sid)
	if sess != nil {
		if data, err := os.ReadFile(filepath.Join(sess.Cwd, ".codehalter", filename)); err == nil {
			return string(data)
		}
	}
	return ""
}

func truncate(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}

func main() {
	// Per-session debug logs (full LLM req/reply, errors) live alongside
	// each session's TOML at .codehalter/session_<id>.log; see sessionLog().
	// Global slog goes to stderr only — Zed captures it for live debugging,
	// and reproducing a session bug from a fresh shell is the session log's
	// job, not a global file's.
	stderrHandler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})
	slog.SetDefault(slog.New(stderrHandler))
	log := slog.New(stderrHandler)

	a := &agent{sessions: make(map[SessionId]*Session), mode: modeInteractive}
	conn := NewAgentSideConnection(a, os.Stdout, os.Stdin, log)
	a.conn = conn

	log.Info("waiting for connection")
	<-conn.Done()
	log.Info("connection closed")
}
