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
	probedConnKey   string
	mode            string // "interactive" | "autopilot"
}

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
	// Detect image support by probing the execute/thinking LLM with a tiny
	// image. We load the global settings only here — project-local settings
	// live under a cwd we do not yet have. If project-local settings override
	// the LLM later, startIndexing will re-probe and update the flag.
	if gs, err := loadGlobalSettings(); err == nil {
		a.settings = gs
		if conn := a.settings.LLMFor("execute", "main"); conn != nil {
			a.imagesSupported = a.probeImageSupport(ctx, conn)
			a.probedConnKey = connKey(conn)
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

// connKey identifies an LLM endpoint by url+model so we can tell when
// project-local settings point at a different model than we probed.
func connKey(c *LLMConnection) string {
	if c == nil {
		return ""
	}
	return c.URL + "|" + c.Model
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
	if err := settings.Validate(); err != nil {
		return err
	}
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

		if a.settings.path != "" {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Using "+a.settings.path+"\n\n")))
		} else {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ No settings.toml found\n\n")))
		}

		a.checkImageSupport(ctx, sid)
		a.checkEnvironment(ctx, sid)
		a.notifyCapabilities(ctx, sid)

		a.ensureGitignore(ctx, cwd, sid)
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
			"⚠ No task runner detected (just, make, npm, go, cargo). Add one so I can build/test/lint.\n\n")))
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

// checkImageSupport re-probes the active LLM if project-local settings point
// at a different model than the one probed in Initialize, then tells the user
// whether image input is available. Runs on the indexing goroutine so it never
// blocks the session handshake.
func (a *agent) checkImageSupport(ctx context.Context, sid SessionId) {
	conn := a.settings.LLMFor("execute", "main")
	if conn == nil {
		a.imagesSupported = false
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ Image support: disabled (no LLM configured)\n\n")))
		return
	}
	if key := connKey(conn); key != a.probedConnKey {
		a.imagesSupported = a.probeImageSupport(ctx, conn)
		a.probedConnKey = key
	}
	if a.imagesSupported {
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Image support: enabled ("+conn.Model+")\n\n")))
	} else {
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Image support: disabled ("+conn.Model+" did not accept image input)\n\n")))
	}
}

// checkEnvironment reports whether codehalter is running inside a container
// and whether Firefox (required for web_search/web_read) is available, so
// devcontainer users see at startup whether the toolchain is wired up.
func (a *agent) checkEnvironment(ctx context.Context, sid SessionId) {
	var b strings.Builder
	kind := containerKind()
	if kind != "" {
		fmt.Fprintf(&b, "Container: %s\n", kind)
	} else {
		b.WriteString("⚠ Container: not detected — codehalter edits files and runs tasks directly on your host. Consider using a devcontainer (see README → Sandboxing with a devcontainer).\n")
	}
	if path, err := findFirefox(); err == nil {
		fmt.Fprintf(&b, "Firefox: %s\n\n", path)
	} else {
		b.WriteString("⚠ Firefox: not found — web_search/web_read disabled. Install firefox or set FIREFOX_PATH.\n\n")
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
