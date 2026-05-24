package main

import (
	"context"
	_ "embed"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

//go:embed docs/PLAN.md
var defaultPlanMD string

//go:embed docs/EXECUTE.md
var defaultExecuteMD string

//go:embed docs/VERIFY.md
var defaultVerifyMD string

//go:embed docs/DOCUMENT.md
var defaultDocumentMD string

//go:embed docs/SKILL-go.md
var skillGo string

//go:embed docs/SKILL-ts.md
var skillTS string

//go:embed docs/SKILL-js.md
var skillJS string

//go:embed docs/SKILL-java.md
var skillJava string

//go:embed docs/SKILL-bash.md
var skillBash string

//go:embed docs/SKILL-container.md
var skillContainer string

//go:embed docs/SKILL-makefile.md
var skillMakefile string

//go:embed docs/SKILL-justfile.md
var skillJustfile string

//go:embed docs/SKILL-alpine.md
var skillAlpine string

//go:embed docs/SKILL-arch.md
var skillArch string

//go:embed docs/SKILL-debian.md
var skillDebian string

//go:embed docs/SKILL-fedora.md
var skillFedora string

//go:embed docs/SKILL-ubuntu.md
var skillUbuntu string

//go:embed docs/Dockerfile.devcontainer.alpine
var defaultDevcontainerDockerfileAlpine string

//go:embed docs/Dockerfile.devcontainer.arch
var defaultDevcontainerDockerfileArch string

//go:embed docs/Dockerfile.devcontainer.debian
var defaultDevcontainerDockerfileDebian string

//go:embed docs/Dockerfile.devcontainer.fedora
var defaultDevcontainerDockerfileFedora string

//go:embed docs/Dockerfile.devcontainer.ubuntu
var defaultDevcontainerDockerfileUbuntu string

//go:embed docs/devcontainer.json
var defaultDevcontainerJSON string

//go:embed docs/settings.toml
var defaultSettingsTOML string

//go:embed docs/mcp.toml
var defaultMCPToml string

// defaultSkills maps a per-stack key (language stack from detectStacks)
// to the embedded skill body. The container / per-OS / per-runner skills
// live in their own seed paths inside ensureSkills, not here, because
// they're not driven by detectStacks.
var defaultSkills = map[string]string{
	"go":   skillGo,
	"ts":   skillTS,
	"js":   skillJS,
	"java": skillJava,
	"bash": skillBash,
}

// osSkills maps an /etc/os-release ID (as returned by readOSInfo) to the
// embedded skill body. Only IDs we have a SKILL-<id>.md for are present;
// readOSInfo filters everything else to "" before lookup.
var osSkills = map[string]string{
	"alpine": skillAlpine,
	"arch":   skillArch,
	"debian": skillDebian,
	"fedora": skillFedora,
	"ubuntu": skillUbuntu,
}

// ensureDefaults copies embedded default files into .codehalter/ if they don't exist.
// Phase prompts (PLAN/EXECUTE/VERIFY/DOCUMENT) and the always-on container
// skill are seeded here. Per-stack / per-runner / per-OS SKILL files are
// handled by ensureSkills on every prepare turn so a stack or runner added
// mid-session takes effect on the next prompt.
func ensureDefaults(cwd string) {
	dir := filepath.Join(cwd, ".codehalter")
	os.MkdirAll(dir, 0o755)
	for _, f := range []struct{ name, content string }{
		{"PLAN.md", defaultPlanMD},
		{"EXECUTE.md", defaultExecuteMD},
		{"VERIFY.md", defaultVerifyMD},
		{"DOCUMENT.md", defaultDocumentMD},
		{"SKILL-container.md", skillContainer},
	} {
		path := filepath.Join(dir, f.name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			os.WriteFile(path, []byte(f.content), 0o644)
		}
	}
	ensureSkills(cwd, detectStacks(cwd), readOSInfo())

	// mcp.toml — only seeded on first run with the bare placeholder. Per-stack
	// MCP wiring (e.g. gopls for Go) is the prepare phase's job: it asks the
	// user before installing tools, and the same flow appends the matching
	// [[server]] entry to this file. Once it exists we never touch it again —
	// the user owns it.
	mcpPath := filepath.Join(dir, "mcp.toml")
	if _, err := os.Stat(mcpPath); os.IsNotExist(err) {
		os.WriteFile(mcpPath, []byte(defaultMCPToml), 0o644)
	}
}

// ensureSkills seeds SKILL-*.md files into .codehalter/ based on what is
// PRESENT in the project tree (justfile / Makefile / language stacks) and
// in the container (/etc/os-release ID), NOT on what tooling is installed
// on PATH. This is load-bearing: the LLM needs SKILL-justfile.md loaded
// BEFORE it installs `just` for the user, otherwise the fix-dispatch turn
// has no idea what justfile syntax looks like.
//
// Per-stack and per-runner seeding is idempotent (writes only when the
// file is missing, so user edits survive). Per-OS handling is stricter:
// every SKILL-<other-os>.md is deleted unconditionally — codehalter
// supports exactly one OS per session, and a stale skill from a prior
// run on a different host would otherwise keep getting concatenated into
// every system prompt. The matching SKILL-<osid>.md is always (re)written
// from the embed with {{KEY}} placeholders substituted from osi.Fields,
// because the resolved version values change whenever the container is
// rebuilt and the LLM needs the current ones to skip the os-release probe.
// User edits to per-OS skills are discarded; the returned []string lists
// what other-OS files were deleted so the caller can rebuild
// sess.SystemPrompt from the cleaned-up directory before the next LLM
// call dispatches.
func ensureSkills(cwd string, stacks []string, osi osInfo) []string {
	dir := filepath.Join(cwd, ".codehalter")
	seed := func(name, body string) {
		if body == "" {
			return
		}
		path := filepath.Join(dir, name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			os.WriteFile(path, []byte(body), 0o644)
		}
	}
	exists := func(names ...string) bool {
		for _, n := range names {
			if _, err := os.Stat(filepath.Join(cwd, n)); err == nil {
				return true
			}
		}
		return false
	}
	for _, stack := range stacks {
		if body, ok := defaultSkills[stack]; ok {
			seed("SKILL-"+stack+".md", body)
		}
	}
	if exists("justfile", "Justfile", ".justfile") {
		seed("SKILL-justfile.md", skillJustfile)
	}
	if exists("Makefile", "makefile", "GNUmakefile") {
		seed("SKILL-makefile.md", skillMakefile)
	}
	var pruned []string
	if osi.ID == "" {
		return pruned
	}
	for other := range osSkills {
		if other == osi.ID {
			continue
		}
		path := filepath.Join(dir, "SKILL-"+other+".md")
		if _, err := os.Stat(path); err != nil {
			continue
		}
		if err := os.Remove(path); err == nil {
			pruned = append(pruned, "SKILL-"+other+".md")
		}
	}
	if body, ok := osSkills[osi.ID]; ok {
		rendered := renderOSSkill(body, osi.Fields)
		path := filepath.Join(dir, "SKILL-"+osi.ID+".md")
		if cur, err := os.ReadFile(path); err != nil || string(cur) != rendered {
			os.WriteFile(path, []byte(rendered), 0o644)
		}
	}
	return pruned
}

// renderOSSkill substitutes {{KEY}} placeholders in the per-OS skill body
// with the matching /etc/os-release field. Unknown placeholders are
// replaced with the empty string so the LLM doesn't see literal `{{X}}`
// when a field is missing on a non-standard image.
func renderOSSkill(body string, fields map[string]string) string {
	var b strings.Builder
	b.Grow(len(body))
	for {
		i := strings.Index(body, "{{")
		if i < 0 {
			b.WriteString(body)
			return b.String()
		}
		j := strings.Index(body[i+2:], "}}")
		if j < 0 {
			b.WriteString(body)
			return b.String()
		}
		b.WriteString(body[:i])
		key := body[i+2 : i+2+j]
		b.WriteString(fields[key])
		body = body[i+2+j+2:]
	}
}

// agent implements acp.Agent.
type agent struct {
	conn            *AgentSideConnection
	mu              sync.Mutex
	cancel          context.CancelFunc
	sessions        map[string]*Session
	settings        Settings
	runners         []taskRunner
	capabilities    capabilities
	emptyProject    bool // true on first session if cwd had no source/manifest files
	indexDone       chan struct{}
	imagesSupported bool
	mode            string // "interactive" | "autopilot"

	// connReachable records whether each configured LLMConnection answered
	// the prepare-phase probe. Keyed by connKey(URL+model). pickBackgroundLLM
	// filters candidates against this so a dead extra slot doesn't burn a
	// timeout on every background summarise call. Populated by probeAllLLMs;
	// nil before the first prepare runs.
	connReachable map[string]bool

	// mainSlotTokens is the per-slot context window for LLM[0] in tokens —
	// the server-reported n_ctx divided by parallelCap, since llama.cpp's
	// `-c N -np K` splits the total across K slots. Discovered by the startup
	// probe (/props.n_ctx or /v1/models --ctx-size). 0 means unknown (probe
	// failed or server didn't report it) — compaction then falls back to the
	// rawBufferTokens constant. compressHistory reads this; nothing else should.
	mainSlotTokens int

	// slotSems caps concurrent LLM calls per configured [[llm]] entry —
	// settings.LLM[i] has a buffered channel at slotSems[i] of capacity
	// LLM[i].parallelCap(). llmStream acquires on entry and releases on exit,
	// so a busy conn naturally queues excess calls instead of over-dispatching
	// to its server. Sized by buildSlotSems on startup and after any settings
	// reload. nil entry → no semaphore (test mocks).
	slotSems []chan struct{}

	// mcpClients holds the spawned MCP server children, keyed by their
	// configured `name`. Tools they advertise are registered into the
	// global tool registry as `<name>__<tool>` so multiple servers can ship
	// a tool called e.g. "search" without colliding. Populated and mutated
	// by reconcileMCP; nil on projects without an mcp.toml.
	mcpClients map[string]*MCPClient

	// mcpReconcileMu serialises reconcileMCP across concurrent callers
	// (startIndexing's banner pass and Prompt's per-turn pass). The lock
	// scope covers the full diff/start/stop sequence so two reconciles
	// can't race on a.mcpClients or the global tool registry.
	mcpReconcileMu sync.Mutex
	// mcpApplied is the set of [[server]] entries the last successful
	// reconcile actually brought up. The next pass diffs the new file
	// against this to decide what to start, stop, or restart. Entries that
	// failed to start are NOT included, so the next reconcile retries them
	// with a fresh StartMCPClient call once the file changes again.
	mcpApplied []MCPServerConfig
	// mcpAppliedMtime is the mtime of .codehalter/mcp.toml at the time of
	// the last reconcile. Unchanged mtime → skip the diff entirely, which
	// also keeps a persistently-broken server from re-emitting the same
	// failed card on every prompt. Zero value means "never reconciled yet".
	mcpAppliedMtime time.Time

	// runCmdStatus is the human-readable outcome of discoverSandbox: either
	// "available" (tool registered) or a specific reason it wasn't (not in a
	// container, shim install failed). notifyCapabilities surfaces it in the
	// consolidated banner so the user sees in chat whether run_command is
	// wired up — silent slog.Info was leaving users wondering why the LLM
	// never called it.
	runCmdStatus string

	// abortReason is set by the bootstrap goroutine when codehalter must not
	// run in this environment (today: started outside a devcontainer). Empty
	// means proceed; non-empty causes Prompt to refuse with this message.
	// Read under mu.
	abortReason string
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

// connKey returns a stable map key for a connection.
func connKey(c *LLMConnection) string { return c.URL + "\x00" + c.Model }

const (
	modeInteractive = "interactive"
	modeAutopilot   = "autopilot"
)

// agentInfo is the implementation block we advertise in the initialize
// response. Static — name and version don't change at runtime.
func agentInfo() any {
	return struct {
		Name    string `json:"name,omitempty"`
		Version string `json:"version,omitempty"`
	}{"llama-acp", "0.1.0"}
}

// sessionModes is the mode state advertised to the client on session
// create/load. The client uses this to render the mode selector.
func (a *agent) sessionModes() *SessionModeState {
	current := a.mode
	if current == "" {
		current = modeInteractive
	}
	return &SessionModeState{
		CurrentModeId: current,
		AvailableModes: []struct {
			Id          string `json:"id"`
			Name        string `json:"name"`
			Description string `json:"description,omitempty"`
		}{
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

// cwdOrDefault resolves the session's working directory to a clean absolute
// path. Clients may pass "." (bench harness) or any relative path; resolvePath
// then prefix-checks against sess.Cwd, and the check breaks when Cwd isn't
// absolute because filepath.Clean drops the leading "./" — read_file("go.mod")
// would resolve to "go.mod" and fail the "outside project directory" check
// even though it's inside the project.
func cwdOrDefault(cwd string) string {
	if cwd == "" {
		cwd, _ = os.Getwd()
	}
	if abs, err := filepath.Abs(cwd); err == nil {
		return abs
	}
	return cwd
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
	defer a.mu.Unlock()
	delete(a.sessions, id)
}

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
	_ = a.conn.SessionUpdate(ctx, sid, u)
}

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
		a.buildSlotSems()
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
	res.AgentInfo = agentInfo()
	res.AuthMethods = []string{}
	return res, nil
}

func (a *agent) Authenticate(_ context.Context) error {
	// No auth needed for local llama.cpp.
	return nil
}

func (a *agent) initSession(cwd string, s *Session) error {
	a.putSession(s)
	ensureDefaults(cwd)
	// Always rebuild SystemPrompt from the freshly-seeded .codehalter/
	// directory. NewSession starts with SystemPrompt == "" so this is the
	// first-and-only build; LoadSession has a possibly-stale SystemPrompt
	// from a prior run on a different host (different OS skill set), and
	// we must overwrite it BEFORE prepare's proposeFix can dispatch an
	// LLM call carrying the stale prefix.
	if sp, err := a.systemPrompt(s.ID); err == nil {
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
	a.buildSlotSems()
	a.discoverRunners(cwd)
	a.discoverSandbox()
	a.registerSubagentTool()
	return nil
}

// startIndexing kicks off the once-per-session bootstrap: interactive
// devcontainer / gitignore prompts that would be hostile to repeat on every
// turn, followed by the first prepare() so the capabilities banner appears
// at session open instead of only after the user's first typed prompt.
// Everything in prepare (LLM probe, env snapshot, MCP reconcile) is also
// re-run from Prompt — calling it here is just the "show the user something
// immediately" entry point. Ordering: devcontainer first because the
// gitignore prompt assumes a sandbox exists. When ensureDevcontainer returns
// false, abortReason is set and the rest is skipped — Prompt refuses every
// turn after that.
func (a *agent) startIndexing(sid string, cwd string) {
	a.indexDone = make(chan struct{})
	slog.Debug("startIndexing: spawning bootstrap goroutine", "sid", sid, "cwd", cwd)
	go func() {
		defer close(a.indexDone)
		defer slog.Debug("startIndexing: bootstrap goroutine done", "sid", sid)
		// Install a.cancel so Zed's Cancel button can interrupt a fix-install
		// runTaskCycle that prepare may dispatch via proposeFix. Same pattern
		// as Prompt(): one cancel slot, last-writer-wins.
		ctx, cancel := context.WithCancel(context.Background())
		a.mu.Lock()
		a.cancel = cancel
		a.mu.Unlock()
		defer cancel()

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

// osInfo holds the result of parsing /etc/os-release. ID is the
// supported-distro slug we use to pick a SKILL-*.md (one of "alpine",
// "arch", "debian", "fedora", "ubuntu"; "" when missing/unsupported).
// Fields is every key=value pair from the file (un-lowercased values,
// quotes stripped) — used to substitute {{VERSION_ID}}, {{PRETTY_NAME}},
// etc. into the per-OS skill body so the LLM doesn't have to probe.
type osInfo struct {
	ID     string
	Fields map[string]string
}

// readOSInfo parses /etc/os-release. ID_LIKE is consulted as a fallback
// so Linux Mint maps to ubuntu, Manjaro to arch, etc. Cheap file read —
// safe to call from prepare on every turn.
func readOSInfo() osInfo {
	info := osInfo{Fields: map[string]string{}}
	data, err := os.ReadFile("/etc/os-release")
	if err != nil {
		return info
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		eq := strings.IndexByte(line, '=')
		if eq <= 0 {
			continue
		}
		k := line[:eq]
		v := strings.Trim(line[eq+1:], `"'`)
		info.Fields[k] = v
	}
	supported := map[string]bool{"alpine": true, "arch": true, "debian": true, "fedora": true, "ubuntu": true}
	if id := strings.ToLower(info.Fields["ID"]); supported[id] {
		info.ID = id
		return info
	}
	for _, alt := range strings.Fields(strings.ToLower(info.Fields["ID_LIKE"])) {
		if supported[alt] {
			info.ID = alt
			return info
		}
	}
	return info
}

func (a *agent) NewSession(_ context.Context, req NewSessionRequest) (NewSessionResponse, error) {
	cwd := cwdOrDefault(req.Cwd)
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
	cwd := cwdOrDefault(req.Cwd)
	slog.Debug("LoadSession: enter", "cwd", cwd, "sid", req.SessionId)
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
	a.replayHistory(ctx, req.SessionId, s)
	a.startIndexing(s.ID, cwd)
	return LoadSessionResponse{Modes: a.sessionModes()}, nil
}

func (a *agent) replayHistory(ctx context.Context, sid string, s *Session) {
	lastRole := ""
	for _, m := range s.Messages {
		if m.Role == lastRole {
			if m.Role == "user" {
				a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: ""}})
			} else {
				a.sendUpdate(ctx, sid, messageChunk{Kind: KindUserMessage, Content: ContentBlock{Type: "text", Text: ""}})
			}
		}
		if m.Role == "user" {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindUserMessage, Content: ContentBlock{Type: "text", Text: m.Content}})
			for _, img := range m.Images {
				a.sendUpdate(ctx, sid, messageChunk{Kind: KindUserMessage, Content: ContentBlock{Type: "image", MimeType: img.MimeType, Data: img.Data}})
			}
		} else {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: m.Content}})
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

func (a *agent) SetSessionMode(ctx context.Context, req SetSessionModeRequest) error {
	if req.ModeId != modeInteractive && req.ModeId != modeAutopilot {
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
	a.mu.Lock()
	if a.cancel != nil {
		a.cancel()
	}
	a.mu.Unlock()
	a.deleteSession(req.SessionId)
	return nil
}

func (a *agent) Cancel(_ context.Context, _ CancelNotification) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.cancel != nil {
		a.cancel()
	}
}

// loadSkills concatenates every SKILL-*.md present in .codehalter/. Detection
// (detectStacks) decides which to seed initially, but loading honors whatever
// the user actually has on disk — drop a SKILL-rust.md in there manually and
// it gets picked up; delete one and it stops loading. Called once per session
// (from systemPrompt) so the concatenated text lives in the first user
// message and stays cache-stable thereafter.
func loadSkills(cwd string) string {
	dir := filepath.Join(cwd, ".codehalter")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		n := e.Name()
		if strings.HasPrefix(n, "SKILL-") && strings.HasSuffix(n, ".md") {
			names = append(names, n)
		}
	}
	if len(names) == 0 {
		return ""
	}
	sort.Strings(names) // deterministic order → stable cache prefix
	var b strings.Builder
	for _, n := range names {
		data, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			continue
		}
		b.Write(data)
		if !strings.HasSuffix(string(data), "\n") {
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}
	return b.String()
}

func (a *agent) systemPrompt(sid string) (string, error) {
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
		"(e.g. npm → also pnpm/yarn; python → also uv/poetry/pypy; " +
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
func (a *agent) sessionLog(sid string) *os.File {
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
func (a *agent) logSession(sid string, tag, format string, args ...any) {
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

func (a *agent) loadPromptFile(sid string, filename string) string {
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
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))

	a := &agent{sessions: make(map[string]*Session), mode: modeInteractive}
	conn := NewAgentSideConnection(a, os.Stdout, os.Stdin)
	a.conn = conn

	slog.Info("waiting for connection")
	<-conn.Done()
	slog.Info("connection closed")
}
