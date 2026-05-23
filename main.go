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

//go:embed docs/SKILL-buildfile.md
var skillBuildfile string

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

//go:embed docs/SKILL-devcontainer.md
var skillDevcontainer string

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

//go:embed docs/BOOTSTRAP-alpine-go.md
var bootstrapAlpineGo string

//go:embed docs/BOOTSTRAP-arch-go.md
var bootstrapArchGo string

//go:embed docs/BOOTSTRAP-debian-go.md
var bootstrapDebianGo string

//go:embed docs/BOOTSTRAP-fedora-go.md
var bootstrapFedoraGo string

//go:embed docs/BOOTSTRAP-ubuntu-go.md
var bootstrapUbuntuGo string

// defaultBootstraps is keyed by "<os>-<stack>". Used by ensureDevcontainer
// to seed BOOTSTRAP-<os>-<stack>.md for each detected stack right after a
// fresh .devcontainer/Dockerfile is written — never re-seeded once the
// .devcontainer dir exists. Only Go is wired in today; add more stacks as
// they get bootstrap templates.
var defaultBootstraps = map[string]string{
	"alpine-go": bootstrapAlpineGo,
	"arch-go":   bootstrapArchGo,
	"debian-go": bootstrapDebianGo,
	"fedora-go": bootstrapFedoraGo,
	"ubuntu-go": bootstrapUbuntuGo,
}

//go:embed docs/settings.toml
var defaultSettingsTOML string

//go:embed docs/mcp.toml
var defaultMCPToml string

var defaultSkills = map[string]string{
	"go":           skillGo,
	"ts":           skillTS,
	"js":           skillJS,
	"java":         skillJava,
	"bash":         skillBash,
	"devcontainer": skillDevcontainer,
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
	stacks := detectStacks(cwd)
	for _, stack := range stacks {
		body, ok := defaultSkills[stack]
		if !ok {
			continue
		}
		path := filepath.Join(dir, "SKILL-"+stack+".md")
		if _, err := os.Stat(path); os.IsNotExist(err) {
			os.WriteFile(path, []byte(body), 0o644)
		}
	}

	// mcp.toml — only seeded on first run. If go.mod is present at seed
	// time, append an uncommented gopls entry so go_symbols / go_references
	// (now provided by `gopls mcp`) are wired up out of the box. Once the
	// file exists we never touch it again — the user owns it.
	mcpPath := filepath.Join(dir, "mcp.toml")
	if _, err := os.Stat(mcpPath); os.IsNotExist(err) {
		content := defaultMCPToml
		for _, s := range stacks {
			if s == "go" {
				if !strings.HasSuffix(content, "\n") {
					content += "\n"
				}
				content += "\n[[server]]\nname = \"gopls\"\ncommand = \"gopls\"\nargs = [\"mcp\"]\n"
				break
			}
		}
		os.WriteFile(mcpPath, []byte(content), 0o644)
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
	// the startup probe. Keyed by connKey(URL+model). pickBackgroundLLM
	// filters candidates against this so a dead extra slot doesn't burn a
	// timeout on every background summarise call. Populated by checkLLM;
	// nil before that.
	connReachable map[string]bool

	// mainContextTokens is LLM[0]'s context window in tokens,
	// discovered by the startup probe (/props.n_ctx or /v1/models --ctx-size).
	// 0 means unknown (probe failed or server didn't report it) — compaction
	// then falls back to the rawBufferTokens constant. compressHistory reads
	// this; nothing else should.
	mainContextTokens int

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
	// container, shim install failed). checkEnvironment surfaces it so the
	// user sees in chat whether run_command is wired up — silent slog.Info
	// was leaving users wondering why the LLM never called it.
	runCmdStatus string

	// abortReason is set by the bootstrap goroutine when codehalter must not
	// run in this environment (today: started outside a devcontainer). Empty
	// means proceed; non-empty causes Prompt to refuse with this message.
	// Read under mu.
	abortReason string
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
	// startIndexing → checkLLM re-probes and updates the flag plus the
	// startup banner.
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
	settings, err := loadSettings(cwd)
	if err != nil {
		return err
	}
	// Empty settings are tolerated (path == "") — startIndexing prompts the
	// user to write a skeleton on first run; running with no LLM until then
	// is graceful (checkLLM prints a warning instead of crashing).
	a.settings = settings
	a.buildSlotSems()
	a.discoverRunners(cwd)
	a.discoverSandbox()
	a.registerSubagentTool()
	return nil
}

func (a *agent) startIndexing(sid string, cwd string) {
	a.indexDone = make(chan struct{})
	go func() {
		defer close(a.indexDone)
		ctx := context.Background()
		a.bootstrapSettings(ctx, cwd, sid)
	}()
}

// bootstrapSettings runs the once-per-session startup sequence: interactive
// prompts (devcontainer/settings/gitignore) that would be hostile to repeat
// on every turn, the heavy one-time LLM probe, the project capabilities
// banner, and the environment summary. The per-prompt re-check is delegated
// to checkSettings, which bootstrapSettings calls in-line so the MCP
// reconcile lands in the same flow.
//
// Order: devcontainer → LLM → gitignore → per-stack bootstrap. The
// devcontainer gate runs first because settings (LLM, MCP) can be fixed
// later inside the container; running unsandboxed at all is the policy
// violation. When ensureDevcontainer returns false, abortReason is set and
// the rest of the sequence is skipped — Prompt then refuses every turn.
func (a *agent) bootstrapSettings(ctx context.Context, cwd string, sid string) {
	if !a.ensureDevcontainer(ctx, cwd, sid) {
		return
	}

	a.ensureSettings(ctx, cwd, sid)

	if a.settings.path != "" {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Using " + a.settings.path + "\n\n"}})
	}

	a.checkLLM(ctx, sid)
	a.notifyCapabilities(ctx, sid)
	a.checkSettings(ctx, cwd, sid)
	a.checkEnvironment(ctx, sid)

	a.ensureGitignore(ctx, cwd, sid)

	// TODO: per-stack bootstrap loop (BOOTSTRAP-<os>-<stack>.md → LLM runs
	// the install commands, verifies). Needs the LLM, hence after checkLLM.
}

// checkSettings is the per-prompt "did anything change?" pass. Every step
// here MUST be silent when nothing has changed — running this on every user
// turn should add zero chat noise during a steady-state conversation, only
// emitting cards when there's something the user needs to see. Today it
// reconciles .codehalter/mcp.toml (mtime-gated); future mid-session checks
// (settings.toml reload triggering an LLM re-probe, capability re-detection
// when a Makefile appears) belong here too.
func (a *agent) checkSettings(ctx context.Context, cwd string, sid string) {
	changes := a.reconcileMCP(ctx, cwd)
	a.renderMCPChanges(ctx, sid, changes)
	a.cleanupGitCommitIfClean(cwd, sid)
}

// notifyCapabilities emits a summary of discovered build/test/lint/format
// runners at session start, flagging any category where nothing was found so
// the user knows to either configure their runner or accept the gap.
func (a *agent) notifyCapabilities(ctx context.Context, sid string) {
	a.mu.Lock()
	caps := a.capabilities
	empty := a.emptyProject
	a.mu.Unlock()

	if empty {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Empty project — I'll ask about language and runner on your first message.\n\n"}})
		return
	}
	if len(caps.runners) == 0 {
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "🟡 No task runner detected (just, make, npm, go, cargo). Add one so I can build/test/lint.\n\n"}})
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
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: b.String()}})
}

// checkLLM probes every configured LLMConnection in parallel, records which
// ones answered, and reports each line to the user. Background summarisation
// then skips the unreachable extras instead of eating a timeout per call.
// Image support is taken from LLM[0] — that's the one the main session's
// turns land on, so its capabilities are the ones the user sees.
func (a *agent) checkLLM(ctx context.Context, sid string) {
	conns := a.settings.allConnections()
	if len(conns) == 0 {
		a.imagesSupported = false
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "🟡 LLM: no [[llm]] in settings.toml — codehalter cannot run until you add one.\n\n"}})
		return
	}
	if settingsLooksPlaceholder(a.settings) {
		a.imagesSupported = false
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "🟡 LLM: " + a.settings.path + " still has the placeholder model \"your-model-id\". Edit it with your real url and model, then restart this Zed session.\n\n"}})
		return
	}

	results := make([]probeResult, len(conns))
	parallel(len(conns), maxParallel, func(i int) {
		c := conns[i]
		results[i] = a.probeLLM(ctx, &c)
	})

	a.connReachable = make(map[string]bool, len(conns))
	// conns[0] is the main entry that owns the foreground session's KV cache.
	// Its context size drives compaction sizing; extra slots' sizes aren't
	// used for that because subagent sessions have their own short lifetimes.
	if len(results) > 0 && results[0].ContextSize > 0 {
		a.mainContextTokens = results[0].ContextSize
	}
	var b strings.Builder
	firstReachable := -1
	// Each LLM gets its own paragraph (\n\n) — markdown collapses single
	// newlines to spaces, so "llm[0]: ...\nllm[1]: ..." would render on one
	// wrapped line and obscure that there are two separate connections.
	for i, r := range results {
		c := conns[i]
		a.connReachable[connKey(&c)] = r.Reachable
		label := fmt.Sprintf("llm[%d]", i)
		if i > 0 && c.Tag != "" {
			label += " " + c.Tag
		}
		switch {
		case !r.Reachable:
			fmt.Fprintf(&b, "🟡 %s: unreachable at %s — start your server or fix the url.\n\n", label, c.URL)
		case r.ModelKnown && !r.ModelLoaded:
			fmt.Fprintf(&b, "🟡 %s: %s reachable but model %q not loaded.\n\n", label, c.URL, c.Model)
		default:
			fmt.Fprintf(&b, "✅ %s: %s @ %s (parallel=%d)\n\n", label, c.Model, c.URL, c.parallelCap())
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
		if a.mainContextTokens > 0 {
			fmt.Fprintf(&b, "✅ Context window: %d tokens (compact at ~%d)\n\n", a.mainContextTokens, a.compactTriggerTokens())
		} else {
			fmt.Fprintf(&b, "🟡 Context window: unknown — server didn't report n_ctx, using default compact trigger %d\n\n", rawBufferTokens)
		}
	}
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: b.String()}})
}

// checkEnvironment reports whether codehalter is running inside a container
// and whether Firefox (required for web_search/web_read) is available, so
// devcontainer users see at startup whether the toolchain is wired up.
func (a *agent) checkEnvironment(ctx context.Context, sid string) {
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
	switch a.runCmdStatus {
	case "available":
		b.WriteString("✅ run_command: available (probes and test installs; `git` is shimmed to exit 127)\n\n")
	case "":
		// discoverSandbox never ran. Should not happen — initSession calls it
		// unconditionally. Stay silent rather than print misleading info.
	default:
		fmt.Fprintf(&b, "🟡 run_command: disabled — %s\n\n", a.runCmdStatus)
	}
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: b.String()}})
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
