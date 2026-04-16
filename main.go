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
)

//go:embed AGENT.md.example
var defaultAgentMD string

//go:embed PLAN.md.example
var defaultPlanMD string

//go:embed EXECUTE.md.example
var defaultExecuteMD string

//go:embed SUMMARY.md.example
var defaultSummaryMD string

//go:embed VERIFY.md.example
var defaultVerifyMD string

// ensureDefaults copies embedded default files into .codehalter/ if they don't exist.
func ensureDefaults(cwd string) {
	dir := filepath.Join(cwd, ".codehalter")
	os.MkdirAll(dir, 0o755)
	for _, f := range []struct{ name, content string }{
		{"AGENT.md", defaultAgentMD},
		{"PLAN.md", defaultPlanMD},
		{"EXECUTE.md", defaultExecuteMD},
		{"SUMMARY.md", defaultSummaryMD},
		{"VERIFY.md", defaultVerifyMD},
	} {
		path := filepath.Join(dir, f.name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			os.WriteFile(path, []byte(f.content), 0o644)
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
	pendingRefs     []CodeRef
	fileCache       *FileCache
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

func (a *agent) sendUpdate(ctx context.Context, sid SessionId, u SessionUpdate) {
	_ = a.conn.SessionUpdate(ctx, SessionNotification{SessionId: sid, Update: u})
}

// phaseNames are the pipeline stages; only the current one is shown to the
// client at a time (the plan UI updates in place as phases progress).
var phaseNames = []string{"Planning", "Executing", "Verifying"}

// sendPhase emits a plan update with a single entry for the current stage,
// replacing any previous entry. When done=true it marks the last stage
// completed so the client renders the pipeline as finished.
func (a *agent) sendPhase(ctx context.Context, sid SessionId, phase int, done bool) {
	var entry PlanEntry
	if done {
		entry = PlanEntry{Content: phaseNames[len(phaseNames)-1], Priority: "medium", Status: "completed"}
	} else {
		entry = PlanEntry{Content: phaseNames[phase], Priority: "medium", Status: "in_progress"}
	}
	a.sendUpdate(ctx, sid, PlanUpdate([]PlanEntry{entry}))
}

func (a *agent) Initialize(ctx context.Context, req InitializeRequest) (InitializeResponse, error) {
	// Detect image support by probing the execute/thinking LLM with a tiny
	// image. We load the global settings only here — project-local settings
	// live under a cwd we do not yet have. If project-local settings override
	// the LLM later, startIndexing will re-probe and update the flag.
	if gs, err := loadGlobalSettings(); err == nil {
		a.settings = gs
		if conn := a.probeTargetConn(); conn != nil {
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

// probeTargetConn returns the LLM connection that receives user images
// (execute, falling back to thinking).
func (a *agent) probeTargetConn() *LLMConnection {
	if c := a.settings.LLM("execute"); c != nil {
		return c
	}
	return a.settings.LLM("thinking")
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

func (a *agent) initSession(cwd string, s *Session) {
	a.putSession(s)
	ensureDefaults(cwd)
	if settings, err := loadSettings(cwd); err == nil {
		a.settings = settings
	}
	a.discoverRunners(cwd)
	a.registerSubagentTool()
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

		a.ensureGitignore(ctx, cwd, sid)
		a.refreshFileCache(ctx, cwd, sid)
	}()
}

// checkImageSupport re-probes the active LLM if project-local settings point
// at a different model than the one probed in Initialize, then tells the user
// whether image input is available. Runs on the indexing goroutine so it never
// blocks the session handshake.
func (a *agent) checkImageSupport(ctx context.Context, sid SessionId) {
	conn := a.probeTargetConn()
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
	a.initSession(cwd, s)
	a.startIndexing(s.ID, cwd)
	return NewSessionResponse{SessionId: s.ID, Modes: a.sessionModes()}, nil
}

func (a *agent) LoadSession(ctx context.Context, req LoadSessionRequest) (LoadSessionResponse, error) {
	cwd := cwdOrDefault(req.Cwd)
	s, err := loadSession(cwd, req.SessionId)
	if err != nil {
		if os.IsNotExist(err) {
			s = newSessionWithID(cwd, req.SessionId)
			a.initSession(cwd, s)
			a.startIndexing(s.ID, cwd)
			return LoadSessionResponse{Modes: a.sessionModes()}, nil
		}
		return LoadSessionResponse{}, fmt.Errorf("loading session: %w", err)
	}
	a.initSession(cwd, s)
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

func (a *agent) refreshFileCache(ctx context.Context, cwd string, sid SessionId) {
	a.fileCache = loadFileCache(cwd)
	stale := updateFileCache(cwd, a.fileCache)
	if len(stale) > 0 {
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("Indexing %d files...\n\n", len(stale)))))
		if err := a.summarizeStaleFiles(ctx, cwd, a.fileCache, stale, sid); err != nil {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("❌ "+err.Error()+"\n\n")))
		} else {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("Indexed %d files.\n\n", len(stale)))))
		}
	}
}

func (a *agent) systemPrompt(sid SessionId) (string, error) {
	sess := a.getSession(sid)
	if sess == nil {
		return "", fmt.Errorf("no session found")
	}

	var b strings.Builder

	content, err := os.ReadFile(filepath.Join(sess.Cwd, ".codehalter", "AGENT.md"))
	if err == nil {
		b.Write(content)
		b.WriteString("\n\n")
	}

	b.WriteString("Project directory: " + sess.Cwd + "\n\n")
	b.WriteString(buildProjectContext(sess.Cwd, a.fileCache))

	return b.String(), nil
}

func (a *agent) replyError(ctx context.Context, sid SessionId, msg string) PromptResponse {
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("❌ Error: "+msg+"\n")))
	if sess := a.getSession(sid); sess != nil {
		sess.AddAssistant("❌ " + msg)
		_ = sess.Save()
	}
	return PromptResponse{StopReason: StopReasonEndTurn}
}

func (a *agent) Prompt(ctx context.Context, req PromptRequest) (PromptResponse, error) {
	ctx, cancel := context.WithCancel(ctx)
	a.mu.Lock()
	a.cancel = cancel
	a.mu.Unlock()
	defer cancel()

	a.waitForIndex()

	// Extract user text and images from prompt blocks.
	var userText string
	var images []ImageData
	for _, block := range req.Content {
		if block.Text != nil {
			userText += block.Text.Text
		}
		if block.Image != nil {
			images = append(images, ImageData{
				MimeType: block.Image.MimeType,
				Data:     block.Image.Data,
			})
		}
	}

	// Store user message. The stored message is the raw userText only —
	// project context and plan/retry prefixes are injected onto the latest
	// prompt at send time (see buildLLMHistory) and are NOT persisted, so
	// history stays cacheable and compact.
	sess := a.getSession(req.SessionId)
	isFirstMessage := sess != nil && len(sess.Messages) == 0 && len(sess.History) == 0
	currentUserIdx := -1
	if sess != nil {
		if len(images) > 0 {
			sess.AddUserWithImages(userText, images)
		} else {
			sess.AddUser(userText)
		}
		currentUserIdx = len(sess.Messages) - 1
		_ = sess.Save()
	}

	// Generate title for new sessions.
	if isFirstMessage {
		go a.generateTitle(context.Background(), sess, userText)
	}

	slog.Info("Prompt", "sid", req.SessionId, "sessions", len(a.sessions))

	// Plan → Execute → Verify. If verify fails, feed the failure context into
	// a fresh plan and try again — the planner can pick a different strategy
	// each pass, so we're not repeating the same failing attempt. Loop until
	// verify is happy, with a hard cap (maxAttempts) as a safety net against
	// a broken LLM that never converges.
	const maxAttempts = 5
	var result toolLoopResult
	var conn *LLMConnection
	planInput := userText

	for attempt := 0; attempt < maxAttempts; attempt++ {
		a.sendPhase(ctx, req.SessionId, 0, false)
		c, planSteps, planToolUses, err := a.planAndRoute(ctx, req.SessionId, planInput)
		if err != nil {
			if sess != nil && len(planToolUses) > 0 {
				sess.AddAssistantWithTools("❌ "+err.Error(), planToolUses)
				_ = sess.Save()
			}
			a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock("❌ "+err.Error()+"\n")))
			return PromptResponse{StopReason: StopReasonEndTurn}, nil
		}
		conn = c

		// stored  = plan prefix + planInput  (persisted to TOML, 1:1 minus cache)
		// content = sysPrompt/projCtx + stored (sent to LLM this turn)
		stored := planInput
		if len(planSteps) > 0 || len(planToolUses) > 0 {
			var planCtx strings.Builder
			if len(planSteps) > 0 {
				planCtx.WriteString("The user approved this plan. Follow these steps exactly:\n")
				for i, step := range planSteps {
					fmt.Fprintf(&planCtx, "%d. %s\n", i+1, step)
				}
			}
			if len(planToolUses) > 0 {
				planCtx.WriteString("\nDuring planning, these tools were already called (do NOT repeat them):\n")
				for _, tu := range planToolUses {
					fmt.Fprintf(&planCtx, "- %s(%s) → %s\n", tu.Name, tu.Input, truncate(tu.Output, 500))
				}
			}
			planCtx.WriteString("\nUser request: ")
			planCtx.WriteString(planInput)
			stored = planCtx.String()
		}

		if sess != nil && currentUserIdx >= 0 && currentUserIdx < len(sess.Messages) {
			sess.Messages[currentUserIdx].Content = stored
			_ = sess.Save()
		}

		content := stored
		if isFirstMessage && attempt == 0 {
			sysPrompt, err := a.systemPrompt(req.SessionId)
			if err != nil {
				return a.replyError(ctx, req.SessionId, err.Error()), nil
			}
			content = sysPrompt + "\n---\n" + content
		} else if sess != nil {
			projCtx := buildProjectContext(sess.Cwd, a.fileCache)
			if projCtx != "" {
				content = projCtx + "\n---\n" + content
			}
		}

		// History is read 1:1 from TOML (including images on user turns). The
		// current user message is skipped here and re-appended with the cache
		// injected, so the cache lives only on the latest prompt and earlier
		// history stays cacheable.
		var messages []llmMessage
		if sess != nil {
			messages = a.buildLLMHistory(sess, currentUserIdx)
		}
		messages = append(messages, a.buildUserMessage(content, images))

		a.sendPhase(ctx, req.SessionId, 1, false)
		result, err = a.execute(ctx, req.SessionId, messages)
		if err != nil {
			return a.replyError(ctx, req.SessionId, err.Error()), nil
		}

		a.sendPhase(ctx, req.SessionId, 2, false)
		var vr *verifyResult
		result, vr, err = a.verify(ctx, req.SessionId, conn, messages, result, planInput, planSteps)
		if err != nil {
			return a.replyError(ctx, req.SessionId, err.Error()), nil
		}
		if vr == nil || vr.Success || len(vr.FixSteps) == 0 {
			break
		}

		if attempt == maxAttempts-1 {
			a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock(
				fmt.Sprintf("⚠ Verification still failing after %d attempts — giving up. Last issues:\n%s\n", maxAttempts, strings.Join(vr.Issues, "\n")),
			)))
			break
		}

		a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock(
			fmt.Sprintf("⚠ Verification failed (attempt %d/%d). Re-planning with the failure context:\n%s\n", attempt+1, maxAttempts, strings.Join(vr.FixSteps, "\n")),
		)))
		planInput = fmt.Sprintf("The previous attempt failed verification.\nIssues: %s\nFix steps: %s\n\nOriginal request: %s\n\nRe-plan with this information. If the fix steps conflict with the original request or the task is infeasible, say so instead of attempting it.",
			strings.Join(vr.Issues, "; "),
			strings.Join(vr.FixSteps, "; "),
			userText)
	}
	a.sendPhase(ctx, req.SessionId, 2, true)

	if sess != nil && result.Text != "" {
		fullResponse := result.Text

		// Update the last assistant message with the final text.
		if len(sess.Messages) > 0 && sess.Messages[len(sess.Messages)-1].Role == "assistant" {
			sess.Messages[len(sess.Messages)-1].Content = fullResponse
		} else {
			sess.AddAssistant(fullResponse)
		}
		_ = sess.Save()
		a.compressHistory(ctx, sess)
	}

	return PromptResponse{StopReason: StopReasonEndTurn}, nil
}

func (a *agent) buildUserMessage(content string, images []ImageData) llmMessage {
	if len(images) == 0 || !a.imagesSupported {
		return llmMessage{Role: "user", Content: content}
	}
	// Build OpenAI-style content array with text and image blocks.
	var parts []any
	parts = append(parts, map[string]any{
		"type": "text",
		"text": content,
	})
	for _, img := range images {
		parts = append(parts, map[string]any{
			"type": "image_url",
			"image_url": map[string]string{
				"url": fmt.Sprintf("data:%s;base64,%s", img.MimeType, img.Data),
			},
		})
	}
	return llmMessage{Role: "user", Content: parts}
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
	logFile, _ := os.OpenFile("/tmp/codehalter_debug.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if logFile != nil {
		defer logFile.Close()
	}
	// Stderr gets all logs (DEBUG+), file gets INFO+ only.
	stderrHandler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})
	if logFile != nil {
		fileHandler := slog.NewTextHandler(logFile, &slog.HandlerOptions{Level: slog.LevelInfo})
		log := slog.New(stderrHandler)
		slog.SetDefault(slog.New(fileHandler))
		_ = log // stderr logger used by jsonrpc via the passed log param
	} else {
		slog.SetDefault(slog.New(stderrHandler))
	}
	log := slog.New(stderrHandler)

	a := &agent{sessions: make(map[SessionId]*Session), mode: modeInteractive}
	conn := NewAgentSideConnection(a, os.Stdout, os.Stdin, log)
	a.conn = conn

	log.Info("waiting for connection")
	<-conn.Done()
	log.Info("connection closed")
}
