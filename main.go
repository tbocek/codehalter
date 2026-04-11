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
	conn     *AgentSideConnection
	mu       sync.Mutex
	cancel   context.CancelFunc
	sessions map[SessionId]*Session
	settings       Settings
	runners        []taskRunner
	pendingRefs    []CodeRef
	fileCache      *FileCache
	indexDone      chan struct{}
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

func (a *agent) sendUpdate(ctx context.Context, sid SessionId, u SessionUpdate) {
	_ = a.conn.SessionUpdate(ctx, SessionNotification{SessionId: sid, Update: u})
}

func (a *agent) Initialize(_ context.Context, req InitializeRequest) (InitializeResponse, error) {
	return InitializeResponse{
		ProtocolVersion: ProtocolVersionNumber,
		AgentCapabilities: AgentCapabilities{
			LoadSession:        true,
			PromptCapabilities: PromptCapabilities{},
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
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Using "+a.settings.path+"\n")))
		} else {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ No settings.toml found\n")))
		}

		a.refreshFileCache(ctx, cwd, sid)
	}()
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
	return NewSessionResponse{SessionId: s.ID, Modes: nil}, nil
}

func (a *agent) LoadSession(ctx context.Context, req LoadSessionRequest) (LoadSessionResponse, error) {
	cwd := cwdOrDefault(req.Cwd)
	s, err := loadSession(cwd, req.SessionId)
	if err != nil {
		if os.IsNotExist(err) {
			s = newSessionWithID(cwd, req.SessionId)
			a.initSession(cwd, s)
			a.startIndexing(s.ID, cwd)
			return LoadSessionResponse{Modes: nil}, nil
		}
		return LoadSessionResponse{}, fmt.Errorf("loading session: %w", err)
	}
	a.initSession(cwd, s)
	a.replayHistory(ctx, req.SessionId, s)
	a.startIndexing(s.ID, cwd)
	return LoadSessionResponse{Modes: nil}, nil
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

func (a *agent) SetSessionMode(_ context.Context, req SetSessionModeRequest) (SetSessionModeResponse, error) {
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
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("Indexing %d files...\n", len(stale)))))
		if err := a.summarizeStaleFiles(ctx, cwd, a.fileCache, stale, sid); err != nil {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("❌ "+err.Error()+"\n")))
		} else {
			a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("Indexed %d files.\n", len(stale)))))
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

	// Extract user text.
	var userText string
	for _, block := range req.Content {
		if block.Text != nil {
			userText += block.Text.Text
		}
	}

	// Store user message.
	sess := a.getSession(req.SessionId)
	isFirstMessage := sess != nil && len(sess.Messages) == 0 && len(sess.History) == 0
	if sess != nil {
		sess.AddUser(userText)
		_ = sess.Save()
	}

	// Generate title for new sessions.
	if isFirstMessage {
		go a.generateTitle(context.Background(), sess, userText)
	}

	// Plan → Execute → Verify cycle with retry.
	const maxRetryCycles = 3
	slog.Info("Prompt", "sid", req.SessionId, "sessions", len(a.sessions))

	originalUserText := userText
	var result toolLoopResult

	for cycle := 0; cycle < maxRetryCycles; cycle++ {
		if cycle > 0 {
			a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock(fmt.Sprintf("\n⚠ Retrying (attempt %d/%d)...\n", cycle+1, maxRetryCycles))))
		}

		// Plan.
		conn, planSteps, planToolUses, err := a.planAndRoute(ctx, req.SessionId, userText)
		if err != nil {
			if sess != nil && len(planToolUses) > 0 {
				sess.AddAssistantWithTools("❌ "+err.Error(), planToolUses)
				_ = sess.Save()
			}
			a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock("❌ "+err.Error()+"\n")))
			return PromptResponse{StopReason: StopReasonEndTurn}, nil
		}

		// Build execution content.
		content := userText
		if len(planSteps) > 0 {
			var planCtx strings.Builder
			planCtx.WriteString("The user approved this plan. Follow these steps exactly:\n")
			for i, step := range planSteps {
				fmt.Fprintf(&planCtx, "%d. %s\n", i+1, step)
			}
			if len(planToolUses) > 0 {
				planCtx.WriteString("\nDuring planning, these tools were already called (do NOT repeat them):\n")
				for _, tu := range planToolUses {
					fmt.Fprintf(&planCtx, "- %s(%s) → %s\n", tu.Name, tu.Input, truncate(tu.Output, 500))
				}
			}
			planCtx.WriteString("\nUser request: ")
			planCtx.WriteString(userText)
			content = planCtx.String()
		}
		if isFirstMessage && cycle == 0 {
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

		// Build message history.
		var messages []llmMessage
		if sess != nil {
			messages = a.buildLLMHistory(sess, originalUserText)
		}
		messages = append(messages, llmMessage{Role: "user", Content: content})

		// Execute.
		result, err = a.runToolLoop(ctx, req.SessionId, conn, messages, false)
		if err != nil {
			return a.replyError(ctx, req.SessionId, err.Error()), nil
		}

		// Verify.
		var vr *verifyResult
		result, vr, err = a.verify(ctx, req.SessionId, conn, messages, result, userText, planSteps)
		if err != nil {
			return a.replyError(ctx, req.SessionId, err.Error()), nil
		}

		// If verification passed or no fix steps, we're done.
		if vr == nil || vr.Success || len(vr.FixSteps) == 0 {
			break
		}

		// Verification failed with fix steps — retry with new context.
		a.sendUpdate(ctx, req.SessionId, AgentMessageChunk(TextBlock(
			fmt.Sprintf("⚠ Verification failed. Fix steps:\n%s\n", strings.Join(vr.FixSteps, "\n")),
		)))
		userText = fmt.Sprintf("Previous attempt failed.\nIssues: %s\nFix steps: %s\n\nOriginal request: %s",
			strings.Join(vr.Issues, "; "),
			strings.Join(vr.FixSteps, "; "),
			originalUserText)
	}

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

	a := &agent{sessions: make(map[SessionId]*Session)}
	conn := NewAgentSideConnection(a, os.Stdout, os.Stdin, log)
	a.conn = conn

	log.Info("waiting for connection")
	<-conn.Done()
	log.Info("connection closed")
}
