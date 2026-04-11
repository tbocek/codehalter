package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
)

// Browser manages a Firefox instance controlled via WebDriver BiDi.
type Browser struct {
	cmd        *exec.Cmd
	profileDir string
	conn       *websocket.Conn
	port       int
	initialTab string // context ID of the tab Firefox opened with

	mu      sync.Mutex
	nextID  atomic.Int64
	pending map[int64]chan json.RawMessage
}

// BiDi message types per W3C spec.

type bidiRequest struct {
	ID     int64  `json:"id"`
	Method string `json:"method"`
	Params any    `json:"params"`
}

type bidiResponse struct {
	Type    string          `json:"type"` // "success" or "error"
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   string          `json:"error,omitempty"`
	Message string          `json:"message,omitempty"`
}

// StartBrowser launches Firefox in private mode with BiDi enabled.
// initialURL is opened in the first tab (use "about:blank" if none).
func StartBrowser(ctx context.Context, port int, initialURL string) (*Browser, error) {
	firefoxPath, err := findFirefox()
	if err != nil {
		return nil, err
	}

	profileDir, err := os.MkdirTemp("", "codehalter-firefox-*")
	if err != nil {
		return nil, fmt.Errorf("creating temp profile: %w", err)
	}

	cmd := exec.CommandContext(ctx, firefoxPath,
		"--private-window",
		"--no-remote",
		"--profile", profileDir,
		fmt.Sprintf("--remote-debugging-port=%d", port),
		initialURL,
	)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		os.RemoveAll(profileDir)
		return nil, fmt.Errorf("starting firefox: %w", err)
	}

	slog.Info("firefox started", "pid", cmd.Process.Pid, "port", port)

	b := &Browser{
		cmd:        cmd,
		profileDir: profileDir,
		port:       port,
		pending:    make(map[int64]chan json.RawMessage),
	}

	// Wait for Firefox to accept connections.
	if err := b.waitReady(ctx); err != nil {
		b.Close()
		return nil, err
	}

	// Connect WebSocket to BiDi endpoint.
	wsURL := fmt.Sprintf("ws://127.0.0.1:%d/session", port)
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		b.Close()
		return nil, fmt.Errorf("connecting websocket to %s: %w", wsURL, err)
	}
	conn.SetReadLimit(10 * 1024 * 1024) // 10MB
	b.conn = conn

	// Start reading messages.
	go b.readLoop()

	// Create a BiDi session.
	result, err := b.Send(ctx, "session.new", map[string]any{
		"capabilities": map[string]any{},
	})
	if err != nil {
		b.Close()
		return nil, fmt.Errorf("creating bidi session: %w", err)
	}
	slog.Info("bidi session created", "result", string(result))

	// Get the initial tab's context ID (Firefox opens with the initialURL).
	treeResult, err := b.Send(ctx, "browsingContext.getTree", map[string]any{})
	if err == nil {
		var tree struct {
			Contexts []struct {
				Context string `json:"context"`
			} `json:"contexts"`
		}
		json.Unmarshal(treeResult, &tree)
		if len(tree.Contexts) > 0 {
			b.initialTab = tree.Contexts[0].Context
			slog.Info("initial tab", "context", b.initialTab)
		}
	}

	return b, nil
}

// Send sends a BiDi command and waits for the response.
func (b *Browser) Send(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := b.nextID.Add(1)

	ch := make(chan json.RawMessage, 1)
	b.mu.Lock()
	b.pending[id] = ch
	b.mu.Unlock()

	defer func() {
		b.mu.Lock()
		delete(b.pending, id)
		b.mu.Unlock()
	}()

	msg := bidiRequest{ID: id, Method: method, Params: params}
	data, _ := json.Marshal(msg)
	slog.Debug("bidi send", "method", method, "id", id)

	if err := b.conn.Write(ctx, websocket.MessageText, data); err != nil {
		return nil, fmt.Errorf("writing bidi message: %w", err)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case raw := <-ch:
		var resp bidiResponse
		json.Unmarshal(raw, &resp)
		if resp.Type == "error" {
			return nil, fmt.Errorf("bidi error: %s: %s", resp.Error, resp.Message)
		}
		return resp.Result, nil
	}
}

// Navigate navigates a tab to the given URL and waits for load.
func (b *Browser) Navigate(ctx context.Context, contextID, url string) error {
	_, err := b.Send(ctx, "browsingContext.navigate", map[string]any{
		"context": contextID,
		"url":     url,
		"wait":    "complete",
	})
	return err
}

// OpenTab creates a new tab, navigates to the URL, and returns the context ID.
func (b *Browser) OpenTab(ctx context.Context, url string) (string, error) {
	result, err := b.Send(ctx, "browsingContext.create", map[string]any{
		"type": "tab",
	})
	if err != nil {
		return "", err
	}
	var ctxResult struct {
		Context string `json:"context"`
	}
	json.Unmarshal(result, &ctxResult)

	err = b.Navigate(ctx, ctxResult.Context, url)
	return ctxResult.Context, err
}

// CloseTab closes a browsing context.
func (b *Browser) CloseTab(ctx context.Context, contextID string) {
	b.Send(ctx, "browsingContext.close", map[string]any{
		"context": contextID,
	})
}

// PageText returns the visible text content of a tab.
func (b *Browser) PageText(ctx context.Context, contextID string) (string, error) {
	result, err := b.Send(ctx, "script.evaluate", map[string]any{
		"expression":   "document.body.innerText",
		"target":       map[string]any{"context": contextID},
		"awaitPromise": false,
	})
	if err != nil {
		return "", err
	}

	var evalResult struct {
		Result struct {
			Type  string `json:"type"`
			Value string `json:"value"`
		} `json:"result"`
	}
	if err := json.Unmarshal(result, &evalResult); err != nil {
		return "", fmt.Errorf("parsing eval result: %w", err)
	}
	return evalResult.Result.Value, nil
}

// EvalJS runs JavaScript in a tab and returns the string result.
func (b *Browser) EvalJS(ctx context.Context, contextID, script string) (string, error) {
	result, err := b.Send(ctx, "script.evaluate", map[string]any{
		"expression":   script,
		"target":       map[string]any{"context": contextID},
		"awaitPromise": false,
	})
	if err != nil {
		return "", err
	}
	var evalResult struct {
		Result struct {
			Type  string `json:"type"`
			Value string `json:"value"`
		} `json:"result"`
	}
	json.Unmarshal(result, &evalResult)
	return evalResult.Result.Value, nil
}

// Close shuts down the browser.
func (b *Browser) Close() {
	if b.conn != nil {
		b.conn.Close(websocket.StatusNormalClosure, "shutdown")
	}
	if b.cmd != nil && b.cmd.Process != nil {
		b.cmd.Process.Kill()
		b.cmd.Wait()
	}
	if b.profileDir != "" {
		os.RemoveAll(b.profileDir)
	}
	slog.Info("browser closed")
}

func (b *Browser) readLoop() {
	for {
		_, data, err := b.conn.Read(context.Background())
		if err != nil {
			slog.Debug("bidi read error", "error", err)
			return
		}
		slog.Debug("bidi recv", "data", string(data))

		var resp bidiResponse
		if err := json.Unmarshal(data, &resp); err != nil {
			continue
		}

		// Dispatch responses (have an ID) to pending callers.
		if resp.ID > 0 {
			b.mu.Lock()
			ch, ok := b.pending[resp.ID]
			b.mu.Unlock()
			if ok {
				ch <- data
			}
		}
		// Events (no ID) are ignored for now.
	}
}

// waitReady polls the TCP port until Firefox accepts connections.
func (b *Browser) waitReady(ctx context.Context) error {
	addr := fmt.Sprintf("127.0.0.1:%d", b.port)
	for i := 0; i < 30; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}

		conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if err == nil {
			conn.Close()
			slog.Info("firefox ready", "port", b.port)
			return nil
		}
	}
	return fmt.Errorf("firefox did not become ready on port %d after 15s", b.port)
}

func findFirefox() (string, error) {
	if p := os.Getenv("FIREFOX_PATH"); p != "" {
		return p, nil
	}
	candidates := []string{"firefox", "firefox-esr"}
	for _, name := range candidates {
		if p, err := exec.LookPath(name); err == nil {
			return p, nil
		}
	}
	// Common paths.
	for _, p := range []string{
		"/usr/bin/firefox",
		"/usr/bin/firefox-esr",
		"/snap/bin/firefox",
		"/Applications/Firefox.app/Contents/MacOS/firefox",
	} {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	// Try flatpak.
	if p, err := exec.LookPath("flatpak"); err == nil {
		out, err := exec.Command(p, "list", "--app", "--columns=application").Output()
		if err == nil {
			for _, line := range filepath.SplitList(string(out)) {
				if line == "org.mozilla.firefox" {
					return "flatpak run org.mozilla.firefox", nil
				}
			}
		}
	}
	return "", fmt.Errorf("firefox not found; set FIREFOX_PATH")
}
