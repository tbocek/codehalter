package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/coder/websocket"
)

type Browser struct {
	cmd        *exec.Cmd
	profileDir string
	conn       *websocket.Conn
	port       int
	initialTab string

	mu      sync.Mutex
	nextID  atomic.Int64
	pending map[int64]chan json.RawMessage
}

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

func StartBrowser(ctx context.Context, port int, initialURL string) (*Browser, error) {
	firefoxPath, err := findFirefox()
	if err != nil {
		return nil, err
	}

	profileDir, err := os.MkdirTemp("", "codehalter-firefox-*")
	if err != nil {
		return nil, fmt.Errorf("creating temp profile: %w", err)
	}

	// Start on about:blank: a CLI-driven load has no load-complete signal, so the
	// real navigation goes through BiDi with wait:"complete" below.
	cmd := exec.CommandContext(ctx, firefoxPath,
		"-headless",
		"--private-window",
		"--no-remote",
		"--profile", profileDir,
		fmt.Sprintf("--remote-debugging-port=%d", port),
		"about:blank",
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

	if err := b.waitReady(ctx); err != nil {
		b.Close()
		return nil, err
	}

	wsURL := fmt.Sprintf("ws://127.0.0.1:%d/session", port)
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		b.Close()
		return nil, fmt.Errorf("connecting websocket to %s: %w", wsURL, err)
	}
	conn.SetReadLimit(10 * 1024 * 1024)
	b.conn = conn

	go b.readLoop()

	result, err := b.Send(ctx, "session.new", map[string]any{
		"capabilities": map[string]any{},
	})
	if err != nil {
		b.Close()
		return nil, fmt.Errorf("creating bidi session: %w", err)
	}
	slog.Info("bidi session created", "result", string(result))

	treeResult, err := b.Send(ctx, "browsingContext.getTree", map[string]any{})
	if err == nil {
		var tree struct {
			Contexts []struct {
				Context string `json:"context"`
			} `json:"contexts"`
		}
		if err := json.Unmarshal(treeResult, &tree); err != nil {
			slog.Debug("getTree: decoding contexts failed", "err", err)
		}
		if len(tree.Contexts) > 0 {
			b.initialTab = tree.Contexts[0].Context
			slog.Info("initial tab", "context", b.initialTab)
		}
	}

	// Capped: wait:"complete" can hang on pages that never fire load (long-poll,
	// streaming). On timeout the partially loaded body still beats failing.
	if b.initialTab != "" && initialURL != "" {
		navCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		_, err := b.Send(navCtx, "browsingContext.navigate", map[string]any{
			"context": b.initialTab,
			"url":     initialURL,
			"wait":    "complete",
		})
		cancel()
		if err != nil && navCtx.Err() == nil {
			b.Close()
			return nil, fmt.Errorf("navigating to %s: %w", initialURL, err)
		}
	}

	return b, nil
}

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
		if err := json.Unmarshal(raw, &resp); err != nil {
			return nil, fmt.Errorf("decoding bidi response: %w", err)
		}
		if resp.Type == "error" {
			return nil, fmt.Errorf("bidi error: %s: %s", resp.Error, resp.Message)
		}
		return resp.Result, nil
	}
}

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
	if err := json.Unmarshal(result, &evalResult); err != nil {
		return "", fmt.Errorf("parsing browser eval result: %w", err)
	}
	return evalResult.Result.Value, nil
}

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

		if resp.ID > 0 {
			b.mu.Lock()
			ch, ok := b.pending[resp.ID]
			b.mu.Unlock()
			if ok {
				ch <- data
			}
		}
	}
}

func (b *Browser) waitReady(ctx context.Context) error {
	addr := fmt.Sprintf("127.0.0.1:%d", b.port)
	for range 30 {
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
	for _, name := range []string{"firefox", "firefox-esr", "firefox-bin"} {
		if p, err := exec.LookPath(name); err == nil {
			return p, nil
		}
	}
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
	return "", fmt.Errorf("no Firefox found (tried firefox, firefox-esr, firefox-bin on PATH and the usual install paths); set FIREFOX_PATH")
}

const maxWebSearchResults = 10

var browserPortCounter atomic.Int32

func nextBrowserPort() int {
	return 9222 + int(browserPortCounter.Add(1))
}

var webTools = []Tool{
	{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "web_search",
			"description": "Search the web with DuckDuckGo. Returns up to 10 results as a numbered list (title, URL, snippet) for you to triage — does NOT fetch page content. Snippets alone are NOT enough to answer factual questions: you MUST follow up by calling web_read on at least one (ideally 1-3) of the most promising URLs. Skipping web_read is only acceptable if every result is clearly off-topic, in which case you should refine the query and search again.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"query"},
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "Keyword-style search query (NOT a natural-language sentence). Use specific technical terms, exact error messages, version numbers, or API/function names. Good: 'golang http.Client timeout context.DeadlineExceeded'. Bad: 'how do I handle timeouts in Go HTTP client'. Quote exact phrases when needed: '\"cannot find package\"'.",
					},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		query := args.str("query")
		if query == "" {
			return "error: query is required", false
		}

		a.logSession(sid, "WEB", "search query: %s", query)

		tcId := a.StartToolCall(ctx, sid, "DuckDuckGo: "+query, "search", nil)

		searchURL := "https://duckduckgo.com/?q=" + url.QueryEscape(query)

		port := nextBrowserPort()
		browser, err := StartBrowser(ctx, port, searchURL)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error starting browser: " + err.Error(), false
		}
		defer browser.Close()
		searchTab := browser.initialTab

		// Polling errors are transient, but the last one is kept so a consistent
		// failure is not reported as "DDG returned nothing".
		var results []ddgResult
		var extractErr error
	poll:
		for range 30 {
			results, extractErr = extractDDGResults(ctx, browser, searchTab)
			if len(results) > 0 {
				break
			}
			select {
			case <-ctx.Done():
				extractErr = ctx.Err()
				break poll
			case <-time.After(500 * time.Millisecond):
			}
		}
		if len(results) == 0 {
			msg := "no search results found"
			if extractErr != nil {
				msg += " (last extraction error: " + extractErr.Error() + ")"
			}
			a.FailToolCall(ctx, sid, tcId, msg)
			return "error: " + msg, false
		}

		if len(results) > maxWebSearchResults {
			results = results[:maxWebSearchResults]
		}

		formatted := formatDDGResults(results)
		a.CompleteToolCallTitled(ctx, sid, tcId,
			fmt.Sprintf("DuckDuckGo: %s (%d results)", query, len(results)),
			[]ToolCallContent{TextContent(formatted)})
		a.say(ctx, sid, "\n"+formatted+"\n")
		a.logSession(sid, "WEB", "results (%d):\n%s", len(results), formatted)
		return formatted, false
	}},

	// Deliberately no "answer my question" mode: the model reads the text
	// itself, without the extra round trip.
	{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "web_read",
			"description": fmt.Sprintf("Open a URL in Firefox and return the page's extracted text, truncated at %d characters. The full body is cached for this session, so `offset`/`limit` read a deeper region without re-fetching. Read it yourself: quote exact versions, names, commands and URLs from it.", maxRawPageChars),
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"url"},
				"properties": map[string]any{
					"url": map[string]any{
						"type":        "string",
						"description": "The URL to read",
					},
					"offset": map[string]any{
						"type":        "integer",
						"description": "Character offset into the page body to start at. Pair with limit to view a specific range of a page already cached from an earlier call (no HTTP re-fetch). Omit (or 0) on the first call.",
					},
					"limit": map[string]any{
						"type":        "integer",
						"description": fmt.Sprintf("Max characters to return starting at offset (hard cap %d). Use after a truncated read to view a deeper region of the cached body. Omit on the first call.", maxWebRangeChars),
					},
				},
			},
		},
	}, Execute: webReadExecute},
}

const (
	maxRawPageChars  = 30000
	maxWebRangeChars = 8000
)

func webReadExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	args := parseArgs(rawArgs)
	targetURL := args.str("url")
	if targetURL == "" {
		return "error: url is required", false
	}
	offset, _ := args.num("offset")
	if offset < 0 {
		offset = 0
	}
	limit, _ := args.num("limit")
	if limit <= 0 || limit > maxWebRangeChars {
		limit = maxWebRangeChars
	}
	// Any supplied `limit` means a slice, even one clamped: presence is the signal.
	rangeRequest := offset > 0 || args.has("limit")

	// A cached body skips a second Firefox launch; a repeat without a range gets
	// the same bytes the first call returned.
	if sess := a.getSession(sid); sess != nil {
		if body, ok := sess.recallWebBody(targetURL); ok {
			tcId := a.StartToolCall(ctx, sid, "Web Read (cached): "+targetURL, "search", nil)
			if rangeRequest {
				slice := sliceWebBody(body, offset, limit)
				a.CompleteToolCallTitled(ctx, sid, tcId,
					fmt.Sprintf("Web Read (cached): %s [%d:%d of %d]", targetURL, offset, offset+len(slice), len(body)),
					[]ToolCallContent{TextContent(fmt.Sprintf("returned %d chars from cache (offset %d, body %d)", len(slice), offset, len(body)))})
				a.logSession(sid, "WEB", "range from cache: url=%s offset=%d limit=%d returned=%d body=%d", targetURL, offset, limit, len(slice), len(body))
				return slice, false
			}
			out := firstPage(body)
			a.CompleteToolCallTitled(ctx, sid, tcId,
				"Web Read (cached): "+targetURL,
				[]ToolCallContent{TextContent(fmt.Sprintf("returned cached result (%d chars, no re-fetch)", len(out)))})
			a.logSession(sid, "WEB", "result from cache: url=%s returned=%d", targetURL, len(out))
			return out, false
		}
	}

	a.logSession(sid, "WEB", "open URL: %s", targetURL)

	tcId := a.StartToolCall(ctx, sid, "Web Read: "+targetURL, "search", nil)

	port := nextBrowserPort()
	browser, err := StartBrowser(ctx, port, targetURL)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "error starting browser: " + err.Error(), false
	}
	defer browser.Close()
	tabID := browser.initialTab

	text, err := browser.EvalJS(ctx, tabID, "document.body.innerText")
	if err != nil {
		a.logSession(sid, "WEB", "page text error: %s", err.Error())
		a.FailToolCall(ctx, sid, tcId, "page text error: "+err.Error())
		return "error getting page text: " + err.Error(), false
	}

	// Cache the full text BEFORE truncation, so range reads can reach past the cap.
	if sess := a.getSession(sid); sess != nil {
		sess.rememberWebBody(targetURL, text)
	}

	// The card is completed, not failed, even on ❌: the model may retry, and Zed
	// would bury a failed card collapsed in red.
	icon, msg := "✅", "Page loaded"
	if issue := pageIssue(text); issue != "" {
		icon, msg = "❌", issue
	}
	a.CompleteToolCallTitled(ctx, sid, tcId, "Web Read: "+targetURL+" "+icon,
		[]ToolCallContent{TextContent(icon + " " + msg)})

	a.logSession(sid, "WEB", "page text (%d chars):\n%s", len(text), text)

	if rangeRequest {
		return sliceWebBody(text, offset, limit), false
	}
	return firstPage(text), false
}

func firstPage(body string) string {
	if len(body) > maxRawPageChars {
		return clipUTF8(body, maxRawPageChars) + "\n... (truncated)"
	}
	return body
}

func sliceWebBody(body string, offset, limit int) string {
	if offset >= len(body) {
		return ""
	}
	for offset < len(body) && !utf8.RuneStart(body[offset]) {
		offset++
	}
	return clipUTF8(body[offset:], limit)
}

var botWallPatterns = []struct {
	needle string
	label  string
}{
	{"just a moment", "Cloudflare interstitial"},
	{"checking your browser", "Cloudflare interstitial"},
	{"verify you are human", "captcha challenge"},
	{"verifying you are human", "captcha challenge"},
	{"are you human", "captcha challenge"},
	{"press and hold", "anti-bot challenge"},
	{"please enable javascript and cookies", "anti-bot wall"},
	{"attention required", "anti-bot wall"},
	{"access denied", "access denied"},
	{"403 forbidden", "403 forbidden"},
	{"too many requests", "rate limited"},
}

func pageIssue(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return "empty page (load failed?)"
	}
	lower := strings.ToLower(trimmed)
	for _, p := range botWallPatterns {
		if strings.Contains(lower, p.needle) {
			return p.label
		}
	}
	if len(trimmed) < 200 {
		return "very short content (load failed or blocked?)"
	}
	return ""
}

type ddgResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

// extractDDGResults drops duplicate URLs: DDG renders the same anchor in several
// sections (organic, "people also viewed", mobile carousel).
func extractDDGResults(ctx context.Context, b *Browser, contextID string) ([]ddgResult, error) {
	js := `JSON.stringify(
		Array.from(document.querySelectorAll('a[data-testid="result-title-a"]')).map(a => {
			const root = a.closest('article') || a.closest('[data-testid="result"]') || a.parentElement;
			const snip = root && (
				root.querySelector('[data-result="snippet"]') ||
				root.querySelector('span[data-testid="result-snippet"]') ||
				root.querySelector('.result__snippet')
			);
			return {
				title: (a.innerText || "").trim(),
				url: a.href,
				snippet: snip ? (snip.innerText || "").trim() : ""
			};
		}).filter(r => r.url.startsWith('http'))
	)`
	raw, err := b.EvalJS(ctx, contextID, js)
	if err != nil {
		return nil, err
	}
	var all []ddgResult
	if err := json.Unmarshal([]byte(raw), &all); err != nil {
		return nil, err
	}
	seen := make(map[string]bool, len(all))
	out := make([]ddgResult, 0, len(all))
	for _, r := range all {
		if seen[r.URL] {
			continue
		}
		seen[r.URL] = true
		out = append(out, r)
	}
	return out, nil
}

func formatDDGResults(rs []ddgResult) string {
	var b strings.Builder
	for i, r := range rs {
		title := r.Title
		if title == "" {
			title = "(no title)"
		}
		fmt.Fprintf(&b, "%d. %s\n   %s\n", i+1, title, r.URL)
		if r.Snippet != "" {
			fmt.Fprintf(&b, "   %s\n", r.Snippet)
		}
	}
	return b.String()
}
