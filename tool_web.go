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
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
)

// ---------------------------------------------------------------------------
// Browser client (WebDriver BiDi over WebSocket → headless Firefox)
// ---------------------------------------------------------------------------

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

	// Start on about:blank, not initialURL: the CLI-driven load gives us no
	// "load complete" signal, so PageText races the renderer and snapshots an
	// empty body on fast servers. We Navigate() to initialURL below with
	// wait:"complete" once BiDi is up.
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

	// Get the initial tab's context ID (Firefox opened on about:blank).
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

	// Drive the real navigation through BiDi so we get a load-complete barrier.
	// Cap at 10s: wait:"complete" can hang on pages that never fire the load
	// event (long-poll chats, sites that keep streaming). On timeout we ignore
	// the error and continue — PageText on a partially-loaded body still beats
	// failing the tool call.
	if b.initialTab != "" && initialURL != "" {
		navCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		err := b.Navigate(navCtx, b.initialTab, initialURL)
		cancel()
		if err != nil && navCtx.Err() == nil {
			b.Close()
			return nil, fmt.Errorf("navigating to %s: %w", initialURL, err)
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
		if err := json.Unmarshal(raw, &resp); err != nil {
			return nil, fmt.Errorf("decoding bidi response: %w", err)
		}
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
	if err := json.Unmarshal(result, &ctxResult); err != nil {
		return "", fmt.Errorf("decoding create-tab response: %w", err)
	}

	err = b.Navigate(ctx, ctxResult.Context, url)
	return ctxResult.Context, err
}

// CloseTab closes a browsing context.
func (b *Browser) CloseTab(ctx context.Context, contextID string) {
	if _, err := b.Send(ctx, "browsingContext.close", map[string]any{
		"context": contextID,
	}); err != nil {
		slog.Debug("CloseTab failed", "context", contextID, "err", err)
	}
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
	if err := json.Unmarshal(result, &evalResult); err != nil {
		return "", fmt.Errorf("parsing browser eval result: %w", err)
	}
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
	return "", fmt.Errorf("firefox not found; set FIREFOX_PATH")
}

// ---------------------------------------------------------------------------
// Tool wrappers
// ---------------------------------------------------------------------------

const maxWebSearchResults = 10

var browserPortCounter atomic.Int32

func init() {
	browserPortCounter.Store(9222)
}

func nextBrowserPort() int {
	return int(browserPortCounter.Add(1))
}

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "web_search",
			"description": "Search the web with DuckDuckGo. Returns up to 10 results as a numbered list (title, URL, snippet) for you to triage — does NOT fetch page content. Snippets alone are NOT enough to answer factual questions: you MUST follow up by calling web_read (summarized) or web_read_raw (raw text — for finding a specific link/string verbatim) on at least one (ideally 1-3) of the most promising URLs. Skipping web_read is only acceptable if every result is clearly off-topic, in which case you should refine the query and search again.",
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
		query := args["query"]
		if query == "" {
			return "error: query is required", false
		}

		a.logSession(sid, "WEB", "search query: %s", query)

		tcId := a.StartToolCall(ctx, sid, "DuckDuckGo: "+query, "search", nil)

		searchURL := "https://duckduckgo.com/?q=" + url.QueryEscape(query)

		// Each search gets its own browser instance.
		port := nextBrowserPort()
		browser, err := StartBrowser(ctx, port, searchURL)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error starting browser: " + err.Error(), false
		}
		defer browser.Close()
		searchTab := browser.initialTab

		// Wait for DDG results to render. Per-iteration errors are transient
		// (page not rendered yet), so we keep polling — but remember the last
		// one so a consistent failure (browser/JS error) surfaces in the result
		// instead of being indistinguishable from "DDG returned nothing".
		var results []ddgResult
		var extractErr error
		for range 30 {
			results, extractErr = extractDDGResults(ctx, browser, searchTab)
			if len(results) > 0 {
				break
			}
			time.Sleep(500 * time.Millisecond)
		}
		browser.CloseTab(ctx, searchTab)
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
		// Surface the list inline in the chat too, so the user can see the
		// URLs and snippets without expanding the tool card.
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "\n" + formatted + "\n"}})
		a.logSession(sid, "WEB", "results (%d):\n%s", len(results), formatted)
		return formatted, false
	}})

	RegisterTool(Tool{Def: webReadDef(
		"web_read",
		"Open a URL in Firefox and return a concise summary of the page content. Use this for unstructured information (what does the page say about X). The user will review the page before the summary is returned.",
	), Execute: makeWebRead(true)})

	RegisterTool(Tool{Def: webReadDef(
		"web_read_raw",
		"Open a URL in Firefox and return the raw extracted text (truncated). Use this when summarization would lose precision: finding a specific download URL on the page, exact version numbers, code snippets, or any string that must be preserved verbatim. The user will review the page before the text is returned.",
	), Execute: makeWebRead(false)})
}

func webReadDef(name, description string) map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        name,
			"description": description,
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
	}
}

const (
	// maxRawPageChars caps the bytes returned by web_read_raw on the FIRST
	// fetch (before truncateForLLM further compresses for the model). Full
	// body is still cached so range reads can dip past this.
	maxRawPageChars = 30000
	// maxWebRangeChars caps a single offset/limit slice from the cached body
	// so a "view more" can't dump megabytes back into the prefix at once.
	maxWebRangeChars = 8000
)

func makeWebRead(summarize bool) func(context.Context, *agent, string, string) (string, bool) {
	return func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		args := parseArgs(rawArgs)
		targetURL := args["url"]
		if targetURL == "" {
			return "error: url is required", false
		}
		offset, _ := strconv.Atoi(args["offset"])
		if offset < 0 {
			offset = 0
		}
		limit, _ := strconv.Atoi(args["limit"])
		if limit <= 0 || limit > maxWebRangeChars {
			limit = maxWebRangeChars
		}
		rangeRequest := offset > 0 || args["limit"] != ""

		// Range request hits the cache first — no second HTTP round-trip when
		// the page was fetched earlier in this session. Cache miss falls
		// through to the regular fetch path so the model can still get a slice
		// (it just costs the fetch the first time).
		if rangeRequest {
			if sess := a.getSession(sid); sess != nil {
				if body, ok := sess.recallWebBody(targetURL); ok {
					slice := sliceWebBody(body, offset, limit)
					tcId := a.StartToolCall(ctx, sid, "Web Read (cached): "+targetURL, "search", nil)
					a.CompleteToolCallTitled(ctx, sid, tcId,
						fmt.Sprintf("Web Read (cached): %s [%d:%d of %d]", targetURL, offset, offset+len(slice), len(body)),
						[]ToolCallContent{TextContent(fmt.Sprintf("returned %d chars from cache (offset %d, body %d)", len(slice), offset, len(body)))})
					a.logSession(sid, "WEB", "range from cache: url=%s offset=%d limit=%d returned=%d body=%d", targetURL, offset, limit, len(slice), len(body))
					return slice, false
				}
			}
		} else {
			// No-range repeat on the same URL+mode: return the previously-rendered
			// output verbatim. Small models often re-ask for the same URL within
			// one phase (or across plan→execute); skipping the fetch + summarize
			// here saves the dominant cost (Firefox launch + page load is 30-60s,
			// summarize is another LLM round-trip). Identical bytes are also nice
			// to the prefix cache if the second call shows up in the same prompt.
			if sess := a.getSession(sid); sess != nil {
				if cached, ok := sess.recallWebResult(targetURL, summarize); ok {
					tcId := a.StartToolCall(ctx, sid, "Web Read (cached): "+targetURL, "search", nil)
					a.CompleteToolCallTitled(ctx, sid, tcId,
						"Web Read (cached): "+targetURL,
						[]ToolCallContent{TextContent(fmt.Sprintf("returned cached result (%d chars, no re-fetch)", len(cached)))})
					a.logSession(sid, "WEB", "result from cache: url=%s summarize=%v returned=%d", targetURL, summarize, len(cached))
					return cached, false
				}
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

		text, err := browser.PageText(ctx, tabID)
		if err != nil {
			a.logSession(sid, "WEB", "page text error: %s", err.Error())
			a.FailToolCall(ctx, sid, tcId, "page text error: "+err.Error())
			return "error getting page text: " + err.Error(), false
		}

		// Cache the full extracted text BEFORE summarization / raw truncation.
		// Later offset/limit calls slice from this — they should be able to
		// reach past the raw 30k cap or the summary's compression.
		if sess := a.getSession(sid); sess != nil {
			sess.rememberWebBody(targetURL, text)
		}

		// Binary success/failure marker after the URL: ✅ when the body looks
		// like real content, ❌ when pageIssue flags a load failure, bot wall,
		// or content too thin to use. The card itself is still marked
		// completed (not failed) because the model can decide to retry with
		// web_read_raw or fall back to web_search; we don't want Zed to bury
		// the result in a red-collapsed card.
		icon, msg := "✅", "Page loaded"
		if issue := pageIssue(text); issue != "" {
			icon, msg = "❌", issue
		}
		a.CompleteToolCallTitled(ctx, sid, tcId, "Web Read: "+targetURL+" "+icon,
			[]ToolCallContent{TextContent(icon + " " + msg)})

		a.logSession(sid, "WEB", "page text (%d chars):\n%s", len(text), stripHTMLAttrs(text))

		// A range request that fell through (cache miss) returns a slice of
		// the freshly-fetched body — honor offset/limit even on the first
		// call so the model gets exactly what it asked for. Range slices are
		// NOT memoized in webResults (offset/limit vary), but the underlying
		// body is in webBodies so the next range call is still free.
		if rangeRequest {
			return sliceWebBody(text, offset, limit), false
		}
		var out string
		if summarize {
			out = a.summarizePage(ctx, sid, "content of "+targetURL, targetURL, text)
		} else {
			out = text
			if len(out) > maxRawPageChars {
				out = out[:maxRawPageChars] + "\n... (truncated)"
			}
		}
		if sess := a.getSession(sid); sess != nil {
			sess.rememberWebResult(targetURL, summarize, out)
		}
		return out, false
	}
}

// sliceWebBody returns up to `limit` characters of `body` starting at `offset`,
// clamping both ends so out-of-range arguments produce a sensible empty/last
// slice instead of a panic. The model can pass offset past the end (e.g. when
// it doesn't know the exact length) — we return "" rather than erroring so the
// model can correct on the next call.
func sliceWebBody(body string, offset, limit int) string {
	if offset >= len(body) {
		return ""
	}
	end := offset + limit
	if end > len(body) {
		end = len(body)
	}
	return body[offset:end]
}

// summarizePage uses the execute LLM to extract only the relevant information
// from a web page. sid scopes the per-session debug log. Routes via the
// session's pin: main → LLM[0], subagent → its pinned LLM[i] entry.
func (a *agent) summarizePage(ctx context.Context, sid string, query, url, pageText string) string {
	conn := a.connForSession(ctx, sid, "execute")

	// Truncate input to avoid overwhelming the LLM.
	const maxInput = 8000
	if len(pageText) > maxInput {
		pageText = pageText[:maxInput]
	}

	prompt := fmt.Sprintf(
		"The user searched for: %q\n\nExtract ONLY the information relevant to this search from the following web page. Be concise and factual. Include specific versions, dates, and facts. Skip navigation, menus, ads, and unrelated content. Max 300 words.\n\nURL: %s\n\n%s",
		query, url, pageText,
	)

	messages := []llmMessage{{Role: "user", Content: prompt}}
	summary, _, err := a.llmStream(ctx, sid, conn, messages, nil, nil, nil)
	if err != nil {
		const maxLen = 2000
		if len(pageText) > maxLen {
			return pageText[:maxLen] + "\n... (truncated)"
		}
		return pageText
	}
	return strings.TrimSpace(summary)
}

// HTML log cleanup: keep semantic structure (headings, lists, tables, code,
// anchors), drop layout chrome and binary blobs. Regex-based; not an HTML
// parser — good enough for log readability, not for security-sensitive use.

// dropBlockTags removes the opening tag, all content, and the closing tag.
// Used for elements whose body is non-text (CSS, JS, vector graphics) or
// universally noisy (forms aren't here because we unwrap them).
var dropBlockTags = []string{"script", "style", "svg", "noscript", "iframe", "canvas", "head"}

// dropVoidTags are self-closing or contentless tags we erase entirely.
var dropVoidTags = map[string]bool{
	"img": true, "br": true, "hr": true, "meta": true, "link": true,
	"base": true, "input": true, "source": true, "track": true, "area": true,
}

// unwrapTags lose their open/close markers but keep inner text.
var unwrapTags = map[string]bool{
	"div": true, "span": true, "nav": true, "header": true, "footer": true,
	"aside": true, "section": true, "article": true, "main": true,
	"body": true, "html": true, "figure": true, "figcaption": true,
	"picture": true, "label": true, "button": true, "form": true,
	"fieldset": true, "legend": true,
}

// dropBlockRes matches each noise-block element (one regex per tag, since
// RE2 has no backreferences). `(?is)` = case-insensitive + dotall.
var dropBlockRes = func() []*regexp.Regexp {
	out := make([]*regexp.Regexp, len(dropBlockTags))
	for i, t := range dropBlockTags {
		out[i] = regexp.MustCompile(`(?is)<` + t + `\b[^>]*>.*?</` + t + `\s*>`)
	}
	return out
}()

var tagRe = regexp.MustCompile(`(?i)<(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*)>`)
var hrefRe = regexp.MustCompile(`(?i)\shref\s*=\s*("[^"]*"|'[^']*'|\S+)`)

// stripHTMLAttrs collapses HTML to its skeleton: noise blocks gone, layout
// wrappers unwrapped, attributes dropped (except href on anchors). Aimed at
// keeping the per-session log readable when we eventually capture raw HTML.
func stripHTMLAttrs(s string) string {
	for _, re := range dropBlockRes {
		s = re.ReplaceAllString(s, "")
	}
	return tagRe.ReplaceAllStringFunc(s, func(match string) string {
		sub := tagRe.FindStringSubmatch(match)
		closing, tag, attrs := sub[1], strings.ToLower(sub[2]), sub[3]
		if dropVoidTags[tag] || unwrapTags[tag] {
			return ""
		}
		if tag == "a" && closing == "" {
			if href := hrefRe.FindString(attrs); href != "" {
				return "<a" + href + ">"
			}
		}
		return "<" + closing + tag + ">"
	})
}

// botWallPatterns are case-insensitive substrings that strongly suggest a
// page is a Cloudflare interstitial, captcha challenge, access denial, or
// rate-limit wall rather than the content the agent asked for. Matched by
// pageIssue against the rendered body text.
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

// pageIssue returns a short label when the rendered body looks like a bot
// wall, captcha, access denial, or failed load. Empty string means the page
// looks normal. Text-heuristic only; misses sophisticated walls, but catches
// the common Cloudflare/captcha/403 patterns that derail web_search runs.
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

// ddgResult is one row from a DuckDuckGo SERP.
type ddgResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

// extractDDGResults pulls title/URL/snippet for each result on a DDG SERP and
// drops duplicate URLs. DDG renders the same anchor in multiple sections
// (organic + "people also viewed" + mobile carousel) so the raw query returns
// the same href several times; we keep first-occurrence order.
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

// formatDDGResults renders the result list as a compact numbered text block
// for the LLM (and the chat panel) — title on one line, URL on the next,
// snippet (when present) indented underneath.
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
