package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"sync/atomic"
	"time"
)

const maxResultTabs = 10

var browserPortCounter atomic.Int32

func init() {
	browserPortCounter.Store(9222)
}

func nextBrowserPort() int {
	return int(browserPortCounter.Add(1))
}

func init() {
	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "web_search",
			"description": "Search the web using DuckDuckGo and open the top results in Firefox tabs for the user to review. Returns the list of result URLs. Follow up with web_read (summarized) or web_read_raw (raw text — for finding a specific link/string on the page).",
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
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
			args := parseArgs(rawArgs)
		query := args["query"]
		if query == "" {
			return "error: query is required"
		}

		a.logSession(sid, "WEB", "search query: %s", query)

		tcId := a.StartToolCall(ctx, sid, "DuckDuckGo: "+query, "search", nil)

		searchURL := "https://duckduckgo.com/?q=" + url.QueryEscape(query)

		// Each search gets its own browser instance.
		port := nextBrowserPort()
		browser, err := StartBrowser(ctx, port, searchURL)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error starting browser: " + err.Error()
		}
		defer browser.Close()
		searchTab := browser.initialTab

		// Wait for DDG results to render.
		var links []string
		for i := 0; i < 30; i++ {
			links, _ = extractDDGLinks(ctx, browser, searchTab)
			if len(links) > 0 {
				break
			}
			time.Sleep(500 * time.Millisecond)
		}
		browser.CloseTab(ctx, searchTab)
		if len(links) == 0 {
			a.FailToolCall(ctx, sid, tcId, "no search results found")
			return "error: no search results found"
		}

		if len(links) > maxResultTabs {
			links = links[:maxResultTabs]
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{
			TextContent(fmt.Sprintf("%d hits, opening in tabs", len(links))),
		})

		// Open each result in its own tab — emit a card per URL so the user
		// sees which sites are being fetched, not just the final list. Peek
		// the rendered text and flag bot walls / captchas so the LLM (and
		// user) know which results aren't trustworthy.
		var opened []string
		issues := map[string]string{}
		for _, link := range links {
			openId := a.StartToolCall(ctx, sid, "Opening: "+link, "search", nil)
			tab, err := browser.OpenTab(ctx, link)
			if err != nil {
				a.FailToolCall(ctx, sid, openId, err.Error())
				continue
			}
			var content []ToolCallContent
			if text, terr := browser.PageText(ctx, tab); terr == nil {
				if issue := pageIssue(text); issue != "" {
					issues[link] = issue
					content = []ToolCallContent{TextContent("🟡 " + issue)}
				}
			}
			a.CompleteToolCall(ctx, sid, openId, content)
			opened = append(opened, link)
		}

		a.logSession(sid, "WEB", "opened tabs (%d):\n%s", len(opened), strings.Join(opened, "\n"))

		// Ask user to review.
		askId := a.StartToolCall(ctx, sid, "Review the tabs in Firefox, then confirm", "think", nil)
		ok, askErr := a.askYesNoAuto(ctx, sid, askId, "OK", "Cancel", true)
		if askErr != nil {
			a.FailToolCall(ctx, sid, askId, askErr.Error())
			return "error asking permission: " + askErr.Error()
		}
		if !ok {
			a.CompleteToolCall(ctx, sid, askId, []ToolCallContent{TextContent("Cancelled")})
			return "user cancelled search"
		}
		a.CompleteToolCall(ctx, sid, askId, []ToolCallContent{TextContent("Returning URLs")})

		var results strings.Builder
		for i, link := range opened {
			if issue := issues[link]; issue != "" {
				fmt.Fprintf(&results, "%d. %s   [🟡 %s]\n", i+1, link, issue)
			} else {
				fmt.Fprintf(&results, "%d. %s\n", i+1, link)
			}
		}
		return results.String()
	}})

	RegisterTool(Tool{ReadOnly: true, Def: webReadDef(
		"web_read",
		"Open a URL in Firefox and return a concise summary of the page content. Use this for unstructured information (what does the page say about X). The user will review the page before the summary is returned.",
	), Execute: makeWebRead(true)})

	RegisterTool(Tool{ReadOnly: true, Def: webReadDef(
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
				},
			},
		},
	}
}

const maxRawPageChars = 30000

func makeWebRead(summarize bool) func(context.Context, *agent, SessionId, string) string {
	return func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
		args := parseArgs(rawArgs)
		targetURL := args["url"]
		if targetURL == "" {
			return "error: url is required"
		}

		a.logSession(sid, "WEB", "open URL: %s", targetURL)

		tcId := a.StartToolCall(ctx, sid, "Opening: "+targetURL, "search", nil)

		port := nextBrowserPort()
		browser, err := StartBrowser(ctx, port, targetURL)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error starting browser: " + err.Error()
		}
		defer browser.Close()
		tabID := browser.initialTab

		openMsg := "Page opened in Firefox."
		if peek, perr := browser.PageText(ctx, tabID); perr == nil {
			if issue := pageIssue(peek); issue != "" {
				openMsg = "🟡 " + issue
			}
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(openMsg)})

		askId := a.StartToolCall(ctx, sid, "Review the page in Firefox, then confirm", "think", nil)
		ok, askErr := a.askYesNoAuto(ctx, sid, askId, "Get content", "Cancel", true)
		if askErr != nil {
			a.FailToolCall(ctx, sid, askId, askErr.Error())
			return "error asking permission: " + askErr.Error()
		}
		if !ok {
			a.CompleteToolCall(ctx, sid, askId, []ToolCallContent{TextContent("Cancelled")})
			return "user cancelled"
		}
		a.CompleteToolCall(ctx, sid, askId, []ToolCallContent{TextContent("Extracting...")})

		text, err := browser.PageText(ctx, tabID)
		if err != nil {
			a.logSession(sid, "WEB", "page text error: %s", err.Error())
			return "error getting page text: " + err.Error()
		}

		a.logSession(sid, "WEB", "page text (%d chars):\n%s", len(text), stripHTMLAttrs(text))

		if summarize {
			return a.summarizePage(ctx, sid, "content of "+targetURL, targetURL, text)
		}
		if len(text) > maxRawPageChars {
			text = text[:maxRawPageChars] + "\n... (truncated)"
		}
		return text
	}
}

// summarizePage uses the execute LLM to extract only the relevant information
// from a web page. sid scopes the per-session debug log.
func (a *agent) summarizePage(ctx context.Context, sid SessionId, query, url, pageText string) string {
	conn := a.pickAvailable(ctx, "execute", a.llmTier(sid))

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
	summary, err := a.llmSimple(ctx, sid, conn, messages)
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

// extractDDGLinks extracts result URLs from a DuckDuckGo search results page.
func extractDDGLinks(ctx context.Context, b *Browser, contextID string) ([]string, error) {
	js := `JSON.stringify(
		Array.from(document.querySelectorAll('a[data-testid="result-title-a"]'))
			.map(a => a.href)
			.filter(h => h.startsWith('http'))
	)`
	raw, err := b.EvalJS(ctx, contextID, js)
	if err != nil {
		return nil, err
	}

	var links []string
	json.Unmarshal([]byte(raw), &links)
	return links, nil
}
