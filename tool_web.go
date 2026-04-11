package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
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
			"description": "Search the web using DuckDuckGo. Opens the top results in Firefox tabs for the user to review, then extracts the text content.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"query"},
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "The search query",
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

		tcId := a.StartToolCall(ctx, sid, "Searching: "+query, "search", nil)

		searchURL := "https://duckduckgo.com/?q=" + url.QueryEscape(query)

		// Each search gets its own browser instance.
		port := nextBrowserPort()
		browser, err := StartBrowser(ctx, port, searchURL)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error starting browser: " + err.Error()
		}
		searchTab := browser.initialTab
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}

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

		// Open each result in a new tab.
		var tabs []string
		for _, link := range links {
			tabID, err := browser.OpenTab(ctx, link)
			if err != nil {
				continue
			}
			tabs = append(tabs, tabID)
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{
			TextContent(fmt.Sprintf("Opened %d results in Firefox:\n%s", len(tabs), strings.Join(links[:len(tabs)], "\n"))),
		})

		// Ask user to review.
		askId := a.StartToolCall(ctx, sid, "Review the tabs in Firefox, then confirm", "think", nil)
		ok, askErr := a.conn.AskYesNo(ctx, sid, askId, "Get results", "Cancel")
		if askErr != nil || !ok {
			a.CompleteToolCall(ctx, sid, askId, []ToolCallContent{TextContent("Cancelled")})
			for _, tabID := range tabs {
				browser.CloseTab(ctx, tabID)
			}
			browser.Close()
			return "user cancelled search"
		}
		a.CompleteToolCall(ctx, sid, askId, []ToolCallContent{TextContent("Extracting...")})

		// Extract text from all tabs, then close them all.
		type tabResult struct {
			link string
			text string
			err  error
		}
		extracted := make([]tabResult, len(tabs))
		for i, tabID := range tabs {
			text, err := browser.PageText(ctx, tabID)
			extracted[i] = tabResult{link: links[i], text: text, err: err}
		}

		// Close all tabs and the browser.
		for _, tabID := range tabs {
			browser.CloseTab(ctx, tabID)
		}
		browser.Close()

		// Summarize each page in parallel using the summary LLM.
		summaries := make([]string, len(extracted))
		for i, r := range extracted {
			if r.err != nil {
				summaries[i] = fmt.Sprintf("error: %s", r.err.Error())
			}
		}
		parallel(len(extracted), func(i int) {
			if extracted[i].err != nil {
				return
			}
			summaries[i] = a.summarizePage(ctx, query, extracted[i].link, extracted[i].text)
		})

		var results strings.Builder
		for i, r := range extracted {
			fmt.Fprintf(&results, "=== Result %d: %s ===\n%s\n\n", i+1, r.link, summaries[i])
		}

		if results.Len() == 0 {
			return "error: failed to extract text from any search results"
		}
		return results.String()
	}})

	RegisterTool(Tool{ReadOnly: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "web_read",
			"description": "Open a URL in Firefox and extract the text content. The user will review the page before results are returned.",
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
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) string {
			args := parseArgs(rawArgs)
		targetURL := args["url"]
		if targetURL == "" {
			return "error: url is required"
		}

		tcId := a.StartToolCall(ctx, sid, "Opening: "+targetURL, "search", nil)

		port := nextBrowserPort()
		browser, err := StartBrowser(ctx, port, targetURL)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error starting browser: " + err.Error()
		}
		tabID := browser.initialTab
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return "error: " + err.Error()
		}

		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Page opened in Firefox.")})

		// Ask user to review.
		askId := a.StartToolCall(ctx, sid, "Review the page in Firefox, then confirm", "think", nil)
		ok, askErr := a.conn.AskYesNo(ctx, sid, askId, "Get content", "Cancel")
		if askErr != nil || !ok {
			a.CompleteToolCall(ctx, sid, askId, []ToolCallContent{TextContent("Cancelled")})
			browser.CloseTab(ctx, tabID)
			browser.Close()
			return "user cancelled"
		}
		a.CompleteToolCall(ctx, sid, askId, []ToolCallContent{TextContent("Extracting...")})

		text, err := browser.PageText(ctx, tabID)
		browser.CloseTab(ctx, tabID)
		browser.Close()
		if err != nil {
			return "error getting page text: " + err.Error()
		}

		return a.summarizePage(ctx, "content of "+targetURL, targetURL, text)
	}})
}

// summarizePage uses the summary LLM to extract only the relevant information from a web page.
func (a *agent) summarizePage(ctx context.Context, query, url, pageText string) string {
	conn := a.settings.SummaryLLM()
	if conn == nil {
		// No summary LLM available, truncate raw text.
		const maxLen = 2000
		if len(pageText) > maxLen {
			return pageText[:maxLen] + "\n... (truncated)"
		}
		return pageText
	}

	// Truncate input to avoid overwhelming the summary LLM.
	const maxInput = 8000
	if len(pageText) > maxInput {
		pageText = pageText[:maxInput]
	}

	prompt := fmt.Sprintf(
		"The user searched for: %q\n\nExtract ONLY the information relevant to this search from the following web page. Be concise and factual. Include specific versions, dates, and facts. Skip navigation, menus, ads, and unrelated content. Max 300 words.\n\nURL: %s\n\n%s",
		query, url, pageText,
	)

	messages := []llmMessage{{Role: "user", Content: prompt}}
	summary, err := a.llmSimple(ctx, conn, messages)
	if err != nil {
		const maxLen = 2000
		if len(pageText) > maxLen {
			return pageText[:maxLen] + "\n... (truncated)"
		}
		return pageText
	}
	return strings.TrimSpace(summary)
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
