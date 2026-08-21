package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// session_insights: mechanical failure analytics over session_*.log files.
//
// Session logs routinely dwarf a small model's context window, and asking a
// 27B model to grep-and-skim them produced shallow, lucky-dip evidence. This tool does the analytics in CODE — parse every
// requested log, extract the failure signals mechanically, return a digest of
// a few KB — so the model spends its context on picking the top problems, not
// on being a log parser. Same lesson as the skills="auto" code-level levers:
// what a weak model won't do reliably from instructions must be done for it.

// logEntry is one `=== <timestamp> [<tag>] ===` block of a session log (the
// exact format logSession writes and logy parses).
type logEntry struct {
	time string
	tag  string
	body string
}

var logHeaderRE = regexp.MustCompile(`^=== (\S+) \[(.+)\] ===$`)

func parseLogEntries(data string) []logEntry {
	var entries []logEntry
	var cur *logEntry
	for _, line := range strings.Split(data, "\n") {
		if m := logHeaderRE.FindStringSubmatch(line); m != nil {
			if cur != nil {
				cur.body = strings.TrimRight(cur.body, "\n")
				entries = append(entries, *cur)
			}
			cur = &logEntry{time: m[1], tag: m[2]}
			continue
		}
		if cur != nil {
			cur.body += line + "\n"
		}
	}
	if cur != nil {
		cur.body = strings.TrimRight(cur.body, "\n")
		entries = append(entries, *cur)
	}
	return entries
}

// failResultRE spots a tool RESULT that reports failure. Matched against the
// first line of the result content only — matching deep in a long output
// would flag results that merely mention the word.
var failResultRE = regexp.MustCompile(`(?i)^\s*(error|err:|❌|failed|exit (code|status) [1-9]|command not found|no such file|cannot |panic:|fatal)`)

// toolCallStat aggregates identical tool calls (same name + same arguments).
type toolCallStat struct {
	name  string
	args  string
	count int
	fails int
	// firstErr is the first line of the first failing result, for the digest.
	firstErr string
}

// sessionDigest is the mechanical summary of one session log.
type sessionDigest struct {
	file       string
	firstTime  string
	lastTime   string
	sizeBytes  int
	requests   int // LLM round-trips
	userTurns  int
	recovers   []string // RECOVER entries, verbatim first lines
	transport  []string // transport/HTTP error entries
	replans    int      // "replan" mentions in message contents
	buildRuns  int      // run_command calls that look like builds
	testRuns   int      // run_command calls that look like test runs
	calls      map[string]*toolCallStat
	parseNotes []string
}

// digestLog computes the mechanical failure signals from one session log. Tool
// calls and their results live INSIDE the logged REQUEST bodies (the full
// OpenAI messages array), so the LAST request of the file — the most complete
// history snapshot — is parsed for call/result pairing, while RECOVER and
// transport entries are collected across all blocks.
func digestLog(path string, data string) sessionDigest {
	d := sessionDigest{file: filepath.Base(path), sizeBytes: len(data), calls: map[string]*toolCallStat{}}
	entries := parseLogEntries(data)
	if len(entries) == 0 {
		d.parseNotes = append(d.parseNotes, "no log entries parsed")
		return d
	}
	d.firstTime, d.lastTime = entries[0].time, entries[len(entries)-1].time

	var lastReq string
	for _, e := range entries {
		switch {
		case strings.HasSuffix(e.tag, " REQUEST"):
			d.requests++
			lastReq = e.body
		case e.tag == "RECOVER":
			d.recovers = append(d.recovers, firstLine(e.body))
		case strings.HasPrefix(e.body, "[transport error]") || strings.HasPrefix(e.body, "[HTTP "):
			d.transport = append(d.transport, firstLine(e.body))
		}
	}
	if lastReq == "" {
		d.parseNotes = append(d.parseNotes, "no REQUEST entry found")
		return d
	}

	var req struct {
		Messages []struct {
			Role      string          `json:"role"`
			Content   json.RawMessage `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
			ToolCallID string `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(lastReq), &req); err != nil {
		d.parseNotes = append(d.parseNotes, "last REQUEST unparseable: "+err.Error())
		return d
	}

	callByID := map[string]*toolCallStat{} // tool_call_id → aggregate bucket
	for _, m := range req.Messages {
		content := flattenContent(m.Content)
		switch m.Role {
		case "user":
			d.userTurns++
		case "assistant":
			for _, tc := range m.ToolCalls {
				key := tc.Function.Name + "\x00" + strings.TrimSpace(tc.Function.Arguments)
				st := d.calls[key]
				if st == nil {
					st = &toolCallStat{name: tc.Function.Name, args: strings.TrimSpace(tc.Function.Arguments)}
					d.calls[key] = st
				}
				st.count++
				callByID[tc.ID] = st
				if tc.Function.Name == "run_command" {
					low := strings.ToLower(tc.Function.Arguments)
					if strings.Contains(low, "test") {
						d.testRuns++
					} else if strings.Contains(low, "build") || strings.Contains(low, "compile") || strings.Contains(low, "vet") {
						d.buildRuns++
					}
				}
			}
		case "tool":
			if failResultRE.MatchString(firstLine(content)) {
				if st := callByID[m.ToolCallID]; st != nil {
					st.fails++
					if st.firstErr == "" {
						st.firstErr = firstLine(content)
					}
				}
			}
		}
		if strings.Contains(strings.ToLower(content), "replan") {
			d.replans++
		}
	}
	return d
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return s
}

// flattenContent renders a message content field (string or multimodal parts)
// to plain text.
func flattenContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			b.WriteString(p.Text)
		}
		return b.String()
	}
	return ""
}

// render writes one session's digest as compact markdown, capping list lengths
// so N sessions still fit a small context.
func (d sessionDigest) render(b *strings.Builder) {
	fmt.Fprintf(b, "## %s (%dKB, %s → %s)\n", d.file, d.sizeBytes/1024, d.firstTime, d.lastTime)
	fmt.Fprintf(b, "%d LLM round-trips, %d user turns, %d build-ish vs %d test-ish run_command calls, %d replan mentions\n",
		d.requests, d.userTurns, d.buildRuns, d.testRuns, d.replans)
	for _, n := range d.parseNotes {
		fmt.Fprintf(b, "- note: %s\n", n)
	}

	// Loops: identical call repeated. The #1 waste signal for a small model.
	var stats []*toolCallStat
	for _, st := range d.calls {
		stats = append(stats, st)
	}
	sort.Slice(stats, func(i, j int) bool { return stats[i].count > stats[j].count })
	printed := 0
	for _, st := range stats {
		if st.count < 2 || printed >= 8 {
			break
		}
		fmt.Fprintf(b, "- LOOP: %s(%s) called %d× with identical args\n", st.name, truncate(st.args, 100), st.count)
		printed++
	}
	// Failing calls.
	sort.Slice(stats, func(i, j int) bool { return stats[i].fails > stats[j].fails })
	printed = 0
	for _, st := range stats {
		if st.fails == 0 || printed >= 8 {
			break
		}
		fmt.Fprintf(b, "- FAIL ×%d: %s(%s) → %s\n", st.fails, st.name, truncate(st.args, 80), truncate(st.firstErr, 100))
		printed++
	}
	for i, r := range d.recovers {
		if i >= 6 {
			fmt.Fprintf(b, "- RECOVER: … %d more\n", len(d.recovers)-i)
			break
		}
		fmt.Fprintf(b, "- RECOVER: %s\n", truncate(r, 140))
	}
	for i, tr := range d.transport {
		if i >= 4 {
			fmt.Fprintf(b, "- TRANSPORT: … %d more\n", len(d.transport)-i)
			break
		}
		fmt.Fprintf(b, "- TRANSPORT: %s\n", truncate(tr, 140))
	}
	if d.testRuns == 0 && d.buildRuns > 0 {
		fmt.Fprintf(b, "- VERIFY GAP: %d build-ish commands but ZERO test-ish commands this session\n", d.buildRuns)
	}
	b.WriteString("\n")
}

const insightsReplyCap = 12000 // keep the digest small-model friendly

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "session_insights",
			"description": "Mechanically analyze codehalter session logs (.codehalter/session_*.log) and return a compact failure digest: repeated identical tool calls (loops), failing tool calls with their first error line, RECOVER events, transport errors, build-vs-test balance, replan mentions. Use this INSTEAD of reading or grepping raw session logs — logs are usually far larger than the context window. Read a raw log only to zoom into one specific spot the digest points at.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"sessions": map[string]any{
						"type":        "integer",
						"description": "How many of the most recent session logs to analyze (default 3, max 10).",
					},
					"file": map[string]any{
						"type":        "string",
						"description": "Analyze one specific log by filename (e.g. session_abc123.log) instead of the most recent ones.",
					},
				},
			},
		},
	}, Execute: insightsExecute})
}

func insightsExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	sess := a.getSession(sid)
	if sess == nil {
		return "error: no session", true
	}
	args := parseArgs(rawArgs)
	dir := filepath.Join(sess.Cwd, ".codehalter")

	var paths []string
	if f := strings.TrimSpace(args.str("file")); f != "" {
		if strings.ContainsAny(f, `/\`) {
			return "error: file must be a bare session_*.log filename", true
		}
		paths = []string{filepath.Join(dir, f)}
	} else {
		n := 3
		if p, ok := args.num("sessions"); ok && p >= 1 {
			n = min(p, 10)
		}
		matches, _ := filepath.Glob(filepath.Join(dir, "session_*.log"))
		sort.Slice(matches, func(i, j int) bool { // newest first, by mtime
			si, _ := os.Stat(matches[i])
			sj, _ := os.Stat(matches[j])
			if si == nil || sj == nil {
				return matches[i] > matches[j]
			}
			return si.ModTime().After(sj.ModTime())
		})
		if len(matches) > n {
			matches = matches[:n]
		}
		paths = matches
	}
	if len(paths) == 0 {
		return "No session logs found in .codehalter/ — nothing to analyze.", false
	}

	var b strings.Builder
	b.WriteString("Mechanical session-log digest (loops, failures, recoveries). Pick evidence from here; only read a raw log to zoom into a specific spot.\n\n")
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			fmt.Fprintf(&b, "## %s: unreadable: %v\n\n", filepath.Base(p), err)
			continue
		}
		digestLog(p, string(data)).render(&b)
		if b.Len() > insightsReplyCap {
			b.WriteString("… digest truncated (size cap) — analyze fewer sessions or one specific file.\n")
			break
		}
	}
	return b.String(), false
}
