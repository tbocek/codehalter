// logy converts a codehalter session_*.log into human-readable markdown that
// shows exactly what was sent to and received from the LLM. Each REQUEST is
// rendered as a DIFF against the previous request in the file: the shared
// message prefix (what a warm KV cache serves for free) is collapsed to one
// line, and only the appended or changed tail is printed in full. On a
// parallel=1 setup this mirrors how the server's prefix cache actually sees
// the traffic. Thinking tokens never appear in requests (they are stripped
// from resent history), so they show up only in RESPONSE blocks — a request
// diff can therefore be smaller than the tokens the server had to re-evaluate.
//
// Usage: logy <session_*.log>            (markdown to stdout)
//        go run ./logy .codehalter/session_xxx.log > session.md
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// entry is one `=== <timestamp> [<tag>] ===` block of the session log.
type entry struct {
	time string
	tag  string
	body string
}

var headerRe = regexp.MustCompile(`^=== (\S+) \[(.+)\] ===$`)

func parseEntries(data string) []entry {
	var entries []entry
	var cur *entry
	for _, line := range strings.Split(data, "\n") {
		if m := headerRe.FindStringSubmatch(line); m != nil {
			if cur != nil {
				cur.body = strings.TrimRight(cur.body, "\n")
				entries = append(entries, *cur)
			}
			cur = &entry{time: m[1], tag: m[2]}
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

// wireMsg is one canonicalised request message: the role plus everything that
// reaches the rendered prompt (content, tool calls, tool_call_id), flattened
// to a comparable string so prefix diffing is byte-honest.
type wireMsg struct {
	role string
	body string // canonical text used for both comparison and display
}

func canonMessages(raw []any) []wireMsg {
	var out []wireMsg
	for _, m := range raw {
		mm, _ := m.(map[string]any)
		role, _ := mm["role"].(string)
		var b strings.Builder
		switch c := mm["content"].(type) {
		case string:
			b.WriteString(c)
		case []any: // multimodal parts
			for _, p := range c {
				pm, _ := p.(map[string]any)
				switch pm["type"] {
				case "text":
					t, _ := pm["text"].(string)
					b.WriteString(t)
				case "image_url":
					b.WriteString("[inline image]")
				default:
					pj, _ := json.Marshal(p)
					b.Write(pj)
				}
			}
		}
		if id, ok := mm["tool_call_id"].(string); ok && id != "" {
			fmt.Fprintf(&b, "\n[tool_call_id: %s]", id)
		}
		if tcs, ok := mm["tool_calls"].([]any); ok {
			for _, tc := range tcs {
				tm, _ := tc.(map[string]any)
				fn, _ := tm["function"].(map[string]any)
				name, _ := fn["name"].(string)
				args, _ := fn["arguments"].(string)
				id, _ := tm["id"].(string)
				fmt.Fprintf(&b, "\n[tool_call %s id=%s args=%s]", name, id, args)
			}
		}
		out = append(out, wireMsg{role: role, body: b.String()})
	}
	return out
}

// commonPrefix returns how many leading messages are identical between the
// two requests — the region a warm prefix cache serves for free.
func commonPrefix(prev, cur []wireMsg) int {
	n := 0
	for n < len(prev) && n < len(cur) && prev[n] == cur[n] {
		n++
	}
	return n
}

// firstDiff returns the index of the first differing byte between two strings
// (== min length when one is a prefix of the other).
func firstDiff(a, b string) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func excerpt(s string, at, radius int) string {
	lo := max(at-radius, 0)
	hi := min(at+radius, len(s))
	e := s[lo:hi]
	e = strings.ReplaceAll(e, "\n", "⏎")
	if lo > 0 {
		e = "…" + e
	}
	if hi < len(s) {
		e += "…"
	}
	return e
}

// fence wraps text in a code fence long enough to not be broken by fences
// inside the text itself.
func fence(text string) string {
	f := "```"
	for strings.Contains(text, f) {
		f += "`"
	}
	return f + "\n" + text + "\n" + f
}

func renderMsg(w *strings.Builder, m wireMsg, note string) {
	fmt.Fprintf(w, "**%s**%s (%d chars)\n\n%s\n\n", m.role, note, len(m.body), fence(m.body))
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: logy <session_*.log>")
		os.Exit(2)
	}
	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "logy:", err)
		os.Exit(1)
	}

	var w strings.Builder
	fmt.Fprintf(&w, "# %s\n\n", os.Args[1])

	var prevMsgs []wireMsg
	var prevTools string
	callNo := 0
	for _, e := range parseEntries(string(data)) {
		switch {
		case strings.HasSuffix(e.tag, "REQUEST"):
			callNo++
			fmt.Fprintf(&w, "## call %d — %s `[%s]`\n\n", callNo, e.time, e.tag)
			var req map[string]any
			if err := json.Unmarshal([]byte(e.body), &req); err != nil {
				fmt.Fprintf(&w, "(unparseable request body: %v)\n\n%s\n\n", err, fence(e.body))
				continue
			}

			// Tools: full dump would drown the log — report size and diff only.
			toolsJSON := ""
			if t, ok := req["tools"]; ok && t != nil {
				tb, _ := json.Marshal(t)
				toolsJSON = string(tb)
				n := 0
				if ta, ok := t.([]any); ok {
					n = len(ta)
				}
				switch {
				case prevTools == "":
					fmt.Fprintf(&w, "tools: %d (%d chars)\n\n", n, len(toolsJSON))
				case toolsJSON == prevTools:
					fmt.Fprintf(&w, "tools: unchanged\n\n")
				default:
					d := firstDiff(prevTools, toolsJSON)
					fmt.Fprintf(&w, "⚠ tools CHANGED at char %d — this breaks the whole prefix cache!\n- was: `%s`\n- now: `%s`\n\n",
						d, excerpt(prevTools, d, 80), excerpt(toolsJSON, d, 80))
				}
			} else if prevTools != "" {
				fmt.Fprintf(&w, "⚠ tools DROPPED (previous call sent %d chars) — this breaks the whole prefix cache!\n\n", len(prevTools))
			}

			msgs := canonMessages(req["messages"].([]any))
			p := commonPrefix(prevMsgs, msgs)
			switch {
			case prevMsgs == nil:
				fmt.Fprintf(&w, "messages: %d (first call — full dump)\n\n", len(msgs))
				for _, m := range msgs {
					renderMsg(&w, m, "")
				}
			case p == len(prevMsgs) && p == len(msgs):
				fmt.Fprintf(&w, "messages: identical to previous call (%d messages)\n\n", p)
			case p == len(prevMsgs):
				fmt.Fprintf(&w, "messages: %d — pure extension: %d unchanged (cache-hot), %d appended\n\n", len(msgs), p, len(msgs)-p)
				for _, m := range msgs[p:] {
					renderMsg(&w, m, " *(new)*")
				}
			default:
				fmt.Fprintf(&w, "⚠ messages: %d — DIVERGED from previous call at message %d (only %d shared — everything after re-evaluates)\n\n", len(msgs), p, p)
				if p < len(msgs) {
					d := firstDiff(prevMsgs[p].body, msgs[p].body)
					fmt.Fprintf(&w, "message %d first differs at char %d:\n- was: `%s`\n- now: `%s`\n\n",
						p, d, excerpt(prevMsgs[p].body, d, 80), excerpt(msgs[p].body, d, 80))
					for _, m := range msgs[p:] {
						renderMsg(&w, m, " *(changed/new)*")
					}
				} else {
					fmt.Fprintf(&w, "context SHRANK: previous had %d messages, this call only %d (a fold/compaction, or a different context)\n\n", len(prevMsgs), len(msgs))
				}
			}
			prevMsgs, prevTools = msgs, toolsJSON

		case strings.HasSuffix(e.tag, "RESPONSE"):
			fmt.Fprintf(&w, "### response — %s\n\n", e.time)
			body := e.body
			// Headline the token/cache line as its own code span.
			if i := strings.Index(body, "\n"); strings.HasPrefix(body, "tokens: ") {
				line := body
				rest := ""
				if i >= 0 {
					line, rest = body[:i], body[i+1:]
				}
				fmt.Fprintf(&w, "`%s`\n\n", line)
				body = rest
			}
			if strings.TrimSpace(body) != "" {
				fmt.Fprintf(&w, "%s\n\n", fence(body))
			}

		default:
			fmt.Fprintf(&w, "### [%s] — %s\n\n", e.tag, e.time)
			if strings.TrimSpace(e.body) != "" {
				fmt.Fprintf(&w, "%s\n\n", fence(e.body))
			}
		}
	}
	os.Stdout.WriteString(w.String())
}
