package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// repairJSON is deliberately not exhaustive: a prefix it can't rescue fails to parse, and
// the caller re-parses on the next delta.
func repairJSON(s string) string {
	var stack []byte
	inStr, esc := false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case esc:
			esc = false
		case inStr && c == '\\':
			esc = true
		case c == '"':
			inStr = !inStr
		case inStr:
			// Brackets inside a string are text, not structure.
		case c == '{' || c == '[':
			stack = append(stack, c)
		case c == '}' || c == ']':
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
		}
	}
	var b strings.Builder
	if inStr {
		b.WriteString(s)
		if esc {
			b.WriteByte('\\') // complete a dangling escape into a literal backslash
		}
		b.WriteByte('"')
	} else {
		// A dangling , or : sits at an object boundary, exactly when a row becomes ready.
		t := strings.TrimRight(s, " \t\r\n")
		switch {
		case strings.HasSuffix(t, ","):
			t = t[:len(t)-1]
		case strings.HasSuffix(t, ":"):
			t += "null"
		}
		b.WriteString(t)
	}
	for i := len(stack) - 1; i >= 0; i-- {
		if stack[i] == '{' {
			b.WriteByte('}')
		} else {
			b.WriteByte(']')
		}
	}
	return b.String()
}

// planTable emits a row only once its subtask object has closed, since the transcript is
// append-only. No title: report_only, which decides it, streams after subtasks.
type planTable struct {
	buf     strings.Builder
	emitted int
	header  bool
}

func (p *planTable) feed(delta string, say func(string)) {
	p.buf.WriteString(delta)
	raw := p.buf.String()

	var partial struct {
		Subtasks []subtask `json:"subtasks"`
	}
	if json.Unmarshal([]byte(repairJSON(raw)), &partial) != nil {
		return
	}
	// Unless raw parses unrepaired, the last subtask is still in flight.
	ready := len(partial.Subtasks)
	if !json.Valid([]byte(raw)) {
		ready--
	}
	for ; p.emitted < ready; p.emitted++ {
		if !p.header {
			say(planTableHead)
			p.header = true
		}
		say(planRow(partial.Subtasks[p.emitted]))
	}
}

// Without the leading blank line a row after agent text is swallowed into that paragraph.
// No number column: Zed gives every column an equal share.
const planTableHead = "\n\n| Subtask | Verify |\n|---|---|\n"

func planRow(st subtask) string {
	return fmt.Sprintf("| %s | %s |\n",
		planCell(st.Description), planCell(strings.Join(st.Verify, "\n")))
}

// planCell keeps a cell on one physical line: a raw `|` adds columns and a newline ends the table.
func planCell(s string) string {
	lines := strings.Split(strings.ReplaceAll(s, "|", `\|`), "\n")
	kept := lines[:0]
	for _, ln := range lines {
		indent := ln[:len(ln)-len(strings.TrimLeft(ln, " \t"))]
		ln = strings.Join(strings.Fields(ln), " ")
		if ln == "" && (len(kept) == 0 || kept[len(kept)-1] == "") {
			continue
		}
		// Plain spaces, not &nbsp;: Zed prints raw HTML as text.
		if ln != "" && indent != "" {
			ln = strings.Repeat(" ", len(indent)+3*strings.Count(indent, "\t")) + ln
		}
		// An odd run of trailing backslashes would escape the `<` of the following <br>.
		if n := len(ln) - len(strings.TrimRight(ln, `\`)); n%2 == 1 {
			ln += " "
		}
		kept = append(kept, ln)
	}
	for len(kept) > 0 && kept[len(kept)-1] == "" {
		kept = kept[:len(kept)-1]
	}
	return strings.Join(kept, "<br>")
}

// planTableSink is display-only: say never touches the LLM message list, so a partial table
// cannot perturb the prefix cache.
func (a *agent) planTableSink(ctx context.Context, sid string) func(int, string, string) {
	if sid == "" {
		return nil
	}
	tbl := &planTable{}
	call, flagged := -1, false
	return func(idx int, name, delta string) {
		if name != submitPlanToolName {
			return
		}
		if idx != call {
			call, tbl, flagged = idx, &planTable{}, false
		}
		tbl.feed(delta, func(s string) { a.say(ctx, sid, s) })
		if flagged || tbl.emitted == 0 {
			return
		}
		flagged = true
		// phaseMu, not sess.mu: this runs inside the SSE read loop and session
		// writers hold sess.mu across long operations.
		if sess := a.getSession(sid); sess != nil {
			sess.phaseMu.Lock()
			sess.planTableShown = true
			sess.phaseMu.Unlock()
		}
	}
}
