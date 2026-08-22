package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// repairJSON closes the strings, arrays and objects left open in a truncated
// JSON prefix so encoding/json can parse what has arrived so far. It does not
// try to be exhaustive: a prefix it can't rescue (a half-written \uXXXX escape,
// a bare `tru`) simply fails to parse, and since the caller re-parses on every
// delta the next few bytes fix it. Failing is a no-op, not a lost row.
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
			// An ordinary character inside a string: brackets here are text, not
			// structure, so they must not reach the stack.
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
		// Outside a string the tail can sit on a separator whose next token hasn't
		// arrived. Both land exactly at object boundaries, which is when a row
		// becomes ready, so they're worth fixing rather than waiting out.
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

// planTable renders submit_plan's streaming arguments as a markdown table that
// only ever grows. The chat transcript is append-only, so a row can never be
// rewritten: a row is therefore emitted only once its subtask object has CLOSED
// in the JSON. Holding the tail back matters even when its description looks
// finished, because `verify` follows `description` in field order and may still
// be filling in.
//
// It carries no title. Whether the planner produced a plan, findings or a
// replan depends on report_only, which the schema emits AFTER subtasks, so at
// the moment the first row is ready the correct heading isn't known yet. A
// wrong label is worse than none, so renderPlan prints the heading by itself
// once the table has streamed.
type planTable struct {
	buf     strings.Builder
	emitted int
	header  bool
}

// feed appends one argument delta and emits whatever rows became final, using
// `say` for each chunk.
func (p *planTable) feed(delta string, say func(string)) {
	p.buf.WriteString(delta)
	raw := p.buf.String()

	var partial struct {
		Subtasks []subtask `json:"subtasks"`
	}
	if json.Unmarshal([]byte(repairJSON(raw)), &partial) != nil {
		return
	}
	// A raw buffer that parses on its own needed no repair, so the array has
	// closed and every element is final. Otherwise the last one is still in
	// flight and must wait.
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

// planTableHead opens the table, and planRow renders one subtask. Both the
// streamed table and renderPlan go through them so a subtask looks the same
// however it reached the screen.
//
// The leading blank line is structural: a table has to start its own block, and
// without it a row arriving straight after a line of agent text is swallowed as
// a lazy continuation of that paragraph.
//
// There is no number column. Zed gives every column an equal share of the pane
// whatever it holds, so one would cost a third of the width to show a single
// digit, and the planner numbers its own subtasks ("1) Add the helper...").
const planTableHead = "\n\n| Subtask | Verify |\n|---|---|\n"

func planRow(st subtask) string {
	return fmt.Sprintf("| %s | %s |\n",
		planCell(st.Description), planCell(strings.Join(st.Verify, "\n")))
}

// planCell makes arbitrary text safe inside a markdown table cell. A pipe row is
// a single physical line, so two things in a subtask would wreck it: an
// unescaped `|` (descriptions name exact commands, and a `grep x | wc -l` splits
// the row into phantom columns) and a newline (it ends the table at that row).
// Pipes are escaped and newlines become <br>. Keeping the shape is what lets the
// cell carry the whole instruction: a real 27B description ran 1114 characters of
// shell pipelines around a heredoc, unreadable if flattened onto one line but
// fine as lines. Note that Zed does NOT render the <br> as a break: it prints
// raw HTML as text, so the tag shows up literally and marks the break instead of
// making it.
func planCell(s string) string {
	lines := strings.Split(strings.ReplaceAll(s, "|", `\|`), "\n")
	kept := lines[:0]
	for _, ln := range lines {
		indent := ln[:len(ln)-len(strings.TrimLeft(ln, " \t"))]
		ln = strings.Join(strings.Fields(ln), " ")
		// Collapse a run of blank lines to one gap, and drop it at the edges: the
		// paragraph break is worth keeping, a heredoc's vertical whitespace is not.
		if ln == "" && (len(kept) == 0 || kept[len(kept)-1] == "") {
			continue
		}
		// Leading whitespace is re-emitted as plain spaces (a tab as four) so the
		// Go source a heredoc carries doesn't render flat at column 0. Only the
		// indent is rebuilt: runs inside the line are alignment padding at worst
		// and collapse harmlessly. Plain spaces, not &nbsp;: Zed prints raw HTML
		// as text, so the entity showed up literally and cost more than the indent
		// it bought. Nothing here starts a line (the cell is one physical line),
		// so a four-space run can't be read as an indented code block.
		if ln != "" && indent != "" {
			ln = strings.Repeat(" ", len(indent)+3*strings.Count(indent, "\t")) + ln
		}
		// A line ending in an odd number of backslashes is a shell continuation. It
		// would escape the `<` of the <br> that follows and print the tag as text,
		// so separate them; a trailing space costs nothing.
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

// planTableSink builds the llmStream onArgs callback that streams submit_plan's
// arguments into the transcript as a growing table, or nil when there is no
// session to render into. Only submit_plan is rendered: every other tool's
// arguments are machinery.
//
// The table is display-only. say goes through sendUpdate, which never touches
// the LLM message list, so a partial or malformed table costs no prompt tokens
// and cannot perturb the prefix cache.
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
		if idx != call { // a different tool call in the same response: start over
			call, tbl, flagged = idx, &planTable{}, false
		}
		tbl.feed(delta, func(s string) { a.say(ctx, sid, s) })
		if flagged || tbl.emitted == 0 {
			return
		}
		flagged = true // once per table: this runs on every delta, hundreds of them
		// Tell renderPlan the subtasks are on screen. Written from inside the SSE
		// read loop, so it takes phaseMu (not sess.mu) for the same reason the
		// phase fields do: session writers hold sess.mu across long operations.
		if sess := a.getSession(sid); sess != nil {
			sess.phaseMu.Lock()
			sess.planTableShown = true
			sess.phaseMu.Unlock()
		}
	}
}
