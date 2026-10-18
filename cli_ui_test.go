package main

import (
	"bufio"
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

func uiFor(tty bool, cols, rows int) (*cliUI, *bytes.Buffer) {
	u := newCLIUI()
	var buf bytes.Buffer
	u.out = bufio.NewWriter(&buf)
	u.tty = tty
	u.echoInput = false
	u.cols, u.rows = cols, rows
	return u, &buf
}

// 20 columns leave 19 for text: a row breaks at its last space, or hard without one.
func TestStreamWrapsAtWidth(t *testing.T) {
	for _, c := range []struct {
		chunks []string
		want   string
	}{
		{[]string{"the quick brown fox ", "jumps over the lazy dog\n"}, "the quick brown\nfox jumps over the\nlazy dog\n"},
		{[]string{"/very/long/path/without/any/spaces\n"}, "/very/long/path/wit\nhout/any/spaces\n"},
	} {
		u, buf := uiFor(false, 20, 24)
		for _, s := range c.chunks {
			u.Stream("", "", s)
		}
		if buf.String() != c.want {
			t.Errorf("wrapped transcript:\n%q\nwant:\n%q", buf.String(), c.want)
		}
	}
}

func TestStreamKeepsBlankLines(t *testing.T) {
	u, buf := uiFor(false, 40, 24)
	u.Stream("", "", "one\n\ntwo\n")
	if want := "one\n\ntwo\n"; buf.String() != want {
		t.Errorf("got %q, want %q", buf.String(), want)
	}
}

func TestStreamHoldsPartialRow(t *testing.T) {
	u, buf := uiFor(false, 40, 24)
	u.Stream("", "", "half a sentence")
	if buf.String() != "" {
		t.Fatalf("partial row was committed early: %q", buf.String())
	}
	flushUI(u)
	if want := "half a sentence\n"; buf.String() != want {
		t.Errorf("got %q, want %q", buf.String(), want)
	}
}

func TestStreamStyleChangeFlushes(t *testing.T) {
	u, buf := uiFor(false, 40, 24)
	u.Stream("", "", "visible")
	u.Stream(ansiDim, "  ", "thinking")
	flushUI(u)
	if want := "visible\n  thinking\n"; buf.String() != want {
		t.Errorf("got %q, want %q", buf.String(), want)
	}
}

func TestClipCountsRunesNotBytes(t *testing.T) {
	u, _ := uiFor(true, 12, 24) // limit is cols-1 = 11
	got := u.clip("äöüäöüäöüäöüäöü")
	if want := "äöüäöüäöüä…"; got != want {
		t.Errorf("clip = %q, want %q", got, want)
	}
}

func TestClipKeepsEscapes(t *testing.T) {
	u, _ := uiFor(true, 8, 24) // limit 7
	got := u.clip(ansiRed + "abcdefghij" + ansiReset)
	if !strings.HasPrefix(got, ansiRed) {
		t.Errorf("clip dropped the opening escape: %q", got)
	}
	if !strings.HasSuffix(got, ansiReset) {
		t.Errorf("clip left the style open: %q", got)
	}
	if visible := stripANSI(got); visible != "abcdef…" {
		t.Errorf("visible text = %q, want %q", visible, "abcdef…")
	}
	if visibleLen(got) != u.cols-1 {
		t.Errorf("clipped row is %d columns, want %d", visibleLen(got), u.cols-1)
	}
}

func stripANSI(s string) string {
	var b strings.Builder
	esc := false
	for _, r := range s {
		switch {
		case esc && r == 'm':
			esc = false
		case esc:
		case r == '\x1b':
			esc = true
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// Erase must move up exactly as many rows as were drawn.
func TestLiveRegionRowAccounting(t *testing.T) {
	u, buf := uiFor(true, 60, 24)
	u.Card(toolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "run tests", Status: "in_progress"})
	if u.liveRows != 1 {
		t.Fatalf("liveRows after one running card = %d, want 1", u.liveRows)
	}
	buf.Reset()
	u.Note("", "transcript line")
	if !strings.HasPrefix(buf.String(), "\x1b[1A\x1b[0J") {
		t.Errorf("expected a 1-row erase before the transcript write, got %q", buf.String())
	}
	if !strings.Contains(buf.String(), "transcript line\n") {
		t.Errorf("transcript line missing: %q", buf.String())
	}
	if u.liveRows != 1 {
		t.Errorf("liveRows after redraw = %d, want 1", u.liveRows)
	}
}

func TestLiveRegionFitsScreen(t *testing.T) {
	u, _ := uiFor(true, 60, 8) // room for 6 live rows
	u.Begin()
	for i := 0; i < 20; i++ {
		u.Card(toolCallUpdate{Kind: "tool_call", ToolCallId: string(rune('a' + i)), Title: "tool", Status: "in_progress"})
	}
	if u.liveRows > u.rows-2 {
		t.Errorf("live region is %d rows on a %d-row screen", u.liveRows, u.rows)
	}
}

func TestLiveRowsFitWidth(t *testing.T) {
	u, _ := uiFor(true, 30, 24)
	u.Begin()
	u.Card(toolCallUpdate{
		Kind: "tool_call", ToolCallId: "1", Status: "in_progress",
		Title:   "a title far longer than thirty columns of terminal",
		Content: []ToolCallContent{{Type: "terminal", TerminalId: "t1"}},
	})
	u.TerminalTail("t1", []string{strings.Repeat("x", 200)})
	u.Plan([]PlanEntry{{Content: strings.Repeat("plan ", 40), Status: "in_progress"}})
	for _, l := range u.liveLines() {
		if w := len([]rune(stripANSI(l))); w > u.cols-1 {
			t.Errorf("live row is %d columns wide (max %d): %q", w, u.cols-1, stripANSI(l))
		}
	}
}

func TestCardCommitsOnCompletion(t *testing.T) {
	u, buf := uiFor(false, 80, 24)
	u.Card(toolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "read main.go", Status: "in_progress"})
	if buf.String() != "" {
		t.Fatalf("a running card must stay in the live region, got %q", buf.String())
	}
	u.Card(toolCallUpdate{
		Kind: "tool_call_update", ToolCallId: "1", Status: "completed",
		Content: []ToolCallContent{{Type: "content", Content: &ContentBlock{Type: "text", Text: "42 lines"}}},
	})
	out := buf.String()
	if !strings.Contains(out, "✓ read main.go") {
		t.Errorf("completed card missing from transcript: %q", out)
	}
	if !strings.Contains(out, "42 lines") {
		t.Errorf("card content missing: %q", out)
	}
	if len(u.order) != 0 || len(u.calls) != 0 {
		t.Errorf("completed card still live: order=%v calls=%d", u.order, len(u.calls))
	}
}

func TestFailedCardIsMarked(t *testing.T) {
	u, buf := uiFor(false, 80, 24)
	u.Card(toolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "run build", Status: "in_progress"})
	u.Card(toolCallUpdate{Kind: "tool_call_update", ToolCallId: "1", Status: "failed"})
	if !strings.Contains(buf.String(), "✗ run build") {
		t.Errorf("failed card not marked: %q", buf.String())
	}
}

func TestEndCommitsInterruptedCards(t *testing.T) {
	u, buf := uiFor(false, 80, 24)
	u.Begin()
	u.Card(toolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "npm test", Status: "in_progress"})
	u.End()
	if !strings.Contains(buf.String(), "npm test (interrupted)") {
		t.Errorf("interrupted card lost: %q", buf.String())
	}
	if u.busy || len(u.order) != 0 {
		t.Errorf("End left state behind: busy=%v order=%v", u.busy, u.order)
	}
}

// Tails are keyed by terminal id, not tool-call id.
func TestCardDropsItsTerminalTail(t *testing.T) {
	u, _ := uiFor(false, 80, 24)
	u.Card(toolCallUpdate{
		Kind: "tool_call", ToolCallId: "call_1", Title: "run", Status: "in_progress",
		Content: []ToolCallContent{{Type: "terminal", TerminalId: "term_1"}},
	})
	u.TerminalTail("term_1", []string{"building…"})
	u.Card(toolCallUpdate{Kind: "tool_call_update", ToolCallId: "call_1", Status: "completed"})
	if _, ok := u.terms["term_1"]; ok {
		t.Errorf("terminal tail outlived its card: %v", u.terms)
	}
}

func TestPromptRowSurvivesAnUpdate(t *testing.T) {
	u, buf := uiFor(true, 60, 24)
	u.Prompt("> ")
	buf.Reset()
	u.Note("", "background note")
	out := stripANSI(buf.String())
	if want := "\nbackground note\n> "; out != want {
		t.Errorf("got %q, want %q", out, want)
	}
	if u.liveRows != 0 {
		t.Errorf("live region drawn over a prompt: liveRows=%d", u.liveRows)
	}
}

func TestSubmittedClearsPrompt(t *testing.T) {
	u, buf := uiFor(true, 60, 24)
	u.Prompt("> ")
	u.Submitted("hello")
	buf.Reset()
	u.Note("", "after enter")
	if out := stripANSI(buf.String()); out != "after enter\n" {
		t.Errorf("got %q, want %q", out, "after enter\n")
	}
}

func TestSubmittedEchoesWhenNothingElseWill(t *testing.T) {
	u, buf := uiFor(false, 60, 24)
	u.echoInput = true
	u.Prompt("❯ ")
	u.Submitted("fix the build")
	u.Note("", "working")
	if want := "❯ fix the build\nworking\n"; buf.String() != want {
		t.Errorf("got %q, want %q", buf.String(), want)
	}
}

func TestDiffLinesTrimsCommonContext(t *testing.T) {
	old := "a\nb\nc\nd\n"
	nw := "a\nB\nc\nd\n"
	got := (&cliUI{}).diffLines("f.txt", &old, nw)
	want := []string{"  f.txt +1 -1", "  - b", "  + B"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("diffLines = %#v, want %#v", got, want)
	}
}

func TestDiffLinesNewFile(t *testing.T) {
	got := (&cliUI{}).diffLines("new.txt", nil, "one\ntwo")
	want := []string{"  new.txt +2 -0", "  + one", "  + two"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("diffLines = %#v, want %#v", got, want)
	}
}

func TestDiffLinesCapsLongHunks(t *testing.T) {
	old := ""
	nw := strings.TrimRight(strings.Repeat("line\n", 40), "\n")
	got := (&cliUI{}).diffLines("big.txt", &old, nw)
	if len(got) != 1+cliDiffLines+1 {
		t.Fatalf("uncapped hunk: %d rows", len(got))
	}
	if last := got[len(got)-1]; !strings.Contains(last, "32 more + lines") {
		t.Errorf("missing elision marker, got %q", last)
	}
}

func TestShortAndElapsed(t *testing.T) {
	for _, c := range []struct {
		n    int
		want string
	}{{7, "7"}, {1500, "2k"}, {124_200, "124k"}, {262_144, "262k"}, {2_500_000, "2.5M"}} {
		if got := short(c.n); got != c.want {
			t.Errorf("short(%d) = %q, want %q", c.n, got, c.want)
		}
	}
	for _, c := range []struct {
		d    time.Duration
		want string
	}{{3 * time.Second, "3s"}, {75 * time.Second, "1m15s"}} {
		if got := elapsed(c.d); got != c.want {
			t.Errorf("elapsed(%v) = %q, want %q", c.d, got, c.want)
		}
	}
}

func TestNonTTYEmitsNoEscapes(t *testing.T) {
	u, buf := uiFor(false, 80, 24)
	u.Begin()
	u.Stream("", "", "hello\n")
	u.Card(toolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "work", Status: "in_progress"})
	u.Tick()
	u.Card(toolCallUpdate{Kind: "tool_call_update", ToolCallId: "1", Status: "completed"})
	u.Note(ansiRed, "an error")
	u.End()
	if strings.Contains(buf.String(), "\x1b") {
		t.Errorf("escape sequences leaked into a non-tty stream: %q", buf.String())
	}
}

func TestSanitizeRemovesCursorMoves(t *testing.T) {
	cases := []struct{ in, want string }{
		{"plain", "plain"},
		{"\x1b[2J\x1b[Hwiped", "[2J[Hwiped"},
		{"bell\a and \bback", "bell and back"},
		{"crlf\r\n", "crlf\n"},
		{"\tindent", "    indent"},
		{"del\x7f", "del"},
	}
	for _, c := range cases {
		if got := sanitize(c.in); got != c.want {
			t.Errorf("sanitize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestStreamStripsEscapes(t *testing.T) {
	u, buf := uiFor(true, 80, 24)
	u.Stream("", "", "before \x1b[10Aafter\n")
	flushUI(u)
	if strings.Contains(buf.String(), "\x1b[10A") {
		t.Errorf("model escape reached the terminal: %q", buf.String())
	}
	if !strings.Contains(buf.String(), "before [10Aafter") {
		t.Errorf("text lost with the escape: %q", buf.String())
	}
}

func TestTerminalTailExpandsTabs(t *testing.T) {
	u, _ := uiFor(true, 80, 24)
	u.TerminalTail("t1", []string{"ok\tPASS\tmain"})
	if got := u.terms["t1"][0]; got != "ok    PASS    main" {
		t.Errorf("tail = %q", got)
	}
}

func TestCardTitleIsSanitized(t *testing.T) {
	u, _ := uiFor(true, 80, 24)
	u.Card(toolCallUpdate{ToolCallId: "t1", Title: "run \x1b[1;1Hls", Status: "in_progress"})
	if got := u.calls["t1"].title; got != "run [1;1Hls" {
		t.Errorf("title = %q", got)
	}
}

func flushUI(u *cliUI) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.flushRow()
	u.drawLive()
}

// screenAfter replays out on a cols-wide terminal that knows what drawLive uses: cursor up,
// erase below, autowrap on and off, and emoji as two cells. It returns the rows left on screen.
func screenAfter(out string, cols int) []string {
	rows, r, c, wrap := []string{""}, 0, 0, true
	put := func(ch rune) {
		w := 1
		if ch >= 0x1F300 || ch >= 0x2600 && ch <= 0x27BF {
			w = 2
		}
		if c+w > cols {
			if !wrap {
				return // the terminal keeps the cursor at the edge and drops the rest
			}
			r, c = r+1, 0
		}
		for len(rows) <= r {
			rows = append(rows, "")
		}
		rows[r] += string(ch)
		c += w
	}
	for i := 0; i < len(out); {
		if strings.HasPrefix(out[i:], "\x1b[") {
			j := i + 2
			for j < len(out) && (out[j] >= '0' && out[j] <= '9' || out[j] == '?') {
				j++
			}
			arg := out[i+2 : j]
			switch out[j] {
			case 'A':
				n := 0
				fmt.Sscan(arg, &n)
				r, c = max(r-n, 0), 0 // a terminal stops at its top row
			case 'J':
				for len(rows) <= r {
					rows = append(rows, "")
				}
				rows = append(rows[:r], string([]rune(rows[r])[:min(c, len([]rune(rows[r])))]))
			case 'C':
				n := 1
				fmt.Sscan(arg, &n)
				c += n
			case 'l', 'h':
				if arg == "?7" {
					wrap = out[j] == 'h'
				}
			}
			i = j + 1
			continue
		}
		ch, size := utf8.DecodeRuneInString(out[i:])
		if ch == '\r' {
			c = 0
		} else if ch == '\n' {
			r, c = r+1, 0
			for len(rows) <= r {
				rows = append(rows, "")
			}
		} else {
			put(ch)
		}
		i += size
	}
	return rows
}

// A live row wider than the terminal in cells (emoji count one each but take two) must not
// wrap: the extra row left the region's top row, "plan 1/2", behind on every redraw.
func TestLiveRegionLeavesNoRowsBehind(t *testing.T) {
	u, buf := uiFor(true, 40, 24)
	u.Begin()
	for range 5 {
		u.Plan([]PlanEntry{{Content: "done", Status: "completed"}, {Content: "check " + strings.Repeat("✅", 30), Status: "in_progress"}})
	}
	if err := u.out.Flush(); err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, row := range screenAfter(buf.String(), u.cols) {
		if strings.Contains(row, "plan 1/2") {
			n++
		}
	}
	if n != 1 {
		t.Errorf("plan header on screen %d times after five redraws, want 1", n)
	}
}

// Rows are measured in cells: an emoji or a CJK glyph takes two, a combining accent none.
// Counted in runes, a row of them was twice the terminal's width and wrapped once more.
func TestRowsAreMeasuredInCells(t *testing.T) {
	for _, c := range []struct {
		s     string
		cells int
	}{{"abc", 3}, {"→ ✓ …", 5}, {"✅🧪", 4}, {"中文", 4}, {"e\u0301", 1}, {"⚠\ufe0f", 1}} {
		if got := visibleLen(c.s); got != c.cells {
			t.Errorf("visibleLen(%q) = %d, want %d", c.s, got, c.cells)
		}
	}
	u, buf := uiFor(false, 21, 24) // 20 cells for text
	u.Stream("", "", strings.Repeat("✅", 15)+"\n"+strings.Repeat("中", 15)+"\n")
	if err := u.out.Flush(); err != nil {
		t.Fatal(err)
	}
	rows := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	for _, row := range rows {
		if w := visibleLen(row); w > 20 {
			t.Errorf("row %q is %d cells, past 20", row, w)
		}
	}
	if len(rows) != 4 { // 30 cells each: a full row of 20 and the rest
		t.Errorf("rows = %q, want two per line", rows)
	}
	if w := visibleLen(u.clip(strings.Repeat("🧪", 30))); w > u.cols-1 {
		t.Errorf("a clipped row of emoji is %d cells, past %d", w, u.cols-1)
	}
}

// While a question's prompt is open, arriving output waits for the answer instead of
// breaking the row being typed on; the main prompt does not hold anything back.
func TestOutputWaitsForAnOpenQuestion(t *testing.T) {
	u, buf := uiFor(true, 60, 24)
	u.Ask("  [1-2] ")
	u.Note("", "background job 3 exited with code 0")
	if strings.Contains(buf.String(), "background job 3") {
		t.Fatalf("printed while the answer was being typed: %q", buf.String())
	}
	u.Submitted("1")
	if !strings.Contains(buf.String(), "background job 3 exited with code 0\n") {
		t.Errorf("held output not printed after the answer: %q", buf.String())
	}
	buf.Reset()
	u.Prompt("\n❯ ")
	u.Note("", "a note at the main prompt")
	if !strings.Contains(buf.String(), "a note at the main prompt") {
		t.Errorf("the main prompt held output back: %q", buf.String())
	}
}

// The input row is the live region's last row with the cursor at its end: transcript lines
// land above it, every redraw replaces it in place, and nothing typed is left behind.
func TestInputRowStaysInPlace(t *testing.T) {
	u, buf := uiFor(true, 40, 24)
	u.Note("", "earlier transcript")
	u.inputOn, u.inPrompt = true, "❯ "
	u.Begin()
	for _, typed := range []string{"s", "sp", "spe", "spec stop"} {
		u.SetInput([]rune(typed))
		u.Tick()
	}
	u.Note("", "a line from the turn")
	u.Tick()
	if err := u.out.Flush(); err != nil {
		t.Fatal(err)
	}
	rows := screenAfter(buf.String(), u.cols)
	for len(rows) > 0 && rows[len(rows)-1] == "" {
		rows = rows[:len(rows)-1] // the row the cursor left when it went back up
	}
	inputs, notes := 0, 0
	if len(rows) == 0 || rows[0] != "earlier transcript" {
		t.Errorf("a redraw reached into the transcript above it:\n%s", strings.Join(rows, "\n"))
	}
	for _, row := range rows {
		if strings.Contains(row, "❯") {
			inputs++
		}
		if strings.Contains(row, "a line from the turn") {
			notes++
		}
	}
	if inputs != 1 || notes != 1 || !strings.Contains(rows[len(rows)-1], "❯ spec stop") {
		t.Errorf("screen after typing and output:\n%s", strings.Join(rows, "\n"))
	}
}

// The status line carries the last call's speeds; a call whose prompt all came from the
// cache measures no prompt speed and leaves the last one shown.
func TestStatusLineShowsTheLastSpeeds(t *testing.T) {
	u, _ := uiFor(true, 120, 24)
	u.Begin()
	u.Usage(50000, 262144)
	u.Speeds(812.4, 51.2)
	u.Speeds(0, 48.6)
	if line := stripANSI(u.statusLine()); !strings.Contains(line, "50k/262k ctx · 812 pp/s · 49 tg/s") {
		t.Errorf("status line = %q", line)
	}
}

// codehalter's phases go into the status row, phase last, instead of plan rows; the update
// that completes them clears it.
func TestPhasesGoIntoTheStatusRow(t *testing.T) {
	u, _ := uiFor(true, 160, 24)
	u.Begin()
	u.Usage(124200, 262144)
	u.Speeds(812, 51)
	u.Plan([]PlanEntry{{Content: "Planning", Status: "completed"},
		{Content: "Executing · 1m · llm[0] ↑426kb ↓37 tok", Status: "in_progress", Meta: &planMeta{Phase: "Executing", Meter: "llm[0] ↑426kb ↓37 tok"}}})
	rows := u.liveLines()
	if len(rows) != 1 || !strings.HasSuffix(stripANSI(rows[0]), " · llm[0] ↑426kb ↓37 tok · 124k/262k ctx · 812 pp/s · 51 tg/s · executing") {
		t.Errorf("live rows = %q", rows)
	}
	u.Plan([]PlanEntry{{Content: "Planning", Status: "completed"}, {Content: "Executing", Status: "completed"}})
	if line := stripANSI(u.statusLine()); strings.Contains(line, "executing") || strings.Contains(line, "llm[0]") {
		t.Errorf("after the phases: %q", line)
	}
}
