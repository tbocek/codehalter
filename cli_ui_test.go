package main

import (
	"bufio"
	"bytes"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/tbocek/codehalter/acp"
)

// uiFor builds a renderer writing into a buffer at a known geometry. Tests are
// in-package, so pointing the fields at a buffer is the whole seam needed.
func uiFor(tty bool, cols, rows int) (*cliUI, *bytes.Buffer) {
	u := newCLIUI()
	var buf bytes.Buffer
	u.out = bufio.NewWriter(&buf)
	u.tty = tty
	u.echoInput = false
	u.cols, u.rows = cols, rows
	return u, &buf
}

func TestBreakRow(t *testing.T) {
	cases := []struct {
		row                string
		width              int
		wantHead, wantTail string
	}{
		{"the quick brown fox ", 19, "the quick brown", "fox "},
		{"hello world", 8, "hello", "world"},
		// No space to break at: a long path is cut rather than pushed past the
		// edge, because one over-long row desyncs the live region for good.
		{"/very/long/path/without/spaces", 10, "/very/long", "/path/without/spaces"},
		{"a b", 2, "a", "b"},
	}
	for _, c := range cases {
		head, tail := breakRow([]rune(c.row), c.width)
		if string(head) != c.wantHead || string(tail) != c.wantTail {
			t.Errorf("breakRow(%q, %d) = %q, %q; want %q, %q",
				c.row, c.width, string(head), string(tail), c.wantHead, c.wantTail)
		}
	}
}

func TestStreamWrapsAtWidth(t *testing.T) {
	u, buf := uiFor(false, 20, 24)
	u.Stream("", "", "the quick brown fox ")
	u.Stream("", "", "jumps over the lazy dog\n")
	want := "the quick brown\nfox jumps over the\nlazy dog\n"
	if buf.String() != want {
		t.Errorf("wrapped transcript:\n%q\nwant:\n%q", buf.String(), want)
	}
}

func TestStreamKeepsBlankLines(t *testing.T) {
	u, buf := uiFor(false, 40, 24)
	u.Stream("", "", "one\n\ntwo\n")
	if want := "one\n\ntwo\n"; buf.String() != want {
		t.Errorf("got %q, want %q", buf.String(), want)
	}
}

// A partial row is held back so more of that sentence can still arrive; flushUI
// is what commits it.
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

// Switching between message and thought must break the row: the two are styled
// differently, and one row can only carry one style.
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

// Escapes are zero-width and must survive clipping, or a truncated colour code
// leaks into the rest of the screen.
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

// The live region is erased by moving up exactly as many rows as were drawn.
// Off by one here eats a transcript row on every redraw.
func TestLiveRegionRowAccounting(t *testing.T) {
	u, buf := uiFor(true, 60, 24)
	u.Card(acp.ToolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "run tests", Status: "in_progress"})
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
		u.Card(acp.ToolCallUpdate{Kind: "tool_call", ToolCallId: string(rune('a' + i)), Title: "tool", Status: "in_progress"})
	}
	if u.liveRows > u.rows-2 {
		t.Errorf("live region is %d rows on a %d-row screen", u.liveRows, u.rows)
	}
}

// Every live row must be at most cols-1 columns wide, or the terminal wraps it
// and the erase arithmetic is wrong from then on.
func TestLiveRowsFitWidth(t *testing.T) {
	u, _ := uiFor(true, 30, 24)
	u.Begin()
	u.Card(acp.ToolCallUpdate{
		Kind: "tool_call", ToolCallId: "1", Status: "in_progress",
		Title:   "a title far longer than thirty columns of terminal",
		Content: []acp.ToolCallContent{{Type: "terminal", TerminalId: "t1"}},
	})
	u.TerminalTail("t1", []string{strings.Repeat("x", 200)})
	u.Plan([]acp.PlanEntry{{Content: strings.Repeat("plan ", 40), Status: "in_progress"}})
	for _, l := range u.liveLines() {
		if w := len([]rune(stripANSI(l))); w > u.cols-1 {
			t.Errorf("live row is %d columns wide (max %d): %q", w, u.cols-1, stripANSI(l))
		}
	}
}

func TestCardCommitsOnCompletion(t *testing.T) {
	u, buf := uiFor(false, 80, 24)
	u.Card(acp.ToolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "read main.go", Status: "in_progress"})
	if buf.String() != "" {
		t.Fatalf("a running card must stay in the live region, got %q", buf.String())
	}
	u.Card(acp.ToolCallUpdate{
		Kind: "tool_call_update", ToolCallId: "1", Status: "completed",
		Content: []acp.ToolCallContent{{Type: "content", Content: &acp.ContentBlock{Type: "text", Text: "42 lines"}}},
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
	u.Card(acp.ToolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "run build", Status: "in_progress"})
	u.Card(acp.ToolCallUpdate{Kind: "tool_call_update", ToolCallId: "1", Status: "failed"})
	if !strings.Contains(buf.String(), "✗ run build") {
		t.Errorf("failed card not marked: %q", buf.String())
	}
}

// A card that was announced and never finished has to reach the transcript
// anyway: End erases the live region, and anything left only there is lost.
func TestEndCommitsInterruptedCards(t *testing.T) {
	u, buf := uiFor(false, 80, 24)
	u.Begin()
	u.Card(acp.ToolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "npm test", Status: "in_progress"})
	u.End()
	if !strings.Contains(buf.String(), "npm test (interrupted)") {
		t.Errorf("interrupted card lost: %q", buf.String())
	}
	if u.busy || len(u.order) != 0 {
		t.Errorf("End left state behind: busy=%v order=%v", u.busy, u.order)
	}
}

// A terminal tail is keyed by terminal id, not tool-call id: dropping the card
// must drop the right map entry or the tails accumulate for the whole session.
func TestCardDropsItsTerminalTail(t *testing.T) {
	u, _ := uiFor(false, 80, 24)
	u.Card(acp.ToolCallUpdate{
		Kind: "tool_call", ToolCallId: "call_1", Title: "run", Status: "in_progress",
		Content: []acp.ToolCallContent{{Type: "terminal", TerminalId: "term_1"}},
	})
	u.TerminalTail("term_1", []string{"building…"})
	u.Card(acp.ToolCallUpdate{Kind: "tool_call_update", ToolCallId: "call_1", Status: "completed"})
	if _, ok := u.terms["term_1"]; ok {
		t.Errorf("terminal tail outlived its card: %v", u.terms)
	}
}

// An update arriving while the cursor sits on an input prompt must break the
// prompt row first, then put the prompt back, or it lands inside what the user
// is typing.
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

// Piped in, nothing echoes the user's typing, so the CLI writes the submitted
// line itself and a redirected run reads back as a conversation.
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
	got := diffLines("f.txt", &old, nw, false)
	want := []string{"  f.txt +1 -1", "  - b", "  + B"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("diffLines = %#v, want %#v", got, want)
	}
}

func TestDiffLinesNewFile(t *testing.T) {
	got := diffLines("new.txt", nil, "one\ntwo", false)
	want := []string{"  new.txt +2 -0", "  + one", "  + two"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("diffLines = %#v, want %#v", got, want)
	}
}

func TestDiffLinesCapsLongHunks(t *testing.T) {
	old := ""
	nw := strings.TrimRight(strings.Repeat("line\n", 40), "\n")
	got := diffLines("big.txt", &old, nw, false)
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
	}{{7, "7"}, {1500, "1.5k"}, {2_500_000, "2.5M"}} {
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

// With no tty there are no escapes at all, so piping the CLI to a file gives a
// readable log rather than a pile of cursor moves.
func TestNonTTYEmitsNoEscapes(t *testing.T) {
	u, buf := uiFor(false, 80, 24)
	u.Begin()
	u.Stream("", "", "hello\n")
	u.Card(acp.ToolCallUpdate{Kind: "tool_call", ToolCallId: "1", Title: "work", Status: "in_progress"})
	u.Tick()
	u.Card(acp.ToolCallUpdate{Kind: "tool_call_update", ToolCallId: "1", Status: "completed"})
	u.Note(ansiRed, "an error")
	u.End()
	if strings.Contains(buf.String(), "\x1b") {
		t.Errorf("escape sequences leaked into a non-tty stream: %q", buf.String())
	}
}

// An ESC that reaches the terminal moves the cursor, and every row the live
// region thinks it owns is then off by however far it moved. Model text and
// tool output are printed verbatim, so they are the ones that have to be clean.
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
	// The ESC is gone, so what was a cursor move is now just text.
	if strings.Contains(buf.String(), "\x1b[10A") {
		t.Errorf("model escape reached the terminal: %q", buf.String())
	}
	if !strings.Contains(buf.String(), "before [10Aafter") {
		t.Errorf("text lost with the escape: %q", buf.String())
	}
}

// A tab counts as one rune and draws as up to eight columns, so a live row
// holding one wraps and throws the row count off for the rest of the session.
func TestTerminalTailExpandsTabs(t *testing.T) {
	u, _ := uiFor(true, 80, 24)
	u.TerminalTail("t1", []string{"ok\tPASS\tmain"})
	if got := u.terms["t1"][0]; got != "ok    PASS    main" {
		t.Errorf("tail = %q", got)
	}
}

func TestCardTitleIsSanitized(t *testing.T) {
	u, _ := uiFor(true, 80, 24)
	u.Card(acp.ToolCallUpdate{ToolCallId: "t1", Title: "run \x1b[1;1Hls", Status: "in_progress"})
	if got := u.calls["t1"].title; got != "run [1;1Hls" {
		t.Errorf("title = %q", got)
	}
}

// flushUI ends the current paragraph the way every non-streamed print does
// internally (flushRow under the lock), so a test can look at a held-back row.
func flushUI(u *cliUI) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.flushRow()
	u.drawLive()
}
