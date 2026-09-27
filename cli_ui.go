package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Inline TUI: rows above the live region are printed once; the bottom liveRows rows are
// reprinted on every render. All writes go through emitLine/drawLive, so liveRows is the only cursor state.

const (
	ansiReset = "\x1b[0m"
	ansiDim   = "\x1b[2m"
	ansiBold  = "\x1b[1m"
	ansiRed   = "\x1b[31m"
	ansiGreen = "\x1b[32m"
	ansiYell  = "\x1b[33m"
	ansiBlue  = "\x1b[34m"
	ansiCyan  = "\x1b[36m"
)

// Every frame must be one column wide; the width arithmetic assumes it.
var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

const (
	cliMinCols      = 40
	cliTermTail     = 3
	cliMaxCards     = 6
	cliDiffLines    = 8
	cliContentLines = 8
)

type cliCall struct {
	id      string
	title   string
	kind    string
	status  string
	started time.Time
	termID  string
}

type cliUI struct {
	mu  sync.Mutex
	out *bufio.Writer
	tty bool

	cols, rows int
	liveRows   int

	// echoInput: stdin is not a terminal, so nothing echoes input and Submitted prints it.
	echoInput bool

	// pend is the incomplete tail of the current wrapped row, drawn at the top of the live
	// region so it reads as a continuation of the committed rows above.
	pend    []rune
	pendSty string
	pendPre string

	calls map[string]*cliCall
	order []string
	terms map[string][]string

	plan []PlanEntry

	mode  string
	used  int
	size  int
	busy  bool
	start time.Time
	frame int

	// promptStr is set while the cursor sits mid-row after an input prompt; emitLine then
	// breaks the row first and drawLive reprints the prompt below.
	promptStr    string
	promptBroken bool
}

func stdoutStyled() bool {
	fi, err := os.Stdout.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0 &&
		os.Getenv("TERM") != "dumb" && os.Getenv("NO_COLOR") == ""
}

func newCLIUI() *cliUI {
	u := &cliUI{
		out:   bufio.NewWriter(os.Stdout),
		calls: map[string]*cliCall{},
		terms: map[string][]string{},
		cols:  80,
		rows:  24,
	}
	u.tty = stdoutStyled()
	u.echoInput = !stdinIsTTY()
	u.refreshSize()
	return u
}

// stty instead of a TIOCGWINSZ ioctl keeps this platform-neutral. Called per turn, not on
// SIGWINCH, so a mid-turn resize applies from the next turn.
func (u *cliUI) refreshSize() {
	if !u.tty {
		return
	}
	cmd := exec.Command("stty", "size")
	cmd.Stdin = os.Stdin
	out, err := cmd.Output()
	if err != nil {
		return
	}
	f := strings.Fields(string(out))
	if len(f) != 2 {
		return
	}
	r, err1 := strconv.Atoi(f[0])
	c, err2 := strconv.Atoi(f[1])
	if err1 != nil || err2 != nil || c < cliMinCols || r < 4 {
		return
	}
	u.mu.Lock()
	u.cols, u.rows = c, r
	u.mu.Unlock()
}

// Exact because drawLive always leaves the cursor at the start of the row after the live region.
func (u *cliUI) eraseLive() {
	if u.liveRows == 0 {
		return
	}
	fmt.Fprintf(u.out, "\x1b[%dA\x1b[0J", u.liveRows)
	u.liveRows = 0
}

func (u *cliUI) emitLine(s string) {
	if u.promptStr != "" && !u.promptBroken {
		u.out.WriteString("\n")
		u.promptBroken = true
	}
	u.eraseLive()
	u.out.WriteString(s)
	u.out.WriteString("\n")
}

func (u *cliUI) drawLive() {
	u.eraseLive()
	if u.promptStr != "" {
		// The cursor must stay where the user types, so no live region under a prompt.
		if u.promptBroken {
			u.out.WriteString(u.style(u.promptStr, ansiBold))
			u.promptBroken = false
		}
		u.out.Flush()
		return
	}
	if !u.tty {
		u.out.Flush()
		return
	}
	for _, l := range u.liveLines() {
		u.out.WriteString(l)
		u.out.WriteString("\n")
		u.liveRows++
	}
	u.out.Flush()
}

// Each entry must be exactly one terminal row (see clip), or liveRows desyncs.
func (u *cliUI) liveLines() []string {
	var out []string
	if len(u.pend) > 0 {
		out = append(out, u.clip(u.pendPre+u.style(string(u.pend), u.pendSty)))
	}
	out = append(out, u.planLines()...)

	running := make([]*cliCall, 0, len(u.order))
	for _, id := range u.order {
		running = append(running, u.calls[id])
	}
	if n := len(running) - cliMaxCards; n > 0 {
		running = running[len(running)-cliMaxCards:]
		out = append(out, u.clip(u.style(fmt.Sprintf("  · +%d more running", n), ansiDim)))
	}
	for _, c := range running {
		out = append(out, u.clip(u.style(u.spin()+" "+c.title, ansiCyan)+
			u.style(" "+elapsed(time.Since(c.started)), ansiDim)))
		for _, l := range u.terms[c.termID] {
			out = append(out, u.clip(u.style("    "+l, ansiDim)))
		}
	}
	if u.busy {
		out = append(out, u.clip(u.statusLine()))
	}

	// A live region taller than the terminal would scroll its top row away and desync
	// liveRows. Keep the first row (streaming sentence or plan header) and the newest rows.
	if limit := u.rows - 2; limit > 1 && len(out) > limit {
		out = append(out[:1], out[len(out)-(limit-1):]...)
	}
	return out
}

func (u *cliUI) planLines() []string {
	if len(u.plan) == 0 {
		return nil
	}
	var out []string
	done := 0
	for _, e := range u.plan {
		if e.Status == "completed" {
			done++
		}
	}
	out = append(out, u.clip(u.style(fmt.Sprintf("  plan %d/%d", done, len(u.plan)), ansiDim)))
	for _, e := range u.plan {
		if e.Status != "in_progress" {
			continue
		}
		out = append(out, u.clip(u.style("  → "+e.Content, ansiYell)))
	}
	return out
}

func (u *cliUI) statusLine() string {
	parts := []string{elapsed(time.Since(u.start))}
	if u.mode != "" {
		parts = append(parts, strings.ToLower(u.mode))
	}
	if u.size > 0 {
		parts = append(parts, fmt.Sprintf("%s/%s ctx", short(u.used), short(u.size)))
	} else if u.used > 0 {
		parts = append(parts, short(u.used)+" ctx")
	}
	if n := len(u.order); n > 0 {
		parts = append(parts, fmt.Sprintf("%d running", n))
	}
	return u.style(u.spin()+" "+strings.Join(parts, " · "), ansiDim)
}

func (u *cliUI) spin() string {
	if !u.tty {
		return "·"
	}
	return spinnerFrames[u.frame%len(spinnerFrames)]
}

func (u *cliUI) style(s, sty string) string {
	if !u.tty || sty == "" {
		return s
	}
	return sty + s + ansiReset
}

// clip keeps rows under cols-1 columns: a row reaching the last column wraps on some
// terminals and desyncs liveRows. Escapes are kept whole so a cut never leaks colour.
func (u *cliUI) clip(s string) string {
	limit := u.cols - 1
	if limit < 2 {
		limit = 2
	}
	if visibleLen(s) <= limit {
		return s
	}
	var b strings.Builder
	width, esc := 0, false
	for _, r := range s {
		if esc {
			b.WriteRune(r)
			if r == 'm' {
				esc = false
			}
			continue
		}
		if r == '\x1b' {
			esc = true
			b.WriteRune(r)
			continue
		}
		if width == limit-1 {
			break
		}
		b.WriteRune(r)
		width++
	}
	b.WriteString("…")
	// Close whatever styling the truncated tail would have closed.
	if strings.Contains(s, "\x1b") {
		b.WriteString(ansiReset)
	}
	return b.String()
}

// Counts runes, not cells: wide (CJK) glyphs are undercounted.
func visibleLen(s string) int {
	n, esc := 0, false
	for _, r := range s {
		switch {
		case esc && r == 'm':
			esc = false
		case esc:
		case r == '\x1b':
			esc = true
		default:
			n++
		}
	}
	return n
}

// sanitize drops cursor-moving control chars from model/tool text and expands tabs (drawn up
// to 8 columns, counted as 1); either would desync liveRows. Newlines are kept.
func sanitize(s string) string {
	if strings.IndexFunc(s, func(r rune) bool { return r < 0x20 && r != '\n' || r == 0x7f }) < 0 {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch {
		case r == '\t':
			b.WriteString("    ")
		case r == '\n':
			b.WriteRune(r)
		case r < 0x20 || r == 0x7f:
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

func elapsed(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.0fs", d.Seconds())
	}
	return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
}

func short(n int) string {
	switch {
	case n >= 1000000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%.1fk", float64(n)/1e3)
	default:
		return strconv.Itoa(n)
	}
}

func (u *cliUI) Stream(sty, pre, s string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	s = sanitize(s)
	if sty != u.pendSty || pre != u.pendPre {
		u.flushRow()
		u.pendSty, u.pendPre = sty, pre
	}
	width := u.cols - 1 - len([]rune(u.pendPre))
	if width < 16 {
		width = 16
	}
	for _, r := range s {
		if r == '\n' {
			u.endRow()
			continue
		}
		u.pend = append(u.pend, r)
		for len(u.pend) > width {
			head, tail := breakRow(u.pend, width)
			u.emitLine(u.pendPre + u.style(string(head), u.pendSty))
			u.pend = tail
		}
	}
	u.drawLive()
}

func breakRow(row []rune, width int) (head, tail []rune) {
	for i := width; i > 0; i-- {
		if row[i-1] == ' ' {
			return row[:i-1], row[i:]
		}
	}
	return row[:width], row[width:]
}

// endRow commits even an empty row, so blank markdown lines survive.
func (u *cliUI) endRow() {
	u.emitLine(u.pendPre + u.style(string(u.pend), u.pendSty))
	u.pend = u.pend[:0]
}

func (u *cliUI) flushRow() {
	if len(u.pend) > 0 {
		u.endRow()
	}
}

func (u *cliUI) Note(sty, s string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.flushRow()
	for _, l := range strings.Split(strings.TrimRight(sanitize(s), "\n"), "\n") {
		u.emitLine(u.clip(u.style(l, sty)))
	}
	u.drawLive()
}

func (u *cliUI) Card(up toolCallUpdate) {
	u.mu.Lock()
	defer u.mu.Unlock()

	c := u.calls[up.ToolCallId]
	if c == nil {
		c = &cliCall{id: up.ToolCallId, started: time.Now()}
		u.calls[up.ToolCallId] = c
		u.order = append(u.order, up.ToolCallId)
	}
	if up.Title != "" {
		c.title = sanitize(up.Title)
	}
	if up.ToolKind != "" {
		c.kind = up.ToolKind
	}
	if up.Status != "" {
		c.status = up.Status
	}
	for _, ct := range up.Content {
		if ct.TerminalId != "" {
			c.termID = ct.TerminalId
		}
	}

	if c.status != "completed" && c.status != "failed" {
		u.drawLive()
		return
	}

	u.flushRow()
	u.dropCall(c)
	icon, sty := "✓", ansiGreen
	if c.status == "failed" {
		icon, sty = "✗", ansiRed
	}
	u.emitLine(u.clip(u.style(icon+" "+c.title, sty) +
		u.style(" "+elapsed(time.Since(c.started)), ansiDim)))
	for _, l := range u.contentLines(up.Content) {
		u.emitLine(u.clip(l))
	}
	u.drawLive()
}

func (u *cliUI) dropCall(c *cliCall) {
	id := c.id
	delete(u.calls, id)
	if c.termID != "" {
		delete(u.terms, c.termID)
	}
	for i, o := range u.order {
		if o == id {
			u.order = append(u.order[:i], u.order[i+1:]...)
			break
		}
	}
}

// Terminal blocks are skipped: their output was shown live and also arrives as text.
func (u *cliUI) contentLines(cs []ToolCallContent) []string {
	var out []string
	for _, c := range cs {
		switch c.Type {
		case "diff":
			out = append(out, u.diffLines(c.Path, c.OldText, c.NewText)...)
		case "content":
			if c.Content == nil || c.Content.Type != "text" {
				continue
			}
			lines := strings.Split(strings.TrimRight(sanitize(c.Content.Text), "\n"), "\n")
			for i, l := range lines {
				if i == cliContentLines {
					out = append(out, u.style(fmt.Sprintf("    … +%d lines", len(lines)-i), ansiDim))
					break
				}
				out = append(out, u.style("  "+strings.TrimRight(l, " \t"), ansiDim))
			}
		}
	}
	return out
}

func (u *cliUI) Plan(entries []PlanEntry) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.plan = make([]PlanEntry, len(entries))
	for i, e := range entries {
		e.Content = sanitize(e.Content)
		u.plan[i] = e
	}
	u.drawLive()
}

func (u *cliUI) Usage(used, size int) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.used, u.size = used, size
	u.drawLive()
}

func (u *cliUI) Mode(mode string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.mode = mode
	u.drawLive()
}

func (u *cliUI) TerminalTail(tid string, lines []string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	for i, l := range lines {
		lines[i] = sanitize(l)
	}
	u.terms[tid] = lines
	u.drawLive()
}

func (u *cliUI) Begin() {
	u.refreshSize()
	u.mu.Lock()
	defer u.mu.Unlock()
	u.busy = true
	u.start = time.Now()
	u.drawLive()
}

func (u *cliUI) End() {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.flushRow()
	for _, id := range append([]string(nil), u.order...) {
		c := u.calls[id]
		if c == nil {
			continue
		}
		u.dropCall(c)
		u.emitLine(u.clip(u.style("· "+c.title+" (interrupted)", ansiDim)))
	}
	u.busy = false
	u.plan = nil
	u.terms = map[string][]string{}
	u.eraseLive()
	u.out.Flush()
}

func (u *cliUI) Tick() {
	u.mu.Lock()
	defer u.mu.Unlock()
	if !u.busy || !u.tty {
		return
	}
	u.frame++
	u.drawLive()
}

// Suspend hides the live region for an inline question; the turn keeps running.
func (u *cliUI) Suspend() {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.flushRow()
	u.busy = false
	u.eraseLive()
	u.out.Flush()
}

func (u *cliUI) Resume() {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.busy = true
	u.drawLive()
}

func (u *cliUI) Prompt(s string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.eraseLive()
	u.promptStr, u.promptBroken = s, false
	u.out.WriteString(u.style(s, ansiBold))
	u.out.Flush()
}

func (u *cliUI) Submitted(text string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.promptStr != "" && u.echoInput {
		u.out.WriteString(text)
		u.out.WriteString("\n")
		u.out.Flush()
	}
	u.promptStr, u.promptBroken = "", false
}

// Trims common head and tail only (no LCS): exact for edit_file's single-region edits, no dependency.
func (u *cliUI) diffLines(path string, oldText *string, newText string) []string {
	var old []string
	if oldText != nil && *oldText != "" {
		old = strings.Split(sanitize(*oldText), "\n")
	}
	var nw []string
	if newText != "" {
		nw = strings.Split(sanitize(newText), "\n")
	}

	pre := 0
	for pre < len(old) && pre < len(nw) && old[pre] == nw[pre] {
		pre++
	}
	suf := 0
	for suf < len(old)-pre && suf < len(nw)-pre && old[len(old)-1-suf] == nw[len(nw)-1-suf] {
		suf++
	}
	rm, add := old[pre:len(old)-suf], nw[pre:len(nw)-suf]

	head := fmt.Sprintf("  %s +%d -%d", sanitize(path), len(add), len(rm))
	out := []string{u.style(head, ansiDim)}
	emit := func(lines []string, sign, sty string) {
		for i, l := range lines {
			if i == cliDiffLines {
				out = append(out, u.style(fmt.Sprintf("    … %d more %s lines", len(lines)-i, sign), ansiDim))
				return
			}
			out = append(out, u.style("  "+sign+" "+strings.TrimRight(l, " \t"), sty))
		}
	}
	emit(rm, "-", ansiRed)
	emit(add, "+", ansiGreen)
	return out
}
