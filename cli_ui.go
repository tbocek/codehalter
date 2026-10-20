package main

import (
	"bufio"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"slices"
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
	status  string
	started time.Time
	termID  string
	text    []string // the live text of a running card (codehalter's own test check), its last lines
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

	used int
	size int
	// pp and tg: the last call's prompt and generation tokens per second, 0 until known.
	pp, tg float64
	// phase and meter: codehalter's phase in progress and what runs in it, from its plan
	// entry's _meta; shown in the status row instead of plan rows.
	phase, meter string
	busy         bool
	start        time.Time
	frame        int

	// promptStr is set while the cursor sits mid-row after an input prompt; emitLine then
	// breaks the row first and drawLive reprints the prompt below.
	promptStr    string
	promptBroken bool
	// asking: a question's prompt is open (Ask). Lines arriving meanwhile are held and printed
	// after the answer: in cooked mode a reprinted prompt row lost what had been typed so far.
	asking bool
	held   []string

	// inputOn: the terminal is in character mode (startInput) and the input row, inPrompt
	// then what has been typed, is the live region's last row, the cursor at its end.
	inputOn     bool
	inPrompt    string
	inBuf       []rune
	cursorInRow bool
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

// stty runs stty on the terminal itself.
func stty(args ...string) (string, error) {
	cmd := exec.Command("stty", args...)
	cmd.Stdin = os.Stdin
	out, err := cmd.Output()
	return string(out), err
}

// startInput puts the terminal in character mode without echo, so the input row is drawn
// here and can be typed in while a turn runs; Ctrl+C still raises SIGINT. Pasted text is
// bracketed so its newlines stay text. It returns what undoes it, safe to call twice.
func (u *cliUI) startInput() func() {
	if !u.tty || !stdinIsTTY() {
		return func() {}
	}
	saved, err := stty("-g")
	if err != nil {
		return func() {}
	}
	if _, err := stty("-icanon", "-echo", "min", "1", "time", "0"); err != nil {
		return func() {}
	}
	u.mu.Lock()
	u.inputOn, u.inPrompt = true, "❯ "
	u.out.WriteString("\x1b[?2004h")
	u.drawLive()
	u.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			u.mu.Lock()
			u.eraseLive()
			u.inputOn = false
			u.out.WriteString("\x1b[?2004l")
			u.out.Flush()
			u.mu.Unlock()
			if _, err := stty(strings.TrimSpace(saved)); err != nil {
				slog.Debug("cli: restoring the terminal", "err", err)
			}
		})
	}
}

func (u *cliUI) inputActive() bool {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.inputOn
}

// SetInput shows what is typed so far.
func (u *cliUI) SetInput(buf []rune) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.inBuf = buf
	u.drawLive()
}

// stty instead of a TIOCGWINSZ ioctl keeps this platform-neutral. Called per turn and, where
// there is one, on SIGWINCH (watchResize).
func (u *cliUI) refreshSize() {
	if !u.tty {
		return
	}
	out, err := stty("size")
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

// Exact because drawLive leaves the cursor at the start of the row after the live region,
// or, with the input row, at the end of that row, the region's last.
func (u *cliUI) eraseLive() {
	if u.liveRows == 0 {
		return
	}
	if u.cursorInRow {
		u.out.WriteString("\r")
		if u.liveRows > 1 {
			fmt.Fprintf(u.out, "\x1b[%dA", u.liveRows-1)
		}
		u.out.WriteString("\x1b[0J")
	} else {
		fmt.Fprintf(u.out, "\x1b[%dA\x1b[0J", u.liveRows)
	}
	u.liveRows, u.cursorInRow = 0, false
}

func (u *cliUI) emitLine(s string) {
	if u.asking {
		u.held = append(u.held, s)
		return
	}
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
	// Autowrap off while drawing: a row wider than the terminal (an emoji or CJK glyph takes
	// two cells but counts as one) is cut at the edge instead of wrapping onto a row liveRows
	// does not count, which left the region's top row behind on every redraw.
	u.out.WriteString("\x1b[?7l")
	lines := u.liveLines()
	for _, l := range lines {
		u.out.WriteString(l)
		u.out.WriteString("\n")
		u.liveRows++
	}
	u.out.WriteString("\x1b[?7h")
	if u.inputOn && len(lines) > 0 {
		// Back to the end of the input row, where the typing shows.
		u.out.WriteString("\x1b[1A\r")
		if col := visibleLen(lines[len(lines)-1]); col > 0 {
			fmt.Fprintf(u.out, "\x1b[%dC", col)
		}
		u.cursorInRow = true
	}
	u.out.Flush()
}

// Each entry must be exactly one terminal row (see clip, and drawLive's autowrap), or liveRows desyncs.
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
		for _, l := range slices.Concat(u.terms[c.termID], c.text) {
			out = append(out, u.clip(u.style("    "+l, ansiDim)))
		}
	}
	if u.busy {
		out = append(out, u.clip(u.statusLine()))
	}
	if u.inputOn {
		// The tail of what is typed, so the cursor's end of it stays in view.
		typed := []rune(strings.ReplaceAll(string(u.inBuf), "\n", "⏎"))
		for room := max(u.cols-1-visibleLen(u.inPrompt), 8); visibleLen(string(typed)) > room; {
			typed = typed[1:]
		}
		out = append(out, u.style(u.inPrompt, ansiBold)+string(typed))
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

// statusLine: "⠸ 67m12s · llm[0] ↑426kb ↓37 tok · 124k/262k ctx · 812 pp/s · 51 tg/s · executing",
// the phase last as the least telling. A running tool is no part of it: its card shows it.
func (u *cliUI) statusLine() string {
	parts := []string{elapsed(time.Since(u.start))}
	if u.meter != "" {
		parts = append(parts, u.meter)
	}
	if u.size > 0 {
		parts = append(parts, fmt.Sprintf("%s/%s ctx", short(u.used), short(u.size)))
	} else if u.used > 0 {
		parts = append(parts, short(u.used)+" ctx")
	}
	for _, s := range []struct {
		rate float64
		unit string
	}{{u.pp, "pp/s"}, {u.tg, "tg/s"}} {
		if s.rate > 0 {
			parts = append(parts, strconv.Itoa(int(s.rate+0.5))+" "+s.unit)
		}
	}
	if u.phase != "" {
		parts = append(parts, strings.ToLower(u.phase))
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
	limit := max(u.cols-1, 2)
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
		if width+runeCells(r) > limit-1 {
			break
		}
		b.WriteRune(r)
		width += runeCells(r)
	}
	b.WriteString("…")
	// Close whatever styling the truncated tail would have closed.
	if strings.Contains(s, "\x1b") {
		b.WriteString(ansiReset)
	}
	return b.String()
}

// visibleLen counts terminal cells, escapes excluded.
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
			n += runeCells(r)
		}
	}
	return n
}

// wideRanges are drawn two cells wide: CJK, Hangul, fullwidth forms, and the emoji with
// emoji presentation (Unicode East Asian Width W). Ambiguous characters count as one,
// as most terminals draw them.
var wideRanges = [][2]rune{
	{0x1100, 0x115F}, {0x231A, 0x231B}, {0x2329, 0x232A}, {0x23E9, 0x23EC}, {0x23F0, 0x23F0},
	{0x23F3, 0x23F3}, {0x25FD, 0x25FE}, {0x2614, 0x2615}, {0x2648, 0x2653}, {0x267F, 0x267F},
	{0x2693, 0x2693}, {0x26A1, 0x26A1}, {0x26AA, 0x26AB}, {0x26BD, 0x26BE}, {0x26C4, 0x26C5},
	{0x26CE, 0x26CE}, {0x26D4, 0x26D4}, {0x26EA, 0x26EA}, {0x26F2, 0x26F3}, {0x26F5, 0x26F5},
	{0x26FA, 0x26FA}, {0x26FD, 0x26FD}, {0x2705, 0x2705}, {0x270A, 0x270B}, {0x2728, 0x2728},
	{0x274C, 0x274C}, {0x274E, 0x274E}, {0x2753, 0x2755}, {0x2757, 0x2757}, {0x2795, 0x2797},
	{0x27B0, 0x27B0}, {0x27BF, 0x27BF}, {0x2B1B, 0x2B1C}, {0x2B50, 0x2B50}, {0x2B55, 0x2B55},
	{0x2E80, 0x303E}, {0x3041, 0x33FF}, {0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xA000, 0xA4CF},
	{0xA960, 0xA97F}, {0xAC00, 0xD7A3}, {0xF900, 0xFAFF}, {0xFE10, 0xFE19}, {0xFE30, 0xFE6F},
	{0xFF00, 0xFF60}, {0xFFE0, 0xFFE6}, {0x1F004, 0x1F004}, {0x1F0CF, 0x1F0CF}, {0x1F18E, 0x1F18E},
	{0x1F191, 0x1F19A}, {0x1F200, 0x1F251}, {0x1F300, 0x1F320}, {0x1F32D, 0x1F335}, {0x1F337, 0x1F37C},
	{0x1F37E, 0x1F393}, {0x1F3A0, 0x1F3CA}, {0x1F3CF, 0x1F3D3}, {0x1F3E0, 0x1F3F0}, {0x1F3F4, 0x1F3F4},
	{0x1F3F8, 0x1F43E}, {0x1F440, 0x1F440}, {0x1F442, 0x1F4FC}, {0x1F4FF, 0x1F53D}, {0x1F54B, 0x1F54E},
	{0x1F550, 0x1F567}, {0x1F57A, 0x1F57A}, {0x1F595, 0x1F596}, {0x1F5A4, 0x1F5A4}, {0x1F5FB, 0x1F64F},
	{0x1F680, 0x1F6C5}, {0x1F6CC, 0x1F6CC}, {0x1F6D0, 0x1F6D2}, {0x1F6D5, 0x1F6D7}, {0x1F6DC, 0x1F6DF},
	{0x1F6EB, 0x1F6EC}, {0x1F6F4, 0x1F6FC}, {0x1F7E0, 0x1F7EB}, {0x1F7F0, 0x1F7F0}, {0x1F90C, 0x1F93A},
	{0x1F93C, 0x1F945}, {0x1F947, 0x1F9FF}, {0x1FA70, 0x1FAFF}, {0x20000, 0x3FFFD},
}

// runeCells is how many terminal cells r takes: 0 for combining marks, zero-width
// characters and variation selectors, 2 for wide ones, else 1.
func runeCells(r rune) int {
	switch {
	case r >= 0x300 && r <= 0x36F, r >= 0x200B && r <= 0x200F, r >= 0x2060 && r <= 0x2064,
		r >= 0xFE00 && r <= 0xFE0F, r >= 0xFE20 && r <= 0xFE2F, r >= 0xE0100 && r <= 0xE01EF:
		return 0
	case r < 0x1100:
		return 1
	}
	if _, wide := slices.BinarySearchFunc(wideRanges, r, func(rg [2]rune, r rune) int {
		switch {
		case rg[1] < r:
			return -1
		case rg[0] > r:
			return 1
		}
		return 0
	}); wide {
		return 2
	}
	return 1
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
		return strconv.Itoa((n+500)/1000) + "k"
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
	width := max(u.cols-1-visibleLen(u.pendPre), 16)
	for _, r := range s {
		if r == '\n' {
			u.endRow()
			continue
		}
		u.pend = append(u.pend, r)
		for visibleLen(string(u.pend)) > width {
			head, tail := breakRow(u.pend, width)
			u.emitLine(u.pendPre + u.style(string(head), u.pendSty))
			u.pend = tail
		}
	}
	u.drawLive()
}

// breakRow splits row after at most width cells, at its last space when there is one.
func breakRow(row []rune, width int) (head, tail []rune) {
	fit, cells := 0, 0
	for fit < len(row) && cells+runeCells(row[fit]) <= width {
		cells += runeCells(row[fit])
		fit++
	}
	fit = max(fit, 1) // a row always moves on, even past a glyph wider than the whole width
	for i := fit; i > 0; i-- {
		if row[i-1] == ' ' {
			return row[:i-1], row[i:]
		}
	}
	return row[:fit], row[fit:]
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
		// One row per card (liveLines): a multi-line command shows its first line and "…".
		title, _, more := strings.Cut(strings.TrimSpace(sanitize(up.Title)), "\n")
		if more {
			title += " …"
		}
		c.title = title
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
		for _, ct := range up.Content {
			if ct.Type == "content" && ct.Content != nil && ct.Content.Type == "text" {
				c.text = strings.Split(strings.TrimRight(sanitize(ct.Content.Text), "\n"), "\n")
			}
		}
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
	delete(u.calls, c.id)
	if c.termID != "" {
		delete(u.terms, c.termID)
	}
	u.order = slices.DeleteFunc(u.order, func(o string) bool { return o == c.id })
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
	// codehalter's phases (each a phase name, or the one in progress with its parts) go into
	// the status row, not plan rows.
	phase := func(e PlanEntry) bool { return e.Meta != nil || slices.Contains(phaseNames, e.Content) }
	if len(entries) > 0 && !slices.ContainsFunc(entries, func(e PlanEntry) bool { return !phase(e) }) {
		u.plan, u.phase, u.meter = nil, "", ""
		for _, e := range entries {
			if e.Meta != nil && e.Status == "in_progress" {
				u.phase, u.meter = e.Meta.Phase, e.Meta.Meter
			}
		}
		u.drawLive()
		return
	}
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

// Speeds keeps the last call's rates; a call with nothing to measure (its whole prompt
// came from the cache) leaves the previous one shown.
func (u *cliUI) Speeds(pp, tg float64) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if pp > 0 {
		u.pp = pp
	}
	if tg > 0 {
		u.tg = tg
	}
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
	for _, id := range slices.Clone(u.order) {
		c := u.calls[id]
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
	if u.inputOn {
		u.inPrompt = strings.TrimLeft(s, "\n")
		u.drawLive()
		return
	}
	u.eraseLive()
	u.promptStr, u.promptBroken = s, false
	u.out.WriteString(u.style(s, ansiBold))
	u.out.Flush()
}

// Ask is Prompt for a question's answer: output arriving before Submitted waits for it,
// unless the input row is drawn apart from the output anyway.
func (u *cliUI) Ask(s string) {
	u.Prompt(s)
	u.mu.Lock()
	u.asking = !u.inputOn
	u.mu.Unlock()
}

func (u *cliUI) Submitted(text string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.inputOn {
		// What was entered stays in the transcript; the row is ready for the next line.
		if text != "" {
			u.emitLine(u.style(u.inPrompt, ansiBold) + strings.ReplaceAll(text, "\n", "\n  "))
		}
		u.inBuf, u.inPrompt = nil, "❯ "
		u.drawLive()
		return
	}
	if u.promptStr != "" && u.echoInput {
		u.out.WriteString(text)
		u.out.WriteString("\n")
	}
	u.promptStr, u.promptBroken, u.asking = "", false, false
	for _, l := range u.held {
		u.out.WriteString(l)
		u.out.WriteString("\n")
	}
	u.held = nil
	u.out.Flush()
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
