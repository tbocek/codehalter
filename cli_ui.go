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

	"github.com/tbocek/codehalter/acp"
)

// ---------------------------------------------------------------------------
// Inline TUI renderer for the standalone CLI client.
//
// The screen is split in two, and the split is the whole design. Everything
// above the cursor is the TRANSCRIPT: rows that have been printed and are never
// touched again, exactly like ordinary program output. The bottom liveRows rows
// are the LIVE REGION: the partial tail of the streaming reply, the current
// plan, in-flight tool cards and the status line, all reprinted from scratch on
// every render.
//
// "Inline" means no alternate screen and no full-screen repaint. Scrollback
// still holds the whole conversation, Ctrl+Shift+C copies what you'd expect,
// and the shell prompt comes back below the last line instead of the screen
// being wiped. It also means the only escape sequences we need are "up N rows"
// and "clear to end of screen"; with tty false we emit neither, so piping
// stdout to a file yields a plain readable log rather than a pile of escapes.
//
// Every write goes through emitLine (transcript) or drawLive (live region), and
// both erase the live region first. liveRows is therefore the single piece of
// cursor state: get it wrong and the display eats real transcript rows.
// ---------------------------------------------------------------------------

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

// spinnerFrames is the braille spinner. Every frame is one column wide, which
// the width arithmetic below assumes of every glyph it prints.
var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

const (
	// cliMinCols is the floor for a terminal that won't tell us its width.
	cliMinCols = 40
	// cliTermTail is how many lines of a running command's output stay visible
	// under its card. Enough to see a build moving, short enough that three
	// concurrent commands still fit.
	cliTermTail = 3
	// cliMaxCards caps the in-flight card list so a burst of tool calls
	// can't push the status line off the screen.
	cliMaxCards = 6
	// cliDiffLines caps each side of a rendered diff hunk.
	cliDiffLines = 8
	// cliContentLines caps a tool result preview in the transcript.
	cliContentLines = 8
)

// cliCall is one tool card. Cards live here while the tool runs and are
// committed to the transcript when it finishes, so the live region only ever
// holds work that is actually in flight.
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

	// echoInput is set when stdin is not a terminal, i.e. nothing is echoing
	// the user's input for us. Then the CLI prints each submitted line itself,
	// so a piped run reads back as a transcript instead of a wall of prompts
	// with no questions attached.
	echoInput bool

	// pend is the not-yet-complete tail of the current wrapped row of streamed
	// text, held back from the transcript because more of that row may still
	// arrive. It is rendered at the TOP of the live region: it is a continuation
	// of the committed rows directly above, so anywhere else would make the
	// sentence jump around as it grows.
	pend    []rune
	pendSty string
	pendPre string

	calls map[string]*cliCall
	order []string
	terms map[string][]string

	plan []acp.PlanEntry

	mode  string
	used  int
	size  int
	busy  bool
	start time.Time
	frame int

	// promptStr is set while the cursor sits at the end of an input prompt,
	// which is the one moment the cursor is NOT at the start of a fresh row.
	// An update arriving then (a background task from a finished turn) would
	// otherwise append itself to whatever the user is typing, so emitLine
	// breaks the row first and drawLive reprints the prompt underneath.
	promptStr    string
	promptBroken bool
}

// stdoutStyled reports whether stdout takes ANSI styling: a terminal, not
// TERM=dumb, and NO_COLOR unset. The CLI's UI uses it, and hands it to the
// launcher for its notice, which prints before any UI exists.
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
	in, err := os.Stdin.Stat()
	u.echoInput = err != nil || in.Mode()&os.ModeCharDevice == 0
	u.refreshSize()
	return u
}

// refreshSize re-reads the terminal geometry. It shells out to stty because the
// alternative is a TIOCGWINSZ ioctl, and codehalter imports neither syscall nor
// x/sys anywhere else: one subprocess per turn is a cheaper price than making
// the whole program platform-specific for a cosmetic number. Called at startup
// and at the top of each turn, NOT on SIGWINCH (which would need syscall too),
// so a mid-turn resize renders at the old width until the next prompt.
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

// ---------------------------------------------------------------------------
// Cursor primitives. Only these three touch liveRows.
// ---------------------------------------------------------------------------

// eraseLive removes the live region and leaves the cursor where its first row
// began. drawLive always ends with the cursor at the start of the row after the
// last live row, so moving up liveRows and clearing to the end of the screen is
// exact regardless of how the rows wrapped.
func (u *cliUI) eraseLive() {
	if u.liveRows == 0 {
		return
	}
	fmt.Fprintf(u.out, "\x1b[%dA\x1b[0J", u.liveRows)
	u.liveRows = 0
}

// emitLine appends one row to the transcript, permanently.
func (u *cliUI) emitLine(s string) {
	if u.promptStr != "" && !u.promptBroken {
		u.out.WriteString("\n")
		u.promptBroken = true
	}
	u.eraseLive()
	u.out.WriteString(s)
	u.out.WriteString("\n")
}

// drawLive reprints the live region from current state. Every public method
// ends with this, so a chunk that commits five transcript rows still costs one
// redraw.
func (u *cliUI) drawLive() {
	u.eraseLive()
	if u.promptStr != "" {
		// A prompt owns the bottom row: reprint it if we broke it, and draw no
		// live region at all, since the cursor has to stay where the user types.
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

// liveLines builds the live region top to bottom. Each entry must occupy
// exactly one terminal row, which is why every builder clips to cols-1: a row
// that reaches the last column makes some terminals wrap, and one wrapped row
// throws liveRows off by one for the rest of the session.
func (u *cliUI) liveLines() []string {
	var out []string
	if len(u.pend) > 0 {
		out = append(out, u.clip(u.pendPre+u.style(string(u.pend), u.pendSty)))
	}
	out = append(out, u.planLines()...)

	// Everything left in u.order is still in flight: dropCall removes a card
	// from both maps the moment it completes.
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

	// Never let the live region fill the screen: the transcript has to stay
	// readable while it renders, and a live region taller than the terminal
	// would scroll its own top row away and desync liveRows. The first row is
	// kept (it is the streaming sentence, or the plan header) and the newest
	// rows below it win, because those are the ones still changing.
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
		// Finished and not-yet-started entries are counted in the header above;
		// spelling them all out would push a 20-step plan over the whole screen.
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

// ---------------------------------------------------------------------------
// Styling and width
// ---------------------------------------------------------------------------

func (u *cliUI) style(s, sty string) string {
	if !u.tty || sty == "" {
		return s
	}
	return sty + s + ansiReset
}

// clip truncates to one row, ellipsis included: the result is at most cols-1
// columns wide, because a row that reaches the last column makes some terminals
// wrap it, and one wrapped row throws liveRows off by one for the rest of the
// session.
//
// It measures only printable runes, so a string that already carries escapes
// keeps them: the styles applied above wrap a whole line, and cutting inside
// one would leak the colour into the rest of the screen.
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

// visibleLen counts printable runes, treating CSI sequences as zero-width.
// Runes, not display cells: a CJK glyph is two columns wide and is undercounted
// here, so the live region keeps to ASCII plus the single-column glyphs above
// and only model-supplied text (which clips conservatively) can be wide.
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

// sanitize strips the control characters that would move the cursor, and
// expands tabs. Text from the model and from tool output is printed verbatim,
// so an ESC in a file being previewed (or in a build log echoed back) would
// otherwise repaint the screen under us and leave liveRows counting rows that
// are no longer there. Tabs go because the width arithmetic counts them as one
// column and the terminal draws up to eight, which wraps a live row and desyncs
// the same counter; four spaces keeps a Go diff's indentation readable and
// makes the count exact. Newlines survive: callers split on them and commit one
// row each.
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

// ---------------------------------------------------------------------------
// Streaming text
// ---------------------------------------------------------------------------

// Stream appends streamed assistant output, wrapping it to the terminal and
// committing every row that is complete. sty/pre select the look: plain for a
// message, dim behind a bar for a thought.
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

// breakRow splits an over-long row at the last space that fits, so wrapping
// falls between words. A single word longer than the row (a path, a URL) has no
// space to break at and is cut hard rather than pushed past the edge.
func breakRow(row []rune, width int) (head, tail []rune) {
	for i := width; i > 0; i-- {
		if row[i-1] == ' ' {
			return row[:i-1], row[i:]
		}
	}
	return row[:width], row[width:]
}

// endRow commits the current row even when empty, which is how a blank line in
// the model's markdown survives into the transcript.
func (u *cliUI) endRow() {
	u.emitLine(u.pendPre + u.style(string(u.pend), u.pendSty))
	u.pend = u.pend[:0]
}

// flushRow commits a partial row, if any. Called when the style changes and at
// the end of a turn, never on a plain newline.
func (u *cliUI) flushRow() {
	if len(u.pend) > 0 {
		u.endRow()
	}
}

// ---------------------------------------------------------------------------
// Transcript entries
// ---------------------------------------------------------------------------

// Note prints one styled line straight into the transcript. Multi-line text is
// split so each row is committed separately and the live region stays aligned.
func (u *cliUI) Note(sty, s string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.flushRow()
	for _, l := range strings.Split(strings.TrimRight(sanitize(s), "\n"), "\n") {
		u.emitLine(u.clip(u.style(l, sty)))
	}
	u.drawLive()
}

// Card records a tool call. A card announced or updated as still running goes
// into the live region; one that reports completed or failed is committed to
// the transcript with its result and removed from the live set.
func (u *cliUI) Card(up acp.ToolCallUpdate) {
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

// contentLines renders a finished card's result blocks. Text is previewed, a
// diff is rendered as a hunk, and a terminal block is dropped: its output was
// already on screen live and the agent sends the same bytes as text.
func (u *cliUI) contentLines(cs []acp.ToolCallContent) []string {
	var out []string
	for _, c := range cs {
		switch c.Type {
		case "diff":
			out = append(out, diffLines(c.Path, c.OldText, c.NewText, u.tty)...)
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

// Plan replaces the current plan. It stays in the live region rather than being
// committed on every change: a plan is current state, and each entry's status
// is rewritten several times per turn.
func (u *cliUI) Plan(entries []acp.PlanEntry) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.plan = make([]acp.PlanEntry, len(entries))
	for i, e := range entries {
		e.Content = sanitize(e.Content)
		u.plan[i] = e
	}
	u.drawLive()
}

// Usage updates the context meter shown in the status line.
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

// TerminalTail publishes the last few lines of a running command so they show
// under its card. Keyed by terminal id, which is what the card's content block
// names.
func (u *cliUI) TerminalTail(tid string, lines []string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if len(lines) > cliTermTail {
		lines = lines[len(lines)-cliTermTail:]
	}
	for i, l := range lines {
		lines[i] = sanitize(l)
	}
	u.terms[tid] = lines
	u.drawLive()
}

// ---------------------------------------------------------------------------
// Turn lifecycle
// ---------------------------------------------------------------------------

// Begin starts a turn: the status line appears and the spinner runs.
func (u *cliUI) Begin() {
	u.refreshSize()
	u.mu.Lock()
	defer u.mu.Unlock()
	u.busy = true
	u.start = time.Now()
	u.drawLive()
}

// End finishes a turn, committing anything still partial and clearing the live
// region. Cards still marked running are committed as interrupted: they were
// announced, so leaving them only in an erased live region would lose them.
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

// Tick advances the spinner and refreshes elapsed times. Driven by the CLI's
// render ticker while a turn runs.
func (u *cliUI) Tick() {
	u.mu.Lock()
	defer u.mu.Unlock()
	if !u.busy || !u.tty {
		return
	}
	u.frame++
	u.drawLive()
}

// Suspend clears the live region and stops it being redrawn, so a question can
// be asked inline and the user's typing is not overwritten. The turn keeps
// running underneath; Resume brings the region back.
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

// Prompt writes the input prompt with no trailing newline, so the terminal's
// own echo of what the user types continues on the same row. This is the one
// place that deliberately leaves the cursor mid-row, which is why the live
// region must already be empty when it runs.
func (u *cliUI) Prompt(s string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.eraseLive()
	u.promptStr, u.promptBroken = s, false
	u.out.WriteString(u.style(s, ansiBold))
	u.out.Flush()
}

// Submitted is called with the line the user just entered. On a terminal the
// echo already happened and the cursor is back at the start of a row, so this
// only clears the prompt state. Piped in, nothing echoed, and the prompt row is
// still open: writing the line here is what closes it.
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

// ---------------------------------------------------------------------------
// Diff rendering
// ---------------------------------------------------------------------------

// diffLines renders an edit as a unified-ish hunk: common leading and trailing
// lines are trimmed and what's left is shown as removed then added. That is not
// a minimal diff (no LCS, so two separate edits in one file come back as one
// wide hunk) but it IS a correct one, it is exact for the single-region edits
// edit_file actually makes, and it costs no dependency.
func diffLines(path string, oldText *string, newText string, color bool) []string {
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

	paint := func(s, sty string) string {
		if !color {
			return s
		}
		return sty + s + ansiReset
	}
	head := fmt.Sprintf("  %s +%d -%d", sanitize(path), len(add), len(rm))
	out := []string{paint(head, ansiDim)}
	emit := func(lines []string, sign, sty string) {
		for i, l := range lines {
			if i == cliDiffLines {
				out = append(out, paint(fmt.Sprintf("    … %d more %s lines", len(lines)-i, sign), ansiDim))
				return
			}
			out = append(out, paint("  "+sign+" "+strings.TrimRight(l, " \t"), sty))
		}
	}
	emit(rm, "-", ansiRed)
	emit(add, "+", ansiGreen)
	return out
}
