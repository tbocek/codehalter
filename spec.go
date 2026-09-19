package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/BurntSushi/toml"
)

// The deterministic half of /spec: what the spec asks for, what the code has,
// and which slice of the spec one round needs. Nothing here calls a model.
//
// The split is the point of the feature. A long spec implemented over dozens of
// turns cannot rely on the model remembering what it has covered: compaction
// folds the history into a bounded summary, and a model asked "is everything
// done?" answers from that summary. So every question with a checkable answer
// is answered by code instead: the requirement ids come from a regex over the
// spec, "done" comes from those ids appearing in test sources that pass, and
// the next item is the first one that isn't. The model only ever sees one item
// and the part of the spec that item needs (spec_loop.go drives the rounds).

// specConfigName is the loop's state file under .codehalter/. Only what cannot
// be recomputed lives in it: the paths and target the user gave, the test
// command if they pinned one, and per-item attempts and blocks. Coverage is
// never stored; it is re-derived from the spec and the code on every round.
const specConfigName = "spec.toml"

// specSetupID is the pseudo-item of the first round: a project that builds and
// tests has to exist before any spec item can be "done".
const specSetupID = "setup"

// specSliceBudget bounds the spec text one round receives (the item's own
// section plus what it cites and links). Sections that don't fit are named as
// paths for the model to read instead. ~28 KB is ~7k tokens: big enough for the
// largest flow section with its parameter rows and a linked format section,
// small enough that dozens of rounds don't each drag a chapter into history.
const specSliceBudget = 28 * 1024

// specPrimaryCap caps the item's own section inside that budget, so one
// oversized section still leaves room for the rows it cites.
const specPrimaryCap = 20 * 1024

// specWholeFileCap is the largest linked file included whole when a link has no
// anchor (the shipped prompts are 1-9 KB; a whole chapter is not).
const specWholeFileCap = 10 * 1024

// defaultSpecIDPatterns are the requirement-id shapes recognised out of the
// box: flows ("F2.3"), parameters ("P.policy.minTakeSeconds") and tools the app
// offers a model ("tool:set_policy"). A spec with other conventions sets
// id_patterns in spec.toml.
var defaultSpecIDPatterns = []string{
	`\bF\d+\.\d+\b`,
	`\bP\.[a-z]+\.[A-Za-z][A-Za-z0-9]*\b`,
	`\btool:[a-z][a-z0-9_]*\b`,
}

// defaultSpecOpenMarkers are the words that mark a decision the spec left open.
// The loop refuses to start over them unless the user accepts them as written:
// under autopilot every open question would otherwise be settled by whichever
// reading the model happened to take.
var defaultSpecOpenMarkers = []string{"REVIEW", "TBD"}

// defaultSpecMaxBlocked is how many items in a row may end blocked before the
// loop stops. Three blocked items back to back is almost never three separate
// problems; it is one systemic one (a broken build, a wrong test command, a
// toolchain that isn't installed), and more rounds only burn time on it.
const defaultSpecMaxBlocked = 3

// specMaxAttempts is how many rounds one item gets before it is blocked: the
// round itself and one retry that is told why the first did not count.
const specMaxAttempts = 2

type specConfig struct {
	SpecDir           string   `toml:"spec_dir"`
	OutDir            string   `toml:"out_dir"`
	Target            string   `toml:"target"`
	TestCmd           string   `toml:"test_cmd,omitempty"`
	IDPatterns        []string `toml:"id_patterns,omitempty"`
	OpenMarkers       []string `toml:"open_markers,omitempty"`
	AcceptOpenMarkers bool     `toml:"accept_open_markers,omitempty"`
	MaxBlocked        int      `toml:"max_blocked,omitempty"`
	// Context lists spec files (relative to spec_dir) that are standing rules
	// for every round rather than items to implement: they are named in every
	// round prompt and never tracked. Empty means the default guess
	// (defaultSpecContext).
	Context []string `toml:"context,omitempty"`
	// Skip lists sections ("file.md#slug") that are not requirements at all and
	// must not become items.
	Skip     []string       `toml:"skip,omitempty"`
	Attempts map[string]int `toml:"attempts,omitempty"`
	Blocked  []specBlock    `toml:"blocked,omitempty"`
}

// specBlock is an item the loop gave up on, with the reason and, when the
// planner asked something under autopilot, the question. The user answers by
// filling Answer; the next /spec picks the item up again with that answer.
type specBlock struct {
	ID       string `toml:"id"`
	Reason   string `toml:"reason"`
	Question string `toml:"question,omitempty"`
	Answer   string `toml:"answer,omitempty"`
}

func specConfigPath(cwd string) string {
	return filepath.Join(cwd, sessionDir, specConfigName)
}

// loadSpecConfig returns the stored loop state, or nil when /spec has never
// been set up in this project.
func loadSpecConfig(cwd string) (*specConfig, error) {
	var cfg specConfig
	if _, err := toml.DecodeFile(specConfigPath(cwd), &cfg); err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading %s: %w", specConfigPath(cwd), err)
	}
	return &cfg, nil
}

func saveSpecConfig(cwd string, cfg *specConfig) error {
	var buf bytes.Buffer
	buf.WriteString("# /spec loop state. spec_dir, out_dir and target come from the /spec command.\n" +
		"# To answer a blocked item, fill its `answer` and run /spec again.\n" +
		"# accept_open_markers = true starts the loop despite open REVIEW/TBD markers.\n" +
		"# test_cmd overrides the detected test command (run from out_dir).\n\n")
	if err := toml.NewEncoder(&buf).Encode(cfg); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(specConfigPath(cwd)), 0o755); err != nil {
		return err
	}
	return os.WriteFile(specConfigPath(cwd), buf.Bytes(), 0o644)
}

func (c *specConfig) idPatterns() []string {
	if len(c.IDPatterns) > 0 {
		return c.IDPatterns
	}
	return defaultSpecIDPatterns
}

func (c *specConfig) openMarkers() []string {
	if c.OpenMarkers != nil {
		return c.OpenMarkers
	}
	return defaultSpecOpenMarkers
}

// defaultSpecContext guesses which spec files are orientation and standing
// rules rather than deliverables: the README, and pages whose name says
// principles, glossary, overview or introduction. A guess, overridable with
// `context` in spec.toml, and shown in /spec status so it is never silent.
func defaultSpecContext(idx *specIndex) []string {
	var out []string
	for _, d := range idx.docs {
		if strings.Contains(d.rel, "/") {
			continue
		}
		low := strings.ToLower(d.rel)
		for _, k := range []string{"readme", "principle", "glossary", "overview", "intro"} {
			if strings.Contains(low, k) {
				out = append(out, d.rel)
				break
			}
		}
	}
	return out
}

func (c *specConfig) context(idx *specIndex) []string {
	if len(c.Context) > 0 {
		return c.Context
	}
	return defaultSpecContext(idx)
}

func (c *specConfig) maxBlocked() int {
	if c.MaxBlocked > 0 {
		return c.MaxBlocked
	}
	return defaultSpecMaxBlocked
}

func (c *specConfig) block(id string) *specBlock {
	for i := range c.Blocked {
		if c.Blocked[i].ID == id {
			return &c.Blocked[i]
		}
	}
	return nil
}

func (c *specConfig) unblock(id string) {
	kept := c.Blocked[:0]
	for _, b := range c.Blocked {
		if b.ID != id {
			kept = append(kept, b)
		}
	}
	c.Blocked = kept
}

// parseSpecArgs reads the text after "/spec". Empty resumes, "status" reports,
// anything else is `<spec-dir> <out-dir> [target prompt...]`: the first two
// tokens are paths and everything after them is the prompt, verbatim, so a
// target like "use gtk4-rs libadwaita" needs no quoting.
func parseSpecArgs(args string) (cmd, specDir, outDir, target string, err error) {
	args = strings.TrimSpace(args)
	switch args {
	case "":
		return "resume", "", "", "", nil
	case "status":
		return "status", "", "", "", nil
	}
	rest := args
	next := func() string {
		rest = strings.TrimLeft(rest, " \t\r\n")
		i := strings.IndexAny(rest, " \t\r\n")
		if i < 0 {
			tok := rest
			rest = ""
			return tok
		}
		tok := rest[:i]
		rest = rest[i:]
		return tok
	}
	specDir, outDir = next(), next()
	if outDir == "" {
		return "", "", "", "", fmt.Errorf("usage: /spec <spec-dir> <out-dir> [technology prompt], /spec to resume, /spec status")
	}
	return "setup", filepath.Clean(specDir), filepath.Clean(outDir), strings.TrimSpace(rest), nil
}

// ---------------------------------------------------------------------------
// Scanning the spec
// ---------------------------------------------------------------------------

type specDoc struct {
	rel   string   // path relative to the spec dir, forward slashes
	lines []string // file content split on "\n"
	// navLine marks navigation lines (a <sub> line or anything between the
	// <!-- nav --> markers): their links point at neighbouring flows and the
	// contents page, never at something the current item needs.
	navLine []bool
	// fenced marks lines inside ``` blocks, where a leading # is not a heading.
	fenced []bool
}

type specSection struct {
	doc        int
	start, end int // [start, end) line range; start is the heading line
	level      int
	title      string
	slug       string
}

// Definition kinds, in priority order: an id's home is the heading that names
// it, else a table row that starts with it (a parameter table), else wherever
// it is first mentioned.
const (
	specDefHeading = iota
	specDefTableRow
	specDefMention
	// specDefSection is a headed section that defines no id of its own (a data
	// format, a screen, a rules list). It becomes an item under a synthetic id,
	// "§<file stem>#<slug>", because otherwise nothing would ever schedule it:
	// measured on one spec, 46 such sections, among them every file format and
	// every chapter's rules, were reachable by no flow's links.
	specDefSection
)

// specRank orders the ledger: headed items and section items interleave in
// document order (so an early chapter's file formats come before the flows
// that write them), then parameter/tool rows, then ids only ever mentioned.
func specRank(kind int) int {
	switch kind {
	case specDefHeading, specDefSection:
		return 0
	case specDefTableRow:
		return 1
	}
	return 2
}

type specItem struct {
	ID    string
	Kind  int
	Doc   int
	Line  int
	Title string // the heading text for a heading-defined item
	prio  int    // definition strength, lower wins (see specDefPrio)
}

// specDefPrio ranks candidate definition sites. A heading that opens with the
// id is its home. A heading that names it further in ("## 2. Tool protocol
// (F6.1)") comes next: it is still the section about that id when no heading
// opens with it, and it beats the index table row that merely lists it. Then a
// table row that starts with it (a parameter table), then a plain mention.
const (
	specPrioHeading = iota
	specPrioHeadingMention
	specPrioTableRow
	specPrioMention
)

type specIndex struct {
	docs     []specDoc
	sections []specSection
	items    map[string]*specItem
	order    []string // ledger order: heading items, then table rows, then mentions
	patterns []*regexp.Regexp
}

var (
	specLinkRe  = regexp.MustCompile(`(!?)\[[^\]]*\]\(([^)\s]+)\)`)
	headingRe   = regexp.MustCompile(`^(#{1,6})\s+(.*?)\s*#*\s*$`)
	specNavOpen = "<!-- nav -->"
	specNavEnd  = "<!-- /nav -->"
)

// scanSpec reads every .md file under root and indexes headings and requirement
// ids. Root-level chapters come before subdirectories, so when an id appears in
// both (a raw inventory repeating a chapter's flow ids) the chapter defines it.
func scanSpec(root string, patterns, context, skip []string) (*specIndex, error) {
	idx := &specIndex{items: map[string]*specItem{}}
	for _, p := range patterns {
		re, err := regexp.Compile(p)
		if err != nil {
			return nil, fmt.Errorf("id pattern %q: %w", p, err)
		}
		idx.patterns = append(idx.patterns, re)
	}

	var files []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.EqualFold(filepath.Ext(path), ".md") {
			rel, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			files = append(files, filepath.ToSlash(rel))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(files, func(i, j int) bool {
		di, dj := strings.Contains(files[i], "/"), strings.Contains(files[j], "/")
		if di != dj {
			return !di
		}
		return files[i] < files[j]
	})

	for _, rel := range files {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return nil, err
		}
		doc := specDoc{rel: rel, lines: strings.Split(string(data), "\n")}
		doc.navLine = make([]bool, len(doc.lines))
		doc.fenced = make([]bool, len(doc.lines))
		inNav, inFence := false, false
		for i, ln := range doc.lines {
			t := strings.TrimSpace(ln)
			if strings.HasPrefix(t, "```") {
				doc.fenced[i] = true
				inFence = !inFence
				continue
			}
			doc.fenced[i] = inFence
			switch {
			case strings.Contains(t, specNavOpen):
				inNav = true
				doc.navLine[i] = true
				continue
			case strings.Contains(t, specNavEnd):
				inNav = false
				doc.navLine[i] = true
				continue
			}
			doc.navLine[i] = inNav || strings.HasPrefix(t, "<sub>")
		}
		di := len(idx.docs)
		idx.docs = append(idx.docs, doc)
		idx.indexDoc(di)
	}
	if context == nil {
		context = defaultSpecContext(idx)
	}
	idx.addSectionItems(context, skip)

	idx.order = make([]string, 0, len(idx.items))
	for id := range idx.items {
		idx.order = append(idx.order, id)
	}
	sort.Slice(idx.order, func(i, j int) bool {
		a, b := idx.items[idx.order[i]], idx.items[idx.order[j]]
		if ra, rb := specRank(a.Kind), specRank(b.Kind); ra != rb {
			return ra < rb
		}
		if a.Doc != b.Doc {
			return a.Doc < b.Doc
		}
		if a.Line != b.Line {
			return a.Line < b.Line
		}
		return a.ID < b.ID
	})
	return idx, nil
}

// indexDoc records one document's sections and the best definition site of
// every id it contains.
func (idx *specIndex) indexDoc(di int) {
	doc := &idx.docs[di]
	var open []int // indices into idx.sections, innermost last
	closeTo := func(level, line int) {
		for len(open) > 0 && idx.sections[open[len(open)-1]].level >= level {
			idx.sections[open[len(open)-1]].end = line
			open = open[:len(open)-1]
		}
	}
	for i, ln := range doc.lines {
		if doc.fenced[i] {
			continue
		}
		if m := headingRe.FindStringSubmatch(ln); m != nil {
			level := len(m[1])
			closeTo(level, i)
			idx.sections = append(idx.sections, specSection{
				doc: di, start: i, end: len(doc.lines), level: level,
				title: m[2], slug: githubSlug(m[2]),
			})
			open = append(open, len(idx.sections)-1)
		}
	}
	closeTo(0, len(doc.lines))

	for i, ln := range doc.lines {
		if doc.navLine[i] {
			continue
		}
		kind := specDefMention
		title := ""
		switch {
		case !doc.fenced[i] && headingRe.MatchString(ln):
			// A heading defines the id it OPENS with ("### F0.7 Derive …"). One
			// that only names an id further in ("### 3.11 Policy derivation
			// (…, F0.7)") is about something else and merely mentions it.
			title = headingRe.FindStringSubmatch(ln)[2]
			kind = specDefHeading
		case strings.HasPrefix(strings.TrimSpace(ln), "|"):
			cells := strings.Split(strings.TrimSpace(ln), "|")
			if len(cells) > 1 && idx.anyID(cells[1]) {
				kind = specDefTableRow
			}
		}
		for _, id := range idx.idsIn(ln) {
			k, prio := kind, specPrioMention
			switch k {
			case specDefHeading:
				prio = specPrioHeading
				if !strings.HasPrefix(strings.TrimSpace(title), id) {
					prio = specPrioHeadingMention
					// A heading that opens with a different id is about that one
					// ("### F1.7 Describe (chunks of P.policy.describeFramesPerReq)"),
					// and one naming several ids is about none of them in
					// particular: either way the ids in it are ordinary mentions.
					if idx.opensWithID(title) || len(idx.idsIn(title)) > 1 {
						k, prio = specDefMention, specPrioMention
					}
				}
			case specDefTableRow:
				prio = specPrioTableRow
				cells := strings.Split(strings.TrimSpace(ln), "|")
				if !containsID(cells[1], id) {
					k, prio = specDefMention, specPrioMention // named in a later cell: a mention, not the row's subject
				}
			}
			if cur, ok := idx.items[id]; !ok || prio < cur.prio {
				t := ""
				if k == specDefHeading {
					t = title
				}
				idx.items[id] = &specItem{ID: id, Kind: k, Doc: di, Line: i, Title: t, prio: prio}
			}
		}
	}
}

func (idx *specIndex) idsIn(s string) []string {
	var out []string
	seen := map[string]bool{}
	for _, re := range idx.patterns {
		for _, m := range re.FindAllString(s, -1) {
			if !seen[m] {
				seen[m] = true
				out = append(out, m)
			}
		}
	}
	return out
}

func (idx *specIndex) anyID(s string) bool { return len(idx.idsIn(s)) > 0 }

// opensWithID reports whether s begins with a requirement id.
func (idx *specIndex) opensWithID(s string) bool {
	s = strings.TrimSpace(s)
	for _, id := range idx.idsIn(s) {
		if strings.HasPrefix(s, id) {
			return true
		}
	}
	return false
}

func containsID(s, id string) bool { return idBoundaryIndex(s, id) >= 0 }

// githubSlug is the anchor GitHub gives a heading: lower-case, punctuation
// dropped (letters, digits, spaces, '-' and '_' survive), spaces to hyphens.
// Links inside a spec are written against it, e.g. "F0.2 Press ▶" → "f02-press-".
func githubSlug(title string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(title) {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '-' || r == '_':
			b.WriteRune(r)
		case r == ' ':
			b.WriteByte('-')
		}
	}
	return b.String()
}

// addSectionItems turns every headed section of a root chapter that defines
// no id into an item of its own, so the parts of a spec written as prose (file
// formats, screens, rules) are scheduled and tracked like the flows are. Not
// turned into items: context files (standing rules), sections listed in skip,
// page titles (H1), sections nested in one that already is an item (the outer
// item carries them), indexes (a table of other items), and parameter tables
// (their rows are items already).
func (idx *specIndex) addSectionItems(context, skip []string) {
	isContext := map[string]bool{}
	for _, c := range context {
		isContext[filepath.ToSlash(filepath.Clean(c))] = true
	}
	isSkip := map[string]bool{}
	for _, s := range skip {
		isSkip[s] = true
	}
	defines := func(doc, start, end int) bool {
		for _, it := range idx.items {
			if it.Doc == doc && it.Line >= start && it.Line < end && (it.Kind == specDefHeading || it.Kind == specDefTableRow) {
				return true
			}
		}
		return false
	}
	var chosen []specSection
	for _, s := range idx.sections {
		d := idx.docs[s.doc]
		if s.level < 2 || strings.Contains(d.rel, "/") || isContext[d.rel] || isSkip[d.rel+"#"+s.slug] {
			continue
		}
		nested := false
		for _, c := range chosen {
			if c.doc == s.doc && s.start > c.start && s.start < c.end {
				nested = true
				break
			}
		}
		if nested || defines(s.doc, s.start, s.end) {
			continue
		}
		text := idx.text(s.doc, s.start, s.end)
		if idx.headingItemsIn(text, "") >= specIndexThreshold || idx.isIDTable(s) {
			continue
		}
		stem := strings.TrimSuffix(d.rel, filepath.Ext(d.rel))
		id := "§" + stem + "#" + s.slug
		idx.items[id] = &specItem{ID: id, Kind: specDefSection, Doc: s.doc, Line: s.start, Title: s.title}
		chosen = append(chosen, s)
	}
}

// isIDTable reports whether a section is mostly a table whose rows each start
// with an id (a chapter's "parameters used" list): those rows are items
// already, so the section adds nothing to track.
func (idx *specIndex) isIDTable(s specSection) bool {
	rows, lines := 0, 0
	for i := s.start + 1; i < s.end; i++ {
		t := strings.TrimSpace(idx.docs[s.doc].lines[i])
		if t == "" || strings.HasPrefix(t, "|---") || strings.HasPrefix(t, "|:") {
			continue
		}
		lines++
		if strings.HasPrefix(t, "|") {
			if cells := strings.Split(t, "|"); len(cells) > 1 && idx.anyID(cells[1]) {
				rows++
			}
		}
	}
	return rows >= 3 && rows*2 >= lines
}

// sectionAt returns the innermost section containing line.
func (idx *specIndex) sectionAt(doc, line int) (specSection, bool) {
	best, found := specSection{}, false
	for _, s := range idx.sections {
		if s.doc == doc && s.start <= line && line < s.end {
			if !found || s.start >= best.start {
				best, found = s, true
			}
		}
	}
	return best, found
}

func (idx *specIndex) sectionBySlug(doc int, slug string) (specSection, bool) {
	for _, s := range idx.sections {
		if s.doc == doc && s.slug == slug {
			return s, true
		}
	}
	return specSection{}, false
}

func (idx *specIndex) docByRel(rel string) int {
	for i, d := range idx.docs {
		if d.rel == rel {
			return i
		}
	}
	return -1
}

func (idx *specIndex) text(doc, start, end int) string {
	return strings.Join(idx.docs[doc].lines[start:end], "\n")
}

// textNoNav is text without the navigation lines (prev/next/contents links),
// which only cost tokens in a round prompt.
func (idx *specIndex) textNoNav(doc, start, end int) string {
	var keep []string
	for i := start; i < end; i++ {
		if !idx.docs[doc].navLine[i] {
			keep = append(keep, idx.docs[doc].lines[i])
		}
	}
	return strings.Join(keep, "\n")
}

// ---------------------------------------------------------------------------
// Slicing: the part of the spec one item needs
// ---------------------------------------------------------------------------

type specSlice struct {
	Text    string
	Related []string // spec paths (with #anchor) that were relevant but did not fit or belong to another item
	Images  []string // images the included text shows, relative to the spec dir
	// Reached identifies every section the slice put in front of the model,
	// for the "never reached" report.
	Reached []string
}

func sectionKey(rel string, start int) string { return fmt.Sprintf("%s:%d", rel, start) }

// slice assembles the spec text for one item: its own section, the table rows
// of the ids it cites, and the sections it links to, within specSliceBudget.
func (idx *specIndex) slice(id, specDirRel string) specSlice {
	var out specSlice
	it, ok := idx.items[id]
	if !ok {
		return out
	}
	used := 0
	seenRel := map[string]bool{}
	var b strings.Builder
	add := func(header, body string) {
		b.WriteString("--- ")
		b.WriteString(header)
		b.WriteString(" ---\n")
		b.WriteString(strings.TrimRight(body, "\n"))
		b.WriteString("\n\n")
		used += len(header) + len(body) + 10
	}
	path := func(doc int) string { return specDirRel + "/" + idx.docs[doc].rel }

	// The item's own text.
	var primary specSection
	var primaryText string
	switch it.Kind {
	case specDefHeading, specDefSection:
		primary, _ = idx.sectionAt(it.Doc, it.Line)
		primaryText = idx.textNoNav(primary.doc, primary.start, primary.end)
	case specDefTableRow:
		primary = specSection{doc: it.Doc, start: it.Line, end: it.Line + 1}
		// The row under its table's header, so the cells keep their column names.
		primaryText = idx.tableHeader(it.Doc, it.Line) + idx.docs[it.Doc].lines[it.Line]
	default:
		primary = idx.paragraphAt(it.Doc, it.Line)
		primaryText = idx.text(primary.doc, primary.start, primary.end)
	}
	if len(primaryText) > specPrimaryCap {
		cut := strings.LastIndexByte(primaryText[:specPrimaryCap], '\n')
		if cut <= 0 {
			cut = len(clipUTF8(primaryText, specPrimaryCap))
		}
		resume := primary.start + strings.Count(primaryText[:cut], "\n") + 2
		primaryText = primaryText[:cut] + fmt.Sprintf("\n[... section continues: read_file %s line=%d]", path(primary.doc), resume)
	}
	title := it.ID
	if it.Title != "" {
		title = it.Title
	}
	add(path(primary.doc)+" · "+title, primaryText)
	out.Reached = append(out.Reached, sectionKey(idx.docs[primary.doc].rel, primary.start))
	seenRel[sectionKey(idx.docs[primary.doc].rel, primary.start)] = true

	var navFree []string // the primary's lines minus navigation, for mentions and links
	for i := primary.start; i < primary.end && i < len(idx.docs[primary.doc].lines); i++ {
		if !idx.docs[primary.doc].navLine[i] {
			navFree = append(navFree, idx.docs[primary.doc].lines[i])
		}
	}
	body := strings.Join(navFree, "\n")

	// Rows of the table-defined ids this item cites (parameters, tools).
	var rows []string
	rowHeader := ""
	for _, other := range idx.idsIn(body) {
		o := idx.items[other]
		if other == id || o == nil || o.Kind != specDefTableRow {
			continue
		}
		if rowHeader == "" {
			rowHeader = idx.tableHeader(o.Doc, o.Line)
		}
		rows = append(rows, idx.docs[o.Doc].lines[o.Line])
		out.Reached = append(out.Reached, sectionKey(idx.docs[o.Doc].rel, o.Line))
	}
	if len(rows) > 0 {
		add("rows for the ids cited above", rowHeader+strings.Join(rows, "\n"))
	}

	// Sections the item links to.
	for _, m := range specLinkRe.FindAllStringSubmatch(body, -1) {
		if m[1] == "!" || strings.Contains(m[2], "://") || strings.HasPrefix(m[2], "mailto:") {
			continue
		}
		target, anchor, _ := strings.Cut(m[2], "#")
		doc := primary.doc
		if target != "" {
			rel := filepath.ToSlash(filepath.Clean(filepath.Join(filepath.Dir(idx.docs[primary.doc].rel), target)))
			doc = idx.docByRel(rel)
			if doc < 0 {
				continue
			}
		}
		var sec specSection
		var ok bool
		if anchor != "" {
			sec, ok = idx.sectionBySlug(doc, anchor)
		} else if target != "" {
			sec, ok = specSection{doc: doc, start: 0, end: len(idx.docs[doc].lines)}, true
			if len(idx.text(doc, 0, sec.end)) > specWholeFileCap {
				out.Related = appendUnique(out.Related, path(doc))
				continue
			}
		}
		if !ok {
			continue
		}
		key := sectionKey(idx.docs[doc].rel, sec.start)
		if seenRel[key] {
			continue
		}
		seenRel[key] = true
		ref := path(doc)
		if anchor != "" {
			ref += "#" + anchor
		}
		// Another heading-defined item's section is that item's round, not this
		// one's: name it so the model can look, don't paste it in.
		if sec.start < len(idx.docs[doc].lines) {
			if other := idx.idsIn(idx.docs[doc].lines[sec.start]); len(other) > 0 && !(len(other) == 1 && other[0] == id) {
				if o := idx.items[other[0]]; o != nil && o.Kind == specDefHeading {
					out.Related = appendUnique(out.Related, ref)
					continue
				}
			}
		}
		text := idx.textNoNav(doc, sec.start, sec.end)
		// An index (a flow table, a contents page) names many items and
		// specifies none of them: pointing at it is enough.
		if idx.headingItemsIn(text, id) >= specIndexThreshold {
			out.Related = appendUnique(out.Related, ref)
			continue
		}
		if used+len(text) > specSliceBudget {
			out.Related = appendUnique(out.Related, ref)
			continue
		}
		add(ref+" (linked)", text)
		out.Reached = append(out.Reached, key)
	}

	for _, m := range specLinkRe.FindAllStringSubmatch(b.String(), -1) {
		if m[1] != "!" || strings.Contains(m[2], "://") {
			continue
		}
		img := filepath.ToSlash(filepath.Clean(filepath.Join(filepath.Dir(idx.docs[primary.doc].rel), m[2])))
		out.Images = appendUnique(out.Images, specDirRel+"/"+img)
	}
	out.Text = b.String()
	return out
}

// specIndexThreshold is how many other heading-defined items a linked section
// may name before it reads as an index rather than as content.
const specIndexThreshold = 5

// headingItemsIn counts the heading-defined items other than self that text
// mentions.
func (idx *specIndex) headingItemsIn(text, self string) int {
	n := 0
	for _, other := range idx.idsIn(text) {
		if o := idx.items[other]; other != self && o != nil && o.Kind == specDefHeading {
			n++
		}
	}
	return n
}

// tableHeader returns the header and separator rows of the table containing
// line, or "" when there is none.
func (idx *specIndex) tableHeader(doc, line int) string {
	lines := idx.docs[doc].lines
	start := line
	for start > 0 && strings.HasPrefix(strings.TrimSpace(lines[start-1]), "|") {
		start--
	}
	if start+1 < len(lines) && start+1 <= line && strings.Contains(lines[start+1], "---") {
		return lines[start] + "\n" + lines[start+1] + "\n"
	}
	return ""
}

// paragraphAt returns the blank-line-delimited block around line.
func (idx *specIndex) paragraphAt(doc, line int) specSection {
	lines := idx.docs[doc].lines
	start, end := line, line+1
	for start > 0 && strings.TrimSpace(lines[start-1]) != "" {
		start--
	}
	for end < len(lines) && strings.TrimSpace(lines[end]) != "" {
		end++
	}
	return specSection{doc: doc, start: start, end: end}
}

func appendUnique(s []string, v string) []string {
	for _, x := range s {
		if x == v {
			return s
		}
	}
	return append(s, v)
}

// openMarkers lists every line holding an open-decision marker, "file:line".
func (idx *specIndex) openMarkers(markers []string, specDirRel string) []string {
	var out []string
	for _, d := range idx.docs {
		for i, ln := range d.lines {
			for _, m := range markers {
				if wordIndex(ln, m) >= 0 {
					out = append(out, fmt.Sprintf("%s/%s:%d", specDirRel, d.rel, i+1))
					break
				}
			}
		}
	}
	return out
}

// wordIndex finds word in s where it is not part of a longer identifier.
func wordIndex(s, word string) int {
	for from := 0; ; {
		i := strings.Index(s[from:], word)
		if i < 0 {
			return -1
		}
		i += from
		before := i == 0 || !isIdentByte(s[i-1])
		after := i+len(word) >= len(s) || !isIdentByte(s[i+len(word)])
		if before && after {
			return i
		}
		from = i + 1
	}
}

func isIdentByte(c byte) bool {
	return c == '_' || c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

// ---------------------------------------------------------------------------
// Coverage: which ids the code's tests name
// ---------------------------------------------------------------------------

// specTestToken is the form an id takes in a test function name: lower-case,
// every run of non-alphanumerics as one '_' ("F2.3" → "f2_3",
// "P.policy.minTakeSeconds" → "p_policy_mintakeseconds", "tool:set_policy" →
// "tool_set_policy"). Rust, Go, Python and JS test names all accept it.
func specTestToken(id string) string {
	// A section id ("§05-cut#7-rules") starts with a digit once normalised,
	// which no test function name can: give it a prefix.
	if rest, ok := strings.CutPrefix(id, "§"); ok {
		return "sec_" + specTestToken(rest)
	}
	var b strings.Builder
	under := false
	for _, r := range strings.ToLower(id) {
		if r < 128 && (unicode.IsLetter(r) || unicode.IsDigit(r)) {
			b.WriteRune(r)
			under = false
		} else if !under && b.Len() > 0 {
			b.WriteByte('_')
			under = true
		}
	}
	return strings.TrimRight(b.String(), "_")
}

// idBoundaryIndex finds id in s where it is a whole id: not preceded by an
// identifier character, and not followed by one or by ".<digit>" (so "F2.3"
// does not match inside "F2.30" or "F2.3.1").
func idBoundaryIndex(s, id string) int {
	for from := 0; ; {
		i := strings.Index(s[from:], id)
		if i < 0 {
			return -1
		}
		i += from
		end := i + len(id)
		ok := i == 0 || !isAlnumByte(s[i-1])
		if ok && end < len(s) {
			if isAlnumByte(s[end]) {
				ok = false
			} else if s[end] == '.' && end+1 < len(s) && s[end+1] >= '0' && s[end+1] <= '9' {
				ok = false
			}
		}
		if ok {
			return i
		}
		from = i + 1
	}
}

func isAlnumByte(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z'
}

// tokenBoundaryIndex is idBoundaryIndex for the test-name form, where '_' is a
// separator: "f2_3" matches in "fn f2_3_s1_greyed" but not in "f2_30" or "xf2_3".
func tokenBoundaryIndex(s, tok string) int {
	for from := 0; ; {
		i := strings.Index(s[from:], tok)
		if i < 0 {
			return -1
		}
		i += from
		end := i + len(tok)
		if (i == 0 || !isAlnumByte(s[i-1])) && (end >= len(s) || !isAlnumByte(s[end])) {
			return i
		}
		from = i + 1
	}
}

// specSkipDirs are never test sources: build output, dependencies, VCS.
var specSkipDirs = map[string]bool{
	"target": true, "node_modules": true, ".git": true, "vendor": true,
	"dist": true, "build": true, ".venv": true, "__pycache__": true,
}

// testSourceText returns the part of a file that is test code, or "" when the
// file is not a test source. A dedicated test file counts whole. A file that is
// only a test file because it carries an inline Rust test module counts from
// the first #[cfg(test)] on, so an id in a production doc comment above it
// does not pass for a test.
func testSourceText(rel, content string) string {
	base := strings.ToLower(filepath.Base(rel))
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	switch {
	case strings.HasSuffix(stem, "_test"), strings.HasPrefix(stem, "test_"),
		strings.HasSuffix(stem, ".test"), strings.HasSuffix(stem, ".spec"),
		strings.HasSuffix(stem, "_tests"), stem == "tests":
		return content
	}
	for _, part := range strings.Split(filepath.ToSlash(filepath.Dir(rel)), "/") {
		if part == "tests" || part == "test" || part == "__tests__" {
			return content
		}
	}
	if i := strings.Index(content, "#[cfg(test)]"); i >= 0 {
		return content[i:]
	}
	if strings.Contains(content, "#[test]") {
		return content
	}
	return ""
}

// specCoverage walks outDir and reports which ids its test sources name, as the
// raw id (in a comment or string) or as its test token (in a test name), plus
// how many test sources it found.
func specCoverage(outAbs string, ids []string) (covered map[string]string, testFiles int, err error) {
	covered = map[string]string{}
	err = filepath.WalkDir(outAbs, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil
			}
			return err
		}
		if d.IsDir() {
			if path != outAbs && (specSkipDirs[d.Name()] || strings.HasPrefix(d.Name(), ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		info, err := d.Info()
		if err != nil || info.Size() > 2<<20 {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil || looksBinary(data) {
			return nil
		}
		rel, _ := filepath.Rel(outAbs, path)
		text := testSourceText(rel, string(data))
		if text == "" {
			return nil
		}
		testFiles++
		lower := strings.ToLower(text)
		for _, id := range ids {
			if _, done := covered[id]; done {
				continue
			}
			if idBoundaryIndex(text, id) >= 0 || tokenBoundaryIndex(lower, specTestToken(id)) >= 0 {
				covered[id] = filepath.ToSlash(rel)
			}
		}
		return nil
	})
	return covered, testFiles, err
}

// detectSpecTestCmd picks the command that runs outDir's tests. A justfile
// `test` recipe wins, because it is where a project puts what plain `cargo
// test` doesn't know (a virtual display for GUI tests, an env var).
func detectSpecTestCmd(outAbs string) string {
	for _, name := range []string{"justfile", "Justfile", ".justfile"} {
		if data, err := os.ReadFile(filepath.Join(outAbs, name)); err == nil {
			for _, ln := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(ln, "test") && (strings.HasPrefix(ln, "test:") || strings.HasPrefix(ln, "test ")) {
					return "just test"
				}
			}
		}
	}
	for _, c := range []struct{ file, cmd string }{
		{"Cargo.toml", "cargo test"},
		{"package.json", "npm test"},
		{"go.mod", "go test ./..."},
	} {
		if fileExists(outAbs, c.file) {
			return c.cmd
		}
	}
	return ""
}

// ---------------------------------------------------------------------------
// Picking the next item and reporting
// ---------------------------------------------------------------------------

// nextSpecItem returns the first item in ledger order that is neither covered
// nor blocked. A blocked item the user has answered is eligible again; its
// answer comes back so the round can carry it.
func nextSpecItem(idx *specIndex, covered map[string]string, cfg *specConfig) (id, answer string) {
	for _, it := range idx.order {
		if _, ok := covered[it]; ok {
			continue
		}
		if b := cfg.block(it); b != nil {
			if strings.TrimSpace(b.Answer) == "" {
				continue
			}
			return it, b.Answer
		}
		return it, ""
	}
	return "", ""
}

// renderSpecStatus is the /spec status report.
func renderSpecStatus(cfg *specConfig, idx *specIndex, covered map[string]string, testFiles int, testCmd string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "**/spec** `%s/` → `%s/`", cfg.SpecDir, cfg.OutDir)
	if cfg.Target != "" {
		fmt.Fprintf(&b, " · target: %s", cfg.Target)
	}
	b.WriteString("\n\n")
	cmd := "`" + testCmd + "`"
	switch {
	case cfg.TestCmd != "":
		cmd = "`" + cfg.TestCmd + "` (from spec.toml)"
	case testCmd == "":
		cmd = "none detected yet (the setup round creates it)"
	}
	fmt.Fprintf(&b, "Test command: %s · %d test source file(s) in `%s/`\n\n", cmd, testFiles, cfg.OutDir)

	type tally struct{ total, done, blocked int }
	perDoc := map[int]*tally{}
	var kinds [4]tally
	for _, id := range idx.order {
		it := idx.items[id]
		t := perDoc[it.Doc]
		if t == nil {
			t = &tally{}
			perDoc[it.Doc] = t
		}
		t.total++
		kinds[it.Kind].total++
		if _, ok := covered[id]; ok {
			t.done++
			kinds[it.Kind].done++
		} else if cfg.block(id) != nil {
			t.blocked++
			kinds[it.Kind].blocked++
		}
	}
	fmt.Fprintf(&b, "Covered: %d/%d flows · %d/%d sections (formats, screens, rules) · %d/%d parameters and tools · %d/%d cited-only ids\n\n",
		kinds[specDefHeading].done, kinds[specDefHeading].total, kinds[specDefSection].done, kinds[specDefSection].total,
		kinds[specDefTableRow].done, kinds[specDefTableRow].total, kinds[specDefMention].done, kinds[specDefMention].total)
	b.WriteString("| file | covered | blocked | total |\n|---|---|---|---|\n")
	docs := make([]int, 0, len(perDoc))
	for d := range perDoc {
		docs = append(docs, d)
	}
	sort.Ints(docs)
	for _, d := range docs {
		t := perDoc[d]
		fmt.Fprintf(&b, "| %s | %d | %d | %d |\n", idx.docs[d].rel, t.done, t.blocked, t.total)
	}
	if next, _ := nextSpecItem(idx, covered, cfg); next != "" {
		fmt.Fprintf(&b, "\nNext: **%s**", next)
		if t := idx.items[next].Title; t != "" {
			fmt.Fprintf(&b, " (%s)", t)
		}
		b.WriteString("\n")
	} else {
		b.WriteString("\nNothing left: every id is covered or blocked.\n")
	}
	if len(cfg.Blocked) > 0 {
		b.WriteString("\nBlocked (fill `answer` in `.codehalter/spec.toml`, then run /spec):\n")
		for _, bl := range cfg.Blocked {
			fmt.Fprintf(&b, "- **%s**: %s", bl.ID, bl.Reason)
			if bl.Question != "" {
				fmt.Fprintf(&b, " · question: %s", bl.Question)
			}
			if bl.Answer != "" {
				b.WriteString(" · answered, will retry")
			}
			b.WriteString("\n")
		}
	}
	if ctx := cfg.context(idx); len(ctx) > 0 {
		fmt.Fprintf(&b, "\nStanding context (named in every round, not tracked as items; set `context` in spec.toml to change): %s\n", strings.Join(ctx, ", "))
	}
	var undefined []string
	for _, id := range idx.order {
		if idx.items[id].Kind == specDefMention {
			it := idx.items[id]
			undefined = append(undefined, fmt.Sprintf("%s (%s:%d)", id, idx.docs[it.Doc].rel, it.Line+1))
		}
	}
	if len(undefined) > 0 {
		fmt.Fprintf(&b, "\nCited but never defined (no heading or table row of their own, likely a gap in the spec): %s\n", strings.Join(undefined, ", "))
	}
	return b.String()
}
