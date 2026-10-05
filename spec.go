package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/BurntSushi/toml"
)

// Nothing in this file calls a model: after compaction a model asked "is everything done?"
// answers from its summary, so every question with a checkable answer is answered by code.

// specConfigName holds only what cannot be recomputed; coverage is re-derived every round.
const specConfigName = "spec.toml"

// specSetupID is the first round's pseudo-item: a project that builds and tests.
const specSetupID = "setup"

// specSliceBudget bounds one round's spec text; sections that don't fit are named as paths.
const specSliceBudget = 28 * 1024

// specPrimaryCap leaves room in the budget for the rows an oversized section cites.
const specPrimaryCap = 20 * 1024

// specWholeFileCap is the largest file an anchorless link pulls in whole.
const specWholeFileCap = 10 * 1024

var defaultSpecIDPatterns = []string{
	`\bF\d+\.\d+\b`,
	`\bP\.[a-z]+\.[A-Za-z][A-Za-z0-9]*\b`,
	`\btool:[a-z][a-z0-9_]*\b`,
}

// defaultSpecOpenMarkers stop the loop unless accepted: under autopilot the model would
// otherwise settle each open decision by whichever reading it happened to take.
var defaultSpecOpenMarkers = []string{"REVIEW", "TBD"}

// defaultSpecMaxBlocked: items blocked back to back are one systemic problem, not several.
const defaultSpecMaxBlocked = 3

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
	// Context lists spec files that are standing rules: named in every round, never tracked.
	Context []string `toml:"context,omitempty"`
	// Skip lists "file.md#slug" sections that must not become items.
	Skip     []string       `toml:"skip,omitempty"`
	Attempts map[string]int `toml:"attempts,omitempty"`
	// Redo: any non-empty value counts, older ledgers hold reason text.
	Redo map[string]string `toml:"redo,omitempty"`
	// Items records the spec text each item was built from, which coverage cannot know.
	Items map[string]specLedger `toml:"items,omitempty"`
	// Final is redone when the item or blocked count has moved since.
	Final *specFinal `toml:"final,omitempty"`
	// LintCmd overrides the detected linter; "off" turns the lint gate off.
	LintCmd string `toml:"lint_cmd,omitempty"`
	// MaxFileLines: a program file over it may barely grow; 0 is the default, below 0 off.
	MaxFileLines int `toml:"max_file_lines,omitempty"`
	// RefactorEvery finished items one round cleans up; 0 is the default, below 0 off.
	RefactorEvery int `toml:"refactor_every,omitempty"`
	// RefactorAt is the ledger size at the last refactor round.
	RefactorAt int `toml:"refactor_at,omitempty"`
	// Bases holds HEAD as an item's first attempt found it: a failed attempt is
	// committed too, and the next attempt's changes count from here.
	Bases map[string]string `toml:"bases,omitempty"`
	// Flaky holds the failure of a suite that failed and then passed with nothing
	// changed; the next item round is asked to make that test deterministic first.
	Flaky string `toml:"flaky,omitempty"`
}

type specFinal struct {
	Items   int       `toml:"items"`
	Blocked int       `toml:"blocked"`
	Commit  string    `toml:"commit,omitempty"`
	At      time.Time `toml:"at"`
	Version string    `toml:"version,omitempty"`
}

// specLedger.Title lets a section moved to another file match as a rename, not a removal.
type specLedger struct {
	Hash      string `toml:"hash"`
	Title     string `toml:"title,omitempty"`
	File      string `toml:"file,omitempty"` // spec file, relative to spec_dir
	CoveredBy string `toml:"covered_by,omitempty"`
	Commit    string `toml:"commit,omitempty"`
	// An adopted item (already covered by a test) has no Commit.
	At      time.Time `toml:"at,omitempty"`
	Version string    `toml:"version,omitempty"`
	// Named: recorded under the rule that only a test's name counts, so reconcile
	// holds it to that; an older entry may rest on a comment and stays done.
	Named bool `toml:"named,omitempty"`
	// Checked: when and by which codehalter the completion check found the item done;
	// a round that rebuilds the item writes a new entry without it.
	Checked string `toml:"checked,omitempty"`
}

func specConfigPath(cwd string) string {
	return filepath.Join(cwd, sessionDir, specConfigName)
}

// loadSpecConfig returns nil, nil when /spec has never been set up.
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
	buf.WriteString("# /spec loop state. spec_dir, out_dir and target come from the /spec command,\n" +
		"# or from the questions the first /spec asked.\n" +
		"# Questions for you are in the spec directory's QUESTIONS.md, not here.\n" +
		"# accept_open_markers = true starts the loop despite open REVIEW/TBD markers.\n" +
		"# test_cmd overrides the detected test command (run from out_dir); lint_cmd the\n" +
		"# detected linter (\"off\" skips linting). max_file_lines is the size budget of a\n" +
		"# program file (1500, -1 off); refactor_every the finished items between two\n" +
		"# refactor rounds (10, -1 off).\n" +
		"# [items] is what the loop finished, with the spec text it was built from:\n" +
		"# editing that section makes the next run redo the item, deleting it makes\n" +
		"# the next run remove its code. Delete an entry to forget an item.\n\n")
	if err := toml.NewEncoder(&buf).Encode(cfg); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(specConfigPath(cwd)), 0o755); err != nil {
		return err
	}
	return writeFileAtomic(specConfigPath(cwd), buf.Bytes(), 0o644)
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

// parseSpecArgs has no positional form: the setup questions read their options off the project.
func parseSpecArgs(args string) (cmd string, err error) {
	switch strings.TrimSpace(args) {
	case "":
		return "resume", nil
	case "status":
		return "status", nil
	case "stop", "abort":
		return "stop", nil
	case "redo":
		return "redo", nil
	}
	return "", fmt.Errorf("usage: /spec (start or resume), /spec status, /spec stop (after the round in flight), /spec abort (at once), /spec redo")
}

// specRedoTargets takes an item id, a section id with a paraphrased slug, or a spec file
// meaning every item it defines.
func specRedoTargets(cfg *specConfig, idx *specIndex, targets []string) (ids, unknown []string) {
	seen := map[string]bool{}
	add := func(id string) {
		if !seen[id] {
			seen[id] = true
			ids = append(ids, id)
		}
	}
	for _, t := range targets {
		if _, ok := idx.items[t]; ok {
			add(t)
			continue
		}
		// A paraphrased slug: the section number still pins it.
		if strings.HasPrefix(t, "§") && strings.Contains(t, "#") {
			if id := sectionByNumber(idx, t); id != "" {
				add(id)
				continue
			}
		}
		file := strings.TrimPrefix(strings.TrimPrefix(t, cfg.SpecDir+"/"), "./")
		found := false
		for d, doc := range idx.docs {
			if doc.rel == file || strings.TrimSuffix(doc.rel, ".md") == file {
				for _, id := range idx.order {
					if idx.items[id].Doc == d {
						add(id)
					}
				}
				found = true
			}
		}
		if !found {
			unknown = append(unknown, t)
		}
	}
	return ids, unknown
}

// sectionByNumber resolves "§<file>#<n>-<anything>" to the unique section numbered n, or "".
func sectionByNumber(idx *specIndex, t string) string {
	hash := strings.Index(t, "#")
	stem, slug := t[:hash+1], t[hash+1:]
	num := slug
	if i := strings.Index(slug, "-"); i > 0 {
		num = slug[:i]
	}
	if num == "" || strings.Trim(num, "0123456789.") != "" {
		return ""
	}
	match := ""
	for _, id := range idx.order {
		if strings.HasPrefix(id, stem) && (strings.HasPrefix(id[len(stem):], num+"-") || id[len(stem):] == num) {
			if match != "" {
				return ""
			}
			match = id
		}
	}
	return match
}

// reopen marks items so their still-present tests do not count as done until a round passes.
func (c *specConfig) reopen(ids []string) {
	if c.Redo == nil {
		c.Redo = map[string]string{}
	}
	for _, id := range ids {
		delete(c.Items, id)
		c.Redo[id] = "redo"
	}
}

// open means a test naming the item exists but proves too little.
func (c *specConfig) open(id string) bool {
	return c.Attempts[id] > 0 || c.Redo[id] != ""
}

// done is the ledger's word: specReconcile keeps it in step with the tests.
func (c *specConfig) done(id string) bool {
	_, ok := c.Items[id]
	return ok
}

type specDoc struct {
	rel   string // path relative to the spec dir, forward slashes
	lines []string
	// navLine marks <sub> lines and <!-- nav --> blocks, whose links never point at what an item needs.
	navLine []bool
	fenced  []bool
}

type specSection struct {
	doc        int
	start, end int // [start, end) line range; start is the heading line
	level      int
	title      string
	slug       string
}

// An id's home is its heading, else a table row that starts with it, else a mention.
const (
	specDefHeading = iota
	specDefTableRow
	specDefMention
	// specDefSection is a headed section with no id of its own, under the id "§<file stem>#<slug>".
	specDefSection
)

type specItem struct {
	ID    string
	Kind  int
	Doc   int
	Line  int
	Title string
	prio  int // lower wins
}

// A heading that names the id further in still beats an index table row that merely lists it.
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
	order    []string
	patterns []*regexp.Regexp
	// questions: the spec's QUESTIONS.md by item id, written by the loop, answered by the user.
	questions map[string][]specQuestion
}

var (
	specLinkRe  = regexp.MustCompile(`(!?)\[[^\]]*\]\(([^)\s]+)\)`)
	headingRe   = regexp.MustCompile(`^(#{1,6})\s+(.*?)\s*#*\s*$`)
	specNavOpen = "<!-- nav -->"
	specNavEnd  = "<!-- /nav -->"
)

// scanSpec reads root-level chapters before subdirectories, so a chapter defines an id an
// inventory repeats.
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
			if strings.EqualFold(rel, specQuestionsFile) {
				return nil // no items: read below
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

	switch data, err := os.ReadFile(filepath.Join(root, specQuestionsFile)); {
	case err == nil:
		idx.questions = parseSpecQuestions(string(data))
	case !os.IsNotExist(err):
		return nil, err
	}
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
	// Headed and section items interleave in document order, so a chapter's file formats
	// come before the flows that write them.
	rank := [...]int{specDefHeading: 0, specDefSection: 0, specDefTableRow: 1, specDefMention: 2}
	sort.Slice(idx.order, func(i, j int) bool {
		a, b := idx.items[idx.order[i]], idx.items[idx.order[j]]
		if rank[a.Kind] != rank[b.Kind] {
			return rank[a.Kind] < rank[b.Kind]
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
				head := strings.TrimSpace(title)
				if !strings.HasPrefix(head, id) {
					prio = specPrioHeadingMention
					// A heading that opens with another id, or names several, is about none of these.
					inHead := idx.idsIn(head)
					if len(inHead) > 1 || slices.ContainsFunc(inHead, func(o string) bool { return strings.HasPrefix(head, o) }) {
						k, prio = specDefMention, specPrioMention
					}
				}
			case specDefTableRow:
				prio = specPrioTableRow
				cells := strings.Split(strings.TrimSpace(ln), "|")
				if !containsID(cells[1], id) {
					k, prio = specDefMention, specPrioMention // a later cell, not the row's subject
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

func containsID(s, id string) bool { return idBoundaryIndex(s, id) >= 0 }

// githubSlug is the anchor GitHub gives a heading, which links inside a spec target.
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

// addSectionItems makes items of prose sections that define no id, or nothing would schedule them.
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

// isIDTable: a section that is mostly id-led table rows adds nothing, the rows are items already.
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

func (idx *specIndex) text(doc, start, end int) string {
	return strings.Join(idx.docs[doc].lines[start:end], "\n")
}

func (idx *specIndex) textNoNav(doc, start, end int) string {
	var keep []string
	for i := start; i < end; i++ {
		if !idx.docs[doc].navLine[i] {
			keep = append(keep, idx.docs[doc].lines[i])
		}
	}
	return strings.Join(keep, "\n")
}

type specSlice struct {
	Text    string
	Related []string // spec paths (with #anchor) that were relevant but did not fit or belong to another item
	Images  []string
	// Reached holds the sectionKey of every section shown, for the "never reached" report.
	Reached []string
}

func sectionKey(rel string, start int) string { return fmt.Sprintf("%s:%d", rel, start) }

// slice assembles an item's own section, the table rows it cites and the sections it links to.
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
		lines := idx.docs[it.Doc].lines
		start, end := it.Line, it.Line+1
		for start > 0 && strings.TrimSpace(lines[start-1]) != "" {
			start--
		}
		for end < len(lines) && strings.TrimSpace(lines[end]) != "" {
			end++
		}
		primary = specSection{doc: it.Doc, start: start, end: end}
		primaryText = idx.text(it.Doc, start, end)
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

	var navFree []string
	for i := primary.start; i < primary.end && i < len(idx.docs[primary.doc].lines); i++ {
		if !idx.docs[primary.doc].navLine[i] {
			navFree = append(navFree, idx.docs[primary.doc].lines[i])
		}
	}
	body := strings.Join(navFree, "\n")

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

	for _, m := range specLinkRe.FindAllStringSubmatch(body, -1) {
		if m[1] == "!" || strings.Contains(m[2], "://") || strings.HasPrefix(m[2], "mailto:") {
			continue
		}
		target, anchor, _ := strings.Cut(m[2], "#")
		doc := primary.doc
		if target != "" {
			rel := filepath.ToSlash(filepath.Clean(filepath.Join(filepath.Dir(idx.docs[primary.doc].rel), target)))
			doc = slices.IndexFunc(idx.docs, func(d specDoc) bool { return d.rel == rel })
			if doc < 0 {
				continue
			}
		}
		var sec specSection
		var ok bool
		if anchor != "" {
			if i := slices.IndexFunc(idx.sections, func(s specSection) bool { return s.doc == doc && s.slug == anchor }); i >= 0 {
				sec, ok = idx.sections[i], true
			}
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
		// Another heading item's section belongs to that item's round: name it, don't paste it.
		if sec.start < len(idx.docs[doc].lines) {
			if other := idx.idsIn(idx.docs[doc].lines[sec.start]); len(other) > 0 && !(len(other) == 1 && other[0] == id) {
				if o := idx.items[other[0]]; o != nil && o.Kind == specDefHeading {
					out.Related = appendUnique(out.Related, ref)
					continue
				}
			}
		}
		text := idx.textNoNav(doc, sec.start, sec.end)
		// An index names many items and specifies none: pointing at it is enough.
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

// specIndexThreshold is how many other heading items a section may name before it is an index.
const specIndexThreshold = 5

func (idx *specIndex) headingItemsIn(text, self string) int {
	n := 0
	for _, other := range idx.idsIn(text) {
		if o := idx.items[other]; other != self && o != nil && o.Kind == specDefHeading {
			n++
		}
	}
	return n
}

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

func appendUnique(s []string, v string) []string {
	for _, x := range s {
		if x == v {
			return s
		}
	}
	return append(s, v)
}

func (idx *specIndex) openMarkers(markers []string, specDirRel string) []string {
	if len(markers) == 0 {
		return nil // an empty alternation would match almost every line
	}
	quoted := make([]string, len(markers))
	for i, m := range markers {
		quoted[i] = regexp.QuoteMeta(m)
	}
	// Not \b: a marker may start or end with punctuation, like "TODO:" or "[open]".
	re := regexp.MustCompile(`(?:^|\W)(?:` + strings.Join(quoted, "|") + `)(?:\W|$)`)
	var out []string
	for _, d := range idx.docs {
		for i, ln := range d.lines {
			if re.MatchString(ln) {
				out = append(out, fmt.Sprintf("%s/%s:%d", specDirRel, d.rel, i+1))
			}
		}
	}
	return out
}

// specTestToken is an id as a test-name token ("F2.3" → "f2_3") valid in Rust, Go, Python and JS.
func specTestToken(id string) string {
	// A section id normalises to a leading digit, which no test name can start with.
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

// idBoundaryIndex matches whole ids only: "F2.3" is not found in "F2.30" or "F2.3.1".
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

var specSkipDirs = map[string]bool{
	"target": true, "node_modules": true, ".git": true, "vendor": true,
	"dist": true, "build": true, ".venv": true, "__pycache__": true,
}

func skipWalkDir(name string) bool { return specSkipDirs[name] || strings.HasPrefix(name, ".") }

var cfgTestModRe = regexp.MustCompile(`^\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{`)

// testSourceText returns a Rust file from #[cfg(test)] on, so an id in a production doc
// comment above it does not pass for a test.
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
	// An inline `#[cfg(test)] mod tests {` starts the test code; the attribute on
	// anything else (`mod tests;`, one `use` or `fn`) marks only that item.
	for from := 0; ; {
		i := strings.Index(content[from:], "#[cfg(test)]")
		if i < 0 {
			break
		}
		i += from
		if cfgTestModRe.MatchString(content[i+len("#[cfg(test)]"):]) {
			return content[i:]
		}
		from = i + 1
	}
	if strings.Contains(content, "#[test]") {
		return content
	}
	return ""
}

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
			if path != outAbs && skipWalkDir(d.Name()) {
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
		text := testNameText(rel, string(data))
		if text == "" {
			return nil
		}
		testFiles++
		for _, id := range ids {
			if _, done := covered[id]; !done && namesItem(text, id) {
				covered[id] = filepath.ToSlash(rel)
			}
		}
		return nil
	})
	return covered, testFiles, err
}

// detectSpecTestCmd prefers a justfile `test` recipe: it carries what `cargo test` doesn't know.
func detectSpecTestCmd(outAbs string) string {
	if justRecipe(outAbs, "test") {
		return "just test"
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

func justRecipe(dir, name string) bool {
	for _, file := range []string{"justfile", "Justfile", ".justfile"} {
		data, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil {
			continue
		}
		for _, ln := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(ln, name+":") || strings.HasPrefix(ln, name+" ") {
				return true
			}
		}
	}
	return false
}

// specItemHash ignores whitespace so a reflowed paragraph does not re-open a finished item.
func specItemHash(idx *specIndex, id string) string {
	it := idx.items[id]
	if it == nil {
		return ""
	}
	// Not the round's slice: its linked sections would make an edit anywhere look like a change here.
	text := ""
	if sec, ok := idx.sectionAt(it.Doc, it.Line); ok && sec.start == it.Line {
		text = idx.text(it.Doc, sec.start, sec.end)
	} else if lines := idx.docs[it.Doc].lines; it.Line >= 0 && it.Line < len(lines) {
		text = lines[it.Line]
	}
	if text == "" {
		return ""
	}
	// An answer is spec text of the item: changing it rebuilds the item.
	for _, q := range idx.questions[id] {
		if q.Answer != "" {
			text += "\n" + q.Question + "\n" + q.Answer
		}
	}
	sum := sha256.Sum256([]byte(strings.Join(strings.Fields(text), " ")))
	return hex.EncodeToString(sum[:8])
}

// specDelta's Renamed and Adopted are already applied to the config by specReconcile.
type specDelta struct {
	Changed []string
	Removed []string
	Renamed []string // "old → new"
	Adopted int      // parameter and tool rows a test names, recorded without a round
	// Reopened: ledger items no test names any more; a comment or a string does not count.
	Reopened []string
	// Unnamed: so many would reopen at once that it reads as a scan problem, and none does.
	Unnamed int
	// Legacy: done before the test-name rule and named by no test; they stay done.
	Legacy []string
	// Vanished: Removed is most of the ledger, which reads as a missing spec, so nothing is removed.
	Vanished bool
}

// specReconcile mutates cfg: renames carry their entry; a done item no test names
// any more is open again; and a parameter or tool row a test names is adopted at its
// current text. A flow or a section is done only by a round of its own.
func specReconcile(cfg *specConfig, idx *specIndex, covered map[string]string) specDelta {
	var d specDelta
	if cfg.Items == nil {
		cfg.Items = map[string]specLedger{}
	}
	ledgered := len(cfg.Items)
	hashes := make(map[string]string, len(idx.order))
	for _, id := range idx.order {
		hashes[id] = specItemHash(idx, id)
	}
	var gone []string
	for id := range cfg.Items {
		if _, live := idx.items[id]; !live {
			gone = append(gone, id)
		}
	}
	sort.Strings(gone)
	for _, old := range gone {
		led := cfg.Items[old]
		match := ""
		for _, id := range idx.order {
			if _, known := cfg.Items[id]; known {
				continue
			}
			if hashes[id] == led.Hash || (led.Title != "" && strings.EqualFold(idx.items[id].Title, led.Title)) {
				match = id
				break
			}
		}
		if match == "" {
			d.Removed = append(d.Removed, old)
			continue
		}
		led.Hash = hashes[match]
		led.Title = idx.items[match].Title
		led.File = idx.docs[idx.items[match].Doc].rel
		cfg.Items[match] = led
		delete(cfg.Items, old)
		d.Renamed = append(d.Renamed, old+" → "+match)
	}
	// Below 4 items "more than half" is one or two deletions, too few to call the spec missing.
	d.Vanished = len(d.Removed) > 0 && (len(d.Removed) == ledgered || ledgered >= 4 && 2*len(d.Removed) > ledgered)
	var unnamed []string
	for _, id := range idx.order {
		switch led, known := cfg.Items[id]; {
		case !known || covered[id] != "":
		case led.Named:
			unnamed = append(unnamed, id)
		default:
			// Reopening these at once would queue a round each (130 in one project):
			// they are reported, and /spec redo rebuilds the ones that matter.
			d.Legacy = append(d.Legacy, id)
		}
	}
	if len(unnamed) > 0 && 2*len(unnamed) > len(cfg.Items) && len(cfg.Items) >= 4 {
		d.Unnamed = len(unnamed)
	} else {
		for _, id := range unnamed {
			delete(cfg.Items, id)
		}
		d.Reopened = unnamed
	}
	for _, id := range idx.order {
		led, known := cfg.Items[id]
		switch {
		// After a failed round the test exists but the suite did not pass: not done.
		case !known && covered[id] != "" && !cfg.open(id) && (idx.items[id].Kind == specDefTableRow || idx.items[id].Kind == specDefMention):
			cfg.Items[id] = specLedger{Hash: hashes[id], Title: idx.items[id].Title,
				File: idx.docs[idx.items[id].Doc].rel, CoveredBy: covered[id],
				At: time.Now().UTC(), Version: versionStamp(), Named: true}
			d.Adopted++
		case known && led.Hash != hashes[id]:
			d.Changed = append(d.Changed, id)
		case known:
			// An attempt count here predates the done rule.
			delete(cfg.Attempts, id)
		}
	}
	return d
}

func specDirCandidates(cwd string) []string {
	nameScore := map[string]int{"spec": 3, "specs": 3, "requirements": 3, "design": 2, "doc": 2, "docs": 2}
	type cand struct {
		rel   string
		score int
		mds   int
		depth int
	}
	var out []cand
	consider := func(rel string, depth int) {
		entries, err := os.ReadDir(filepath.Join(cwd, rel))
		if err != nil {
			return
		}
		mds := 0
		for _, e := range entries {
			if !e.IsDir() && strings.EqualFold(filepath.Ext(e.Name()), ".md") {
				mds++
			}
		}
		if mds < 2 {
			return
		}
		out = append(out, cand{rel: rel, score: nameScore[strings.ToLower(filepath.Base(rel))], mds: mds, depth: depth})
	}
	consider(".", 0)
	top, err := os.ReadDir(cwd)
	if err != nil {
		return nil
	}
	for _, e := range top {
		if !e.IsDir() || skipWalkDir(e.Name()) {
			continue
		}
		consider(e.Name(), 1)
		sub, err := os.ReadDir(filepath.Join(cwd, e.Name()))
		if err != nil {
			continue
		}
		for _, s := range sub {
			if s.IsDir() && !skipWalkDir(s.Name()) {
				consider(filepath.ToSlash(filepath.Join(e.Name(), s.Name())), 2)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].score != out[j].score {
			return out[i].score > out[j].score
		}
		if out[i].mds != out[j].mds {
			return out[i].mds > out[j].mds
		}
		if out[i].depth != out[j].depth {
			return out[i].depth < out[j].depth
		}
		return out[i].rel < out[j].rel
	})
	rels := make([]string, 0, len(out))
	for _, c := range out {
		// A directory inside one already offered is part of that spec, not a rival.
		nested := false
		for _, kept := range rels {
			if kept != "." && strings.HasPrefix(c.rel, kept+"/") {
				nested = true
				break
			}
		}
		if !nested {
			rels = append(rels, c.rel)
		}
	}
	return rels
}

func specEntryPage(specAbs string) (rel, content string) {
	entries, err := os.ReadDir(specAbs)
	if err != nil {
		return "", ""
	}
	var mds []string
	for _, e := range entries {
		if !e.IsDir() && strings.EqualFold(filepath.Ext(e.Name()), ".md") {
			mds = append(mds, e.Name())
		}
	}
	if len(mds) == 0 {
		return "", ""
	}
	sort.Strings(mds)
	pick := mds[0]
	for _, name := range mds {
		if l := strings.ToLower(name); l == "readme.md" || l == "index.md" {
			pick = name
			break
		}
	}
	data, err := os.ReadFile(filepath.Join(specAbs, pick))
	if err != nil {
		return "", ""
	}
	return pick, string(data)
}

func specOutDirOptions(cwd, specDir, suggested string) []string {
	var out []string
	add := func(d string) {
		d = strings.Trim(filepath.ToSlash(filepath.Clean(d)), "/")
		if d == "" || d == "." || d == specDir || slices.Contains(out, d) || len(out) >= 3 {
			return
		}
		out = append(out, d)
	}
	add(suggested)
	entries, _ := os.ReadDir(cwd)
	for _, e := range entries {
		if !e.IsDir() || skipWalkDir(e.Name()) {
			continue
		}
		if lang, _ := manifestStack(filepath.Join(cwd, e.Name())); lang != "" {
			add(e.Name())
		}
	}
	return out
}

func specTargetOptions(cwd, outDir, suggested, entry string) []string {
	var out []string
	add := func(t string) {
		t = strings.TrimSpace(t)
		if t == "" || slices.Contains(out, t) || len(out) >= 3 {
			return
		}
		out = append(out, t)
	}
	add(suggested)
	if lang, deps := manifestStack(filepath.Join(cwd, outDir)); lang != "" {
		if len(deps) > 0 {
			add(lang + " with " + strings.Join(deps, ", "))
		} else {
			add(lang)
		}
	}
	if _, guess := specGuessTarget(entry); guess != "" {
		add(guess)
	}
	return out
}

// manifestStack is not a build-system parser: a manifest it cannot read yields no dependencies.
func manifestStack(dir string) (lang string, deps []string) {
	read := func(name string) (string, bool) {
		b, err := os.ReadFile(filepath.Join(dir, name))
		return string(b), err == nil
	}
	first3 := func(names []string) []string {
		if len(names) > 3 {
			names = names[:3]
		}
		return names
	}
	if s, ok := read("Cargo.toml"); ok {
		var names []string
		in := false
		for _, line := range strings.Split(s, "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "[") {
				in = line == "[dependencies]"
				continue
			}
			if name, _, found := strings.Cut(line, "="); in && found && name != "" && !strings.HasPrefix(name, "#") {
				names = append(names, strings.TrimSpace(name))
			}
		}
		return "rust", first3(names)
	}
	if s, ok := read("go.mod"); ok {
		var names []string
		in := false
		for _, line := range strings.Split(s, "\n") {
			line = strings.TrimSpace(line)
			switch {
			case line == "require (":
				in = true
			case line == ")":
				in = false
			case strings.HasPrefix(line, "require "):
				names = append(names, path.Base(strings.Fields(line)[1]))
			case in && line != "" && !strings.HasPrefix(line, "//"):
				if f := strings.Fields(line); len(f) >= 1 && !strings.Contains(line, "// indirect") {
					names = append(names, path.Base(f[0]))
				}
			}
		}
		return "go", first3(names)
	}
	if s, ok := read("package.json"); ok {
		var pkg struct {
			Dependencies map[string]string `json:"dependencies"`
		}
		var names []string
		if json.Unmarshal([]byte(s), &pkg) == nil {
			for name := range pkg.Dependencies {
				names = append(names, name)
			}
			sort.Strings(names)
		}
		lang = "javascript"
		if _, err := os.Stat(filepath.Join(dir, "tsconfig.json")); err == nil {
			lang = "typescript"
		}
		return lang, first3(names)
	}
	if _, ok := read("pyproject.toml"); ok {
		return "python", nil
	}
	if _, ok := read("CMakeLists.txt"); ok {
		return "c", nil
	}
	if _, ok := read("build.zig"); ok {
		return "zig", nil
	}
	return "", nil
}

// specTargetGuess is ordered: a spec that says "rust" and "gtk4-rs" is a GTK project.
var specTargetGuess = []struct {
	keywords []string
	outDir   string
	target   string
}{
	{[]string{"gtk4-rs", "gtk 4", "gtk4", "libadwaita"}, "rust", "rust with gtk4-rs and libadwaita"},
	{[]string{"tauri"}, "src-tauri", "rust with tauri"},
	{[]string{"cargo", "rust"}, "rust", "rust"},
	{[]string{"go.mod", "golang", "goroutine"}, "go", "go"},
	{[]string{"react", "typescript", "tsx"}, "web", "typescript with react"},
	{[]string{"fastapi", "django", "python"}, "py", "python"},
}

func specGuessTarget(entry string) (outDir, target string) {
	low := strings.ToLower(entry)
	for _, g := range specTargetGuess {
		for _, k := range g.keywords {
			if strings.Contains(low, k) {
				return g.outDir, g.target
			}
		}
	}
	return "", ""
}

// specSectionFromText parses rather than using the live index: it reads a file recovered from git.
func specSectionFromText(content, id string) string {
	slug := id
	if i := strings.LastIndex(id, "#"); i >= 0 {
		slug = id[i+1:]
	}
	lines := strings.Split(content, "\n")
	heading := func(l string) (level int, title string) {
		t := strings.TrimSpace(l)
		if !strings.HasPrefix(t, "#") {
			return 0, ""
		}
		level = len(t) - len(strings.TrimLeft(t, "#"))
		return level, strings.TrimSpace(t[level:])
	}
	start, level := -1, 0
	for i, l := range lines {
		if h, title := heading(l); h > 0 && githubSlug(title) == slug {
			start, level = i, h
			break
		}
	}
	if start < 0 {
		for _, l := range lines {
			if containsID(l, id) {
				return strings.TrimSpace(l)
			}
		}
		return ""
	}
	for i := start + 1; i < len(lines); i++ {
		if h, _ := heading(lines[i]); h > 0 && h <= level {
			return strings.Join(lines[start:i], "\n")
		}
	}
	return strings.Join(lines[start:], "\n")
}

func renderSpecDelta(d specDelta) string {
	// Adoptions alone are bookkeeping, not news.
	if len(d.Changed) == 0 && len(d.Removed) == 0 && len(d.Renamed) == 0 && len(d.Reopened) == 0 && d.Unnamed == 0 && len(d.Legacy) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("**/spec changes since the last run**\n\n")
	if len(d.Changed) > 0 {
		fmt.Fprintf(&b, "- %d item(s) changed in the spec and will be redone: %s\n", len(d.Changed), strings.Join(d.Changed, ", "))
	}
	switch {
	case d.Vanished:
		fmt.Fprintf(&b, "- ⚠ %d built item(s) are gone at once, more than half of the ledger: the spec looks missing or unreadable (directory moved or deleted, or ids no longer recognised), so nothing is deleted\n", len(d.Removed))
	case len(d.Removed) > 0:
		fmt.Fprintf(&b, "- %d item(s) are gone from the spec, their code and tests are removed first: %s\n", len(d.Removed), strings.Join(d.Removed, ", "))
	}
	if len(d.Renamed) > 0 {
		fmt.Fprintf(&b, "- %d item(s) moved or were retitled, their record travelled with them: %s\n", len(d.Renamed), strings.Join(d.Renamed, ", "))
	}
	if len(d.Reopened) > 0 {
		fmt.Fprintf(&b, "- %d done item(s) are open again: no test is named after them, and a comment or a string that mentions one does not count: %s\n", len(d.Reopened), strings.Join(d.Reopened, ", "))
	}
	if d.Unnamed > 0 {
		fmt.Fprintf(&b, "- ⚠ %d done item(s) are named by no test, more than half of the ledger: that looks like a problem reading the tests, not missing work, so none is reopened\n", d.Unnamed)
	}
	if d.Adopted > 0 {
		fmt.Fprintf(&b, "- %d parameter or tool row(s) a test is named after were recorded at their current text\n", d.Adopted)
	}
	if len(d.Legacy) > 0 {
		ids := strings.Join(d.Legacy[:min(len(d.Legacy), 12)], ", ")
		if len(d.Legacy) > 12 {
			ids += ", …"
		}
		fmt.Fprintf(&b, "- %d item(s) were counted done before only a test's name counted, and no test is named after them (a comment or a string mentions them). They stay done; `/spec redo <ids>` rebuilds any of them: %s\n", len(d.Legacy), ids)
	}
	b.WriteString("\n")
	return b.String()
}

// nextSpecItem passes over an item waiting on its question, and those skip names.
func nextSpecItem(idx *specIndex, cfg *specConfig, skip func(string) bool) string {
	for _, it := range idx.order {
		if !cfg.done(it) && !idx.asked(it) && (skip == nil || !skip(it)) {
			return it
		}
	}
	return ""
}

type specTally struct{ total, done, asked int }

func specTallies(cfg *specConfig, idx *specIndex) (kinds [4]specTally, perDoc []specTally) {
	perDoc = make([]specTally, len(idx.docs))
	for _, id := range idx.order {
		it := idx.items[id]
		k, d := &kinds[it.Kind], &perDoc[it.Doc]
		k.total++
		d.total++
		switch {
		case cfg.done(id):
			k.done++
			d.done++
		case idx.asked(id):
			k.asked++
			d.asked++
		}
	}
	return kinds, perDoc
}

func renderSpecStatus(cfg *specConfig, idx *specIndex, testFiles int, testCmd string) string {
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

	kinds, perDoc := specTallies(cfg, idx)
	fmt.Fprintf(&b, "Covered: %d/%d flows · %d/%d sections (formats, screens, rules) · %d/%d parameters and tools · %d/%d cited-only ids\n\n",
		kinds[specDefHeading].done, kinds[specDefHeading].total, kinds[specDefSection].done, kinds[specDefSection].total,
		kinds[specDefTableRow].done, kinds[specDefTableRow].total, kinds[specDefMention].done, kinds[specDefMention].total)
	b.WriteString("| file | covered | waiting on a question | total |\n|---|---|---|---|\n")
	for d, t := range perDoc {
		if t.total > 0 {
			fmt.Fprintf(&b, "| %s | %d | %d | %d |\n", idx.docs[d].rel, t.done, t.asked, t.total)
		}
	}
	if next := nextSpecItem(idx, cfg, nil); next != "" {
		fmt.Fprintf(&b, "\nNext: **%s**", next)
		if t := idx.items[next].Title; t != "" {
			fmt.Fprintf(&b, " (%s)", t)
		}
		b.WriteString("\n")
	} else {
		b.WriteString("\nNothing left: every id is covered or waits on a question.\n")
	}
	if open := specOpenQuestions(idx); len(open) > 0 {
		fmt.Fprintf(&b, "\nWaiting on your answer in `%s/%s` (write it after **Answer:**, then run /spec):\n", cfg.SpecDir, specQuestionsFile)
		for _, q := range open {
			fmt.Fprintf(&b, "- **%s**: %s%s\n", q.ID, q.Question, specStaleNote(q))
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
