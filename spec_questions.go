package main

import (
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"
)

// specQuestionsFile sits in the spec directory: a question is a gap in the spec,
// so its answer is spec too. codehalter writes the questions, the user the answers;
// it is no item source, and scanSpec reads it into specIndex.questions.
const specQuestionsFile = "QUESTIONS.md"

const specQuestionsHead = "# Questions\n\n" +
	"The /spec loop writes a question here when the spec does not decide something a round needs, " +
	"and skips that item until the question has an answer. Write the answer after **Answer:**, in your " +
	"own words or as the number of an option, then run /spec. An answered question is part of the spec " +
	"until its item is built with it; then codehalter removes it from here.\n"

type specOption struct {
	Choice  string `json:"choice"`
	Example string `json:"example"`
}

type specQuestion struct {
	ID       string
	Question string
	Quote    string
	QuoteAt  string // file:line, found by codehalter, not claimed by the model
	Options  []specOption
	Stopped  string // a stuck item's question: what stopped its last attempt
	// Parsed back from the file.
	Text    string // the whole section as the file holds it
	Answer  string
	Version string // the codehalter that asked; "" in a file from before it was written down
}

var (
	specQuestionHeadRe = regexp.MustCompile(`^##\s+(\S+)\s+·\s*(.*?)\s*$`)
	specAnswerRe       = regexp.MustCompile(`^\*{0,2}Answer\*{0,2}:\*{0,2}\s*(.*)$`)
	specAskedByRe      = regexp.MustCompile(`^Asked \S+ by codehalter (.+?) while building `)
)

// parseSpecQuestions keys the sections by item id; an item may have asked more than once.
func parseSpecQuestions(text string) map[string][]specQuestion {
	out := map[string][]specQuestion{}
	_, secs := parseSpecQuestionFile(text)
	for _, q := range secs {
		out[q.ID] = append(out[q.ID], q)
	}
	return out
}

// parseSpecQuestionFile splits the file into the text before its first question
// and its questions in order. A `---` line is the separator between two
// questions, not part of either.
func parseSpecQuestionFile(text string) (head string, secs []specQuestion) {
	var cur *specQuestion
	var body, headLines []string
	inAnswer, inFence := false, false
	flush := func() {
		if cur != nil {
			// Newlines only: the empty answer line keeps its space after "**Answer:**".
			cur.Text = strings.Trim(strings.Join(body, "\n"), "\n")
			cur.Answer = strings.TrimSpace(cur.Answer)
			secs = append(secs, *cur)
		}
	}
	for _, ln := range strings.Split(text, "\n") {
		// A test's output inside a fence may hold a heading or an "Answer:" of its own.
		if strings.HasPrefix(strings.TrimSpace(ln), "```") {
			inFence = !inFence
		}
		if !inFence && strings.TrimSpace(ln) == "---" {
			continue
		}
		if m := specQuestionHeadRe.FindStringSubmatch(ln); m != nil && !inFence {
			flush()
			cur, body, inAnswer = &specQuestion{ID: m[1], Question: m[2]}, nil, false
		}
		if cur == nil {
			headLines = append(headLines, ln)
			continue
		}
		body = append(body, ln)
		if m := specAskedByRe.FindStringSubmatch(ln); m != nil && cur.Version == "" && !inFence {
			cur.Version = m[1]
		}
		switch m := specAnswerRe.FindStringSubmatch(strings.TrimSpace(ln)); {
		case m != nil && !inFence:
			inAnswer = true
			cur.Answer = m[1]
		case inAnswer:
			cur.Answer += "\n" + ln
		}
	}
	flush()
	return strings.TrimSpace(strings.Join(headLines, "\n")), secs
}

// writeSpecQuestionFile writes the head and the questions, `---` between two.
func writeSpecQuestionFile(path, head string, secs []string) error {
	var b strings.Builder
	b.WriteString(strings.TrimSpace(head) + "\n")
	for i, s := range secs {
		if i > 0 {
			b.WriteString("\n---\n")
		}
		b.WriteString("\n" + strings.Trim(s, "\n") + "\n")
	}
	return writeFileAtomic(path, []byte(b.String()), 0o644)
}

// removeAnsweredQuestions takes the item's answered questions out of the file once
// a round has built the item with them; an unanswered one stays.
func removeAnsweredQuestions(specAbs, id string) (removed int, err error) {
	path := filepath.Join(specAbs, specQuestionsFile)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	head, secs := parseSpecQuestionFile(string(data))
	var kept []string
	for _, q := range secs {
		if q.ID == id && q.Answer != "" {
			removed++
			continue
		}
		kept = append(kept, q.Text)
	}
	if removed == 0 {
		return 0, nil
	}
	return removed, writeSpecQuestionFile(path, head, kept)
}

// asked: the item has a question without an answer, so a round would only ask again.
func (idx *specIndex) asked(id string) bool {
	for _, q := range idx.questions[id] {
		if q.Answer == "" {
			return true
		}
	}
	return false
}

// specOpenQuestions in spec order, for the reports.
func specOpenQuestions(idx *specIndex) []specQuestion {
	var out []specQuestion
	seen := map[string]bool{}
	for _, id := range append(slices.Clone(idx.order), slices.Sorted(maps.Keys(idx.questions))...) {
		if seen[id] {
			continue
		}
		seen[id] = true
		for _, q := range idx.questions[id] {
			if q.Answer == "" {
				out = append(out, q)
			}
		}
	}
	return out
}

// specStaleNote says a question came from another codehalter than this one, whose
// cause (three questions once came from a codehalter bug) may be gone.
func specStaleNote(q specQuestion) string {
	switch q.Version {
	case versionStamp():
		return ""
	case "":
		return " (asked by an older codehalter: if its cause was in codehalter, answer `1` to try again)"
	}
	return " (asked by codehalter " + q.Version + ", not this one: if its cause was in codehalter, answer `1` to try again)"
}

// answered is every answered question of the item, as the file words it.
func (idx *specIndex) answered(id string) string {
	var parts []string
	for _, q := range idx.questions[id] {
		if q.Answer != "" {
			parts = append(parts, q.Text)
		}
	}
	return strings.Join(parts, "\n\n")
}

func appendSpecQuestion(specAbs, title string, q specQuestion) error {
	path := filepath.Join(specAbs, specQuestionsFile)
	old, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	head, secs := specQuestionsHead, []specQuestion(nil)
	if len(old) > 0 {
		head, secs = parseSpecQuestionFile(string(old))
	}
	var b strings.Builder
	fmt.Fprintf(&b, "## %s · %s\n\n", q.ID, q.Question)
	// The version, so a question a codehalter bug caused can be told apart once fixed.
	fmt.Fprintf(&b, "Asked %s by codehalter %s while building %s.\n\n", time.Now().Format("2006-01-02"), versionStamp(), title)
	if q.Quote != "" {
		fmt.Fprintf(&b, "The spec comes closest in `%s`:\n\n", q.QuoteAt)
		for _, ln := range strings.Split(strings.TrimSpace(q.Quote), "\n") {
			b.WriteString("> " + ln + "\n")
		}
		b.WriteString("\n")
	}
	if q.Stopped != "" {
		// A fence inside the output would end this one early.
		b.WriteString("What stopped the last attempt:\n\n```\n" + strings.ReplaceAll(strings.TrimSpace(q.Stopped), "```", "` ` `") + "\n```\n\n")
		b.WriteString("Options:\n\n")
	} else {
		b.WriteString("Options (the first is the planner's pick):\n\n")
	}
	for i, o := range q.Options {
		fmt.Fprintf(&b, "%d. %s. Example: %s\n", i+1, strings.TrimRight(strings.TrimSpace(o.Choice), "."), strings.TrimSpace(o.Example))
	}
	b.WriteString("\n**Answer:** \n")
	texts := make([]string, 0, len(secs)+1)
	for _, s := range secs {
		texts = append(texts, s.Text)
	}
	return writeSpecQuestionFile(path, head, append(texts, b.String()))
}

// specStuckStopped caps what a stuck item's question quotes of its failure: the
// end of a test run is enough to see why, and the file is read by a person.
const specStuckStopped = 2000

// specStuckQuestion words a stuck item's question in code: only the user can tell
// whether its cause is gone.
func specStuckQuestion(idx *specIndex, specDir, id, reason string) specQuestion {
	q := specQuestion{ID: id, Question: "This item did not pass its attempts. How should the rounds go on?",
		Stopped: clipUTF8(strings.TrimSpace(reason), specStuckStopped)}
	file := specDir + "/"
	if it := idx.items[id]; it != nil {
		q.Quote = strings.TrimSpace(strings.TrimLeft(idx.docs[it.Doc].lines[it.Line], "#"))
		q.QuoteAt = fmt.Sprintf("%s:%d", idx.docs[it.Doc].rel, it.Line+1)
		file = specDir + "/" + idx.docs[it.Doc].rel
	}
	q.Options = []specOption{
		{Choice: "Try again as the spec says", Example: "right when the cause was elsewhere and is gone now, such as a build another item broke or a tool installed since: answer `1`, and the next /spec gives it fresh attempts"},
		{Choice: "Change what the spec asks", Example: "edit the item's section in `" + file + "` (a smaller step, another format, a service left out) and answer `2`; the next /spec builds the new text"},
		{Choice: "Tell the rounds how to get past it", Example: "answer `3: test the upload against a local fake server on port 8080` or `3: install libadwaita 1.5 in the Dockerfile first`, and every round of the item reads it"},
	}
	return q
}

// specQuestionFrom returns why the planner's question is not answerable from the
// spec as asked, or "": a quote found in the spec and options with examples.
func specQuestionFrom(p *planResult, specAbs string) (q specQuestion, wrong string) {
	q = specQuestion{Question: strings.TrimSpace(p.Question), Quote: strings.TrimSpace(p.SpecQuote)}
	for _, o := range p.Options {
		if o.Choice = strings.TrimSpace(o.Choice); o.Choice != "" {
			o.Example = strings.TrimSpace(o.Example)
			q.Options = append(q.Options, o)
		}
	}
	switch {
	case q.Question == "":
		return q, "it has no `question`"
	case q.Quote == "":
		return q, "it has no `spec_quote`"
	case len(q.Options) < 2 || len(q.Options) > 3:
		return q, fmt.Sprintf("it has %d `options`, not 2 or 3", len(q.Options))
	}
	for _, o := range q.Options {
		if o.Example == "" || strings.EqualFold(o.Example, o.Choice) {
			return q, fmt.Sprintf("the option %q has no `example` of what the user would see or the program would do", o.Choice)
		}
	}
	at, err := findSpecQuote(specAbs, q.Quote)
	switch {
	case err != nil:
		return q, "codehalter could not read the spec to find its `spec_quote` (" + err.Error() + ")"
	case at == "":
		return q, "its `spec_quote` is not in the spec; copy the text exactly, from a spec file"
	}
	q.QuoteAt = at
	return q, ""
}

// findSpecQuote compares by words, so line breaks, indentation and a quote's own
// `> ` or quotation marks do not matter.
func findSpecQuote(specAbs, quote string) (string, error) {
	var qw []string
	for _, ln := range strings.Split(quote, "\n") {
		qw = append(qw, strings.Fields(strings.TrimPrefix(strings.TrimSpace(ln), ">"))...)
	}
	needle := strings.Trim(strings.Join(qw, " "), "\"'“”„ ")
	if needle == "" {
		return "", nil
	}
	found := ""
	err := filepath.WalkDir(specAbs, func(path string, d fs.DirEntry, err error) error {
		if err != nil || found != "" {
			return err
		}
		if d.IsDir() || !strings.EqualFold(filepath.Ext(path), ".md") || strings.EqualFold(d.Name(), specQuestionsFile) && filepath.Dir(path) == specAbs {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var norm strings.Builder
		var starts []int
		for _, ln := range strings.Split(string(data), "\n") {
			starts = append(starts, norm.Len())
			norm.WriteString(strings.Join(strings.Fields(ln), " ") + " ")
		}
		at := strings.Index(norm.String(), needle)
		if at < 0 {
			return nil
		}
		line := 0
		for i, s := range starts {
			if s <= at {
				line = i
			}
		}
		found = fmt.Sprintf("%s:%d", filepath.ToSlash(strings.TrimPrefix(path, specAbs+string(filepath.Separator))), line+1)
		return nil
	})
	return found, err
}

func specQuestionCorrective(wrong string) string {
	return "Your question was not recorded: " + wrong + ". Read the spec again first: a layout, a label, an order or a value it states anywhere (the chapter's screen description, its picture, a table, another item) decides it, so follow that and call `submit_plan` with the subtasks. " +
		"Only if the spec truly does not decide it, call `submit_plan` again with clear=false, no subtasks, and: `question`, one sentence naming what the user must decide; `spec_quote`, the spec text that comes closest to deciding it, copied exactly from a spec file; `options`, 2 or 3 ways to decide it with your pick first, each a short `choice` and an `example` of what the user would see or the program would do with it (a label, a layout, a value, a line of a file)."
}
