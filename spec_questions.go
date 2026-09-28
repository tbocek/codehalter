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
	"own words or as the number of an option, then run /spec. An answered question is part of the spec: " +
	"every later round of the item reads it, and changing the answer rebuilds the item.\n"

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
	Text   string // the whole section as the file holds it
	Answer string
}

var (
	specQuestionHeadRe = regexp.MustCompile(`^##\s+(\S+)\s+·\s*(.*?)\s*$`)
	specAnswerRe       = regexp.MustCompile(`^\*{0,2}Answer\*{0,2}:\*{0,2}\s*(.*)$`)
)

// parseSpecQuestions keys the sections by item id; an item may have asked more than once.
func parseSpecQuestions(text string) map[string][]specQuestion {
	out := map[string][]specQuestion{}
	var cur *specQuestion
	var body []string
	inAnswer, inFence := false, false
	flush := func() {
		if cur != nil {
			cur.Text = strings.TrimSpace(strings.Join(body, "\n"))
			cur.Answer = strings.TrimSpace(cur.Answer)
			out[cur.ID] = append(out[cur.ID], *cur)
		}
	}
	for _, ln := range strings.Split(text, "\n") {
		// A test's output inside a fence may hold a heading or an "Answer:" of its own.
		if strings.HasPrefix(strings.TrimSpace(ln), "```") {
			inFence = !inFence
		}
		if m := specQuestionHeadRe.FindStringSubmatch(ln); m != nil && !inFence {
			flush()
			cur, body, inAnswer = &specQuestion{ID: m[1], Question: m[2]}, nil, false
		}
		if cur == nil {
			continue
		}
		body = append(body, ln)
		switch m := specAnswerRe.FindStringSubmatch(strings.TrimSpace(ln)); {
		case m != nil && !inFence:
			inAnswer = true
			cur.Answer = m[1]
		case inAnswer:
			cur.Answer += "\n" + ln
		}
	}
	flush()
	return out
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
	var b strings.Builder
	if len(old) == 0 {
		b.WriteString(specQuestionsHead)
	} else {
		b.WriteString(strings.TrimRight(string(old), "\n") + "\n")
	}
	fmt.Fprintf(&b, "\n## %s · %s\n\n", q.ID, q.Question)
	fmt.Fprintf(&b, "Asked %s while building %s.\n\n", time.Now().Format("2006-01-02"), title)
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
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// specStuckStopped caps what a stuck item's question quotes of its failure: the
// end of a test run is enough to see why, and the file is read by a person.
const specStuckStopped = 2000

// specStuckQuestion turns an item that did not pass its attempts into a question
// codehalter words itself, so it does not hang on the model asking well: the
// item's spec line, what stopped it, and how it can go on. Nine of ten blocks in
// one run had causes only the user could see were gone (a bug fixed, a tool
// installed), and a blind retry every run would spend the attempts again.
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

// specQuestionFrom holds a /spec question to what makes it answerable without
// opening the code: what to decide, the spec text that comes closest (found, not
// claimed), and options that each show what the user would get. F2.2 asked
// "side by side or stacked?" when the spec's toolbar line already said.
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
