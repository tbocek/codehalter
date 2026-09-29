package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The gate: what to decide, the spec text found where the model said it is, 2 or 3 options with examples.
func TestSpecQuestionFrom(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "sub"), 0o755)
	os.WriteFile(filepath.Join(dir, "sub", "05-cut.md"), []byte("# Cut\n\n## Toolbar\n\nThe toolbar groups, left to right:\n  ▶ recording, ▶✂ cut.\n"), 0o644)
	// QUESTIONS.md is no source: quoting an earlier question back proves nothing.
	os.WriteFile(filepath.Join(dir, specQuestionsFile), []byte("## F1.1 · colour?\n\n> Buttons are teal.\n"), 0o644)
	two := []specOption{{Choice: "Side by side", Example: "[▶] [▶✂] in one row"}, {Choice: "Stacked", Example: "▶ above ▶✂"}}
	for _, tc := range []struct {
		name string
		p    planResult
		want string // part of wrong; "" is a good question
	}{
		{"grounded", planResult{Question: "Which order?", SpecQuote: "> The toolbar groups, left to right: ▶ recording,\n> ▶✂ cut.", Options: two}, ""},
		{"no question", planResult{SpecQuote: "The toolbar groups", Options: two}, "no `question`"},
		{"no quote", planResult{Question: "Which order?", Options: two}, "no `spec_quote`"},
		{"one option", planResult{Question: "Which order?", SpecQuote: "The toolbar groups", Options: two[:1]}, "1 `options`"},
		{"no example", planResult{Question: "Which order?", SpecQuote: "The toolbar groups",
			Options: []specOption{{Choice: "Side by side", Example: "side by side"}, two[1]}}, `"Side by side" has no`},
		{"invented quote", planResult{Question: "Which order?", SpecQuote: "The toolbar is vertical.", Options: two}, "not in the spec"},
		{"quote from QUESTIONS.md", planResult{Question: "Which colour?", SpecQuote: "Buttons are teal.", Options: two}, "not in the spec"},
	} {
		q, wrong := specQuestionFrom(&tc.p, dir)
		if tc.want == "" {
			if wrong != "" || q.QuoteAt != "sub/05-cut.md:5" {
				t.Errorf("%s: wrong=%q at=%q, want accepted at sub/05-cut.md:5", tc.name, wrong, q.QuoteAt)
			}
		} else if !strings.Contains(wrong, tc.want) {
			t.Errorf("%s: wrong=%q, want it to name %q", tc.name, wrong, tc.want)
		}
	}
}

// Written by the loop, answered by hand, read back: open until answered, and not an item source.
func TestSpecQuestionsFile(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "01.md"), []byte("# 01\n\n### F0.1 Store\n\nKeep the notes.\n"), 0o644)
	q := specQuestion{ID: "F0.1", Question: "Which store?", Quote: "Keep the notes.", QuoteAt: "01.md:5",
		Options: []specOption{{Choice: "SQLite.", Example: "notes.db beside the project"}, {Choice: "JSON", Example: "notes.json"}}}
	for range 2 {
		if err := appendSpecQuestion(dir, "F0.1 Store", q); err != nil {
			t.Fatal(err)
		}
	}
	data, _ := os.ReadFile(filepath.Join(dir, specQuestionsFile))
	if strings.Count(string(data), "# Questions") != 1 || !strings.Contains(string(data), "1. SQLite. Example: notes.db beside the project\n") {
		t.Errorf("file:\n%s", data)
	}
	idx, err := scanSpec(dir, defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range idx.order {
		if idx.docs[idx.items[id].Doc].rel == specQuestionsFile {
			t.Errorf("QUESTIONS.md became item %s", id)
		}
	}
	before := specItemHash(idx, "F0.1")
	if !idx.asked("F0.1") || idx.answered("F0.1") != "" || len(specOpenQuestions(idx)) != 2 {
		t.Fatalf("unanswered: asked=%v answered=%q", idx.asked("F0.1"), idx.answered("F0.1"))
	}

	// One answered on its own line below the marker, over two lines; the other still open.
	text := strings.Replace(string(data), "**Answer:** \n", "**Answer:**\nSQLite,\nin .notes/\n", 1)
	os.WriteFile(filepath.Join(dir, specQuestionsFile), []byte(text), 0o644)
	if idx, err = scanSpec(dir, defaultSpecIDPatterns, nil, nil); err != nil {
		t.Fatal(err)
	}
	if qs := idx.questions["F0.1"]; len(qs) != 2 || qs[0].Question != "Which store?" || qs[0].Answer != "SQLite,\nin .notes/" {
		t.Fatalf("parsed = %+v", qs)
	}
	if !idx.asked("F0.1") {
		t.Error("the second question is still open")
	}
	if got := idx.answered("F0.1"); !strings.Contains(got, "## F0.1 · Which store?") || !strings.Contains(got, "in .notes/") {
		t.Errorf("answered = %q", got)
	}
	if specItemHash(idx, "F0.1") == before {
		t.Error("an answer must change the item's fingerprint")
	}

	// A stuck item's failure output is fenced: its own heading or "Answer:" line is not the file's.
	stuck := specStuckQuestion(idx, "spec", "F0.1", "the test command did not pass:\n## F9.9 · panic\n**Answer:** 42\n```inner```")
	if err := appendSpecQuestion(dir, "F0.1 Store", stuck); err != nil {
		t.Fatal(err)
	}
	if idx, err = scanSpec(dir, defaultSpecIDPatterns, nil, nil); err != nil {
		t.Fatal(err)
	}
	if qs := idx.questions["F0.1"]; len(qs) != 3 || qs[2].Answer != "" || len(idx.questions["F9.9"]) != 0 || stuck.QuoteAt != "01.md:3" {
		t.Errorf("stuck question parsed as %+v (F9.9: %v), quote at %q", qs, idx.questions["F9.9"], stuck.QuoteAt)
	}
}

// A question says which codehalter asked it, so one a since-fixed bug caused can be told apart.
func TestSpecQuestionsKnowWhoAsked(t *testing.T) {
	dir := t.TempDir()
	if err := appendSpecQuestion(dir, "F0.1 Store", specQuestion{ID: "F0.1", Question: "Which store?",
		Options: []specOption{{Choice: "SQLite", Example: "notes.db"}, {Choice: "JSON", Example: "notes.json"}}}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, specQuestionsFile))
	if err != nil {
		t.Fatal(err)
	}
	text := string(data) + "\n## F0.2 · Which font?\n\nAsked 2026-09-28 by codehalter v118 (2026-09-28, 993ef45) while building F0.2 Font.\n\n**Answer:** \n" +
		"\n## F0.3 · Which size?\n\nAsked 2026-09-27 while building F0.3 Size.\n\n**Answer:** \n"
	qs := parseSpecQuestions(text)
	if q := qs["F0.1"][0]; q.Version != versionStamp() || specStaleNote(q) != "" {
		t.Errorf("this codehalter's question: version %q, note %q", q.Version, specStaleNote(q))
	}
	if note := specStaleNote(qs["F0.2"][0]); !strings.Contains(note, "codehalter v118 (2026-09-28, 993ef45), not this one") {
		t.Errorf("another version's question: note %q", note)
	}
	if note := specStaleNote(qs["F0.3"][0]); !strings.Contains(note, "older codehalter") {
		t.Errorf("a question from before versions were written: note %q", note)
	}
}
