package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSamplesRoundTripWithQuestionHash(t *testing.T) {
	path := filepath.Join(t.TempDir(), "samples.jsonl")
	q := Question{Question: "install foo", Rubric: "tries dnf first", Tools: []string{"run_command"}}
	rec := sampleRecord{
		ClaimID:      "go#aa",
		Skill:        "go",
		QuestionHash: questionHash(q),
		Pairs:        []samplePair{{AnswerA: "a", AnswerB: "b"}},
	}
	if err := appendSamples(path, rec); err != nil {
		t.Fatal(err)
	}
	got := readSamples(path)
	r, ok := got["go#aa"]
	if !ok || len(r.Pairs) != 1 || r.Pairs[0].AnswerB != "b" {
		t.Fatalf("round trip lost data: %+v", got)
	}
	if !r.matches(q) {
		t.Fatal("hash must match the question that produced the pairs")
	}
	// A re-authored question (any field changed) must invalidate the pairs.
	for _, changed := range []Question{
		{Question: "install BAR", Rubric: q.Rubric, Tools: q.Tools},
		{Question: q.Question, Rubric: "different rubric", Tools: q.Tools},
		{Question: q.Question, Rubric: q.Rubric, Tools: nil},
	} {
		if r.matches(changed) {
			t.Fatalf("stale pairs matched a changed question: %+v", changed)
		}
	}
	// Legacy row without a hash never matches.
	if (sampleRecord{Pairs: r.Pairs}).matches(q) {
		t.Fatal("legacy (hashless) record must not match")
	}
}

func TestReadSamplesLastLineWins(t *testing.T) {
	path := filepath.Join(t.TempDir(), "samples.jsonl")
	if err := appendSamples(path, sampleRecord{ClaimID: "x#1", Pairs: []samplePair{{AnswerA: "old"}}}); err != nil {
		t.Fatal(err)
	}
	if err := appendSamples(path, sampleRecord{ClaimID: "x#1", Pairs: []samplePair{{AnswerA: "new"}}}); err != nil {
		t.Fatal(err)
	}
	got := readSamples(path)
	if got["x#1"].Pairs[0].AnswerA != "new" {
		t.Fatalf("last line must win: %+v", got["x#1"])
	}
}

func TestReadSamplesMissingFile(t *testing.T) {
	if got := readSamples(filepath.Join(t.TempDir(), "nope.jsonl")); len(got) != 0 {
		t.Fatalf("missing file should give an empty map, got %d", len(got))
	}
}

func TestReadSamplesSkipsMalformed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "samples.jsonl")
	if err := os.WriteFile(path, []byte("{not json\n"+`{"claim_id":"x#2","pairs":[{"answer_a":"a","answer_b":"b"}]}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := readSamples(path)
	if len(got) != 1 || got["x#2"].Pairs[0].AnswerA != "a" {
		t.Fatalf("malformed line handling wrong: %+v", got)
	}
}
