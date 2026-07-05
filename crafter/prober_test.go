package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGenerateSamples(t *testing.T) {
	// Count calls and verify the B run carries the claim as system prompt.
	var calls, withInstruction int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.Messages[0].Role == "system" && strings.Contains(req.Messages[0].Content, "always check errors") {
			withInstruction++
		}
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":"answer %d"}}]}`, calls))))
	}))
	defer ts.Close()

	claim := Claim{ID: "go#x", Skill: "go", Text: "always check errors"}
	q := Question{Question: "write a file reader", Rubric: "checks the error"}
	pairs, err := generateSamples(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, claim, q, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 2 || calls != 4 || withInstruction != 2 {
		t.Fatalf("pairs=%d calls=%d instructed=%d — want 2/4/2", len(pairs), calls, withInstruction)
	}
	if pairs[0].AnswerA == "" || pairs[0].AnswerB == "" || pairs[0].AnswerA == pairs[0].AnswerB {
		t.Fatalf("A/B answers not captured distinctly: %+v", pairs[0])
	}
}

func TestJudgeSamplesMajority(t *testing.T) {
	// Judge returns keep, drop, keep → majority keep.
	verdicts := []string{
		`{"a_satisfies":false,"b_satisfies":true,"similar":false,"verdict":"keep","reason":"B fixed it"}`,
		`{"a_satisfies":true,"b_satisfies":true,"similar":true,"verdict":"drop","reason":"same"}`,
		`{"a_satisfies":false,"b_satisfies":true,"similar":false,"verdict":"keep","reason":"B fixed it"}`,
	}
	call := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		v := verdicts[call%len(verdicts)]
		call++
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, v))))
	}))
	defer ts.Close()

	judge := ModelSpec{Name: "main", Server: ts.URL, Model: "j"}
	target := ModelSpec{Name: "gemma"}
	claim := Claim{ID: "go#x", Skill: "go", Text: "claim"}
	q := Question{Question: "q", Rubric: "r"}
	pairs := []samplePair{{"a1", "b1"}, {"a2", "b2"}, {"a3", "b3"}}

	// Also assert the progress callback fires once per sample.
	var progressed int
	res := judgeSamples(context.Background(), judge, target, claim, q, pairs,
		func(i, n int, verdict string) { progressed++ })
	if progressed != len(pairs) {
		t.Fatalf("progress fired %d times, want %d", progressed, len(pairs))
	}
	if res.Err != "" {
		t.Fatal(res.Err)
	}
	if !res.Keep || res.Verdict != "keep" || len(res.Samples) != 3 {
		t.Fatalf("majority fold wrong: keep=%v verdict=%s samples=%d", res.Keep, res.Verdict, len(res.Samples))
	}
	if res.Model != "gemma" || res.ClaimID != "go#x" {
		t.Fatalf("result identity wrong: %+v", res)
	}
}

func TestJudgeSamplesNormalizesOffSpecVerdict(t *testing.T) {
	// No usable verdict string → derived from booleans: B satisfies, A doesn't → keep.
	body := `{"a_satisfies":false,"b_satisfies":true,"similar":false,"verdict":"unsure","reason":"?"}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, body))))
	}))
	defer ts.Close()

	res := judgeSamples(context.Background(),
		ModelSpec{Name: "main", Server: ts.URL, Model: "j"}, ModelSpec{Name: "t"},
		Claim{ID: "x#1"}, Question{Question: "q", Rubric: "r"}, []samplePair{{"a", "b"}}, nil)
	if !res.Keep || res.Samples[0].Verdict != "keep" {
		t.Fatalf("off-spec verdict not normalized to keep: %+v", res)
	}
}
