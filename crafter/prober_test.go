package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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
	pairs := []samplePair{{AnswerA: "a1", AnswerB: "b1"}, {AnswerA: "a2", AnswerB: "b2"}, {AnswerA: "a3", AnswerB: "b3"}}

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

func TestGenerateSamplesWithToolsCapturesCalls(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req["tools"] == nil {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"tools missing"}}`))
			return
		}
		_, _ = w.Write([]byte(sse(
			`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"web_search","arguments":"{\"query\":\"x\"}"}}]}}]}`,
		)))
	}))
	defer ts.Close()

	q := Question{Question: "install foo", Rubric: "calls web_search first", Tools: []string{"web_search"}}
	pairs, err := generateSamples(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, Claim{ID: "x#1", Text: "web_search before claiming unavailable"}, q, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 1 || len(pairs[0].ACalls) != 1 || len(pairs[0].BCalls) != 1 {
		t.Fatalf("calls not captured: %+v", pairs)
	}
	if pairs[0].ACalls[0] != `web_search({"query":"x"})` {
		t.Fatalf("rendered call = %q", pairs[0].ACalls[0])
	}
}

func TestJudgeSamplesShowsToolCalls(t *testing.T) {
	var judgeSaw string
	verdict := `{"a_satisfies":false,"b_satisfies":true,"similar":false,"verdict":"keep","reason":"B called the tool"}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		judgeSaw = req.Messages[len(req.Messages)-1].Content
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, verdict))))
	}))
	defer ts.Close()

	q := Question{Question: "q", Rubric: "calls web_search", Tools: []string{"web_search"}}
	pairs := []samplePair{{AnswerA: "", ACalls: nil, AnswerB: "", BCalls: []string{`web_search({"query":"docs"})`}}}
	res := judgeSamples(context.Background(),
		ModelSpec{Name: "main", Server: ts.URL, Model: "j"}, ModelSpec{Name: "t"},
		Claim{ID: "x#1"}, q, pairs, nil)
	if res.Err != "" {
		t.Fatal(res.Err)
	}
	for _, want := range []string{"TOOL CALLS A:\n(none)", `TOOL CALLS B:` + "\n" + `web_search({"query":"docs"})`, "(no text — see tool calls)"} {
		if !strings.Contains(judgeSaw, want) {
			t.Fatalf("judge prompt missing %q:\n%s", want, judgeSaw)
		}
	}
	if !res.Keep || res.Samples[0].BCalls[0] != `web_search({"query":"docs"})` {
		t.Fatalf("verdict/ledger wrong: %+v", res)
	}
}

func TestQuestionCachePromptHashInvalidation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "go.json")
	if err := writeQuestionCache(path, map[string]Question{"go#1": {Question: "q", Rubric: "r"}}); err != nil {
		t.Fatal(err)
	}
	if got := readQuestionCache(path); len(got) != 1 {
		t.Fatalf("fresh cache should hit, got %d", len(got))
	}
	// Old-format cache (bare map, pre-wrapper) must be treated as stale.
	if err := os.WriteFile(path, []byte(`{"go#1":{"question":"q","rubric":"r"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readQuestionCache(path); len(got) != 0 {
		t.Fatalf("legacy cache must miss, got %d", len(got))
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
		Claim{ID: "x#1"}, Question{Question: "q", Rubric: "r"}, []samplePair{{AnswerA: "a", AnswerB: "b"}}, nil)
	if !res.Keep || res.Samples[0].Verdict != "keep" {
		t.Fatalf("off-spec verdict not normalized to keep: %+v", res)
	}
}
