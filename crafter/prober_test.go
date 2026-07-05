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
	"sync/atomic"
	"testing"
	"time"
)

// concurrencyServer replies to every chat request after a short hold, tracking
// how many requests were in flight at once. Reply is a fixed SSE body.
func concurrencyServer(t *testing.T, reply string) (url string, peak *int32, total *int32) {
	t.Helper()
	var inFlight, pk, tot int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&tot, 1)
		cur := atomic.AddInt32(&inFlight, 1)
		for {
			p := atomic.LoadInt32(&pk)
			if cur <= p || atomic.CompareAndSwapInt32(&pk, p, cur) {
				break
			}
		}
		time.Sleep(30 * time.Millisecond)
		atomic.AddInt32(&inFlight, -1)
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(ts.Close)
	return ts.URL, &pk, &tot
}

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

func TestGenerateSamplesParallelArms(t *testing.T) {
	url, peak, total := concurrencyServer(t, sse(`{"choices":[{"delta":{"content":"answer"}}]}`))
	target := ModelSpec{Name: "t", Server: url, Model: "m", Parallel: 2, sem: make(chan struct{}, 2)}
	pairs, err := generateSamples(context.Background(), target, Claim{ID: "x#1", Text: "claim"}, Question{Question: "q", Rubric: "r"}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(pairs) != 3 {
		t.Fatalf("pairs = %d, want 3", len(pairs))
	}
	// The A and B arms run concurrently — peak in-flight must reach 2.
	if p := atomic.LoadInt32(peak); p != 2 {
		t.Fatalf("peak in-flight = %d, want 2 (A and B arms concurrent)", p)
	}
	if c := atomic.LoadInt32(total); c != 6 {
		t.Fatalf("total calls = %d, want 6 (3 samples × 2 arms)", c)
	}
}

func TestGenerateSamplesSerialWhenParallelOne(t *testing.T) {
	url, peak, total := concurrencyServer(t, sse(`{"choices":[{"delta":{"content":"answer"}}]}`))
	target := ModelSpec{Name: "t", Server: url, Model: "m", Parallel: 1, sem: make(chan struct{}, 1)}
	if _, err := generateSamples(context.Background(), target, Claim{ID: "x#1", Text: "c"}, Question{Question: "q", Rubric: "r"}, 3); err != nil {
		t.Fatal(err)
	}
	if p := atomic.LoadInt32(peak); p != 1 {
		t.Fatalf("parallel=1 must serialize; peak = %d", p)
	}
	if c := atomic.LoadInt32(total); c != 6 {
		t.Fatalf("total calls = %d, want 6", c)
	}
}

func TestJudgeSamplesParallel(t *testing.T) {
	verdict := `{"a_satisfies":false,"b_satisfies":true,"similar":false,"verdict":"keep","reason":"ok"}`
	url, peak, _ := concurrencyServer(t, sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, verdict)))
	judge := ModelSpec{Name: "main", Server: url, Model: "j", Parallel: 3, sem: make(chan struct{}, 3)}
	pairs := []samplePair{{AnswerA: "a1", AnswerB: "b1"}, {AnswerA: "a2", AnswerB: "b2"}, {AnswerA: "a3", AnswerB: "b3"}}

	var progressed int32
	res := judgeSamples(context.Background(), judge, ModelSpec{Name: "gemma"}, Claim{ID: "x#1"}, Question{Question: "q", Rubric: "r"}, pairs,
		func(i, n int, verdict string) { atomic.AddInt32(&progressed, 1) })
	if res.Err != "" {
		t.Fatal(res.Err)
	}
	if !res.Keep || len(res.Samples) != 3 {
		t.Fatalf("majority fold wrong: keep=%v samples=%d", res.Keep, len(res.Samples))
	}
	if atomic.LoadInt32(&progressed) != 3 {
		t.Fatalf("progress fired %d times, want 3", progressed)
	}
	if p := atomic.LoadInt32(peak); p < 2 {
		t.Fatalf("judge samples did not run concurrently; peak = %d", p)
	}
	// Ledger order preserved despite concurrent judging.
	for i, s := range res.Samples {
		if s.AnswerA != fmt.Sprintf("a%d", i+1) {
			t.Fatalf("sample %d out of order: %s", i, s.AnswerA)
		}
	}
}

func TestJudgeSamplesParallelPropagatesError(t *testing.T) {
	// One judge call 500s; the whole probe must surface an error (a partial
	// majority would bias the verdict).
	var n int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) == 2 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"boom"}}`))
			return
		}
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"{\"verdict\":\"drop\",\"reason\":\"x\"}"}}]}`)))
	}))
	defer ts.Close()
	judge := ModelSpec{Name: "main", Server: ts.URL, Model: "j", Parallel: 3, sem: make(chan struct{}, 3)}
	res := judgeSamples(context.Background(), judge, ModelSpec{Name: "t"}, Claim{ID: "x#1"},
		Question{Question: "q", Rubric: "r"}, []samplePair{{}, {}, {}}, nil)
	if res.Err == "" {
		t.Fatal("a failed judge sample must error the probe")
	}
}

func TestAuthorClaimsParallel(t *testing.T) {
	url, peak, _ := concurrencyServer(t, sse(`{"choices":[{"delta":{"content":"{\"question\":\"q\",\"rubric\":\"r\"}"}}]}`))
	judge := ModelSpec{Name: "main", Server: url, Model: "j", Parallel: 3, sem: make(chan struct{}, 3)}
	claims := []Claim{{ID: "go#0", Text: "a"}, {ID: "go#1", Text: "b"}, {ID: "go#2", Text: "c"}, {ID: "go#3", Text: "d"}}
	cache := filepath.Join(t.TempDir(), "go.json")

	got, err := authorClaims(context.Background(), judge, claims, cache, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Fatalf("authored %d, want 4", len(got))
	}
	if p := atomic.LoadInt32(peak); p < 2 {
		t.Fatalf("authoring did not run concurrently; peak = %d", p)
	}
	// Second call hits the cache — no new authoring.
	url2, _, total2 := concurrencyServer(t, sse(`{"choices":[{"delta":{"content":"{\"question\":\"q\",\"rubric\":\"r\"}"}}]}`))
	judge2 := ModelSpec{Name: "main", Server: url2, Model: "j", Parallel: 3, sem: make(chan struct{}, 3)}
	if _, err := authorClaims(context.Background(), judge2, claims, cache, nil); err != nil {
		t.Fatal(err)
	}
	if c := atomic.LoadInt32(total2); c != 0 {
		t.Fatalf("cached re-run made %d calls, want 0", c)
	}
}

func TestAuthorClaimsParallelIsRaceFree(t *testing.T) {
	// Stress the shared cache map under the worker pool — the -race detector
	// catches an unsynchronized write.
	url, _, _ := concurrencyServer(t, sse(`{"choices":[{"delta":{"content":"{\"question\":\"q\",\"rubric\":\"r\"}"}}]}`))
	judge := ModelSpec{Name: "main", Server: url, Model: "j", Parallel: 4, sem: make(chan struct{}, 4)}
	var claims []Claim
	for i := 0; i < 20; i++ {
		claims = append(claims, Claim{ID: fmt.Sprintf("go#%02d", i), Text: "t"})
	}
	var counter int32
	got, err := authorClaims(context.Background(), judge, claims, filepath.Join(t.TempDir(), "c.json"),
		func(done, total int, id string, d time.Duration) { atomic.AddInt32(&counter, 1) })
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 20 || atomic.LoadInt32(&counter) != 20 {
		t.Fatalf("authored %d, progress %d, want 20/20", len(got), counter)
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
