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
	// Count calls and verify arm B carries the supplied system prompt, arm A none.
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
	pairs, err := generateSamples(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, claim, q, 2, claim.Text)
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
	res := judgeSamples(context.Background(), judge, target, claim, q, pairs, 2, // threshold 2 = strict majority of 3
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
	if !res.Unstable {
		t.Fatal("keep/drop/keep is non-unanimous — should be flagged Unstable")
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
	pairs, err := generateSamples(context.Background(), ModelSpec{Name: "t", Server: ts.URL, Model: "m"}, Claim{ID: "x#1", Text: "web_search before claiming unavailable"}, q, 1, "skill")
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
		Claim{ID: "x#1"}, q, pairs, 1, nil)
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
	pairs, err := generateSamples(context.Background(), target, Claim{ID: "x#1", Text: "claim"}, Question{Question: "q", Rubric: "r"}, 3, "skill")
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
	if _, err := generateSamples(context.Background(), target, Claim{ID: "x#1", Text: "c"}, Question{Question: "q", Rubric: "r"}, 3, "skill"); err != nil {
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
	res := judgeSamples(context.Background(), judge, ModelSpec{Name: "gemma"}, Claim{ID: "x#1"}, Question{Question: "q", Rubric: "r"}, pairs, 1,
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
		Question{Question: "q", Rubric: "r"}, []samplePair{{}, {}, {}}, 1, nil)
	if res.Err == "" {
		t.Fatal("a failed judge sample must error the probe")
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
		Claim{ID: "x#1"}, Question{Question: "q", Rubric: "r"}, []samplePair{{AnswerA: "a", AnswerB: "b"}}, 1, nil)
	if !res.Keep || res.Samples[0].Verdict != "keep" {
		t.Fatalf("off-spec verdict not normalized to keep: %+v", res)
	}
}

// TestJudgeSamplesKeepThreshold pins the keep/drop fold: keepThreshold is the
// minimum "keep" votes to keep the statement, and any non-unanimous vote is
// flagged Unstable. Threshold 1 (the default) keeps unless the samples agree to
// drop; threshold 2 restores strict majority for 3 samples.
func TestJudgeSamplesKeepThreshold(t *testing.T) {
	keep := `{"a_satisfies":false,"b_satisfies":true,"similar":false,"verdict":"keep","reason":"B fixed it"}`
	drop := `{"a_satisfies":true,"b_satisfies":true,"similar":true,"verdict":"drop","reason":"same"}`
	cases := []struct {
		name                   string
		verdicts               []string
		threshold              int
		wantKeep, wantUnstable bool
	}{
		{"one keep, threshold 1 → keep", []string{drop, keep, drop}, 1, true, true},
		{"one keep, threshold 2 → drop", []string{drop, keep, drop}, 2, false, true},
		{"unanimous drop → drop, stable", []string{drop, drop, drop}, 1, false, false},
		{"unanimous keep → keep, stable", []string{keep, keep, keep}, 1, true, false},
		{"two keep, threshold 2 → keep", []string{keep, drop, keep}, 2, true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var call int32
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				i := atomic.AddInt32(&call, 1) - 1
				_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, tc.verdicts[i]))))
			}))
			defer ts.Close()
			res := judgeSamples(context.Background(),
				ModelSpec{Name: "main", Server: ts.URL, Model: "j"}, ModelSpec{Name: "t"},
				Claim{ID: "x#1"}, Question{Question: "q", Rubric: "r"}, make([]samplePair, len(tc.verdicts)), tc.threshold, nil)
			if res.Err != "" {
				t.Fatal(res.Err)
			}
			if res.Keep != tc.wantKeep {
				t.Fatalf("keep = %v, want %v", res.Keep, tc.wantKeep)
			}
			if res.Unstable != tc.wantUnstable {
				t.Fatalf("unstable = %v, want %v", res.Unstable, tc.wantUnstable)
			}
		})
	}
}

// TestJudgeSamplesIneffective pins the "model ignores it" signature: a drop
// where every sample failed the rubric even with the skill loaded (all
// B_sat=false) is flagged Ineffective; any B success (or a keep) is not.
func TestJudgeSamplesIneffective(t *testing.T) {
	dropBoth := `{"a_satisfies":false,"b_satisfies":false,"similar":true,"verdict":"drop","reason":"ignored"}`
	dropKnown := `{"a_satisfies":true,"b_satisfies":true,"similar":true,"verdict":"drop","reason":"already knows"}`
	cases := []struct {
		name     string
		verdicts []string
		want     bool
	}{
		{"all B fail → ineffective", []string{dropBoth, dropBoth, dropBoth}, true},
		{"already knows → not ineffective", []string{dropKnown, dropKnown, dropKnown}, false},
		{"mixed B → not ineffective", []string{dropBoth, dropKnown, dropBoth}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var call int32
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				i := atomic.AddInt32(&call, 1) - 1
				_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, tc.verdicts[i]))))
			}))
			defer ts.Close()
			res := judgeSamples(context.Background(),
				ModelSpec{Name: "main", Server: ts.URL, Model: "j"}, ModelSpec{Name: "t"},
				Claim{ID: "x#1"}, Question{Question: "q", Rubric: "r"}, make([]samplePair, len(tc.verdicts)), 1, nil)
			if res.Err != "" {
				t.Fatal(res.Err)
			}
			if res.Keep {
				t.Fatal("all-drop verdicts must not keep")
			}
			if res.Ineffective != tc.want {
				t.Fatalf("ineffective = %v, want %v", res.Ineffective, tc.want)
			}
		})
	}
}

// TestStrengthenClaim: the judge rewrite is accepted when non-empty and
// changed, rejected when empty or identical to the original.
func TestStrengthenClaim(t *testing.T) {
	for _, tc := range []struct {
		reply  string
		wantOK bool
	}{
		{`{"text":"- Order: STRICTLY 1) dnf 2) upstream — even when you know a COPR package exists."}`, true},
		{`{"text":""}`, false},
		{`{"text":"- original wording"}`, false}, // unchanged → useless
	} {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, tc.reply))))
		}))
		got, ok := strengthenClaim(context.Background(),
			ModelSpec{Name: "main", Server: ts.URL, Model: "j"},
			Claim{ID: "x#1", Source: "- original wording", Text: "original"},
			Question{Question: "q", Rubric: "r"},
			[]samplePair{{AnswerB: "wrong answer"}})
		ts.Close()
		if ok != tc.wantOK {
			t.Fatalf("reply %s: ok = %v, want %v (got %q)", tc.reply, ok, tc.wantOK, got)
		}
	}
}

// Zero pairs must never fold into a verdict — no evidence, no drop.
func TestJudgeSamplesEmptyPairsErrors(t *testing.T) {
	res := judgeSamples(context.Background(),
		ModelSpec{Name: "main"}, ModelSpec{Name: "t"},
		Claim{ID: "x#1"}, Question{Question: "q", Rubric: "r"}, nil, 1, nil)
	if res.Err == "" {
		t.Fatalf("empty pairs must produce an errored result, got verdict %q", res.Verdict)
	}
	if res.Keep || res.Verdict != "" {
		t.Fatalf("errored result must carry no verdict: %+v", res)
	}
}

// The probe catalog must resolve every agent-default tool AUTHOR.md offers —
// a name in the prompt but not the catalog silently downgrades those probes
// to text-only (measured: the judge requested write_file, which was missing).
func TestProbeToolCatalogCoversAuthorList(t *testing.T) {
	for _, name := range []string{"run_command", "read_file", "write_file", "edit_file", "list_files", "search_text", "web_search"} {
		if !strings.Contains(authorPrompt, "`"+name+"`") {
			t.Errorf("AUTHOR.md does not offer %q", name)
		}
		if got := buildProbeTools([]string{name}); len(got) != 1 {
			t.Errorf("catalog cannot resolve %q", name)
		}
	}
}
