package main

import (
	"context"
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

// pipelineFixture spins up fake target and judge servers for probePipeline
// tests. The target counts its calls; the judge can be given a handler hook.
func pipelineFixture(t *testing.T, judgeHook func(targetCalls *atomic.Int32)) (target, judge ModelSpec, targetCalls *atomic.Int32) {
	t.Helper()
	targetCalls = &atomic.Int32{}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := targetCalls.Add(1)
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":"answer %d"}}]}`, n))))
	}))
	t.Cleanup(ts.Close)
	verdict := `{"a_satisfies":false,"b_satisfies":true,"similar":false,"verdict":"keep","reason":"ok"}`
	js := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if judgeHook != nil {
			judgeHook(targetCalls)
		}
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, verdict))))
	}))
	t.Cleanup(js.Close)
	return ModelSpec{Name: "target", Server: ts.URL, Model: "t"},
		ModelSpec{Name: "main", Server: js.URL, Model: "j"}, targetCalls
}

func makePending(n int) []pendingClaim {
	var pend []pendingClaim
	for i := 0; i < n; i++ {
		pend = append(pend, pendingClaim{
			idx:   i,
			claim: Claim{ID: fmt.Sprintf("go#%02d", i), Skill: "go", Text: "claim"},
			q:     Question{Question: "q", Rubric: "r"},
		})
	}
	return pend
}

// The core promise of the pipeline: while the judge scores claim 1, the target
// is already generating claim 2. The judge's first call blocks until the
// target has served claim 2's A/B requests — if generation were serialized
// behind judging, this would deadlock (and fail via the poll deadline).
func TestProbePipelineOverlapsGenerationAndJudging(t *testing.T) {
	var judgeFirstSaw atomic.Int32
	judgeFirstSaw.Store(-1)
	var first atomic.Bool
	first.Store(true)
	target, judge, targetCalls := pipelineFixture(t, func(tc *atomic.Int32) {
		if !first.CompareAndSwap(true, false) {
			return
		}
		deadline := time.Now().Add(5 * time.Second)
		for tc.Load() < 4 && time.Now().Before(deadline) { // claim1 A+B + claim2 A+B
			time.Sleep(time.Millisecond)
		}
		judgeFirstSaw.Store(tc.Load())
	})

	var got []ProbeResult
	ev := pipelineEvents{onResult: func(pc pendingClaim, res ProbeResult, genMs int64) error {
		got = append(got, res)
		return nil
	}}
	if err := probePipeline(context.Background(), judge, target, makePending(3), 1, ev); err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("results = %d, want 3", len(got))
	}
	// Delivered in pend order despite concurrent generation.
	for i, r := range got {
		if r.ClaimID != fmt.Sprintf("go#%02d", i) || !r.Keep {
			t.Fatalf("result %d = %s/%s", i, r.ClaimID, r.Verdict)
		}
	}
	if s := judgeFirstSaw.Load(); s < 4 {
		t.Fatalf("no overlap: target had served only %d calls while judge held claim 1 (want ≥4)", s)
	}
	if c := targetCalls.Load(); c != 6 {
		t.Fatalf("target calls = %d, want 6 (3 claims × A/B)", c)
	}
}

// A result that cannot be persisted must abort the pipeline (and not hang on
// the generator's pending send).
func TestProbePipelineStopsWhenRecordFails(t *testing.T) {
	target, judge, _ := pipelineFixture(t, nil)
	boom := fmt.Errorf("disk full")
	ev := pipelineEvents{onResult: func(pc pendingClaim, res ProbeResult, genMs int64) error {
		return boom
	}}
	done := make(chan error, 1)
	go func() { done <- probePipeline(context.Background(), judge, target, makePending(4), 1, ev) }()
	select {
	case err := <-done:
		if err != boom {
			t.Fatalf("err = %v, want %v", err, boom)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("pipeline hung after record failure — generator not unblocked")
	}
}

// Cancelling mid-run stops generation at the claim boundary; everything
// already in flight still lands via onResult, and the pipeline returns.
func TestProbePipelineCancellation(t *testing.T) {
	target, judge, _ := pipelineFixture(t, nil)
	ctx, cancel := context.WithCancel(context.Background())
	var got int32
	ev := pipelineEvents{onResult: func(pc pendingClaim, res ProbeResult, genMs int64) error {
		if atomic.AddInt32(&got, 1) == 1 {
			cancel() // interrupt after the first recorded verdict
		}
		return nil
	}}
	done := make(chan error, 1)
	go func() { done <- probePipeline(ctx, judge, target, makePending(10), 1, ev) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("cancelled pipeline should return nil, got %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("pipeline hung after cancellation")
	}
	if n := atomic.LoadInt32(&got); n < 1 || n > 3 {
		t.Fatalf("recorded %d results after early cancel, want 1–3 (first + at most in-flight/buffered)", n)
	}
}

// An errored generation is recorded as an errored result, and the pipeline
// carries on with the remaining claims.
func TestProbePipelineRecordsGenerationErrors(t *testing.T) {
	var calls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 { // first claim's A call fails
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"boom"}}`))
			return
		}
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"fine"}}]}`)))
	}))
	defer ts.Close()
	verdict := `{"a_satisfies":false,"b_satisfies":true,"similar":false,"verdict":"keep","reason":"ok"}`
	js := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, verdict))))
	}))
	defer js.Close()

	var got []ProbeResult
	ev := pipelineEvents{onResult: func(pc pendingClaim, res ProbeResult, genMs int64) error {
		got = append(got, res)
		return nil
	}}
	err := probePipeline(context.Background(),
		ModelSpec{Name: "main", Server: js.URL, Model: "j"},
		ModelSpec{Name: "target", Server: ts.URL, Model: "t"},
		makePending(2), 1, ev)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("results = %d, want 2", len(got))
	}
	if got[0].Err == "" || !strings.Contains(got[0].Err, "boom") {
		t.Fatalf("first result should carry the generation error, got %+v", got[0])
	}
	if got[1].Err != "" || !got[1].Keep {
		t.Fatalf("second claim should succeed, got %+v", got[1])
	}
}

func TestDiscoverSkills(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"SKILL-go.md", "SKILL-base.md", "SKILL-ts.md", "README.md"} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// No filter: every SKILL-*.md, README ignored, sorted.
	all, err := discoverSkills(dir, Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if got := skillNames(all); len(got) != 3 || got[0] != "base" || got[1] != "go" || got[2] != "ts" {
		t.Fatalf("discover all = %v", got)
	}
	// Filtered.
	some, err := discoverSkills(dir, Settings{Skills: []string{"go"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := skillNames(some); len(got) != 1 || got[0] != "go" {
		t.Fatalf("discover filtered = %v", got)
	}
}

func TestResultsRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "results.jsonl")
	r1 := ProbeResult{ClaimID: "go#00", Verdict: "keep", Keep: true}
	r2 := ProbeResult{ClaimID: "go#01", Verdict: "drop"}
	// Two lines for go#00: the later one wins on read.
	r1b := ProbeResult{ClaimID: "go#00", Verdict: "drop", Keep: false}
	for _, r := range []ProbeResult{r1, r2, r1b} {
		if err := appendResult(path, r); err != nil {
			t.Fatal(err)
		}
	}
	got := readResults(path)
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	if got["go#00"].Keep {
		t.Fatalf("last write for go#00 should win (drop)")
	}
	if got["go#01"].Verdict != "drop" {
		t.Fatalf("go#01 = %+v", got["go#01"])
	}
}

func TestReadResultsSkipsMalformed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "results.jsonl")
	// A good row, a half-written (malformed) row, a blank line, another good row.
	content := `{"claim_id":"go#00","verdict":"keep","keep":true}
{"claim_id":"go#01","verdict":"dr
` + "\n" + `{"claim_id":"go#02","verdict":"drop"}
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	got := readResults(path)
	// Malformed go#01 skipped (re-probed later); the two valid rows survive.
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2 (%v)", len(got), got)
	}
	if _, ok := got["go#00"]; !ok {
		t.Fatal("go#00 missing")
	}
	if _, ok := got["go#02"]; !ok {
		t.Fatal("go#02 missing")
	}
	if _, ok := got["go#01"]; ok {
		t.Fatal("malformed go#01 should not be present")
	}
}

func TestReadResultsMissingFile(t *testing.T) {
	got := readResults(filepath.Join(t.TempDir(), "nope.jsonl"))
	if len(got) != 0 {
		t.Fatalf("missing file should yield empty map, got %d", len(got))
	}
}

func TestFmtDuration(t *testing.T) {
	cases := []struct {
		sec  int
		want string
	}{
		{0, "00:00:00"},
		{5, "00:00:05"},
		{65, "00:01:05"},
		{3661, "01:01:01"},
	}
	for _, c := range cases {
		if got := fmtDuration(time.Duration(c.sec) * time.Second); got != c.want {
			t.Fatalf("fmtDuration(%ds) = %q, want %q", c.sec, got, c.want)
		}
	}
}

func TestCachedOr(t *testing.T) {
	if got := cachedOr(50 * time.Millisecond); got != "cached" {
		t.Fatalf("fast step = %q, want cached", got)
	}
	if got := cachedOr(65 * time.Second); got != "00:01:05" {
		t.Fatalf("slow step = %q", got)
	}
}

func TestTally(t *testing.T) {
	ms := ModelStats{Skills: []SkillStats{
		{Kept: 2, Dropped: 1, Errored: 0},
		{Kept: 1, Dropped: 3, Errored: 1},
	}}
	k, d, e := tally(ms)
	if k != 3 || d != 4 || e != 1 {
		t.Fatalf("tally = %d/%d/%d, want 3/4/1", k, d, e)
	}
}

func TestPct(t *testing.T) {
	cases := []struct {
		orig, pruned int
		want         string
	}{
		{100, 66, "-34%"},
		{100, 100, "+0%"},
		{0, 0, "0%"},
		{100, 120, "+20%"},
	}
	for _, c := range cases {
		if got := pct(c.orig, c.pruned); got != c.want {
			t.Fatalf("pct(%d,%d) = %q, want %q", c.orig, c.pruned, got, c.want)
		}
	}
}
