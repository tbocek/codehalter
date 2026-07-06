package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// schedCounts tracks how many of each judge/target stage the fake cluster served.
type schedCounts struct{ author, judge, gen atomic.Int32 }

// schedServers spins up a fake judge (authors a fixed question, then scores each
// sample with `verdict`) and a fake target (returns "answer N" after targetDelay).
// The judge tells author calls from judge calls by the system prompt. Prep never
// hits these — tests supply a fake prepFunc — so author+judge counts are exact.
func schedServers(t *testing.T, verdict string, targetDelay time.Duration) (judge, target ModelSpec, cnt *schedCounts) {
	t.Helper()
	cnt = &schedCounts{}
	js := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if len(req.Messages) > 0 && req.Messages[0].Content == authorPrompt {
			cnt.author.Add(1)
			_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"{\"question\":\"q\",\"rubric\":\"r\"}"}}]}`)))
			return
		}
		cnt.judge.Add(1)
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, verdict))))
	}))
	t.Cleanup(js.Close)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cnt.gen.Add(1)
		if targetDelay > 0 {
			time.Sleep(targetDelay)
		}
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"answer"}}]}`)))
	}))
	t.Cleanup(ts.Close)
	return ModelSpec{Name: "main", Server: js.URL, Model: "j"},
		ModelSpec{Name: "gemma", Server: ts.URL, Model: "t"}, cnt
}

const keepVerdict = `{"a_satisfies":false,"b_satisfies":true,"similar":false,"verdict":"keep","reason":"ok"}`

func newTestStore(t *testing.T, skills ...string) *probeStore {
	t.Helper()
	dir := t.TempDir()
	q := map[string]map[string]Question{}
	qp := map[string]string{}
	for _, s := range skills {
		q[s] = map[string]Question{}
		qp[s] = filepath.Join(dir, "authored-"+s+".json")
	}
	return &probeStore{
		questions:    q,
		questionPath: qp,
		samplesPath:  filepath.Join(dir, "samples.jsonl"),
		resultsPath:  filepath.Join(dir, "results.jsonl"),
	}
}

// fakeSkills / fakePrep stand in for streamline+segment: fakePrep returns n
// deterministic claims per skill with no LLM call, so tests exercise the
// scheduler (not segmentation).
func fakeSkills(stacks ...string) []skillSource {
	var out []skillSource
	for _, s := range stacks {
		out = append(out, skillSource{stack: s, path: s + ".md"})
	}
	return out
}

func fakePrep(n int) prepFunc {
	return func(_ ModelSpec, sk skillSource) ([]byte, []Claim, error) {
		var claims []Claim
		for i := 0; i < n; i++ {
			claims = append(claims, Claim{ID: fmt.Sprintf("%s#%02d", sk.stack, i), Skill: sk.stack, Text: "claim"})
		}
		return []byte("orig-" + sk.stack), claims, nil
	}
}

// resume helpers.
func freshResume(Claim) (Question, bool, []samplePair, bool, bool) {
	return Question{}, false, nil, false, false // everything needs authoring
}
func authoredResume(Claim) (Question, bool, []samplePair, bool, bool) {
	return Question{Question: "q", Rubric: "r"}, true, nil, false, false // skip authoring
}
func generatedResume(Claim) (Question, bool, []samplePair, bool, bool) {
	return Question{Question: "q", Rubric: "r"}, true, []samplePair{{AnswerA: "a", AnswerB: "b"}}, true, false // skip to judging
}

// collector gathers onResult/onSkillDone concurrently.
type collector struct {
	mu      sync.Mutex
	results []ProbeResult
	skills  []string
}

func (c *collector) ev() schedEvents {
	return schedEvents{
		onResult: func(_ ModelSpec, _ *probeItem, res ProbeResult, _ int64) {
			c.mu.Lock()
			c.results = append(c.results, res)
			c.mu.Unlock()
		},
		onSkillDone: func(s string, _ []byte, _ []Claim) {
			c.mu.Lock()
			c.skills = append(c.skills, s)
			c.mu.Unlock()
		},
	}
}

// The full path: PREP → AUTHOR → GENERATE → SCORE, every artifact persisted.
func TestRunProbePassFullPathPersists(t *testing.T) {
	judge, target, cnt := schedServers(t, keepVerdict, 0)
	store := newTestStore(t, "go")
	var c collector
	if err := runProbePass(context.Background(), []ModelSpec{judge}, target,
		fakeSkills("go"), fakePrep(3), freshResume, 1, 1, store, c.ev()); err != nil {
		t.Fatal(err)
	}
	if len(c.results) != 3 {
		t.Fatalf("results = %d, want 3", len(c.results))
	}
	for _, r := range c.results {
		if r.Err != "" || !r.Keep {
			t.Fatalf("result not a clean keep: %+v", r)
		}
	}
	if cnt.author.Load() != 3 || cnt.gen.Load() != 6 || cnt.judge.Load() != 3 {
		t.Fatalf("stage counts author=%d gen=%d judge=%d, want 3/6/3", cnt.author.Load(), cnt.gen.Load(), cnt.judge.Load())
	}
	if got := readResults(store.resultsPath); len(got) != 3 {
		t.Fatalf("results.jsonl has %d, want 3", len(got))
	}
	if got := readSamples(store.samplesPath); len(got) != 3 {
		t.Fatalf("samples.jsonl has %d, want 3", len(got))
	}
	if got := readQuestionCache(store.questionPath["go"]); len(got) != 3 {
		t.Fatalf("question cache has %d, want 3", len(got))
	}
	if len(c.skills) != 1 || c.skills[0] != "go" {
		t.Fatalf("onSkillDone = %v, want [go] exactly once", c.skills)
	}
}

// Resume from persisted samples: claims enter at SCORE, so nothing is authored or
// generated.
func TestRunProbePassResumeFromSamples(t *testing.T) {
	judge, target, cnt := schedServers(t, keepVerdict, 0)
	store := newTestStore(t, "go")
	var c collector
	if err := runProbePass(context.Background(), []ModelSpec{judge}, target,
		fakeSkills("go"), fakePrep(4), generatedResume, 1, 1, store, c.ev()); err != nil {
		t.Fatal(err)
	}
	if cnt.gen.Load() != 0 || cnt.author.Load() != 0 {
		t.Fatalf("resume-from-samples still authored/generated: gen=%d author=%d", cnt.gen.Load(), cnt.author.Load())
	}
	if cnt.judge.Load() != 4 || len(c.results) != 4 {
		t.Fatalf("judge=%d results=%d, want 4/4", cnt.judge.Load(), len(c.results))
	}
}

// Resume from a cached question: authoring is skipped, generation + judging run.
func TestRunProbePassResumeFromQuestion(t *testing.T) {
	judge, target, cnt := schedServers(t, keepVerdict, 0)
	store := newTestStore(t, "go")
	var c collector
	if err := runProbePass(context.Background(), []ModelSpec{judge}, target,
		fakeSkills("go"), fakePrep(2), authoredResume, 1, 1, store, c.ev()); err != nil {
		t.Fatal(err)
	}
	if cnt.author.Load() != 0 {
		t.Fatalf("cached question still re-authored: author=%d", cnt.author.Load())
	}
	if cnt.gen.Load() != 4 || cnt.judge.Load() != 2 {
		t.Fatalf("gen=%d judge=%d, want 4/2", cnt.gen.Load(), cnt.judge.Load())
	}
}

// A generation error is recorded (so resume retries it) and the pass carries on.
func TestRunProbePassGenerationErrorRecorded(t *testing.T) {
	var genN atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if genN.Add(1) == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"boom"}}`))
			return
		}
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"fine"}}]}`)))
	}))
	defer ts.Close()
	js := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, keepVerdict))))
	}))
	defer js.Close()

	store := newTestStore(t, "go")
	var c collector
	err := runProbePass(context.Background(),
		[]ModelSpec{{Name: "main", Server: js.URL, Model: "j"}},
		ModelSpec{Name: "gemma", Server: ts.URL, Model: "t"},
		fakeSkills("go"), fakePrep(3), authoredResume, 1, 1, store, c.ev())
	if err != nil {
		t.Fatal(err)
	}
	if len(c.results) != 3 {
		t.Fatalf("results = %d, want 3", len(c.results))
	}
	errored := 0
	for _, r := range c.results {
		if r.Err != "" {
			errored++
		}
	}
	if errored != 1 {
		t.Fatalf("errored results = %d, want exactly 1", errored)
	}
}

// A judge that returns an empty question can't author — the claim stays untested
// (no result row) and the pass still completes and fires onSkillDone.
func TestRunProbePassAuthorFailureUntested(t *testing.T) {
	js := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"{\"question\":\"\",\"rubric\":\"\"}"}}]}`)))
	}))
	defer js.Close()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"x"}}]}`)))
	}))
	defer ts.Close()

	store := newTestStore(t, "go")
	var authorFails atomic.Int32
	var c collector
	ev := c.ev()
	ev.onAuthorFail = func(ModelSpec, *probeItem) { authorFails.Add(1) }
	if err := runProbePass(context.Background(),
		[]ModelSpec{{Name: "main", Server: js.URL, Model: "j"}},
		ModelSpec{Name: "gemma", Server: ts.URL, Model: "t"},
		fakeSkills("go"), fakePrep(3), freshResume, 1, 1, store, ev); err != nil {
		t.Fatal(err)
	}
	if authorFails.Load() != 3 {
		t.Fatalf("author failures = %d, want 3", authorFails.Load())
	}
	if len(c.results) != 0 {
		t.Fatalf("unauthored claims must not produce results, got %d", len(c.results))
	}
	if len(c.skills) != 1 {
		t.Fatalf("skill should still complete once, got %v", c.skills)
	}
}

// Cancelling mid-pass returns nil, persists what finished, and does not fire
// onSkillDone for the interrupted skill.
func TestRunProbePassCancellation(t *testing.T) {
	judge, target, _ := schedServers(t, keepVerdict, 25*time.Millisecond)
	store := newTestStore(t, "go")
	ctx, cancel := context.WithCancel(context.Background())
	var got, skillDone atomic.Int32
	ev := schedEvents{
		onResult: func(_ ModelSpec, _ *probeItem, _ ProbeResult, _ int64) {
			if got.Add(1) == 1 {
				cancel()
			}
		},
		onSkillDone: func(string, []byte, []Claim) { skillDone.Add(1) },
	}
	done := make(chan error, 1)
	go func() {
		done <- runProbePass(ctx, []ModelSpec{judge}, target,
			fakeSkills("go"), fakePrep(12), authoredResume, 1, 1, store, ev)
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("cancelled pass should return nil, got %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("pass hung after cancellation")
	}
	if n := got.Load(); n < 1 || n >= 12 {
		t.Fatalf("recorded %d results after cancel, want 1..11", n)
	}
	if skillDone.Load() != 0 {
		t.Fatal("onSkillDone must not fire for an interrupted skill")
	}
}

// A result that cannot be persisted aborts the pass.
func TestRunProbePassPersistenceErrorFatal(t *testing.T) {
	judge, target, _ := schedServers(t, keepVerdict, 0)
	store := newTestStore(t, "go")
	store.resultsPath = filepath.Join(t.TempDir(), "missing-dir", "results.jsonl")
	var c collector
	err := runProbePass(context.Background(), []ModelSpec{judge}, target,
		fakeSkills("go"), fakePrep(2), generatedResume, 1, 1, store, c.ev())
	if err == nil {
		t.Fatal("a failed result write must abort the pass")
	}
}

// Under real concurrency (multi-slot judge + target) the pass
// stays race-free and complete: every claim across skills lands, each skill
// finalizes once. Run with -race.
func TestRunProbePassConcurrentIsRaceFree(t *testing.T) {
	judge, target, _ := schedServers(t, keepVerdict, 0)
	judge.Parallel, judge.sem = 4, make(chan struct{}, 4)
	target.Parallel, target.sem = 4, make(chan struct{}, 4)
	store := newTestStore(t, "go", "bash")
	var c collector
	if err := runProbePass(context.Background(), []ModelSpec{judge}, target,
		fakeSkills("go", "bash"), fakePrep(10), freshResume, 2, 1, store, c.ev()); err != nil {
		t.Fatal(err)
	}
	if len(c.results) != 20 {
		t.Fatalf("results = %d, want 20", len(c.results))
	}
	if got := readResults(store.resultsPath); len(got) != 20 {
		t.Fatalf("results.jsonl has %d, want 20", len(got))
	}
	seen := map[string]bool{}
	for _, s := range c.skills {
		if seen[s] {
			t.Fatalf("skill %s finalized twice", s)
		}
		seen[s] = true
	}
	if !seen["go"] || !seen["bash"] || len(seen) != 2 {
		t.Fatalf("skills finalized = %v, want go+bash once each", c.skills)
	}
}

// Overlap: while the judge scores claim 0, the target is already generating later
// claims. The judge's first score blocks until the target has run ahead.
func TestRunProbePassOverlapsGenerationAndJudging(t *testing.T) {
	var genServed atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		genServed.Add(1)
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"answer"}}]}`)))
	}))
	defer ts.Close()
	var judgeFirstSaw atomic.Int32
	judgeFirstSaw.Store(-1)
	var first atomic.Bool
	first.Store(true)
	js := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if first.CompareAndSwap(true, false) {
			deadline := time.Now().Add(5 * time.Second)
			for genServed.Load() < 4 && time.Now().Before(deadline) {
				time.Sleep(time.Millisecond)
			}
			judgeFirstSaw.Store(genServed.Load())
		}
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, keepVerdict))))
	}))
	defer js.Close()

	store := newTestStore(t, "go")
	var c collector
	if err := runProbePass(context.Background(),
		[]ModelSpec{{Name: "main", Server: js.URL, Model: "j"}},
		ModelSpec{Name: "gemma", Server: ts.URL, Model: "t"},
		fakeSkills("go"), fakePrep(3), authoredResume, 1, 1, store, c.ev()); err != nil {
		t.Fatal(err)
	}
	if len(c.results) != 3 {
		t.Fatalf("results = %d, want 3", len(c.results))
	}
	if s := judgeFirstSaw.Load(); s < 4 {
		t.Fatalf("no overlap: target served only %d gen calls while the judge held claim 0 (want >=4)", s)
	}
	if g := genServed.Load(); g != 6 {
		t.Fatalf("target gen calls = %d, want 6", g)
	}
}

// Queues are unbounded: the target must NOT block on a slow judge — it generates
// every claim while the judge is stalled.
func TestRunProbePassUnboundedRunsAhead(t *testing.T) {
	release := make(chan struct{})
	var genServed atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		genServed.Add(1)
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"answer"}}]}`)))
	}))
	defer ts.Close()
	js := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, keepVerdict))))
	}))
	defer js.Close()

	const n = 8
	store := newTestStore(t, "go")
	var c collector
	done := make(chan error, 1)
	go func() {
		done <- runProbePass(context.Background(),
			[]ModelSpec{{Name: "main", Server: js.URL, Model: "j"}},
			ModelSpec{Name: "gemma", Server: ts.URL, Model: "t"},
			fakeSkills("go"), fakePrep(n), authoredResume, 1, 1, store, c.ev())
	}()
	want := int32(n * 2)
	deadline := time.Now().Add(3 * time.Second)
	for genServed.Load() < want && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if g := genServed.Load(); g != want {
		t.Fatalf("unbounded: target generated %d gen calls while the judge stalled, want all %d", g, want)
	}
	close(release)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("pass hung after releasing the judge")
	}
	if len(c.results) != n {
		t.Fatalf("results = %d, want %d", len(c.results), n)
	}
}

// With two judge endpoints, both take work.
func TestRunProbePassSpreadsAcrossJudges(t *testing.T) {
	newJudge := func(name string) (ModelSpec, *atomic.Int32) {
		var calls atomic.Int32
		s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			var req chatRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if len(req.Messages) > 0 && req.Messages[0].Content == authorPrompt {
				_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"{\"question\":\"q\",\"rubric\":\"r\"}"}}]}`)))
				return
			}
			_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, keepVerdict))))
		}))
		t.Cleanup(s.Close)
		return ModelSpec{Name: name, Server: s.URL, Model: "j"}, &calls
	}
	j1, c1 := newJudge("main")
	j2, c2 := newJudge("judge2")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"answer"}}]}`)))
	}))
	defer ts.Close()
	target := ModelSpec{Name: "gemma", Server: ts.URL, Model: "t"}
	store := newTestStore(t, "go")
	var c collector
	if err := runProbePass(context.Background(), []ModelSpec{j1, j2}, target,
		fakeSkills("go"), fakePrep(20), freshResume, 1, 1, store, c.ev()); err != nil {
		t.Fatal(err)
	}
	if len(c.results) != 20 {
		t.Fatalf("results = %d, want 20", len(c.results))
	}
	if c1.Load() == 0 || c2.Load() == 0 {
		t.Fatalf("work not spread across judges: main=%d judge2=%d calls", c1.Load(), c2.Load())
	}
}

func TestPreferAuthor(t *testing.T) {
	cases := []struct {
		genQueued, slots int
		want             bool
	}{
		{0, 2, true},  // target idle-risk → author
		{1, 2, true},  // still below one-per-slot → author
		{2, 2, false}, // one per slot queued → score/prep
		{5, 2, false}, // well fed
		{0, 0, true},  // slots clamps to >=1, so 0 < 1 → author
	}
	for _, c := range cases {
		if got := preferAuthor(c.genQueued, c.slots); got != c.want {
			t.Fatalf("preferAuthor(%d,%d) = %v, want %v", c.genQueued, c.slots, got, c.want)
		}
	}
}

// strengthenFixture: a judge whose score verdicts are scripted per call and
// whose strengthen pass returns a fixed rewrite, plus a target that records
// every system prompt it was given.
func strengthenFixture(t *testing.T, scoreReplies []string) (judge, target ModelSpec, strengthens *atomic.Int32, targetSystems *[]string, mu *sync.Mutex) {
	t.Helper()
	strengthens = &atomic.Int32{}
	var scoreN atomic.Int32
	js := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		sys := ""
		if len(req.Messages) > 0 {
			sys = req.Messages[0].Content
		}
		switch sys {
		case strengthenPrompt:
			strengthens.Add(1)
			_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"{\"text\":\"- STRONGER: never deviate, even when you know better.\"}"}}]}`)))
		case judgePrompt:
			i := int(scoreN.Add(1)) - 1
			if i >= len(scoreReplies) {
				i = len(scoreReplies) - 1
			}
			_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, scoreReplies[i]))))
		default:
			t.Errorf("unexpected judge system prompt: %.60s", sys)
		}
	}))
	t.Cleanup(js.Close)

	mu = &sync.Mutex{}
	systems := []string{}
	targetSystems = &systems
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		sys := ""
		if len(req.Messages) > 0 && req.Messages[0].Role == "system" {
			sys = req.Messages[0].Content
		}
		mu.Lock()
		systems = append(systems, sys)
		*targetSystems = systems
		mu.Unlock()
		_, _ = w.Write([]byte(sse(`{"choices":[{"delta":{"content":"answer"}}]}`)))
	}))
	t.Cleanup(ts.Close)
	return ModelSpec{Name: "main", Server: js.URL, Model: "j"},
		ModelSpec{Name: "gemma", Server: ts.URL, Model: "t"}, strengthens, targetSystems, mu
}

const skillWithWeak = "line one\n- WEAK STATEMENT\nline three"

func weakPrep(_ ModelSpec, sk skillSource) ([]byte, []Claim, error) {
	return []byte(skillWithWeak), []Claim{{
		ID: sk.stack + "#weak", Skill: sk.stack, Text: "the weak statement",
		Source: "- WEAK STATEMENT", StartLine: 2, EndLine: 2,
	}}, nil
}

const ineffDrop = `{"a_satisfies":false,"b_satisfies":false,"similar":true,"verdict":"drop","reason":"ignored the statement"}`

// An ineffective drop triggers exactly one strengthened retry: the judge
// rewrites the statement, the target regenerates ONLY arm B with the rewrite
// spliced into the skill, and the retry's keep wins (with StrengthenedText set).
func TestRunProbePassStrengthenRetryKeeps(t *testing.T) {
	judge, target, strengthens, systems, mu := strengthenFixture(t, []string{ineffDrop, keepVerdict})
	store := newTestStore(t, "go")
	var c collector
	if err := runProbePass(context.Background(), []ModelSpec{judge}, target,
		fakeSkills("go"), weakPrep, authoredResume, 1, 1, store, c.ev()); err != nil {
		t.Fatal(err)
	}

	if len(c.results) != 2 {
		t.Fatalf("results = %d, want 2 (ineffective drop + strengthened keep)", len(c.results))
	}
	first, second := c.results[0], c.results[1]
	if first.Keep || !first.Ineffective || first.StrengthenedText != "" {
		t.Fatalf("first row should be a plain ineffective drop: %+v", first)
	}
	if !second.Keep || !strings.Contains(second.StrengthenedText, "STRONGER") {
		t.Fatalf("second row should be a strengthened keep: %+v", second)
	}
	if strengthens.Load() != 1 {
		t.Fatalf("strengthen calls = %d, want 1", strengthens.Load())
	}
	// Ledger: last row wins on resume → the keep.
	if got := readResults(store.resultsPath); !got["go#weak"].Keep || got["go#weak"].StrengthenedText == "" {
		t.Fatalf("resume ledger should end on the strengthened keep: %+v", got["go#weak"])
	}
	// Target calls: first round A("")+B(skill) = 2, retry = 1 (B only) = 3 total,
	// and the retry's system is the skill with the rewrite spliced over the span.
	mu.Lock()
	defer mu.Unlock()
	if len(*systems) != 3 {
		t.Fatalf("target calls = %d, want 3 (A+B, then B-only retry)", len(*systems))
	}
	retrySys := (*systems)[2]
	if !strings.Contains(retrySys, "STRONGER") || strings.Contains(retrySys, "WEAK STATEMENT") ||
		!strings.Contains(retrySys, "line one") || !strings.Contains(retrySys, "line three") {
		t.Fatalf("retry B system not the spliced skill: %q", retrySys)
	}
}

// If the strengthened retry STILL fails, the claim ends as a drop (ineffective,
// with the attempted wording recorded) and there is no second retry.
func TestRunProbePassStrengthenRetryStillIneffective(t *testing.T) {
	judge, target, strengthens, systems, mu := strengthenFixture(t, []string{ineffDrop, ineffDrop})
	store := newTestStore(t, "go")
	var c collector
	if err := runProbePass(context.Background(), []ModelSpec{judge}, target,
		fakeSkills("go"), weakPrep, authoredResume, 1, 1, store, c.ev()); err != nil {
		t.Fatal(err)
	}
	if len(c.results) != 2 {
		t.Fatalf("results = %d, want 2", len(c.results))
	}
	final := c.results[1]
	if final.Keep || !final.Ineffective || !strings.Contains(final.StrengthenedText, "STRONGER") {
		t.Fatalf("final row should be an ineffective drop recording the attempt: %+v", final)
	}
	if strengthens.Load() != 1 {
		t.Fatalf("strengthen calls = %d, want exactly 1 (no retry loop)", strengthens.Load())
	}
	mu.Lock()
	n := len(*systems)
	mu.Unlock()
	if n != 3 {
		t.Fatalf("target calls = %d, want 3", n)
	}
	if len(c.skills) != 1 {
		t.Fatalf("skill must still finalize once, got %v", c.skills)
	}
}

// An empty judge pool is a hard error, not a hang (the worker accounting would
// otherwise wait forever on a judge worker that was never spawned).
func TestRunProbePassNoJudgesErrors(t *testing.T) {
	store := newTestStore(t, "go")
	done := make(chan error, 1)
	go func() {
		done <- runProbePass(context.Background(), nil, ModelSpec{Name: "t"},
			fakeSkills("go"), fakePrep(1), freshResume, 1, 1, store, collectorEv(t))
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("no judges must be an error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("runProbePass hung on an empty judge pool")
	}
}

// collectorEv is a throwaway event sink for tests that only care about errors.
func collectorEv(t *testing.T) schedEvents {
	t.Helper()
	return schedEvents{}
}
