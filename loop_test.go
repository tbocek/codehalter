package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// TestStartToolMeter pins the tool status meter's lifecycle: stop() halts the
// ticker and joins its goroutine promptly (no deadlock, no leak), and a cancelled
// ctx also lets stop() return. It does not assert the 1s-tick text (time-based,
// like streamWaitMeter, which is likewise not unit-tested).
func TestStartToolMeter(t *testing.T) {
	a, s := newTestAgent(t)

	// Normal stop joins quickly.
	stop := a.startToolMeter(context.Background(), s.ID, "web_search")
	if stop == nil {
		t.Fatal("startToolMeter returned a nil stop")
	}
	done := make(chan struct{})
	go func() { stop(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("stop() did not return: meter goroutine leaked or deadlocked")
	}

	// A cancelled ctx also unblocks stop().
	ctx, cancel := context.WithCancel(context.Background())
	stop2 := a.startToolMeter(ctx, s.ID, "run_command")
	cancel()
	done2 := make(chan struct{})
	go func() { stop2(); close(done2) }()
	select {
	case <-done2:
	case <-time.After(2 * time.Second):
		t.Fatal("stop() after ctx cancel did not return")
	}
}

// TestThrottledStream pins the token batcher that keeps per-token streaming from
// flooding the editor at high tg/s: the first token emits immediately, tokens
// within the interval batch into one emit, and flush drains the tail (and is a
// no-op when nothing is buffered).
func TestThrottledStream(t *testing.T) {
	var emits []string
	sink, flush := throttledStream(func(chunk string) { emits = append(emits, chunk) })

	sink("a") // first token: lastSent is zero, so it emits right away
	if len(emits) != 1 || emits[0] != "a" {
		t.Fatalf("first token should emit immediately: %v", emits)
	}
	sink("b") // within the interval → batched, not emitted
	sink("c")
	if len(emits) != 1 {
		t.Errorf("tokens within the interval should batch, got %v", emits)
	}
	flush() // emit the tail
	if len(emits) != 2 || emits[1] != "bc" {
		t.Errorf("flush should emit the batched tail: %v", emits)
	}
	flush() // nothing buffered → no emit
	if len(emits) != 2 {
		t.Errorf("empty flush should not emit: %v", emits)
	}
}

// TestImproveExecuteSkipsVerify pins that during an /improve execute pass the
// build/test runners are SKIPPED at the tool layer (the edits are .md only): the
// model's run_task call never executes, while a normal execute pass runs it. The
// skip is non-failing (see TestSkipToolCall) so it can't condemn the subtask.
func TestImproveExecuteSkipsVerify(t *testing.T) {
	withFreshToolRegistry(t)
	var ran int
	RegisterTool(Tool{
		Def: map[string]any{"type": "function", "function": map[string]any{
			"name": "run_task", "description": "x",
			"parameters": map[string]any{"type": "object"}}},
		Execute: func(_ context.Context, _ *agent, _, _ string) (string, bool) {
			ran++
			return "ran the task", false
		},
	})

	run := func(improve bool) {
		// run_task, then two empty (text) rounds: with a terminal policy the first
		// empty round is nudged, the second exits.
		mock := newMockLLM(t, sseToolCall("c0", "run_task", `{"task":"just:build"}`), sseText("a"), sseText("done"))
		defer mock.Close()
		a, s := newTestAgent(t)
		a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
		s.improving.Store(improve)
		a.runExecutePhase(context.Background(), s.ID, subtask{Description: "apply"}, 0, 1)
	}

	ran = 0
	run(false)
	if ran != 1 {
		t.Errorf("normal execute: run_task should run (ran=%d, want 1)", ran)
	}
	ran = 0
	run(true)
	if ran != 0 {
		t.Errorf("/improve execute: run_task must be SKIPPED, but it ran (ran=%d, want 0)", ran)
	}
}

// TestImproveRespondFunnel pins the /improve funnel: a bare `respond` must NOT
// end the turn before submit_improvement ran. The loop nudges respond toward
// submit_improvement; once that's called, codehalter applies the change in code.
// Proof: PLAN.md is edited only if the gated respond let the loop continue to the
// submit_improvement call (a model that just analysed and respond'd would leave it
// untouched — the original bug).
func TestImproveRespondFunnel(t *testing.T) {
	a, s := newTestAgent(t)
	a.mu.Lock()
	a.mode = "Autopilot" // auto-answer the Apply card (no editor conn in the test)
	a.mu.Unlock()
	s.improving.Store(true)

	ch := filepath.Join(s.Cwd, ".codehalter")
	os.MkdirAll(ch, 0o755)
	os.WriteFile(filepath.Join(ch, "PLAN.md"), []byte("alpha OLD omega"), 0o644)
	// No LICENSE: the edit applies locally, submission is refused (no HTTP call).

	subArgs := submitImprovementArgs(t, map[string]string{
		"improvements": `[{"title":"t","file":"PLAN.md","type":"replace","original":"OLD","new":"NEW","reasoning":"r"}]`,
	})
	// Round 1: a premature prose respond (the observed failure). Round 2: the
	// structured submit_improvement the funnel pushes it toward.
	mock := newMockLLM(t,
		sseToolCall("c0", "respond", `{"message":"Here is my analysis of the session."}`),
		sseToolCall("c1", "submit_improvement", subArgs),
	)
	defer mock.Close()
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}

	a.runExecutePhase(context.Background(), s.ID, subtask{Description: "apply the analysis"}, 0, 1)

	if b, _ := os.ReadFile(filepath.Join(ch, "PLAN.md")); string(b) != "alpha NEW omega" {
		t.Errorf("respond was not funnelled to submit_improvement (PLAN.md unchanged): %q", b)
	}
}

// TestPlanResultSubtasksDeserialize ensures the planner's JSON output (an
// array of `{description, verify}` objects under `subtasks`) round-trips into
// the planResult / subtask structs the orchestrator consumes. This is the
// contract between PLAN.md and runExecutePhase.
func TestPlanResultSubtasksDeserialize(t *testing.T) {
	raw := `{
		"clear": true,
		"subtasks": [
			{"description": "refactor storage", "verify": ["go build ./...", "go test ./storage/..."]},
			{"description": "update API", "verify": ["curl /healthz"]},
			{"description": "write migration"}
		]
	}`
	var p planResult
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(p.Subtasks) != 3 {
		t.Fatalf("subtasks: want 3, got %d", len(p.Subtasks))
	}
	if p.Subtasks[0].Description != "refactor storage" {
		t.Errorf("subtasks[0].Description = %q", p.Subtasks[0].Description)
	}
	if !slices.Equal(p.Subtasks[0].Verify, []string{"go build ./...", "go test ./storage/..."}) {
		t.Errorf("subtasks[0].Verify = %v", p.Subtasks[0].Verify)
	}
	if len(p.Subtasks[2].Verify) != 0 {
		t.Errorf("subtasks[2].Verify should be empty (omitempty), got %v", p.Subtasks[2].Verify)
	}

	// report_only round-trips so confirmPlan can skip the gate.
	rawReport := `{"clear": true, "report_only": true, "subtasks": [{"description": "summarise X"}]}`
	var p2 planResult
	if err := json.Unmarshal([]byte(rawReport), &p2); err != nil {
		t.Fatalf("unmarshal report_only: %v", err)
	}
	if !p2.ReportOnly {
		t.Errorf("expected report_only=true")
	}
}

// TestIssueBagTokenisation pins the bag-of-words tokeniser used for fuzzy
// failure matching: lowercase, punctuation-stripped, order-independent. The
// reworded-near-duplicate case ("missing import" vs "import is missing") is
// the one that motivates the fuzzy approach over exact key matching.
func TestIssueBagTokenisation(t *testing.T) {
	// Casing, punctuation and word order are all discarded.
	a := issueBag([]string{"Missing import!", "Syntax error."})
	b := issueBag([]string{"syntax  ERROR", "missing\timport"})
	if !slices.Equal(sortedKeys(a), sortedKeys(b)) {
		t.Errorf("expected equivalent bags, got %v vs %v", sortedKeys(a), sortedKeys(b))
	}

	// Adjacent non-alphanumeric runs collapse to a single separator (no empty
	// tokens leak into the bag).
	bag := issueBag([]string{"foo--bar...baz"})
	want := []string{"bar", "baz", "foo"}
	if !slices.Equal(sortedKeys(bag), want) {
		t.Errorf("got %v, want %v", sortedKeys(bag), want)
	}
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

// TestJaccardSimilarity covers the failure-loop bail decision. The reworded
// near-duplicate must score above the configured threshold so the retry
// loop bails; unrelated failures must stay below.
func TestJaccardSimilarity(t *testing.T) {
	// Two empty bags are treated as identical (degenerate but well-defined).
	if got := jaccard(map[string]bool{}, map[string]bool{}); got != 1 {
		t.Errorf("empty/empty: got %v, want 1", got)
	}

	// Reworded duplicate: {"missing","import"} vs {"import","is","missing"}.
	// |∩|=2, |∪|=3 → 0.666… → must exceed the threshold so a retry bails.
	a := issueBag([]string{"missing import"})
	b := issueBag([]string{"import is missing"})
	if s := jaccard(a, b); s < failureSimilarityThreshold {
		t.Errorf("reworded duplicate: got %v, want >= %v", s, failureSimilarityThreshold)
	}

	// Unrelated failures must NOT collapse — exact wording chosen so the
	// Jaccard score is comfortably under the threshold.
	c := issueBag([]string{"missing import in foo.go"})
	d := issueBag([]string{"unused variable x"})
	if s := jaccard(c, d); s >= failureSimilarityThreshold {
		t.Errorf("disjoint issues: got %v, want < %v", s, failureSimilarityThreshold)
	}

	// Symmetric.
	if jaccard(a, b) != jaccard(b, a) {
		t.Errorf("expected jaccard to be symmetric")
	}
}

// TestCapHitLadder pins the cap-hit recovery in runToolLoopSeeded: a generation
// truncated AT max_tokens first retries with a be-concise nudge appended to the
// wire context, then once more on a doubled cap, and a response that then
// succeeds ends the loop normally. Discarded partials never reach the result.
func TestCapHitLadder(t *testing.T) {
	mock := newMockLLM(t,
		sseTruncatedContent("way too long", 1000, defaultMaxTokens),
		sseTruncatedContent("still too long", 1000, defaultMaxTokens),
		sseText("done"),
	)
	defer mock.Close()
	a, s := newTestAgent(t)
	a.mainSlotTokens = 85248 // ample n_ctx room: these are cap hits, not the ceiling

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", false, 0)
	if err != nil {
		t.Fatalf("ladder should recover: %v", err)
	}
	if got := mock.callCount(); got != 3 {
		t.Fatalf("callCount = %d, want 3 (cap, nudged cap, success)", got)
	}
	if res.Text != "done" {
		t.Errorf("res.Text = %q, want the successful reply only (partials discarded)", res.Text)
	}

	// Rung 1: the second request must carry the be-concise nudge as the
	// trailing user message.
	req2 := mock.request(1)
	msgs, _ := req2["messages"].([]any)
	if len(msgs) == 0 {
		t.Fatalf("second request has no messages")
	}
	last, _ := msgs[len(msgs)-1].(map[string]any)
	if last["role"] != "user" || !strings.Contains(fmt.Sprint(last["content"]), "output limit") {
		t.Errorf("second request should end with the be-concise nudge, got: %v", last)
	}
	if mt, ok := req2["max_tokens"].(float64); !ok || int(mt) != defaultMaxTokens {
		t.Errorf("second request max_tokens = %v, want unchanged %d", req2["max_tokens"], defaultMaxTokens)
	}

	// Rung 2: the third request runs on the doubled cap.
	req3 := mock.request(2)
	if mt, ok := req3["max_tokens"].(float64); !ok || int(mt) != 2*defaultMaxTokens {
		t.Errorf("third request max_tokens = %v, want doubled %d", req3["max_tokens"], 2*defaultMaxTokens)
	}
}

// TestCapHitLadderExhausted pins the ladder's exit: a third consecutive cap hit
// stops retrying and surfaces the cap error into the normal failure path
// (replan), instead of doubling forever.
func TestCapHitLadderExhausted(t *testing.T) {
	mock := newMockLLM(t,
		sseTruncatedContent("too long", 1000, defaultMaxTokens),
		sseTruncatedContent("too long", 1000, defaultMaxTokens),
		sseTruncatedContent("too long", 1000, 2*defaultMaxTokens),
	)
	defer mock.Close()
	a, s := newTestAgent(t)
	a.mainSlotTokens = 85248

	_, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", false, 0)
	if asCapHit(err) == nil {
		t.Fatalf("exhausted ladder should surface the cap error, got: %v", err)
	}
	if got := mock.callCount(); got != 3 {
		t.Errorf("callCount = %d, want 3 (no retries past the ladder)", got)
	}
}

// TestStuckLadderFuzzyOutput pins the Jaccard extension of the repetition
// ladder: a re-issued identical call whose output differs only in noise (an
// elapsed-time / attempt counter, i.e. a timestamped failing build) counts as
// reproduced, so the loop bails at stuckBailRounds instead of spinning while
// the exact output hash keeps changing.
func TestStuckLadderFuzzyOutput(t *testing.T) {
	const toolName = "noisy_probe_test_tool"
	attempt := 0
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        toolName,
			"description": "test-only failing probe with noisy output",
			"parameters":  map[string]any{"type": "object"},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		attempt++
		// Long fixed error text + two noise tokens (elapsed, attempt) → Jaccard
		// ≈ 0.93 between consecutive outputs: above stuckOutputSimilarity while
		// the fnv hash differs every time.
		return fmt.Sprintf("build FAILED: cannot load package example.com/foo/bar: import cycle not allowed in dependency graph involving widget factory manager controller service repository handler adapter transport codec parser lexer scanner tokenizer emitter renderer scheduler dispatcher broker queue worker pool cache index shard replica leader follower quorum consensus journal snapshot compaction segment (elapsed %dms, attempt %d)", 1200+attempt*7, attempt), true
	}})
	t.Cleanup(func() { UnregisterToolsByPrefix(toolName) })

	call := sseToolCall("c1", toolName, `{}`)
	responses := make([]string, 12)
	for i := range responses {
		responses[i] = call
	}
	mock := newMockLLM(t, responses...)
	defer mock.Close()
	a, s := newTestAgent(t)
	a.mainSlotTokens = 85248

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", false, 0)
	if err != nil {
		t.Fatalf("stuck bail is a graceful exit, got error: %v", err)
	}
	if res.RespondCalled {
		t.Errorf("bail must not report a terminal exit")
	}
	// Round 1 registers the first output; rounds 2-6 are fuzzy-reproduced stuck
	// rounds, and the ladder bails at stuckBailRounds — 6 calls total.
	if got := mock.callCount(); got != 1+stuckBailRounds {
		t.Errorf("callCount = %d, want %d (bail at stuckBailRounds via fuzzy match)", got, 1+stuckBailRounds)
	}
}
