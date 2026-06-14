package main

import (
	"context"
	"encoding/json"
	"slices"
	"testing"
)

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
		s.improveFlow = improve
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
