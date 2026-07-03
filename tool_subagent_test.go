package main

import (
	"context"
	"strings"
	"testing"
)

// TestSubagentFoldRelaunch pins the launch_subagent context-full recovery: a
// subagent that dies on the n_ctx ceiling triggers a fold of the FOREGROUND
// session (whose history is what the pool and the history-inheriting subagent
// are full of), and the failed task is relaunched once per fold step instead
// of surfacing straight to a replan. Call order against the mock: subagent
// attempt (ceiling) → the fold's synchronous summarise → relaunched subagent
// (respond terminal).
func TestSubagentFoldRelaunch(t *testing.T) {
	mock := newMockLLM(t,
		sseTruncatedContent("partial", 1000, 100), // ceiling: truncated BELOW the cap
		sseText("folded-note"),                    // foldHistory's synchronous summarise
		sseToolCall("r1", respondToolName, `{"message":"sub ok"}`),
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	a.mainSlotTokens = 85248
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
	a.registerSubagentTool()

	// Parent history to fold: keepWindowStart has no server token stamps here,
	// so the first fold step keeps only the last assistant message and folds
	// everything before it.
	s.AddUser("question one")
	s.AddAssistant("answer one")
	s.AddUser("question two")
	s.AddAssistant("answer two")
	before := len(s.Messages)

	tc := toolCall{ID: "tc1"}
	tc.Function.Name = "launch_subagent"
	tc.Function.Arguments = `{"tasks":[{"instructions":"do the thing","task":"execute"}]}`
	out, failed := a.executeTool(context.Background(), s.ID, tc)
	if failed {
		t.Fatalf("fold-relaunch should recover the subagent, got failed=true: %s", out)
	}
	if !strings.Contains(out, "sub ok") {
		t.Errorf("output should carry the relaunched subagent's result, got: %s", out)
	}
	if got := mock.callCount(); got != 3 {
		t.Errorf("callCount = %d, want 3 (attempt, fold summarise, relaunch)", got)
	}
	if len(s.Messages) >= before {
		t.Errorf("foreground history should have folded: %d messages, was %d", len(s.Messages), before)
	}
	if s.Summary == "" {
		t.Errorf("fold should have rotated the older turns into the summary")
	}
}

// TestSubagentFoldRelaunchGivesUp pins the recovery's bound: when the relaunch
// dies on the ceiling again, the deeper fold step finds nothing more to free
// (step 1 already folded down to the last assistant message, so step 2 is a
// no-op) and the fan-out stops and reports the failure (failed=true on a
// fully-failed batch) instead of folding or relaunching forever.
func TestSubagentFoldRelaunchGivesUp(t *testing.T) {
	mock := newMockLLM(t,
		sseTruncatedContent("partial", 1000, 100), // attempt 1: ceiling
		sseText("folded-note"),                    // fold step 1 summarise
		sseTruncatedContent("partial", 1000, 100), // relaunch 1: ceiling again; step 2 frees nothing → stop
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	a.mainSlotTokens = 85248
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
	a.registerSubagentTool()
	s.AddUser("question one")
	s.AddAssistant("answer one")
	s.AddUser("question two")
	s.AddAssistant("answer two")

	tc := toolCall{ID: "tc1"}
	tc.Function.Name = "launch_subagent"
	tc.Function.Arguments = `{"tasks":[{"instructions":"do the thing","task":"execute"}]}`
	out, failed := a.executeTool(context.Background(), s.ID, tc)
	if !failed {
		t.Errorf("a batch that never recovered should report failed=true, got: %s", out)
	}
	if !strings.Contains(out, "FAILED") {
		t.Errorf("output should mark the subagent FAILED, got: %s", out)
	}
	if got := mock.callCount(); got != 3 {
		t.Errorf("callCount = %d, want 3 (attempt, fold summarise, one relaunch — no second fold to try)", got)
	}
}
