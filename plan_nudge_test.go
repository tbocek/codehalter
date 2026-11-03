package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// sseToolCallUsing is sseToolCall with the call's completion tokens reported.
func sseToolCallUsing(id, name, args string, completion int) string {
	call, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]any{"tool_calls": []map[string]any{{
		"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": args},
	}}}}}})
	usage, _ := json.Marshal(map[string]any{"choices": []any{}, "usage": map[string]any{"prompt_tokens": 100, "completion_tokens": completion}})
	return fmt.Sprintf("data: %s\n\ndata: %s\n\ndata: [DONE]\n\n", call, usage)
}

// Once, after the planner has generated the budget, and naming the blocker comes first.
func TestPlanTokenNudgeAsksForANewAngle(t *testing.T) {
	a, s := newTestAgent(t)
	withTools(a, fakeTool("read_file", func(string) (string, bool) { return "hit", false }))
	var resp []string
	for i := 0; i < 6; i++ {
		resp = append(resp, sseToolCallUsing(fmt.Sprintf("c%d", i), "read_file", fmt.Sprintf(`{"path":"f%d.rs"}`, i), 2500))
	}
	resp = append(resp, sseText("done"))
	mock := newMockLLM(t, resp...)
	defer mock.Close()
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "test-model"}}}

	if _, err := a.runToolLoopSeeded(context.Background(), s.ID, a.connFor("thinking"),
		[]llmMessage{{Role: "user", Content: "plan"}}, phasePolicy{}, "plan", false, 0); err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	nudges := 0
	for i := 0; i < mock.callCount(); i++ {
		msgs, _ := mock.request(i)["messages"].([]any)
		last, _ := msgs[len(msgs)-1].(map[string]any)
		if c, _ := last["content"].(string); last["role"] == "user" && strings.Contains(c, "what is blocking you") {
			nudges++
			// 2500 per call: 7500 after the third, so before the fourth request.
			if i != 3 {
				t.Errorf("nudge arrived before call %d, want before call 3", i)
			}
		}
	}
	if nudges != 1 {
		t.Errorf("nudges = %d, want exactly 1", nudges)
	}
}

// An answer alone is checked once: "same" keeps it, subtasks replace it.
func TestPlanAnswerIsCheckedOnce(t *testing.T) {
	answer := `{"answer":"Planned a subtask that runs ./slow.sh and reports the number."}`
	t.Run("kept", func(t *testing.T) {
		a, s, mock := planPhaseAgent(t, sseToolCall("p1", submitPlanToolName, `{"answer":"Format(5) returns 0.05 €."}`),
			sseToolCall("p2", submitPlanToolName, `{"answer":"same"}`))
		plan, err := a.runPlanPhase(context.Background(), s.ID, "")
		if err != nil || plan == nil || plan.answer != "Format(5) returns 0.05 €." || mock.callCount() != 2 {
			t.Fatalf("plan %+v, err %v, calls %d: want the first answer kept after one check", plan, err, mock.callCount())
		}
		if last := fmt.Sprint(mock.request(1)["messages"]); !strings.Contains(last, "no `subtasks`, so nothing will run") {
			t.Errorf("the check did not reach the planner: %s", last)
		}
	})
	t.Run("replaced", func(t *testing.T) {
		a, s, _ := planPhaseAgent(t, sseToolCall("p1", submitPlanToolName, answer),
			sseToolCall("p2", submitPlanToolName, `{"subtasks":[{"description":"run ./slow.sh","verify":["it printed a number"]}]}`))
		plan, err := a.runPlanPhase(context.Background(), s.ID, "")
		if err != nil || plan == nil || len(plan.Subtasks) != 1 {
			t.Fatalf("plan %+v, err %v: want the subtasks the check produced", plan, err)
		}
	})
}
