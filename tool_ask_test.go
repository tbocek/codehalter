package main

import (
	"context"
	"strings"
	"testing"
)

// TestImproveAskCap pins the code-level /improve cap end-to-end through
// executeTool: while an /improve turn is armed, ask_user "Apply/Skip" prompts
// are allowed for the top improveAskCap changes and blocked with a steering
// note afterward, while the Yes/No submit prompt is exempt. Outside an armed
// turn there is no cap. Autopilot auto-answers so no editor conn is needed.
func TestImproveAskCap(t *testing.T) {
	a, s := newTestAgent(t)
	a.mode = "Autopilot" // auto-answer Apply/Skip without an editor conn

	ask := func(yes, no string) string {
		var tc toolCall
		tc.Function.Name = "ask_user"
		tc.Function.Arguments = `{"question":"q","yes_label":"` + yes + `","no_label":"` + no + `"}`
		out, _ := a.executeTool(context.Background(), s.ID, tc)
		return out
	}
	const capNote = "improve cap"

	// Not armed → no cap; every Apply/Skip is answered normally.
	for i := 0; i < improveAskCap+2; i++ {
		if out := ask("Apply", "Skip"); strings.Contains(out, capNote) {
			t.Fatalf("unarmed turn must not cap, got: %s", out)
		}
	}

	// Armed → the first improveAskCap Apply/Skip prompts pass, the next is blocked.
	s.improving.Store(true)
	s.improveAsks = 0
	for i := 0; i < improveAskCap; i++ {
		if out := ask("Apply", "Skip"); strings.Contains(out, capNote) {
			t.Fatalf("improvement %d should be allowed, got: %s", i+1, out)
		}
	}
	if out := ask("Apply", "Skip"); !strings.Contains(out, capNote) {
		t.Errorf("a prompt past the cap should be blocked, got: %s", out)
	}
	// The Yes/No submit prompt is exempt even after the cap is reached.
	if out := ask("Yes", "No"); strings.Contains(out, capNote) {
		t.Errorf("submit prompt should be exempt from the cap, got: %s", out)
	}

	// Re-arming a fresh turn resets the counter.
	s.improving.Store(true)
	s.improveAsks = 0
	if out := ask("Apply", "Skip"); strings.Contains(out, capNote) {
		t.Errorf("re-armed turn should start fresh, got: %s", out)
	}
}
