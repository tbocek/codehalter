package main

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The in-round check has a fresh call review the tests the round wrote, shown the diff (a new
// file too) and the item's text: what it finds holds the round back for a turn, an empty review
// lets it through, and the final judgement asks nothing.
func TestSpecRoundTestsAreReviewed(t *testing.T) {
	a, sess := newTestAgent(t)
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	writeTree(t, sess.Cwd, map[string]string{".gitignore": ".codehalter/\n", "rust/src/lib.rs": "// v1\n", "rust/tests/base.rs": "#[test]\nfn base_builds() {}\n"})
	gitInit(t, sess.Cwd)
	found := `{"findings": [{"test": "f0_1_switches", "problem": "3: sleeps 200 ms for the tab to switch", "fix": "wait for the tab's state through the shared wait helper"}]}`
	m := newMockLLM(t, sseText("Reviewing.\n"+found), sseText(`{"findings": []}`))
	defer m.Close()
	a.settings = Settings{LLM: []LLMConnection{*m.conn("execute")}}
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: "true", LintCmd: "off", Items: map[string]specLedger{}}
	r := &specRun{a: a, sid: sess.ID, sess: sess, cfg: cfg, idx: idx, outAbs: filepath.Join(sess.Cwd, "rust"), reasons: map[string]string{}}
	writeTree(t, sess.Cwd, map[string]string{"rust/tests/switch.rs": "#[test]\nfn f0_1_switches() { std::thread::sleep(std::time::Duration::from_millis(200)); }\n"})
	look := func(runTests bool) specRoundResult {
		t.Helper()
		res := specRoundResult{Mode: specModeItem, Committed: true}
		if _, _, err := r.inspect(t.Context(), specWork{Item: "F0.1", Mode: specModeItem}, nil, time.Now(), runTests, &res); err != nil {
			t.Fatal(err)
		}
		return res
	}

	res := look(false)
	if len(res.TestReview) != 1 || !strings.Contains(res.TestReview[0], "`f0_1_switches`: 3: sleeps 200 ms") {
		t.Fatalf("review = %q, want the finding", res.TestReview)
	}
	prompt, _ := m.request(0)["messages"].([]any)[0].(map[string]any)["content"].(string)
	if !strings.Contains(prompt, "+fn f0_1_switches() { std::thread::sleep") || !strings.Contains(prompt, "S1 Switch.") || !strings.Contains(prompt, "rust/tests/base.rs") {
		t.Errorf("the review was not shown the new test, the item's text and the other test files:\n%s", prompt)
	}
	if done, _, why, _ := specDecide(&specConfig{}, "F0.1", res, ""); done || !strings.Contains(why, "independent review of the tests") {
		t.Errorf("done=%v why=%q, want the round held back by the review", done, why)
	}

	if res := look(false); len(res.TestReview) != 0 {
		t.Errorf("an empty review held the round back: %q", res.TestReview)
	}
	if look(true); m.callCount() != 2 {
		t.Errorf("%d review calls, want none from the final judgement", m.callCount())
	}
}
