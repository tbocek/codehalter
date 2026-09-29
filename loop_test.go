package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func meterCall(name, args string) toolCall {
	tc := toolCall{ID: "tc1"}
	tc.Function.Name = name
	tc.Function.Arguments = args
	return tc
}

// stop() joins the ticker goroutine promptly, also after a ctx cancel.
func TestStartToolMeter(t *testing.T) {
	a, s := newTestAgent(t)

	stop := a.startToolMeter(context.Background(), s.ID, meterCall("web_search", `{"query":"goldmark tables"}`))
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
	stop2 := a.startToolMeter(ctx, s.ID, meterCall("run_command", `{"command":"sleep 1"}`))
	cancel()
	done2 := make(chan struct{})
	go func() { stop2(); close(done2) }()
	select {
	case <-done2:
	case <-time.After(2 * time.Second):
		t.Fatal("stop() after ctx cancel did not return")
	}
}

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

// The contract between PLAN.md and runExecutePhase.
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

	// report_only round-trips so renderPlan can label it "Findings:".
	rawReport := `{"clear": true, "report_only": true, "subtasks": [{"description": "summarise X"}]}`
	var p2 planResult
	if err := json.Unmarshal([]byte(rawReport), &p2); err != nil {
		t.Fatalf("unmarshal report_only: %v", err)
	}
	if !p2.ReportOnly {
		t.Errorf("expected report_only=true")
	}
}

// Lowercase, punctuation-stripped, order-independent, so rewordings match.
func TestIssueBagTokenisation(t *testing.T) {
	a := issueBag([]string{"Missing import!", "Syntax error."})
	b := issueBag([]string{"syntax  ERROR", "missing\timport"})
	if !slices.Equal(sortedKeys(a), sortedKeys(b)) {
		t.Errorf("expected equivalent bags, got %v vs %v", sortedKeys(a), sortedKeys(b))
	}

	// No empty tokens from adjacent separators.
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

// A reworded near-duplicate scores above the threshold, unrelated failures below.
func TestJaccardSimilarity(t *testing.T) {
	// Two empty bags are treated as identical (degenerate but well-defined).
	if got := jaccard(map[string]bool{}, map[string]bool{}); got != 1 {
		t.Errorf("empty/empty: got %v, want 1", got)
	}

	// |∩|=2, |∪|=3: 0.67, above the threshold.
	a := issueBag([]string{"missing import"})
	b := issueBag([]string{"import is missing"})
	if s := jaccard(a, b); s < failureSimilarityThreshold {
		t.Errorf("reworded duplicate: got %v, want >= %v", s, failureSimilarityThreshold)
	}

	c := issueBag([]string{"missing import in foo.go"})
	d := issueBag([]string{"unused variable x"})
	if s := jaccard(c, d); s >= failureSimilarityThreshold {
		t.Errorf("disjoint issues: got %v, want < %v", s, failureSimilarityThreshold)
	}

	if jaccard(a, b) != jaccard(b, a) {
		t.Errorf("expected jaccard to be symmetric")
	}
}

// A cap hit retries with a be-concise nudge; the discarded partial never reaches the result.
func TestCapHitLadder(t *testing.T) {
	mock := newMockLLM(t,
		sseTruncatedContent("way too long", 1000, defaultMaxTokens),
		sseText("done"),
	)
	defer mock.Close()
	a, s := newTestAgent(t)
	a.mainSlotTokens.Store(85248) // ample n_ctx room: these are cap hits, not the ceiling

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", false, 0)
	if err != nil {
		t.Fatalf("ladder should recover: %v", err)
	}
	if got := mock.callCount(); got != 2 {
		t.Fatalf("callCount = %d, want 2 (cap, nudged success)", got)
	}
	if res.Text != "done" {
		t.Errorf("res.Text = %q, want the successful reply only (partials discarded)", res.Text)
	}

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
}

// A reply that needs more than the cap (a large file) survives the nudge; one
// retry on a doubled cap gets it through.
func TestCapHitLadderDoubles(t *testing.T) {
	mock := newMockLLM(t,
		sseTruncatedContent("big file", 1000, defaultMaxTokens),
		sseTruncatedContent("big file", 1000, defaultMaxTokens),
		sseText("done"),
	)
	defer mock.Close()
	a, s := newTestAgent(t)
	a.mainSlotTokens.Store(85248)

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", false, 0)
	if err != nil {
		t.Fatalf("the doubled cap should recover: %v", err)
	}
	if got := mock.callCount(); got != 3 || res.Text != "done" {
		t.Fatalf("callCount = %d, text = %q; want 3 calls ending in the doubled retry's reply", got, res.Text)
	}
	if mt, _ := mock.request(2)["max_tokens"].(float64); int(mt) != 2*defaultMaxTokens {
		t.Errorf("third request max_tokens = %v, want %d", mock.request(2)["max_tokens"], 2*defaultMaxTokens)
	}
	if mt, _ := mock.request(1)["max_tokens"].(float64); int(mt) != defaultMaxTokens {
		t.Errorf("the nudged request max_tokens = %v, want the unchanged %d", mock.request(1)["max_tokens"], defaultMaxTokens)
	}
}

func TestCapHitLadderExhausted(t *testing.T) {
	mock := newMockLLM(t,
		sseTruncatedContent("too long", 1000, defaultMaxTokens),
		sseTruncatedContent("too long", 1000, defaultMaxTokens),
		sseTruncatedContent("too long", 1000, 2*defaultMaxTokens),
	)
	defer mock.Close()
	a, s := newTestAgent(t)
	a.mainSlotTokens.Store(85248)

	_, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", false, 0)
	var ce *capHitError
	if !errors.As(err, &ce) {
		t.Fatalf("exhausted ladder should surface the cap error, got: %v", err)
	}
	if got := mock.callCount(); got != 3 {
		t.Errorf("callCount = %d, want 3 (cap, nudge, doubled cap, then give up)", got)
	}
}

// Output differing only in noise (a timestamped failing build) counts as reproduced.
func TestStuckLadderFuzzyOutput(t *testing.T) {
	const toolName = "noisy_probe_test_tool"
	attempt := 0
	var testTools []Tool
	testTools = append(testTools, Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        toolName,
			"description": "test-only failing probe with noisy output",
			"parameters":  map[string]any{"type": "object"},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		attempt++
		// Jaccard ≈ 0.93 between outputs: above stuckOutputSimilarity, a new hash every time.
		return fmt.Sprintf("build FAILED: cannot load package example.com/foo/bar: import cycle not allowed in dependency graph involving widget factory manager controller service repository handler adapter transport codec parser lexer scanner tokenizer emitter renderer scheduler dispatcher broker queue worker pool cache index shard replica leader follower quorum consensus journal snapshot compaction segment (elapsed %dms, attempt %d)", 1200+attempt*7, attempt), true
	}})

	call := sseToolCall("c1", toolName, `{}`)
	responses := make([]string, 12)
	for i := range responses {
		responses[i] = call
	}
	mock := newMockLLM(t, responses...)
	defer mock.Close()
	a, s := newTestAgent(t)
	a.tools.add(testTools...)
	a.mainSlotTokens.Store(85248)

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", false, 0)
	if err != nil {
		t.Fatalf("stuck bail is a graceful exit, got error: %v", err)
	}
	if res.Terminal != "" {
		t.Errorf("bail must not report a terminal exit")
	}
	// Round 1 is new output; rounds 2-6 are stuck, and the ladder bails.
	if got := mock.callCount(); got != 1+stuckBailRounds {
		t.Errorf("callCount = %d, want %d (bail at stuckBailRounds via fuzzy match)", got, 1+stuckBailRounds)
	}
}

// A step ended for repeating one call hands it on: the turn's next step gets
// that call's output back instead of a run, until it writes something. On one
// /spec item five replanned steps each opened with the grep that ended the one before.
func TestStuckCallIsNotRunAgainInTheSameTurn(t *testing.T) {
	const toolName = "same_probe_test_tool"
	runs := 0
	a, s := newTestAgent(t)
	a.tools.add()
	a.tools.add(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        toolName,
			"description": "test-only probe with the same answer every time",
			"parameters":  map[string]any{"type": "object"},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		runs++
		return "13:![The Cut page](img/05-cut.png)", false
	}})
	a.mainSlotTokens.Store(85248)
	call := sseToolCall("c1", toolName, `{}`)
	var step1 []string
	for range 1 + stuckBailRounds {
		step1 = append(step1, call)
	}
	write, _ := json.Marshal(map[string]string{"path": "note.txt", "content": "changed\n"})
	mock := newMockLLM(t, append(step1,
		call, // step 2 opens with the same call: not run
		sseToolCall("w1", "write_file", string(write)),
		call, // after a write it is a re-check: runs
		sseToolCall("r1", respondToolName, `{"message":"done"}`),
	)...)
	defer mock.Close()
	step := func() toolLoopResult {
		t.Helper()
		res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
			[]llmMessage{{Role: "user", Content: "look at the picture"}}, phasePolicy{terminals: map[string]bool{respondToolName: true}}, "execute", false, 0)
		if err != nil {
			t.Fatal(err)
		}
		return res
	}
	if res := step(); res.Terminal != "" || runs != 1+stuckBailRounds {
		t.Fatalf("step 1: terminal=%q runs=%d, want ended after %d runs", res.Terminal, runs, 1+stuckBailRounds)
	}
	if res := step(); res.Terminal != respondToolName {
		t.Fatalf("step 2 did not finish: %+v", res)
	}
	if runs != 2+stuckBailRounds {
		t.Errorf("runs = %d, want %d: refused once, run again after the write", runs, 2+stuckBailRounds)
	}
	if got := mock.request(len(step1) + 1); !strings.Contains(fmt.Sprint(got["messages"]), "not run: an earlier attempt at this task ran exactly this call") ||
		!strings.Contains(fmt.Sprint(got["messages"]), "img/05-cut.png") {
		t.Errorf("the refused call did not carry the note and the earlier output:\n%v", got["messages"])
	}
}

func TestToolMeterShowsTheArgument(t *testing.T) {
	h := newTerminalHarness(t)
	a, s := h.agent, h.sess
	s.phaseMu.Lock()
	s.phaseActive, s.phaseCurrent = true, 0
	s.phaseMu.Unlock()

	long := strings.Repeat("x", 200)
	for _, tc := range []struct {
		name, args, want string
	}{
		{"run_command", `{"command":"go build ./..."}`, "run_command go build ./..."},
		{"read_file", `{"path":"loop.go","limit":40}`, "read_file loop.go"},
		{"web_search", `{"query":"line one\nline two"}`, "web_search line one line two"}, // no row-breaking newline
		{"run_command", `{"command":"` + long + `"}`, "run_command " + strings.Repeat("x", toolMeterArgRunes)},
		{"read_file", `not json at all`, "read_file"},
	} {
		stop := a.startToolMeter(context.Background(), s.ID, meterCall(tc.name, tc.args))
		stop()
		if !h.waitFor(func() bool { return len(h.updatesOfKind("plan")) > 0 }) {
			t.Fatalf("%s: no plan update", tc.name)
		}
		var got string
		for _, u := range h.updatesOfKind("plan") {
			entries, _ := u["entries"].([]any)
			for _, e := range entries {
				em, _ := e.(map[string]any)
				if em["status"] == "in_progress" {
					got, _ = em["content"].(string)
				}
			}
		}
		if want := " (running " + tc.want + "…)"; !strings.HasSuffix(got, want) {
			t.Errorf("%s(%s) row = %q, want it to end in %q", tc.name, tc.args, got, want)
		}
		h.mu.Lock()
		h.updates = nil
		h.mu.Unlock()
	}
}

// A wire-only corrective would drop out of the MIDDLE of history on the next rebuild.
func TestAddCorrectiveSurvivesRebuild(t *testing.T) {
	a, s := newTestAgent(t)
	s.AddUser("do the thing")
	s.AddAssistant("I'll get right on it")

	wire := a.addCorrective(s.ID, a.buildLLMContext(s), "Call a tool. Do not reply in prose.")

	rebuilt := a.buildLLMContext(s)
	if len(rebuilt) != len(wire) {
		t.Fatalf("rebuild has %d messages, wire had %d — the corrective did not persist", len(rebuilt), len(wire))
	}
	for i := range wire {
		if rebuilt[i].Role != wire[i].Role || fmt.Sprint(rebuilt[i].Content) != fmt.Sprint(wire[i].Content) {
			t.Errorf("message %d diverges\n wire: %s %q\nrebuilt: %s %q",
				i, wire[i].Role, wire[i].Content, rebuilt[i].Role, rebuilt[i].Content)
		}
	}
}

func TestPlanRecoversFromMalformedSubmitPlanArguments(t *testing.T) {
	broken := sseToolCall("c1", submitPlanToolName, `{"clear":true,"subtasks":[{"description":"do the thing"`)
	fixed := sseToolCall("c2", submitPlanToolName,
		`{"clear":true,"subtasks":[{"description":"do the thing","verify":["go build ./..."]}],"report_only":false}`)
	a, s, _ := planPhaseAgent(t, broken, fixed)

	plan, err := a.runPlanPhase(context.Background(), s.ID, "")
	if err != nil {
		t.Fatalf("malformed arguments should recover via the corrective retry, got: %v", err)
	}
	if plan == nil || len(plan.Subtasks) != 1 || plan.Subtasks[0].Description != "do the thing" {
		t.Fatalf("plan = %+v, want the retry's single subtask", plan)
	}
}

// The shapes Qwen3.8 sent six rounds in a row: the plan inside a `plan` key, as
// an object or as JSON text holding a list. They are the plan, with no retry.
func TestPlanUnwrapsAWrappedPlan(t *testing.T) {
	for _, args := range []string{
		`{"plan": {"clear": true, "report_only": false, "subtasks": [{"description": "fix the compile break"}]}}`,
		`{"plan": "[{\"clear\": true, \"report_only\": false, \"subtasks\": [{\"description\": \"fix the compile break\"}]}]"}`,
	} {
		a, s, mock := planPhaseAgent(t, sseToolCall("p1", submitPlanToolName, args))
		plan, err := a.runPlanPhase(context.Background(), s.ID, "")
		if err != nil || plan == nil || len(plan.Subtasks) != 1 || plan.Subtasks[0].Description != "fix the compile break" {
			t.Errorf("%s: plan=%+v err=%v, want the wrapped subtask", args, plan, err)
		}
		if mock.callCount() != 1 {
			t.Errorf("%s: %d calls, want 1 (no retry)", args, mock.callCount())
		}
	}
}

// A plan with nothing in it gets the one retry even without clear set.
func TestPlanRetriesAnEmptyPlan(t *testing.T) {
	a, s, mock := planPhaseAgent(t,
		sseToolCall("p1", submitPlanToolName, `{"clear": false, "report_only": false}`),
		sseToolCall("p2", submitPlanToolName, `{"clear": true, "subtasks": [{"description": "do it"}]}`),
	)
	plan, err := a.runPlanPhase(context.Background(), s.ID, "")
	if err != nil || plan == nil || len(plan.Subtasks) != 1 {
		t.Fatalf("plan=%+v err=%v, want the retry's subtask", plan, err)
	}
	if mock.callCount() != 2 || !strings.Contains(fmt.Sprint(mock.request(1)["messages"]), "not inside another key") {
		t.Errorf("%d calls, want 2 with the corrective", mock.callCount())
	}
}

func planPhaseAgent(t *testing.T, responses ...string) (*agent, *Session, *mockLLM) {
	t.Helper()
	mock := newMockLLM(t, responses...)
	t.Cleanup(mock.Close)
	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
	return a, s, mock
}

// Prose beside subtasks is a preamble and costs no corrective round trip.
func TestPlanPreambleIsNotAnAnswer(t *testing.T) {
	args := `{"clear":true,"report_only":false,"subtasks":[{"description":"delete out/test and rebuild","verify":["go build ./..."]}]}`
	a, s, mock := planPhaseAgent(t, sseContentThenToolCall(
		"The request is clear: delete the out/test build output and recompile the site.",
		"p1", submitPlanToolName, args))

	plan, err := a.runPlanPhase(context.Background(), s.ID, "")
	if err != nil {
		t.Fatalf("runPlanPhase: %v", err)
	}
	if got := mock.callCount(); got != 1 {
		t.Errorf("a preamble cost %d LLM calls, want 1 (no corrective round trip)", got)
	}
	if plan == nil || len(plan.Subtasks) != 1 || plan.Subtasks[0].Description != "delete out/test and rebuild" {
		t.Fatalf("plan = %+v, want the submitted subtask intact", plan)
	}
}

// Via the closed-think prefill on the wire; the stored turn is the prompt and nothing else.
func TestExecutePhaseTurnsReasoningOff(t *testing.T) {
	mock := newMockLLM(t, sseToolCall("r1", respondToolName, `{"message":"done"}`))
	defer mock.Close()

	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}

	out := a.runExecutePhase(context.Background(), s.ID, subtask{Description: "do the thing"}, 0, 1)
	if out.Reason != "" {
		t.Fatalf("executor error: %s", out.Reason)
	}

	body := mock.request(0)
	if body["continue_final_message"] != true || body["add_generation_prompt"] != false {
		t.Errorf("execute call did not ask the server to continue a prefill: "+
			"continue_final_message=%v add_generation_prompt=%v",
			body["continue_final_message"], body["add_generation_prompt"])
	}
	if _, set := body["chat_template_kwargs"]; set {
		t.Errorf("execute re-rendered the prompt instead of appending: %v", body["chat_template_kwargs"])
	}
	// A grammar, not a re-render, so the prefix is untouched.
	if body["tool_choice"] != "required" {
		t.Errorf("execute call tool_choice = %v, want required", body["tool_choice"])
	}
	sent, _ := body["messages"].([]any)
	if len(sent) == 0 {
		t.Fatal("no messages on the wire")
	}
	last, _ := sent[len(sent)-1].(map[string]any)
	if last["role"] != "assistant" || last["content"] != noThinkPrefillContent {
		t.Errorf("last wire message = %v, want the assistant prefill %q", last, noThinkPrefillContent)
	}

	var stored []string
	for _, m := range s.Messages {
		if m.Role == "user" {
			stored = append(stored, m.Content)
		}
	}
	if len(stored) != 1 {
		t.Fatalf("want exactly the subtask prompt stored, got %d user turns: %q", len(stored), stored)
	}
	if !strings.HasSuffix(stored[0], "do the thing\n") {
		t.Errorf("stored subtask prompt carries a wire-only suffix: %q", stored[0])
	}
}

func TestToolLoopRecordsToolUses(t *testing.T) {
	var testTools []Tool
	const testToolName = "test_echo_tool_9d7f"
	testTools = append(testTools, Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        testToolName,
				"description": "echoes the input.msg field (test only)",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"msg": map[string]any{"type": "string"}},
				},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			args := parseArgs(rawArgs)
			return "echo: " + args.str("msg"), false
		},
	})

	mock := newMockLLM(t,
		sseToolCall("call_1", testToolName, `{"msg":"hello"}`),
		sseText("All done."),
	)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("please echo hello")
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	a := &agent{
		sessions: map[string]*Session{s.ID: s},
	}
	withTools(a, testTools...)

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "please echo hello"}}, phasePolicy{}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}

	if res.Text != "All done." {
		t.Errorf("final text: got %q, want %q", res.Text, "All done.")
	}
	if len(res.ToolUses) != 1 {
		t.Fatalf("ToolUses: got %d, want 1", len(res.ToolUses))
	}
	if res.ToolUses[0].Output != "echo: hello" {
		t.Errorf("ToolUse output: got %q", res.ToolUses[0].Output)
	}

	// Each assistant turn is stored verbatim, never merged: user, tool turn, text turn.
	if got := len(s.Messages); got != 3 {
		t.Fatalf("session messages: got %d, want 3", got)
	}
	if s.Messages[1].Role != "assistant" || len(s.Messages[1].ToolUses) != 1 {
		t.Errorf("msg[1]: want assistant with 1 tool use, got role=%q tools=%d", s.Messages[1].Role, len(s.Messages[1].ToolUses))
	}
	if s.Messages[2].Role != "assistant" || s.Messages[2].Content != "All done." {
		t.Errorf("msg[2]: want assistant content %q, got role=%q content=%q", "All done.", s.Messages[2].Role, s.Messages[2].Content)
	}

	loaded, err := loadSession(dir, s.ID)
	if err != nil {
		t.Fatalf("loadSession: %v", err)
	}
	if got := len(loaded.Messages); got != 3 {
		t.Fatalf("persisted messages: got %d, want 3", got)
	}
	if got := len(loaded.Messages[1].ToolUses); got != 1 {
		t.Fatalf("persisted tool uses: got %d, want 1", got)
	}
	if loaded.Messages[1].ToolUses[0].Output != "echo: hello" {
		t.Errorf("persisted output mismatch")
	}

	if mock.callCount() != 2 {
		t.Errorf("LLM call count: got %d, want 2", mock.callCount())
	}
}

// respond exits on the same iteration, with no second round trip.
func TestToolLoopRespondExits(t *testing.T) {
	mock := newMockLLM(t,
		sseToolCall("call_1", respondToolName, `{"message":"final answer"}`),
	)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("answer me")
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	a := &agent{sessions: map[string]*Session{s.ID: s}}

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "answer me"}}, phasePolicy{terminals: map[string]bool{respondToolName: true}}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}

	if res.Text != "final answer" {
		t.Errorf("res.Text: got %q, want %q", res.Text, "final answer")
	}
	if mock.callCount() != 1 {
		t.Errorf("LLM call count: got %d, want 1 (respond should exit on the same iteration)", mock.callCount())
	}
	if len(res.ToolUses) != 1 || res.ToolUses[0].Name != respondToolName {
		t.Errorf("ToolUses: want one respond call, got %+v", res.ToolUses)
	}
}

// A denied tool stays in the array; dispatch rejects it and the loop continues.
func TestRunToolLoopDenyGate(t *testing.T) {
	var testTools []Tool
	var execs int
	testTools = append(testTools, Tool{
		Def: map[string]any{"type": "function", "function": map[string]any{
			"name": "mutate", "description": "x", "parameters": map[string]any{"type": "object"}}},
		Execute: func(ctx context.Context, a *agent, sid string, raw string) (string, bool) {
			execs++
			return "did it", false
		},
	})
	mock := newMockLLM(t,
		sseToolCall("c1", "mutate", `{}`),
		sseText("ok"),
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	withTools(a, testTools...)
	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}},
		phasePolicy{deny: map[string]bool{"mutate": true}}, "plan", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if execs != 0 {
		t.Errorf("denied tool executed %d times, want 0", execs)
	}
	if len(res.ToolUses) != 1 || !res.ToolUses[0].Failed {
		t.Fatalf("denied call should be a single failed tooluse, got %+v", res.ToolUses)
	}
	if !strings.Contains(res.ToolUses[0].Output, "not available") {
		t.Errorf("deny message missing: %q", res.ToolUses[0].Output)
	}
}

func TestRunToolLoopMultiTerminalUpsert(t *testing.T) {
	planArgs := `{"clear":true,"subtasks":[{"description":"do y"}]}`
	mock := newMockLLM(t, sseToolCall("c1", submitPlanToolName, planArgs))
	defer mock.Close()

	a, s := newTestAgent(t)
	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}},
		phasePolicy{terminals: map[string]bool{respondToolName: true, submitPlanToolName: true}}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if res.Terminal != submitPlanToolName {
		t.Errorf("Terminal: got %q, want %q", res.Terminal, submitPlanToolName)
	}
	if !strings.Contains(res.Text, "do y") {
		t.Errorf("res.Text should carry the upsert plan JSON: %q", res.Text)
	}
}

func TestToolLoopNoTerminalKeepsTextExit(t *testing.T) {
	mock := newMockLLM(t,
		sseText(`{"clear": true, "steps": ["do x"]}`),
	)
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	s.AddUser("plan something")
	if err := s.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	a := &agent{sessions: map[string]*Session{s.ID: s}}

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}},
		phasePolicy{}, "document", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if mock.callCount() != 1 {
		t.Errorf("LLM call count: got %d, want 1 (no nudge when no terminal tool is exposed)", mock.callCount())
	}
	if !strings.Contains(res.Text, "do x") {
		t.Errorf("res.Text missing text content: got %q", res.Text)
	}
}

// The plan lands in res.Text, the prose beside it in res.Content.
func TestPlanSubmitPlanSeparatesAnswer(t *testing.T) {
	planArgs := `{"clear":true,"subtasks":[],"report_only":true}`
	mock := newMockLLM(t, sseContentThenToolCall("Active servers: gopls.", "tc1", submitPlanToolName, planArgs))
	defer mock.Close()

	dir := t.TempDir()
	s, err := newSession(dir)
	if err != nil {
		t.Fatalf("newSession: %v", err)
	}
	a := &agent{sessions: map[string]*Session{s.ID: s}}

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("thinking"),
		[]llmMessage{{Role: "user", Content: "what servers?"}},
		phasePolicy{terminals: map[string]bool{submitPlanToolName: true, respondToolName: true}}, "plan", false, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if res.Terminal != submitPlanToolName {
		t.Errorf("Terminal: got %q, want submit_plan (the plan terminal)", res.Terminal)
	}
	if !strings.Contains(res.Text, `"report_only":true`) {
		t.Errorf("res.Text should carry the submit_plan args (the plan JSON): got %q", res.Text)
	}
	if !strings.Contains(res.Content, "Active servers: gopls.") {
		t.Errorf("res.Content should carry the prose answer separately: got %q", res.Content)
	}
}

// Identical calls each execute: a cached read after a mutating command would be stale.
func TestToolLoopNoDedup(t *testing.T) {
	var testTools []Tool
	const readName = "test_read"
	const writeName = "test_write"
	var reads, writes int
	testTools = append(testTools, Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": readName, "description": "read",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			reads++
			return "read-ok", false
		},
	})
	testTools = append(testTools, Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": writeName, "description": "write",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			writes++
			return "wrote", false
		},
	})

	mock := newMockLLM(t,
		sseToolCall("c1", readName, `{}`),
		sseToolCall("c2", readName, `{}`),
		sseToolCall("c3", writeName, `{}`),
		sseToolCall("c4", writeName, `{}`),
		sseText("done"),
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	withTools(a, testTools...)
	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if reads != 2 {
		t.Errorf("read tool executed %d times, want 2 (no dedup)", reads)
	}
	if writes != 2 {
		t.Errorf("write tool executed %d times, want 2 (no dedup)", writes)
	}
	if len(res.ToolUses) != 4 {
		t.Fatalf("ToolUses: got %d, want 4", len(res.ToolUses))
	}
	for i, tu := range res.ToolUses {
		if strings.HasPrefix(tu.Output, "[deduped:") {
			t.Errorf("ToolUses[%d] should not be deduped, got %q", i, tu.Output)
		}
	}
}

// A nudge each stuck round, a swap to the thinking sampler (keeping a forced
// tool_choice) at stuckEscalateRounds, a graceful bail at stuckBailRounds.
// An execute loop that only reads is nudged and moved to the thinking sampler at
// readStreakEscalate, and ends at readStreakBail with a reason the replan sees.
func TestToolLoopEndsAReadOnlyStreak(t *testing.T) {
	a, s := newTestAgent(t)
	withTools(a, Tool{
		Def: map[string]any{"type": "function", "function": map[string]any{
			"name": "read_file", "description": "read", "parameters": map[string]any{"type": "object"}}},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			return "the text of " + parseArgs(rawArgs).str("path"), false
		},
	})
	var resp []string
	for i := range readStreakBail + 5 {
		resp = append(resp, sseToolCall(fmt.Sprintf("c%d", i), "read_file", fmt.Sprintf(`{"path":"file%c%c.rs"}`, 'a'+i/26, 'a'+i%26)))
	}
	mock := newMockLLM(t, resp...)
	defer mock.Close()
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m",
		ParamsExecute: map[string]any{"temperature": 0.3}, ParamsThinking: map[string]any{"temperature": 1.0}}}}

	_, err := a.runToolLoopSeeded(context.Background(), s.ID, a.connFor("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{terminals: map[string]bool{respondToolName: true}}, "execute", true, 0)
	if err == nil || !strings.Contains(err.Error(), "in a row without an edit") {
		t.Fatalf("err = %v, want the read-streak bail", err)
	}
	if got := mock.callCount(); got != readStreakBail {
		t.Errorf("model calls = %d, want %d", got, readStreakBail)
	}
	next := mock.request(readStreakEscalate)
	if fmt.Sprint(next["messages"]) == "" || !strings.Contains(fmt.Sprint(next["messages"]), "You have read for 20 calls") {
		t.Error("the call after the escalate threshold carries no nudge")
	}
	if next["temperature"] != 1.0 || mock.request(readStreakEscalate - 1)["temperature"] != 0.3 {
		t.Errorf("temperature before/after the threshold = %v/%v, want 0.3/1.0 (the thinking sampler)",
			mock.request(readStreakEscalate - 1)["temperature"], next["temperature"])
	}
}

func TestToolLoopRepetitionLadder(t *testing.T) {
	var testTools []Tool
	const toolName = "test_probe_a3f"
	var execs int
	testTools = append(testTools, Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": toolName, "description": "probe",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			execs++
			return "same result", false
		},
	})

	var resp []string
	for i := 0; i < 8; i++ {
		resp = append(resp, sseToolCall(fmt.Sprintf("c%d", i), toolName, `{}`))
	}
	mock := newMockLLM(t, resp...)
	defer mock.Close()

	a, s := newTestAgent(t)
	withTools(a, testTools...)
	a.settings = Settings{
		LLM: []LLMConnection{{
			Server:         mock.ts.URL,
			Model:          "test-model",
			ParamsExecute:  map[string]any{"temperature": 0.3},
			ParamsThinking: map[string]any{"temperature": 1.0},
		}},
	}
	conn := a.connFor("execute")
	if conn == nil {
		t.Fatalf("connFor(execute) returned nil")
	}
	conn = conn.withBody("tool_choice", "required")

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: want graceful nil error, got %v", err)
	}
	if res.Terminal != "" {
		t.Errorf("Terminal: got %q, want none (the loop bailed, never called respond)", res.Terminal)
	}
	// Round 1 is productive; the 5th stuck round (round 6) bails.
	if mock.callCount() != 6 {
		t.Errorf("LLM calls: got %d, want 6 (bail at stuckBailRounds)", mock.callCount())
	}
	if execs != 6 {
		t.Errorf("tool execs: got %d, want 6", execs)
	}
	var found bool
	for i := 2; i < mock.callCount(); i++ {
		msgs, _ := mock.request(i)["messages"].([]any)
		for _, m := range msgs {
			mm, _ := m.(map[string]any)
			if mm["role"] == "user" {
				if c, _ := mm["content"].(string); strings.Contains(c, "makes no progress") {
					found = true
				}
			}
		}
	}
	if !found {
		t.Errorf("no repeat-corrective user message found in the stuck rounds")
	}
	// The swap fires at the END of round 4 (stuckRounds reaches 3).
	for i := 0; i < 4; i++ {
		if temp, _ := mock.request(i)["temperature"].(float64); temp != 0.3 {
			t.Errorf("request %d temperature: got %v, want 0.3 (pre-escalation)", i, temp)
		}
	}
	for i := 4; i < 6; i++ {
		if temp, _ := mock.request(i)["temperature"].(float64); temp != 1.0 {
			t.Errorf("request %d temperature: got %v, want 1.0 (post-escalation)", i, temp)
		}
	}
	for i := 0; i < 6; i++ {
		if tc := mock.request(i)["tool_choice"]; tc != "required" {
			t.Errorf("request %d tool_choice: got %v, want required on both sides of the swap", i, tc)
		}
	}
}

// A green re-run after an edit is a re-verify, never a repeat.
func TestRepetitionLadderExemptsSuccessfulRunCommand(t *testing.T) {
	var testTools []Tool
	var execs int
	testTools = append(testTools, Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "run_command", "description": "probe",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			execs++
			return "go build -o codehalter .", false // identical green output, success
		},
	}, Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "edit_file", "description": "probe",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			return "edited", false
		},
	})

	var resp []string
	for i := 0; i < 6; i++ {
		resp = append(resp, sseToolCall(fmt.Sprintf("e%d", i), "edit_file", fmt.Sprintf(`{"path":"a.go","old_text":"%d","new_text":"%d"}`, i, i+1)))
		resp = append(resp, sseToolCall(fmt.Sprintf("c%d", i), "run_command", `{"command":"just build"}`))
	}
	resp = append(resp, sseText("all green, done"))
	mock := newMockLLM(t, resp...)
	defer mock.Close()

	a, s := newTestAgent(t)
	withTools(a, testTools...)
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "test-model"}}}
	conn := a.connFor("execute")
	if conn == nil {
		t.Fatalf("connFor(execute) returned nil")
	}

	if _, err := a.runToolLoopSeeded(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", true, 0); err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if execs != 6 {
		t.Errorf("tool execs: got %d, want 6 (no early bail on successful run_command re-runs)", execs)
	}
	for i := 0; i < mock.callCount(); i++ {
		msgs, _ := mock.request(i)["messages"].([]any)
		for _, m := range msgs {
			mm, _ := m.(map[string]any)
			if c, _ := mm["content"].(string); mm["role"] == "user" && strings.Contains(c, "makes no progress") {
				t.Errorf("request %d carried a repeat-corrective for a successful run_command re-run", i)
			}
		}
	}
}

// Fan-out over distinct arguments returns new output each call, so no round is stuck.
func TestToolLoopDoesNotEscalateOnDistinctArgs(t *testing.T) {
	var testTools []Tool
	const toolName = "test_grep_q9z"
	var execs int
	testTools = append(testTools, Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": toolName, "description": "grep",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			execs++
			return "no match", false
		},
	})

	mock := newMockLLM(t,
		sseToolCall("c1", toolName, `{"q":"x1"}`),
		sseToolCall("c2", toolName, `{"q":"x2"}`),
		sseToolCall("c3", toolName, `{"q":"x3"}`),
		sseToolCall("c4", toolName, `{"q":"x4"}`),
		sseToolCall("c5", toolName, `{"q":"x5"}`),
		sseText("done."),
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	withTools(a, testTools...)
	a.settings = Settings{
		LLM: []LLMConnection{{
			Server:         mock.ts.URL,
			Model:          "test-model",
			ParamsExecute:  map[string]any{"temperature": 0.3},
			ParamsThinking: map[string]any{"temperature": 1.0},
		}},
	}

	conn := a.connFor("execute")
	if conn == nil {
		t.Fatalf("connFor(execute) returned nil")
	}
	_, err := a.runToolLoopSeeded(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}

	if execs != 5 {
		t.Errorf("tool execs: got %d, want 5", execs)
	}
	if mock.callCount() != 6 {
		t.Errorf("LLM calls: got %d, want 6", mock.callCount())
	}

	for i := 0; i < mock.callCount(); i++ {
		req := mock.request(i)
		if req == nil {
			t.Fatalf("request %d missing", i)
		}
		temp, _ := req["temperature"].(float64)
		if temp != 0.3 {
			t.Errorf("request %d temperature: got %v, want 0.3 (no escalation expected on distinct args)", i, temp)
		}
	}
}

// The phase contract: prose is an exit only where the phase allows it
// (the documenter); elsewhere it gets one nudge naming the terminal, and a
// second prose reply in a row ends the loop without a Terminal.
func TestPhaseContractNudge(t *testing.T) {
	run := func(t *testing.T, policy phasePolicy, responses ...string) (toolLoopResult, *mockLLM, *Session) {
		t.Helper()
		mock := newMockLLM(t, responses...)
		t.Cleanup(mock.Close)
		a, s := newTestAgent(t)
		res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
			[]llmMessage{{Role: "user", Content: "go"}}, policy, "execute", true, 0)
		if err != nil {
			t.Fatalf("runToolLoop: %v", err)
		}
		return res, mock, s
	}
	execPolicy := phasePolicy{terminals: map[string]bool{respondToolName: true}}

	t.Run("documenter prose ends the loop", func(t *testing.T) {
		res, mock, _ := run(t, phasePolicy{terminals: map[string]bool{respondToolName: true}, proseEnds: true},
			sseText("No documentation change needed."))
		if mock.callCount() != 1 || res.Text != "No documentation change needed." {
			t.Errorf("calls %d, res %+v: want one call and the prose", mock.callCount(), res)
		}
	})
	t.Run("prose then respond", func(t *testing.T) {
		res, mock, s := run(t, execPolicy,
			sseText("Done, the file is fixed."),
			sseToolCall("c1", respondToolName, `{"message":"Done."}`))
		if mock.callCount() != 2 || res.Terminal != respondToolName {
			t.Fatalf("calls %d, terminal %q: want the nudge and then respond", mock.callCount(), res.Terminal)
		}
		if got := lastUserMessage(s); !strings.Contains(got, "Call `respond` if the step is done") {
			t.Errorf("stored nudge = %q", got)
		}
	})
	t.Run("prose twice ends the loop", func(t *testing.T) {
		res, mock, _ := run(t, execPolicy, sseText("one"), sseText("two"))
		if mock.callCount() != 2 || res.Terminal != "" {
			t.Errorf("calls %d, terminal %q: want one nudge only, then the loop ends", mock.callCount(), res.Terminal)
		}
	})
}

// Picked up before the next model call as an ordinary, stored user message.
func TestSteeringLandsBetweenRounds(t *testing.T) {
	a, s := newTestAgent(t)
	testTools := []Tool{{
		Def: map[string]any{"type": "function", "function": map[string]any{
			"name": "probe", "description": "x", "parameters": map[string]any{"type": "object"}}},
		Execute: func(ctx context.Context, a *agent, sid string, raw string) (string, bool) {
			s.addSteer("also update the README")
			return "probed", false
		},
	}}
	mock := newMockLLM(t,
		sseToolCall("c1", "probe", `{}`),
		sseToolCall("c2", respondToolName, `{"message":"done"}`),
	)
	defer mock.Close()
	withTools(a, append(testTools, respondTool)...)

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}},
		phasePolicy{terminals: map[string]bool{respondToolName: true}}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if res.Terminal != respondToolName {
		t.Fatalf("the loop did not finish: %+v", res)
	}
	if mock.callCount() != 2 {
		t.Fatalf("LLM calls: got %d, want 2", mock.callCount())
	}

	msgs, _ := mock.request(1)["messages"].([]any)
	last, _ := msgs[len(msgs)-1].(map[string]any)
	if last["role"] != "user" || last["content"] != "also update the README" {
		t.Errorf("the steer should be the last message of the next round, got %v", last)
	}
	found := false
	for _, m := range s.Messages {
		found = found || (m.Role == "user" && m.Content == "also update the README")
	}
	if !found {
		t.Error("the steer was not stored in the session")
	}
	if q := s.takePending(); len(q) != 0 {
		t.Errorf("the queue should be empty after it was picked up, got %v", q)
	}
}

func TestBackgroundNoteReachesTheLoopMidTurn(t *testing.T) {
	a, s := newTestAgent(t)
	withTools(a, Tool{
		Def: map[string]any{"type": "function", "function": map[string]any{
			"name": "run_command", "description": "probe", "parameters": map[string]any{"type": "object"}}},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			// The job finishes while this round's tool is still running.
			s.addBgNote(bgNote{line: "background job 7 `just test` exited with code 0 after 2m20s",
				full: "[codehalter, not the user: background job 7 `just test` exited with code 0 after 2m20s. Last output:]\n\ntest result: ok. 12 passed"})
			return "edited", false
		},
	})
	mock := newMockLLM(t, sseToolCall("c1", "run_command", `{"command":"sed -i s/a/b/ x.rs"}`), sseText("done"))
	defer mock.Close()
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "test-model"}}}

	if _, err := a.runToolLoopSeeded(context.Background(), s.ID, a.connFor("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", true, 0); err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if mock.callCount() != 2 {
		t.Fatalf("calls = %d, want 2", mock.callCount())
	}
	msgs, _ := mock.request(1)["messages"].([]any)
	last, _ := msgs[len(msgs)-1].(map[string]any)
	if c, _ := last["content"].(string); last["role"] != "user" || !strings.Contains(c, "12 passed") {
		t.Fatalf("the finished job's note did not reach the model before its next call; last message: %v", last)
	}
	if s.hasPending() {
		t.Error("the note was delivered but left queued, so the turn end would deliver it again")
	}
}

// The note came first, so it leads: arrival order, not one kind before the other.
func TestSteerAndNoteShareOneMessage(t *testing.T) {
	a, s := newTestAgent(t)
	note := bgNote{line: "background job 7 `just test` exited with code 0 after 2m20s",
		full: "[codehalter, not the user: background job 7 `just test` exited with code 0 after 2m20s. Last output:]\n\ntest result: ok. 12 passed"}
	withTools(a, Tool{
		Def: map[string]any{"type": "function", "function": map[string]any{
			"name": "probe", "description": "x", "parameters": map[string]any{"type": "object"}}},
		Execute: func(ctx context.Context, a *agent, sid string, raw string) (string, bool) {
			s.addBgNote(note)
			s.addSteer("also update the README")
			return "probed", false
		},
	}, respondTool)
	mock := newMockLLM(t,
		sseToolCall("c1", "probe", `{}`),
		sseToolCall("c2", respondToolName, `{"message":"done"}`),
	)
	defer mock.Close()

	if _, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}},
		phasePolicy{terminals: map[string]bool{respondToolName: true}}, "execute", true, 0); err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if mock.callCount() != 2 {
		t.Fatalf("LLM calls: got %d, want 2", mock.callCount())
	}

	want := note.full + "\n\nalso update the README"
	msgs, _ := mock.request(1)["messages"].([]any)
	last, _ := msgs[len(msgs)-1].(map[string]any)
	prev, _ := msgs[len(msgs)-2].(map[string]any)
	if last["role"] != "user" || last["content"] != want {
		t.Errorf("the next round should end on one user message, note then steer; got %v", last)
	}
	if prev["role"] != "tool" {
		t.Errorf("the note and the steer reached the model as separate messages; before the last: %v", prev)
	}
	stored := 0
	for _, m := range s.Messages {
		if m.Role == "user" && m.Content == want {
			stored++
		}
	}
	if stored != 1 {
		t.Errorf("the joined message was stored %d times, want 1", stored)
	}
}

// Once, as a user message; the execute phase never sees it.
func TestPlanRoundNudgeAsksToSubmit(t *testing.T) {
	a, s := newTestAgent(t)
	withTools(a, Tool{
		Def: map[string]any{"type": "function", "function": map[string]any{
			"name": "read_file", "description": "probe", "parameters": map[string]any{"type": "object"}}},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) { return "hit", false },
	})
	var resp []string
	for i := 0; i <= planRoundNudge+1; i++ {
		resp = append(resp, sseToolCall(fmt.Sprintf("c%d", i), "read_file", fmt.Sprintf(`{"path":"f%d.rs"}`, i)))
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
		if c, _ := last["content"].(string); last["role"] == "user" && strings.Contains(c, "Call `submit_plan` NOW") {
			nudges++
			if i != planRoundNudge {
				t.Errorf("nudge arrived before call %d, want before call %d", i, planRoundNudge)
			}
		}
	}
	if nudges != 1 {
		t.Errorf("nudges = %d, want exactly 1", nudges)
	}
}

// A string that is not an array is still an error.
func TestPlanResultAcceptsStringifiedSubtasks(t *testing.T) {
	var direct, quoted planResult
	if err := json.Unmarshal([]byte(`{"clear":true,"subtasks":[{"description":"do x","verify":["run just test via run_command"]}]}`), &direct); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"clear":true,"subtasks":"[{\"description\":\"do x\",\"verify\":[\"run just test via run_command\"]}]"}`), &quoted); err != nil {
		t.Fatalf("stringified subtasks rejected: %v", err)
	}
	if !direct.Clear || len(direct.Subtasks) != 1 || len(quoted.Subtasks) != 1 || quoted.Subtasks[0].Description != "do x" || len(quoted.Subtasks[0].Verify) != 1 {
		t.Errorf("shapes differ: direct=%+v quoted=%+v", direct, quoted)
	}
	var bad planResult
	if err := json.Unmarshal([]byte(`{"clear":true,"subtasks":"do x"}`), &bad); err == nil {
		t.Error("a plain string that is not an array parsed as subtasks")
	}
	var none planResult
	if err := json.Unmarshal([]byte(`{"clear":true,"report_only":true}`), &none); err != nil || !none.ReportOnly {
		t.Errorf("a plan with no subtasks must still parse: %v %+v", err, none)
	}
}

// User input resumes the parked loop, which may park again; only after the job
// exits does a respond end the turn.
func TestToolLoopParksOnRespondWhileJobRuns(t *testing.T) {
	h := newTerminalHarness(t)
	defer h.agent.shutdownBackground()
	oldWait, oldPoll := cmdHandoverWait, parkPoll
	cmdHandoverWait, parkPoll = 100*time.Millisecond, 20*time.Millisecond
	defer func() { cmdHandoverWait, parkPoll = oldWait, oldPoll }()

	mock := newMockLLM(t,
		sseToolCall("c1", "run_command", `{"command":"sleep 1.2; echo suite-green"}`),
		sseToolCall("c2", respondToolName, `{"message":"Waiting for job 1."}`),
		sseToolCall("c3", respondToolName, `{"message":"Still waiting, as you asked."}`),
		sseToolCall("c4", respondToolName, `{"message":"final: suite green"}`),
	)
	defer mock.Close()
	h.agent.tools.add(Tool{Def: map[string]any{"type": "function", "function": map[string]any{"name": "run_command", "parameters": map[string]any{"type": "object"}}}, Execute: runCmdExecute})

	// The user interjects once the turn is parked.
	go func() {
		deadline := time.Now().Add(3 * time.Second)
		for time.Now().Before(deadline) {
			for _, u := range h.updatesOfKind(KindAgentMessage) {
				if c, _ := u["content"].(map[string]any); c != nil && strings.Contains(fmt.Sprint(c["text"]), "⏸ Waiting for job 1") {
					h.sess.addSteer("how is it going?")
					return
				}
			}
			time.Sleep(10 * time.Millisecond)
		}
	}()

	res, err := h.agent.runToolLoopSeeded(context.Background(), h.sess.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "run the suite and report"}}, phasePolicy{terminals: map[string]bool{respondToolName: true}}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if res.Text != "final: suite green" {
		t.Errorf("res.Text = %q, want the respond that came after the job reported", res.Text)
	}
	if mock.callCount() != 4 {
		t.Errorf("LLM calls = %d, want 4: run, park, resume on the user's text and park again, resume on the exit", mock.callCount())
	}
	var users []string
	for _, m := range h.sess.Messages {
		if m.Role == "user" {
			users = append(users, m.Content)
		}
	}
	joined := strings.Join(users, "\n")
	for _, want := range []string{"how is it going?", "exited with code 0", "suite-green", "Continue the work that was waiting"} {
		if !strings.Contains(joined, want) {
			t.Errorf("the resumes did not reach the model as user messages; missing %q in:\n%s", want, joined)
		}
	}
	if strings.Index(joined, "how is it going?") > strings.Index(joined, "exited with code 0") {
		t.Error("the user's interjection must arrive before the job's exit, not after")
	}
}

// A run_background test run parks a waiting respond like a handed-over command;
// a server does not, or the step would wait forever.
func TestToolLoopParksForBackgroundTestRun(t *testing.T) {
	gate := filepath.Join(t.TempDir(), "gate.mk")
	if err := os.WriteFile(gate, []byte("all:\n\t@sleep 0.6; echo test-suite-green\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		cmd   string
		calls int
	}{
		{"sleep 5; echo serve-forever", 2},
		{"make -s -f " + gate + " && sleep 5", 2}, // built, then something that stays up
	}
	if _, err := exec.LookPath("make"); err == nil {
		cases = append(cases, struct {
			cmd   string
			calls int
		}{"make -s -f " + gate, 3})
	}
	for _, tc := range cases {
		h := newTerminalHarness(t)
		oldGrace, oldPoll := bgJobGrace, parkPoll
		bgJobGrace, parkPoll = 50*time.Millisecond, 20*time.Millisecond
		mock := newMockLLM(t,
			sseToolCall("c1", "run_background", fmt.Sprintf(`{"command":%q}`, tc.cmd)),
			sseToolCall("c2", respondToolName, `{"message":"Waiting for job 1."}`),
			sseToolCall("c3", respondToolName, `{"message":"final"}`),
		)
		h.agent.tools.add(Tool{Def: map[string]any{"type": "function", "function": map[string]any{"name": "run_background", "parameters": map[string]any{"type": "object"}}}, Execute: runBackgroundExecute})
		_, err := h.agent.runToolLoopSeeded(context.Background(), h.sess.ID, mock.conn("execute"),
			[]llmMessage{{Role: "user", Content: "run it"}}, phasePolicy{terminals: map[string]bool{respondToolName: true}}, "execute", true, 0)
		bgJobGrace, parkPoll = oldGrace, oldPoll
		h.agent.shutdownBackground()
		mock.Close()
		if err != nil {
			t.Fatalf("%s: %v", tc.cmd, err)
		}
		if got := mock.callCount(); got != tc.calls {
			t.Errorf("%s: %d model calls, want %d", tc.cmd, got, tc.calls)
		}
	}
}

// Needs no message text: a server that forces the tool call returns none.
func TestPlanAnswerInArgument(t *testing.T) {
	a, s, mock := planPhaseAgent(t, sseToolCall("p1", submitPlanToolName,
		`{"clear":true,"report_only":true,"subtasks":[],"answer":"Only the Prepare page builds widgets; Cut, Narrate and Produce are bare."}`))
	plan, err := a.runPlanPhase(context.Background(), s.ID, "")
	if err != nil {
		t.Fatalf("runPlanPhase: %v", err)
	}
	if got := mock.callCount(); got != 1 {
		t.Errorf("an answer in the argument cost %d LLM calls, want 1 (no nudge)", got)
	}
	if plan == nil || !strings.HasPrefix(plan.answer, "Only the Prepare page") || len(plan.Subtasks) != 0 {
		t.Fatalf("plan = %+v, want the argument as the answer", plan)
	}
}

func TestPlanRedoIsAPlan(t *testing.T) {
	a, s, mock := planPhaseAgent(t, sseToolCall("p1", submitPlanToolName,
		`{"clear":true,"report_only":false,"subtasks":[],"redo":["F0.9","§03-shell#1-screen"]}`))
	plan, err := a.runPlanPhase(context.Background(), s.ID, "")
	if err != nil {
		t.Fatalf("runPlanPhase: %v", err)
	}
	if got := mock.callCount(); got != 1 {
		t.Errorf("a redo submission cost %d LLM calls, want 1", got)
	}
	if plan == nil || strings.Join(plan.Redo, ",") != "F0.9,§03-shell#1-screen" {
		t.Fatalf("plan = %+v, want the redo ids", plan)
	}
}

func TestPlanStringifiedSubtasksWithStrayBrace(t *testing.T) {
	var p planResult
	raw := `{"clear": true, "report_only": false, "subtasks": "[{\"description\": \"look\", \"verify\": [\"a\"]}]}"}`
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(p.Subtasks) != 1 || p.Subtasks[0].Description != "look" {
		t.Errorf("subtasks = %+v", p.Subtasks)
	}
}

// read A, read B, read B: the first B carries a batching note the second does
// not, and the second must still count as a repeat.
func TestStuckLadderSeesRepeatPastBatchNote(t *testing.T) {
	a, s := newTestAgent(t)
	rt := &repetitionTracker{hash: map[string]uint64{}, bag: map[string]map[string]bool{}}
	for _, name := range []string{"a.toml", "b.toml"} {
		if err := os.WriteFile(filepath.Join(s.Cwd, name), []byte("k = 1\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	read := func(path string) bool {
		s.markReplyStart()
		var tc toolCall
		tc.Function.Name = "read_file"
		tc.Function.Arguments = fmt.Sprintf(`{"path":%q}`, path)
		tu, _ := a.runToolCall(context.Background(), s.ID, tc)
		return rt.sawAgain(tc, tu)
	}
	read("a.toml")
	read("b.toml")
	if !read("b.toml") {
		t.Error("the first identical re-read after a batching note did not count as a repeat")
	}
}

// Three blind spots of the old key: a fresh echo label, a range revisited after
// another one, and a read-only heredoc that counted as a write.
func TestStuckLadderKeysCommandsBySubstance(t *testing.T) {
	rt := &repetitionTracker{hash: map[string]uint64{}, bag: map[string]map[string]bool{}}
	run := func(cmd, out string) bool {
		var tc toolCall
		tc.Function.Name = "run_command"
		b, _ := json.Marshal(map[string]string{"command": cmd})
		tc.Function.Arguments = string(b)
		return rt.sawAgain(tc, ToolUse{Name: "run_command", Output: "exit 0\n\n" + out})
	}
	run(`echo "=== the prompt ==="; grep -n prompt spec/09.md | head -3`, "=== the prompt ===\n12:prompt")
	if !run(`echo "=== system message ==="; grep -n prompt spec/09.md | head -3`, "=== system message ===\n12:prompt") {
		t.Error("the same grep under a new echo label did not count as a repeat")
	}
	run(`sed -n '10,20p' src/a.rs`, "ten to twenty")
	run(`sed -n '30,40p' src/a.rs`, "thirty to forty")
	if !run(`sed -n '10,20p' src/a.rs`, "ten to twenty") {
		t.Error("a range read again after another range did not count as a repeat")
	}
	heredoc := "python3 - <<'EOF'\nprint(open('a.rs').read().count('fn'))\nEOF"
	run(heredoc, "7")
	if !run(heredoc, "7") {
		t.Error("a read-only python heredoc run twice counted as a write, not a repeat")
	}
	// A script that writes, with the double quotes JSON escapes, makes the re-check new.
	run("go build ./...", "ok")
	run("python3 - <<'EOF'\nopen(\"a.go\", \"w\").write(\"x\")\nEOF", "")
	if run("go build ./...", "ok") {
		t.Error("the build after a python write counted as a repeat")
	}
	// An echo piped into a program is its input, not a label.
	run(`echo "hello" | ./parse`, "error: bad input")
	if run(`echo '{"a":1}' | ./parse`, "error: bad input") {
		t.Error("two different inputs piped into the same program counted as one call")
	}
}

// The same command straight after itself, exit 0 and the same output, is a spin.
func TestStuckLadderCatchesSuccessfulCommandSpin(t *testing.T) {
	rt := &repetitionTracker{hash: map[string]uint64{}, bag: map[string]map[string]bool{}}
	mk := func(id, name, args string) toolCall {
		var tc toolCall
		tc.ID, tc.Function.Name, tc.Function.Arguments = id, name, args
		return tc
	}
	probe := mk("c", "run_command", `{"command":"ls -la shots/x.png"}`)
	edit := mk("e", "edit_file", `{"path":"a.rs"}`)
	out := ToolUse{Name: "run_command", Output: "exit 0\n\n-rw-r--r-- 1 dev dev 33037 x.png\n"}

	other := mk("o", "run_command", `{"command":"grep -n foo a.rs"}`)
	otherOut := ToolUse{Name: "run_command", Output: "exit 0\n\n12:foo\n"}

	if rt.sawAgain(probe, out) {
		t.Fatal("the first run is new")
	}
	if !rt.sawAgain(probe, out) {
		t.Error("the same successful command straight after itself did not count as a repeat")
	}
	// Alternating two probes changes nothing either.
	rt.sawAgain(other, otherOut)
	if !rt.sawAgain(probe, out) {
		t.Error("a successful command re-run after only another probe did not count as a repeat")
	}
	if !rt.sawAgain(other, otherOut) {
		t.Error("the alternating probe did not count as a repeat")
	}
	rt.sawAgain(edit, ToolUse{Name: "edit_file", Output: "ok"})
	if rt.sawAgain(probe, out) {
		t.Error("a re-run after an edit is a re-verify, not a repeat")
	}
	rt.sawAgain(mk("s", "run_command", `{"command":"sed -i s/a/b/ a.rs"}`), ToolUse{Name: "run_command", Output: "exit 0\n"})
	if rt.sawAgain(probe, out) {
		t.Error("a re-run after an in-place shell write is a re-verify, not a repeat")
	}
	rt.sawAgain(mk("p", "run_command", "{\"command\":\"cd rust && python3 - <<'PY'\\nopen('a.rs','w').write('x')\\nPY\"}"), ToolUse{Name: "run_command", Output: "exit 0\n"})
	if rt.sawAgain(probe, out) {
		t.Error("a re-run after a Python heredoc edit is a re-verify, not a repeat")
	}
	// A counter bumped into the command changes neither the key nor the answer.
	for i := 1; i <= 3; i++ {
		poll := mk("r", "run_command", fmt.Sprintf(`{"command":"ls -l shotview/cut.png | cut -c1-70; echo READY%d"}`, i))
		pollOut := ToolUse{Name: "run_command", Output: fmt.Sprintf("exit 0\n\n-rw-r--r-- 1 dev dev 115300 Sep 26 22:20 shotview/cut.\nREADY%d\n", i)}
		if got := rt.sawAgain(poll, pollOut); got != (i > 1) {
			t.Errorf("poll %d: repeat = %v, want %v", i, got, i > 1)
		}
	}
	// A redirect to a log is not a change to the tree.
	snap := mk("n", "run_command", `{"command":"cd rust && just snapshot 04-prepare > /tmp/snap.log 2>&1"}`)
	snapOut := ToolUse{Name: "run_command", Output: "exit 0\n\n-> shots/04-prepare.png\n"}
	rt.sawAgain(snap, snapOut)
	if !rt.sawAgain(snap, snapOut) {
		t.Error("a repeated snapshot render with only a log redirect did not count as a repeat")
	}
	if !rt.sawAgain(probe, ToolUse{Name: "run_command", Output: "exit 1\n\nboom", Failed: true}) {
		// New output is not a repeat; the next identical failure is.
		t.Log("first failure is new output")
	}
	if !rt.sawAgain(probe, ToolUse{Name: "run_command", Output: "exit 1\n\nboom", Failed: true}) {
		t.Error("a repeated failure did not count")
	}
}

// A clear submit_plan with neither an answer nor any work gets the one
// planner retry, and the retry's plan is taken.
func TestPlanNeitherAnswerNorWorkRetries(t *testing.T) {
	a, s, mock := planPhaseAgent(t,
		sseToolCall("p1", submitPlanToolName, `{"clear":true,"report_only":false,"subtasks":[]}`),
		sseToolCall("p2", submitPlanToolName, `{"clear":true,"report_only":false,"subtasks":[{"description":"fix it","verify":["go test ./..."]}]}`))
	plan, err := a.runPlanPhase(context.Background(), s.ID, "")
	if err != nil {
		t.Fatalf("runPlanPhase: %v", err)
	}
	if mock.callCount() != 2 || plan == nil || len(plan.Subtasks) != 1 {
		t.Fatalf("calls %d, plan %+v: want the retry's subtask", mock.callCount(), plan)
	}
	if got := lastUserMessage(s); !strings.Contains(got, "neither an answer nor subtasks") {
		t.Errorf("stored corrective = %q", got)
	}
}

// A server that refuses connections is waited for with growing pauses, and the
// call goes through once it listens again; past the patience it is an error.
func TestToolLoopWaitsForARestartingServer(t *testing.T) {
	oldBackoff, oldPatience := transientStreamBackoff, serverDownPatience
	transientStreamBackoff, serverDownPatience = 20*time.Millisecond, 5*time.Second
	defer func() { transientStreamBackoff, serverDownPatience = oldBackoff, oldPatience }()

	mock := newMockLLM(t, sseText("back"))
	defer mock.Close()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()
	go func() {
		time.Sleep(150 * time.Millisecond)
		if ln, err := net.Listen("tcp", addr); err == nil {
			go http.Serve(ln, mock.ts.Config.Handler)
			t.Cleanup(func() { ln.Close() })
		}
	}()
	a, s := newTestAgent(t)
	conn := &LLMConnection{Server: "http://" + addr, Model: "m", Tag: "execute"}
	res, err := a.runToolLoopSeeded(context.Background(), s.ID, conn, []llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", false, 0)
	if err != nil || res.Text != "back" {
		t.Fatalf("res=%q err=%v, want the reply once the server listens again", res.Text, err)
	}

	serverDownPatience = 50 * time.Millisecond
	closed, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	conn.Server = "http://" + closed.Addr().String()
	if _, err := a.runToolLoopSeeded(context.Background(), s.ID, conn, []llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", false, 0); err == nil || !strings.Contains(err.Error(), "refused connections") {
		t.Errorf("err = %v, want the give-up after the patience", err)
	}
}

// A step still editing and building at the call cap gets one extension to finish;
// one that did not edit hits the cap as before.
func TestToolLoopExtendsAProductiveStepOnce(t *testing.T) {
	fake := func(name, out string) Tool {
		return Tool{Def: map[string]any{"type": "function", "function": map[string]any{"name": name, "parameters": map[string]any{"type": "object"}}},
			Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
				return out + rawArgs, false
			}}
	}
	run := func(editing bool) (toolLoopResult, error, int) {
		var resp []string
		for i := range maxToolLoopIterations {
			switch {
			case editing && i%2 == 0:
				resp = append(resp, sseToolCall(fmt.Sprintf("c%d", i), "edit_file", fmt.Sprintf(`{"path":"src/f%d.rs"}`, i)))
			default:
				resp = append(resp, sseToolCall(fmt.Sprintf("c%d", i), "run_command", fmt.Sprintf(`{"command":"cargo build --bin b%d"}`, i)))
			}
		}
		resp = append(resp, sseToolCall("done", respondToolName, `{"message":"page built"}`))
		mock := newMockLLM(t, resp...)
		defer mock.Close()
		a, s := newTestAgent(t)
		withTools(a, fake("edit_file", "file written successfully "), fake("run_command", "exit 0\n\nbuilt "))
		res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
			[]llmMessage{{Role: "user", Content: "build the page"}}, phasePolicy{terminals: map[string]bool{respondToolName: true}}, "execute", false, 0)
		return res, err, mock.callCount()
	}
	if res, err, calls := run(true); err != nil || res.Terminal != respondToolName || calls != maxToolLoopIterations+1 {
		t.Errorf("editing step: terminal=%q err=%v calls=%d, want it to finish on the extension", res.Terminal, err, calls)
	}
	if _, err, calls := run(false); err == nil || !strings.Contains(err.Error(), "exceeded") || calls != maxToolLoopIterations {
		t.Errorf("step without edits: err=%v calls=%d, want the cap", err, calls)
	}
}
