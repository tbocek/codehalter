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

// meterCall builds the one tool call shape startToolMeter reads: a name and a
// raw JSON argument string.
func meterCall(name, args string) toolCall {
	tc := toolCall{ID: "tc1"}
	tc.Function.Name = name
	tc.Function.Arguments = args
	return tc
}

// TestStartToolMeter pins the tool status meter's lifecycle: stop() halts the
// ticker and joins its goroutine promptly (no deadlock, no leak), and a cancelled
// ctx also lets stop() return. It does not assert the 1s-tick text: that is
// startStatusMeter's, and it is time-based.
func TestStartToolMeter(t *testing.T) {
	a, s := newTestAgent(t)

	// Normal stop joins quickly.
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
		// Long fixed error text + two noise tokens (elapsed, attempt) → Jaccard
		// ≈ 0.93 between consecutive outputs: above stuckOutputSimilarity while
		// the fnv hash differs every time.
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

// TestToolMeterShowsTheArgument pins what the status row is FOR: "run_command"
// sitting at 77s says something is slow but not what, and the arguments have
// scrolled away in the transcript by then. The row carries the command itself.
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
		{"run_task", `{"task":"just:build"}`, "run_task just:build"},
		{"search_text", `{"query":"LoadAll","path":"src"}`, "search_text LoadAll"}, // query wins over path
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

// TestAddCorrectiveSurvivesRebuild pins the invariant that cost one turn 9998 of
// 15346 re-evaluated tokens: a corrective turn appended to the wire only is
// gone from the next runToolLoop rebuild, which drops it out of the MIDDLE of
// history and shifts everything after it. The wire and a rebuild must agree.
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

// TestPlanRecoversFromMalformedSubmitPlanArguments pins the recovery hole: a
// planner that CALLS submit_plan but writes arguments which aren't valid JSON
// used to skip the corrective retry entirely (the guard also demanded
// !RespondCalled) and fail the whole turn on the first malformed argument list.
func TestPlanRecoversFromMalformedSubmitPlanArguments(t *testing.T) {
	broken := sseToolCall("c1", submitPlanToolName, `{"clear":true,"subtasks":[{"description":"do the thing"`)
	fixed := sseToolCall("c2", submitPlanToolName,
		`{"clear":true,"subtasks":[{"description":"do the thing","verify":["go build ./..."]}],"report_only":false}`)
	mock := newMockLLM(t, broken, fixed)
	defer mock.Close()
	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
	// An empty PLAN.md disables planning outright, so seed one: the content is
	// irrelevant here, only its presence gates the phase.
	if err := os.MkdirAll(filepath.Join(s.Cwd, ".codehalter"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(s.Cwd, ".codehalter", "PLAN.md"), []byte("plan things"), 0o644); err != nil {
		t.Fatal(err)
	}

	plan, _, err := a.runPlanPhase(context.Background(), s.ID, "")
	if err != nil {
		t.Fatalf("malformed arguments should recover via the corrective retry, got: %v", err)
	}
	if plan == nil || len(plan.Subtasks) != 1 || plan.Subtasks[0].Description != "do the thing" {
		t.Fatalf("plan = %+v, want the retry's single subtask", plan)
	}
}

// planPhaseAgent wires a mock LLM to a session and seeds a PLAN.md, the two
// things runPlanPhase needs before it will run at all: no [[llm]] is an error
// and an empty PLAN.md disables planning outright. The PLAN.md content is
// irrelevant to these tests, only its presence gates the phase.
func planPhaseAgent(t *testing.T, responses ...string) (*agent, *Session, *mockLLM) {
	t.Helper()
	mock := newMockLLM(t, responses...)
	t.Cleanup(mock.Close)
	a, s := newTestAgent(t)
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "m"}}}
	if err := os.MkdirAll(filepath.Join(s.Cwd, ".codehalter"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(s.Cwd, ".codehalter", "PLAN.md"), []byte("plan things"), 0o644); err != nil {
		t.Fatal(err)
	}
	return a, s, mock
}

// TestPlanPreambleIsNotAnAnswer pins the detector's good path: prose written
// alongside subtasks with report_only=false is a PREAMBLE, and preambles cost
// nothing. orchestrate already drops that prose (it surfaces answer only when
// there are no subtasks), so there is nothing to pick between and no reason to
// spend a corrective round trip asking. Measured on two real sessions: 10 of 35
// plan submissions looked exactly like this one, every one report_only=false,
// every one nudged for a plan it had already submitted.
func TestPlanPreambleIsNotAnAnswer(t *testing.T) {
	args := `{"clear":true,"report_only":false,"subtasks":[{"description":"delete out/test and rebuild","verify":["go build ./..."]}]}`
	a, s, mock := planPhaseAgent(t, sseContentThenToolCall(
		"The request is clear: delete the out/test build output and recompile the site.",
		"p1", submitPlanToolName, args))

	plan, _, err := a.runPlanPhase(context.Background(), s.ID, "")
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

// TestPlanReportOnlyProseStillNudges pins the other side: report_only=true IS
// the real fork. Those subtasks only relay findings, so prose alongside them
// may already have delivered the answer the subtasks would go re-derive. The
// planner has to pick, and the corrective round is what makes it.
func TestPlanReportOnlyProseStillNudges(t *testing.T) {
	ambiguous := sseContentThenToolCall(
		"The three helpers live in a.go, b.go and c.go.",
		"p1", submitPlanToolName,
		`{"clear":true,"report_only":true,"subtasks":[{"description":"list the helpers","verify":["ls"]}]}`)
	picked := sseContentThenToolCall(
		"The three helpers live in a.go, b.go and c.go.",
		"p2", submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[]}`)
	a, s, mock := planPhaseAgent(t, ambiguous, picked)

	plan, _, err := a.runPlanPhase(context.Background(), s.ID, "")
	if err != nil {
		t.Fatalf("runPlanPhase: %v", err)
	}
	if got := mock.callCount(); got != 2 {
		t.Errorf("an answer + report_only subtasks cost %d LLM calls, want 2 (the nudge)", got)
	}
	if plan == nil || len(plan.Subtasks) != 0 {
		t.Fatalf("plan = %+v, want the corrected answer-only submission", plan)
	}
}

// TestExecutePhaseTurnsReasoningOff pins how the execute role stops reasoning:
// the connection carries the closed-<think> prefill, so the wire ends in an
// assistant message the server is told to continue, and every earlier token is
// untouched.
//
// It used to be a "/no_think" suffix glued onto the STORED subtask prompt. That
// bought nothing twice over: 237 of 388 execute responses carrying it reasoned
// anyway over one 11.6h session, and the suffix then rode the history for the
// rest of it. So the stored turn is asserted too — it must be the prompt and
// nothing else.
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
	// This phase ends only on a terminal tool, so prose is always a slip: the
	// server is told to answer with a tool call rather than being nudged into
	// one afterwards. A grammar, not a re-render, so the prefix is untouched.
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

// TestToolLoopRecordsToolUses verifies that when the LLM returns a tool call,
// the tool loop executes it, appends the ToolUse to the session, and persists
// to disk — before the second LLM turn produces the final text.
func TestToolLoopRecordsToolUses(t *testing.T) {
	// Isolate from the package-level registry so the synthetic `respond` tool
	// (registered in tool_phase_end.go init) isn't in scope — its presence would
	// flip the loop's empty-tool-call branch from "exit with allText" to a
	// nudge, which is a different code path tested elsewhere.
	var testTools []Tool
	// A stub tool, so we don't depend on the filesystem tool implementations.
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
		// Turn 1: LLM asks to call the stub tool.
		sseToolCall("call_1", testToolName, `{"msg":"hello"}`),
		// Turn 2: LLM produces the final assistant text.
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

	// The loop now stores each assistant turn VERBATIM (cache-faithful replay):
	// [user, assistant{tool turn: echo}, assistant{text turn: "All done."}].
	// The tool turn carries the tool use; the final text turn carries the text —
	// no merge, no post-hoc patch.
	if got := len(s.Messages); got != 3 {
		t.Fatalf("session messages: got %d, want 3", got)
	}
	if s.Messages[1].Role != "assistant" || len(s.Messages[1].ToolUses) != 1 {
		t.Errorf("msg[1]: want assistant with 1 tool use, got role=%q tools=%d", s.Messages[1].Role, len(s.Messages[1].ToolUses))
	}
	if s.Messages[2].Role != "assistant" || s.Messages[2].Content != "All done." {
		t.Errorf("msg[2]: want assistant content %q, got role=%q content=%q", "All done.", s.Messages[2].Role, s.Messages[2].Content)
	}

	// Persistence: the turns are on disk from the incremental Save in the loop.
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

// TestToolLoopRespondExits verifies that when the model calls the synthetic
// `respond` terminal tool, the loop exits with the message arg as res.Text on
// the same iteration — no second LLM round-trip to "produce final text".
// This is the post-respond exit semantic (vs the legacy "empty tool_calls
// means done" path covered by TestToolLoopRecordsToolUses with a fresh
// registry).
func TestToolLoopRespondExits(t *testing.T) {
	// The agent's own tools, so respond is in scope.
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

// TestRunToolLoopDenyGate pins the dispatch gate that replaced array pruning: a
// tool the phase policy denies is REJECTED without executing, recorded as a
// failed tool use with a teaching message, and the loop continues so the model
// can correct. (The tool is still in the array — only dispatch blocks it.)
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

// TestRunToolLoopMultiTerminalUpsert pins multi-terminal exit: in an execute
// loop exposing BOTH respond and submit_plan, calling submit_plan ends the loop
// with Terminal=submit_plan and the plan JSON in res.Text — the signal the
// orchestrator reads as a plan-upsert.
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

// TestToolLoopNoTerminalKeepsTextExit verifies that a policy with NO terminals
// keeps the legacy text-only exit: a no-tool-calls turn returns immediately
// instead of nudging. (Phases set their terminals explicitly now; an empty
// phasePolicy is the no-terminal case.)
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

// TestPlanSubmitPlanSeparatesAnswer verifies the plan phase's terminal split:
// when the planner writes a direct answer AND calls submit_plan, the structured
// plan lands in res.Text (submit_plan's echoed args) while the user-facing prose
// lands in res.Content — the two channels never mix, so the old "answer mashed
// into the plan JSON" bug can't recur.
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
	if !res.RespondCalled {
		t.Errorf("RespondCalled: got false, want true (submit_plan is the plan terminal)")
	}
	if !strings.Contains(res.Text, `"report_only":true`) {
		t.Errorf("res.Text should carry the submit_plan args (the plan JSON): got %q", res.Text)
	}
	if !strings.Contains(res.Content, "Active servers: gopls.") {
		t.Errorf("res.Content should carry the prose answer separately: got %q", res.Content)
	}
}

// TestToolLoopNoDedup verifies that identical tool calls in the same tool
// loop each execute the underlying tool. The dedup cache used to suppress
// the second call, which broke read-after-write: a mutator (sed via
// run_command) would change state, then a re-issued read returned the
// pre-mutation cached value and the model concluded the mutation failed.
// Now every call executes; the model gets a fresh result every time.
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

// TestToolLoopRepeatNudgeAndBail covers the repetition recovery path: the
// second consecutive identical tool call still executes (read-after-write
// must keep working) but appends a user-role nudge; the third identical
// call bails before executing. ~3 iterations of stuck behavior fails
// fast instead of waiting for the 50-iter cap.
// TestToolLoopRepetitionLadder exercises the unified repeat ladder end to end: a
// tool that returns identical output every call makes no progress, so consecutive
// rounds climb ONE ladder — a corrective nudge each stuck round, a one-shot swap
// to the thinking sampler at stuckEscalateRounds, then a GRACEFUL bail (nil error,
// RespondCalled=false) at stuckBailRounds. Replaces the old separate
// signature-nudge/bail and per-name-escalation guards.
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

	// Identical call every round → identical output → stuck from round 2 on.
	// 8 queued is more than enough; the loop bails at round 6 (stuckBailRounds).
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
	conn := a.connForSession(context.Background(), s.ID, "execute")
	if conn == nil {
		t.Fatalf("connForSession(execute) returned nil")
	}

	res, err := a.runToolLoopSeeded(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: want graceful nil error, got %v", err)
	}
	if res.RespondCalled {
		t.Errorf("RespondCalled: got true, want false (the loop bailed, never called respond)")
	}
	// round 1 is productive (first output); rounds 2-6 are stuck; the 5th stuck
	// round (round 6) hits stuckBailRounds and bails after executing.
	if mock.callCount() != 6 {
		t.Errorf("LLM calls: got %d, want 6 (bail at stuckBailRounds)", mock.callCount())
	}
	if execs != 6 {
		t.Errorf("tool execs: got %d, want 6", execs)
	}
	// Corrective shows up once a round goes stuck (request index 2 onward).
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
	// stuckEscalateRounds=3: the swap fires at the END of round 4 (stuckRounds
	// hits 3), so requests 0-3 use the execute sampler (0.3), 4-5 use thinking (1.0).
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
}

// TestRepetitionLadderExemptsSuccessfulRunTask pins the build/test re-verify
// carve-out: re-running run_task with identical SUCCESSFUL output (e.g. just:build
// green both times after an edit) is NOT counted as no-progress, so the loop never
// nudges or bails on it — unlike the generic repeating tool in the ladder test.
func TestRepetitionLadderExemptsSuccessfulRunTask(t *testing.T) {
	var testTools []Tool
	var execs int
	testTools = append(testTools, Tool{
		Def: map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": "run_task", "description": "probe",
				"parameters": map[string]any{"type": "object"},
			},
		},
		Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
			execs++
			return "go build -o codehalter .", false // identical green output, success
		},
	})

	// Six identical successful run_task calls, then a plain-text exit.
	var resp []string
	for i := 0; i < 6; i++ {
		resp = append(resp, sseToolCall(fmt.Sprintf("c%d", i), "run_task", `{"task":"just:build"}`))
	}
	resp = append(resp, sseText("all green, done"))
	mock := newMockLLM(t, resp...)
	defer mock.Close()

	a, s := newTestAgent(t)
	withTools(a, testTools...)
	a.settings = Settings{LLM: []LLMConnection{{Server: mock.ts.URL, Model: "test-model"}}}
	conn := a.connForSession(context.Background(), s.ID, "execute")
	if conn == nil {
		t.Fatalf("connForSession(execute) returned nil")
	}

	if _, err := a.runToolLoopSeeded(context.Background(), s.ID, conn,
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", true, 0); err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if execs != 6 {
		t.Errorf("tool execs: got %d, want 6 (no early bail on successful run_task re-runs)", execs)
	}
	for i := 0; i < mock.callCount(); i++ {
		msgs, _ := mock.request(i)["messages"].([]any)
		for _, m := range msgs {
			mm, _ := m.(map[string]any)
			if c, _ := mm["content"].(string); mm["role"] == "user" && strings.Contains(c, "makes no progress") {
				t.Errorf("request %d carried a repeat-corrective for a successful run_task re-run", i)
			}
		}
	}
}

// TestToolLoopDoesNotEscalateOnDistinctArgs verifies that legitimate fan-out
// across distinct arguments (e.g. read_file on go.mod, examples/go.mod, …
// when surveying a multi-module repo) never climbs the repetition ladder: each
// call returns NEW output, so no round is "stuck", the sampler stays on the
// execute role, and the loop never bails.
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

	// 5 tool calls with *different* args (the surveying-pattern that used
	// to trip the old per-name counter). With distinct-args counting, none
	// of these count as redundant, so no escalation should fire.
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

	conn := a.connForSession(context.Background(), s.ID, "execute")
	if conn == nil {
		t.Fatalf("connForSession(execute) returned nil")
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

	// Every call must still use the execute sampler — no escalation.
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

// TestToolLoopNudgesReasonedButEmpty pins the "calculated a lot, then nothing"
// guard: a turn with a big reasoning block but EMPTY visible content (no tool
// call) is nudged to write the answer as plain text, not silently accepted as an
// empty result. After the nudge the model writes the answer and it's returned.
func TestToolLoopNudgesReasonedButEmpty(t *testing.T) {
	mock := newMockLLM(t,
		sseReasoning(strings.Repeat("thinking hard. ", 80)), // ~1.1 KB reasoning, empty content, no calls
		sseText("here is the answer"),                       // after the nudge, the visible answer
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}}, phasePolicy{}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if mock.callCount() != 2 {
		t.Errorf("calls: got %d, want 2 (reasoned-but-empty must be nudged, not accepted)", mock.callCount())
	}
	if res.Text != "here is the answer" {
		t.Errorf("res.Text: got %q, want the post-nudge answer", res.Text)
	}
}

// TestNoToolCallNudgesEscalate pins the ladder for a model that answers in
// prose in a phase that only ends on a terminal tool: three re-asks whose
// wording escalates, then the text exit so it cannot spin forever. One polite
// nudge used to be the whole budget, and the weaker models answered in prose
// again on the next round, ending the turn with the work unfinished.
func TestNoToolCallNudgesEscalate(t *testing.T) {
	mock := newMockLLM(t,
		sseText("I will now do the thing"),
		sseText("I am doing the thing"),
		sseText("still prose"),
		sseText("final prose"),
	)
	defer mock.Close()

	a, s := newTestAgent(t)
	res, err := a.runToolLoopSeeded(context.Background(), s.ID, mock.conn("execute"),
		[]llmMessage{{Role: "user", Content: "go"}},
		phasePolicy{terminals: map[string]bool{respondToolName: true}}, "execute", true, 0)
	if err != nil {
		t.Fatalf("runToolLoop: %v", err)
	}
	if mock.callCount() != noCallNudges+1 {
		t.Fatalf("LLM calls: got %d, want %d (one per nudge plus the accepted text exit)", mock.callCount(), noCallNudges+1)
	}

	// Each re-ask carries its own wording, and the last one is the imperative.
	var nudges []string
	for i := 1; i < mock.callCount(); i++ {
		msgs, _ := mock.request(i)["messages"].([]any)
		last, _ := msgs[len(msgs)-1].(map[string]any)
		nudges = append(nudges, last["content"].(string))
	}
	for _, want := range []string{"no tool call", "Prose again", "STOP. You MUST call"} {
		found := false
		for _, n := range nudges {
			found = found || strings.Contains(n, want)
		}
		if !found {
			t.Errorf("no nudge contained %q; got %q", want, nudges)
		}
	}
	if nudges[0] == nudges[1] || nudges[1] == nudges[2] {
		t.Errorf("the wording must change with each re-ask, got %q", nudges)
	}
	if res.RespondCalled {
		t.Error("the model never called respond; the loop must not report one")
	}
	if !strings.Contains(res.Text, "final prose") {
		t.Errorf("the accepted text exit should carry the prose, got %q", res.Text)
	}
}

// TestSteeringLandsBetweenRounds pins what a prompt typed mid-turn does: it is
// queued, the running turn picks it up before its next model call, and it
// arrives as an ordinary user message (an append, so the prefix cache holds).
// Before this, typing cancelled the turn in flight and started a new one, which
// during a long /spec round threw away the whole round.
func TestSteeringLandsBetweenRounds(t *testing.T) {
	a, s := newTestAgent(t)
	// The user types while the tool runs, which is when they actually do.
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
	if !res.RespondCalled {
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
	// It is part of the conversation, not a one-off: the session keeps it.
	found := false
	for _, m := range s.Messages {
		found = found || (m.Role == "user" && m.Content == "also update the README")
	}
	if !found {
		t.Error("the steer was not stored in the session")
	}
	if q := s.takeSteer(); len(q) != 0 {
		t.Errorf("the queue should be empty after it was picked up, got %v", q)
	}
}
