package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"time"
)

// This file owns the plan phase and the per-subtask loop; the orchestrator
// (prompt.go) drives plan → subtasks → replan and the closing document phase.
// Each subtask carries its own verify recipe and runs as ONE tool loop in which
// the executor reads, edits, installs and verifies before calling `respond`.
// There is no separate verify call, and the only ceiling is
// maxToolLoopIterations.

// ---------------------------------------------------------------------------
// Plan phase
// ---------------------------------------------------------------------------

// subtask is one piece of decomposed work. Description is the self-
// contained instruction the executor acts on; Verify is the ordered list
// of concrete checks the executor MUST run (via tools) before calling
// respond. An empty Verify is legal only for pure-lookup subtasks where
// no files are edited.
type subtask struct {
	Description string   `json:"description"`
	Verify      []string `json:"verify,omitempty"`
}

// UnmarshalJSON accepts `subtasks` both as the array the schema asks for and
// as that array serialised into a string, which the planner sent six times in
// one afternoon ("subtasks": "[{\"description\": ...}]"). Each rejection cost a
// whole planning call to retry; decoding the string costs nothing.
func (p *planResult) UnmarshalJSON(b []byte) error {
	type plain planResult
	var raw struct {
		plain
		Subtasks json.RawMessage `json:"subtasks"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	*p = planResult(raw.plain)
	if len(raw.Subtasks) == 0 || string(raw.Subtasks) == "null" {
		return nil
	}
	if raw.Subtasks[0] == '"' {
		var inner string
		if err := json.Unmarshal(raw.Subtasks, &inner); err != nil {
			return err
		}
		raw.Subtasks = json.RawMessage(inner)
	}
	return json.Unmarshal(raw.Subtasks, &p.Subtasks)
}

type planResult struct {
	Clear    bool      `json:"clear"`
	Choices  []string  `json:"choices"`
	Question string    `json:"question"`
	Subtasks []subtask `json:"subtasks"`
	// ReportOnly is true when the planner already has the answer in hand
	// (pure-lookup tasks: "where is X?", "what version of Y?") and every
	// subtask only relays findings — no file edits, no commands. When
	// true the orchestrator skips the "Execute this plan?" confirmation.
	ReportOnly bool `json:"report_only"`
	// Redo and Spec are the planner's third exit, for a request that is more
	// work than one plan should carry (see specFromPlan): the ids of spec
	// items to rebuild when the project has a spec, or a spec to write when
	// it has none, with where it goes, where to build and with what.
	Redo    []string   `json:"redo"`
	Spec    []specFile `json:"spec"`
	SpecDir string     `json:"spec_dir"`
	OutDir  string     `json:"out_dir"`
	Target  string     `json:"target"`
	// Answer is the planner's user-facing prose when it answered a lookup
	// directly (report_only with no subtasks), given as an argument. It has
	// its own field because a server that forces the tool call (Halogen)
	// returns no message text beside it: a planner told "write the answer as
	// your message" wrote a full audit into its reasoning twice, and the user
	// saw "I couldn't produce a clear answer".
	Answer string `json:"answer"`
	// answer is the resolved answer: Answer when given, else the message
	// text beside the call (runPlanPhase sets it from the loop's separate
	// content channel), so orchestrate can surface it instead of silently
	// dropping a finished answer.
	answer string
}

// resolveAnswer fills plan.answer from the argument or, failing that, from the
// prose the model wrote beside the call (a free-text plan has it mashed with
// the JSON, so the JSON object is dropped out).
func (p *planResult) resolveAnswer(res toolLoopResult) {
	if a := strings.TrimSpace(p.Answer); a != "" {
		p.answer = a
		return
	}
	if res.RespondCalled {
		p.answer = strings.TrimSpace(res.Content)
	} else {
		p.answer = strings.TrimSpace(strings.Replace(res.Text, trimJSON(res.Text), "", 1))
	}
}

// runPlanPhase runs the planner and returns its plan, resolving any
// clarification round trip. It does not ask "Execute this plan?": the
// orchestrator owns that, so the user sees the whole subtask list first.
//
// A nil plan means "proceed without one": no PLAN.md, the loop failed, or the
// reply would not parse even after a corrective retry. errUserCancelled means
// the user aborted a clarification. replanContext is "" on the first pass; on a
// replan it is a short note, the failure detail being in history already.
func (a *agent) runPlanPhase(ctx context.Context, sid string, replanContext string) (*planResult, []ToolUse, error) {
	thinking := a.connFor("thinking")
	if thinking == nil {
		return nil, nil, fmt.Errorf("no [[llm]] in .codehalter/settings.toml")
	}
	// PLAN.md is in the system prompt now (the stable, cached prefix); here we
	// inject only the phase trigger + any replan context. An empty PLAN.md still
	// disables planning. Keeping the primer out of the history is what stops
	// re-plans from stacking 7 KB copies and re-bloating the context.
	if a.loadPromptFile(sid, "PLAN.md") == "" {
		return nil, nil, nil
	}
	marker := "Begin the PLANNING phase — produce the plan now (planning guidance is in the system prompt)."
	if replanContext != "" {
		marker = "Begin the PLANNING phase again.\n\n" + replanContext
	}

	sess := a.getSession(sid)
	if sess != nil {
		sess.AddUser(marker)
		sess.saveOrLog()
		// Only renderPlan clears this, and a plan phase that streamed its first row
		// and then errored or was cancelled never reaches it. Clearing here means
		// the only thing that can suppress this phase's list is this phase's own
		// table, never a leftover from the turn before.
		sess.phaseMu.Lock()
		sess.planTableShown = false
		sess.phaseMu.Unlock()
	}

	// Planning is read-only, enforced at dispatch: edit_file/write_file are
	// denied, since planner edits leak into history (`sed -i` cannot be blocked
	// here; PLAN.md forbids it in prose). Terminals: submit_plan, whose
	// arguments ARE the plan, or respond for a direct answer with no work.
	policy := phasePolicy{
		deny:      map[string]bool{"write_file": true, "edit_file": true},
		terminals: map[string]bool{submitPlanToolName: true, respondToolName: true},
	}

	// stream=false: planning output is machinery, and orchestrate renders the
	// result. The planner ends on submit_plan, so planRes.Text is clean JSON and
	// planRes.Content any answer prose. A model that emits the plan as free text
	// instead falls through to the parse below and one corrective retry.
	var plan planResult
	planRes, err := a.runToolLoop(ctx, sid, thinking, policy, "plan", false, 0)
	if err != nil {
		return nil, planRes.ToolUses, err
	}
	// respond as the plan terminal = a direct answer: the request needs no work
	// (a question, or already done), so there's no plan to execute. Surface it as
	// a report_only answer with no subtasks; orchestrate prints it and stops.
	if planRes.Terminal == respondToolName {
		// The respond turn's text is already in the session (the loop stored it +
		// respond's tool result); just surface the answer to orchestrate.
		return &planResult{Clear: true, ReportOnly: true, answer: strings.TrimSpace(planRes.Text)}, planRes.ToolUses, nil
	}
	parseErr := json.Unmarshal([]byte(trimJSON(planRes.Text)), &plan)
	if parseErr != nil {
		// Two failures land here and need different correctives: the planner
		// answered in prose without calling submit_plan, or it called it with
		// arguments that are not valid JSON. Both get the retry.
		corrective := "Call the `submit_plan` tool with your plan as its arguments. Do not reply in prose."
		wrong := "the planner replied in prose instead of calling submit_plan"
		if planRes.RespondCalled {
			corrective = fmt.Sprintf("Your `submit_plan` arguments were not valid JSON (%v). Call it again, emitting the arguments as one well-formed JSON object.", parseErr)
			wrong = fmt.Sprintf("submit_plan's arguments were not valid JSON (%v)", parseErr)
		}
		// Rows only stream once their object closes, so some may already be on
		// screen. Name the count: the user just watched a plan appear and has to
		// know it is not the one that will run.
		salvaged := ""
		var partial struct {
			Subtasks []subtask `json:"subtasks"`
		}
		if json.Unmarshal([]byte(repairJSON(planRes.Text)), &partial) == nil && len(partial.Subtasks) > 0 {
			salvaged = fmt.Sprintf(" %d subtask(s) already reached the table and are not final.", len(partial.Subtasks))
		}
		// Say it out loud. Planning runs with stream=false, so without this the
		// user watches the Planning row sit through an entire extra round trip
		// with no hint that anything went wrong.
		a.say(ctx, sid, fmt.Sprintf("\n⚠ Planning went wrong! %s.%s Asking the planner to try again.\n", wrong, salvaged))
		slog.Info("planner produced no parsable plan; retrying with corrective",
			"sid", sid, "calledSubmitPlan", planRes.RespondCalled, "err", parseErr, "snippet", truncate(planRes.Text, 200))
		// runToolLoop builds fresh from the session, so the files the planner read
		// this turn stay in front of it; the corrective rides as a trailing turn.
		retry, retryErr := a.runToolLoop(ctx, sid, thinking, policy, "plan", false, 0, corrective)
		planRes.Text = retry.Text
		planRes.Content = retry.Content
		planRes.RespondCalled = retry.RespondCalled
		planRes.ToolUses = append(planRes.ToolUses, retry.ToolUses...)
		planRes.DurationMs += retry.DurationMs
		if retryErr != nil {
			return nil, planRes.ToolUses, retryErr
		}
		parseErr = json.Unmarshal([]byte(trimJSON(planRes.Text)), &plan)
	}
	if parseErr != nil {
		a.say(ctx, sid, fmt.Sprintf("\n⚠ Planning failed! The planner could not produce a valid plan even after a corrective retry (%v). Nothing will run.\n", parseErr))
		return nil, planRes.ToolUses, fmt.Errorf("plan not valid JSON: %w", parseErr)
	}

	// Direct answer: the `answer` argument, else the prose beside the call.
	// orchestrate shows this when the plan has no subtasks.
	plan.resolveAnswer(planRes)

	// The planner must submit EITHER an answer (a message, no subtasks) OR a
	// plan, never both and never neither. Structural violations get ONE nudge to
	// pick a lane, then a re-parse. A "message" that is really a promise ("I'll
	// summarize") reads as an answer here; keeping the model off that is
	// PLAN.md's job, not a brittle string match.
	hasPlan := len(plan.Subtasks) > 0 || len(plan.Redo) > 0 || len(plan.Spec) > 0
	// Prose alongside subtasks is a PREAMBLE, not an answer: report_only=false
	// says the plan is meant to run, so there is nothing to choose between.
	// Nudging on it cost a round trip (4-50s) on 10 of 35 measured submissions, to
	// arrive at the plan already submitted. report_only=true is the real fork and
	// still nudges: those subtasks relay findings the message may already carry.
	hasAnswer := plan.answer != "" && (!hasPlan || plan.ReportOnly)
	if plan.Clear && hasPlan == hasAnswer {
		nudge := "You submitted neither a usable answer nor a plan: the `answer` argument and your message are empty, or only promise to act (\"I'll…\"). Your reasoning is never shown. Either call submit_plan again with the COMPLETE answer in its `answer` argument (report_only=true, no subtasks), OR submit subtasks that produce it. Never write \"I'll…\" / \"let me…\"."
		if hasPlan {
			nudge = "You submitted BOTH a final answer and a plan. Pick one: answer the user completely now (report_only=true, no subtasks), OR drop the message and submit only the subtasks to execute."
		}
		wrong := "submitted neither an answer nor any subtasks"
		if hasPlan {
			wrong = "submitted a final answer AND a plan at once"
		}
		a.say(ctx, sid, "\n⚠ The planner "+wrong+"! Asking it to pick one.\n")
		slog.Info("planner: ambiguous submission, nudging to pick one", "sid", sid, "hasPlan", hasPlan, "hasAnswer", hasAnswer)
		// runToolLoop builds fresh from the session (reads from this turn stay in
		// context) and stores the nudge as the trailing turn.
		retry, rerr := a.runToolLoop(ctx, sid, thinking, policy, "plan", false, 0, nudge)
		switch {
		case rerr != nil:
			// Recoverable: the ambiguous submission still stands and orchestrate can
			// work with it. Not silent, though, because what runs next is then the
			// thing that was just called ambiguous.
			a.say(ctx, sid, fmt.Sprintf("⚠ The corrective round failed (%v)! Going ahead with the planner's first submission.\n", rerr))
		default:
			planRes.ToolUses = append(planRes.ToolUses, retry.ToolUses...)
			var rp planResult
			switch {
			case retry.Terminal == respondToolName:
				plan = planResult{Clear: true, ReportOnly: true, answer: strings.TrimSpace(retry.Text)}
			case json.Unmarshal([]byte(trimJSON(retry.Text)), &rp) == nil:
				plan = rp
				plan.resolveAnswer(retry)
			default:
				a.say(ctx, sid, "⚠ The corrected reply was not valid JSON either! Going ahead with the planner's first submission.\n")
			}
		}
	}

	// The planner's turn (its prose + the submit_plan call/result) is already in
	// the session, stored verbatim by the loop — no post-hoc patch.

	toolUses := planRes.ToolUses
	if !plan.Clear && len(plan.Choices) > 0 {
		question := plan.Question
		if question == "" {
			question = "I'm not sure what you mean. Which of these?"
		}
		a.say(ctx, sid, question)

		// Inside a /spec loop under autopilot nobody is there to answer, and
		// taking the first option would let the model settle an open point of the
		// spec by itself. Park the question instead: the loop blocks the item
		// with it and moves on (spec_loop.go).
		if sess != nil && sess.specFence() != "" && a.isAutopilot() {
			appendAssistantNote(sess, "Question parked for the user by the spec loop: "+question)
			sess.saveOrLog()
			return nil, toolUses, &specQuestionError{Question: question, Choices: plan.Choices}
		}

		tcId := a.StartToolCall(ctx, sid, "Clarification needed", "think", nil)
		choice, err := a.askChoiceAuto(ctx, sid, tcId, question, plan.Choices)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + choice)})

		// err = the card's ctx was cancelled (the turn was stopped while the card
		// was open); choice=="abort" is the user answering the card with a stop.
		if err != nil {
			appendAssistantNote(sess, "Clarification cancelled.")
			if sess != nil {
				sess.saveOrLog()
			}
			return nil, toolUses, err
		}
		if choice == "abort" {
			appendAssistantNote(sess, "User aborted on clarification.")
			if sess != nil {
				sess.saveOrLog()
			}
			return nil, toolUses, errUserCancelled
		}

		appendAssistantNote(sess, "User chose: "+choice)
		if sess != nil {
			sess.saveOrLog()
		}
		a.say(ctx, sid, "Understood: "+choice+"\n")

		// Re-run the planner now that the user has clarified. The session already
		// carries "User chose: X" so the model sees the full context and should
		// produce subtasks. Without this, orchestrate receives the original empty
		// plan (subtasks=[]) and hits the "couldn't produce a plan" fallback.
		p, u, err := a.runPlanPhase(ctx, sid, replanContext)
		return p, append(toolUses, u...), err
	}

	return &plan, toolUses, nil
}

// ---------------------------------------------------------------------------
// Subtask loop: one bounded tool-calling pass that self-verifies
// ---------------------------------------------------------------------------

// subtaskOutcome captures what runExecutePhase produced. Success is true only
// when the model called respond AND no tool in the loop returned Failed=true.
// Reason summarises why a failed subtask failed — it feeds the orchestrator's
// replan context and the Jaccard duplicate-failure check.
type subtaskOutcome struct {
	Result  toolLoopResult
	Success bool
	Reason  string
	// Upsert is non-nil when the executor called submit_plan to revise the
	// remaining plan; the orchestrator adopts it and continues (completed work
	// stays, nothing cancelled). Success=false, Reason empty in this case.
	Upsert *planResult
}

// runExecutePhase runs one subtask as a single tool-calling loop. EXECUTE.md
// plus the subtask description and verify recipe open the loop; the
// executor runs with all execute tools (web tools excluded — those live in
// planning). The executor self-verifies via the recipe before calling
// respond. Bounded only by the hard maxToolLoopIterations backstop in this file.
func (a *agent) runExecutePhase(ctx context.Context, sid string, st subtask, idx, total int) subtaskOutcome {
	sess := a.getSession(sid)

	// EXECUTE.md is in the system prompt now; inject only the specific subtask +
	// verify recipe, not the full primer (which stacked in the history per subtask).
	var prompt strings.Builder
	fmt.Fprintf(&prompt, "EXECUTION phase — Task %d/%d\n\n%s\n", idx+1, total, st.Description)
	if len(st.Verify) > 0 {
		prompt.WriteString("\n## Verify recipe — run every entry via tools before calling respond\n\n")
		for i, v := range st.Verify {
			fmt.Fprintf(&prompt, "%d. %s\n", i+1, v)
		}
	}

	if sess != nil {
		sess.AddUser(prompt.String())
		sess.saveOrLog()
	}

	// Execute allows EVERY tool (web_search/web_read usable mid-edit). Terminals:
	// respond ends the subtask; submit_plan revises the remaining plan in place
	// (the orchestrator adopts it — see subtaskOutcome).
	policy := phasePolicy{terminals: map[string]bool{respondToolName: true, submitPlanToolName: true}}
	// Reasoning off for the whole subtask: withThinkingDisabled appends a closed
	// <think></think> for the model to continue, which suppresses it without
	// changing a single earlier token (see llm.go).
	// ... and every round must be a tool call: this phase ends only on a
	// terminal tool, so prose is always a slip (withToolChoice).
	conn := a.connFor("execute").withThinkingDisabled().withToolChoice("required")
	res, err := a.runToolLoop(ctx, sid, conn, policy, "execute", true, executeFailCap)
	// The executor's turns (prose + respond's call/result) are already in the
	// session, stored verbatim by the loop — no post-hoc patch.

	out := subtaskOutcome{Result: res}
	if err != nil {
		out.Reason = "executor error: " + err.Error()
		return out
	}
	// Plan-upsert: submit_plan's args ARE the revised plan (in res.Text). Hand it
	// up for the orchestrator to adopt; a malformed one falls through to failure.
	if res.Terminal == submitPlanToolName {
		var up planResult
		if err := json.Unmarshal([]byte(trimJSON(res.Text)), &up); err == nil && len(up.Subtasks) > 0 {
			out.Upsert = &up
			return out
		}
		out.Reason = "executor called submit_plan with no usable subtasks"
		return out
	}
	if !res.RespondCalled {
		out.Reason = "executor exited without calling respond"
		return out
	}
	// Exit-code authority, last state only. Failed flags override the model's
	// own verdict, since small models declare success over a non-zero exit. But
	// only the LAST run of each distinct call (name + arguments) counts: a verify
	// command that failed, got fixed and re-ran green must not condemn the
	// subtask.
	type callKey struct{ name, input string }
	lastFailed := map[callKey]bool{}
	var order []callKey
	for _, u := range res.ToolUses {
		k := callKey{u.Name, u.Input}
		if _, seen := lastFailed[k]; !seen {
			order = append(order, k)
		}
		lastFailed[k] = u.Failed
	}
	var failedNames []string
	for _, k := range order {
		// Only run_command exit codes are verdicts. Other tools set Failed to feed
		// the fail cap without being one (an edit_file usage error is recovered by
		// a later edit or caught by the verify recipe; view_image on a missing
		// image is benign), and condemning the subtask on those replans for nothing.
		if lastFailed[k] && k.name == "run_command" {
			failedNames = append(failedNames, k.name)
		}
	}
	if len(failedNames) > 0 {
		out.Reason = "failed tools: " + strings.Join(failedNames, ", ")
		return out
	}
	out.Success = true
	return out
}

// ---------------------------------------------------------------------------
// Document phase (runs once at end of a successful prompt)
// ---------------------------------------------------------------------------

// runDocumentPhase runs once every subtask has succeeded. It is part of the
// FOREGROUND turn: same connection as execute and the full conversation, so it
// reuses execute's warm prefix and sees the actual edits rather than a digest.
// DOCUMENT.md self-skips when no documentation update is warranted.
func (a *agent) runDocumentPhase(ctx context.Context, sid string, exec toolLoopResult) (toolLoopResult, error) {
	docPrompt := a.loadPromptFile(sid, "DOCUMENT.md")
	if docPrompt == "" {
		return exec, nil
	}

	sess := a.getSession(sid)
	if sess == nil {
		return exec, nil
	}

	// Documentation is part of the FOREGROUND turn (plan → execute → document),
	// not background work — only the summariser and git-commit drafter belong on
	// the background LLM. Run it on the SAME connection as execute so it reuses
	// execute's warm KV prefix instead of cold-prefilling a separate slot.
	// Reasoning off, like the executor it follows (see withThinkingDisabled).
	conn := a.connFor("execute").withThinkingDisabled()
	if conn == nil {
		return exec, nil
	}

	// The doc instruction lands in the session as the trailing user turn, so
	// buildLLMContext hands the documenter the FULL turn — the edits the executor
	// actually made, not a lossy digest — continuing the cached lineage.
	sess.AddUser(docPrompt)
	sess.saveOrLog()

	// Blank line before the documenter streams, so its output (often just "No
	// documentation change needed.") starts a fresh markdown paragraph instead
	// of running into the executor's final sentence.
	a.say(ctx, sid, "\n\n")
	// Documentation wraps up a finished turn: deny submit_plan (no looping back
	// to planning). respond is a terminal here too: the model writes docs and
	// then calls respond to signal "documentation complete" — the same intent as
	// in execute phase. Without this, the loop sees respond as a normal tool
	// call, continues, and the model calls respond again → stuck-repetition.
	docPolicy := phasePolicy{
		deny:      map[string]bool{submitPlanToolName: true},
		terminals: map[string]bool{respondToolName: true},
	}
	docRes, err := a.runToolLoop(ctx, sid, conn, docPolicy, "document", true, 0)
	if err != nil {
		slog.Warn("document phase failed", "err", err)
		return exec, nil
	}
	// The document turn is already in the session, stored verbatim by the loop.
	exec.ToolUses = append(exec.ToolUses, docRes.ToolUses...)
	return exec, nil
}

// planRoundNudge is the planning round after which the planner is told to
// submit. It sits above the median plan (16 rounds) and below the runaways
// (32); an executor that reads for itself makes what is left cheap.
const planRoundNudge = 20

// maxToolLoopIterations is runToolLoop's hard runaway backstop: a model
// emitting "different enough" tool calls forever can't spin past it. One
// iteration is one LLM round-trip; a complex execute pass is usually 10-20, so
// 100 leaves headroom while still bailing genuine runaways. The repetition
// ladder catches the common stuck patterns earlier.
const maxToolLoopIterations = 100

// keepSmallTurnTokens caps the verbatim keep-window of completed small turns the
// 400 recovery preserves, sized by the server's real prompt_tokens (not an
// estimate). The unfinished small turn is always kept on top; everything older
// folds into Summary. See Session.keepWindowStart.
const keepSmallTurnTokens = 10_000

// reasoningNudgeBytes is the "the model clearly worked" bar: more reasoning than
// this with an EMPTY visible message means the answer is stuck in the (never-
// shown) reasoning channel, so the loop nudges it to write the answer as text.
const reasoningNudgeBytes = 512

// A mid-response connection drop (server/router closed the stream) is usually
// momentary, so the recovery loop re-sends the same request a few times with a
// short backoff before surfacing a clear message instead of a raw EOF.
const (
	maxTransientStreamRetries = 5
	transientStreamBackoff    = 3 * time.Second
)

// maxStreamRuleRetries caps how many times one round re-asks after a stream
// rule aborted the generation (see rules.go). Two: the first retry carries the
// reminder, the second is the benefit of the doubt. A model still emitting the
// same off-format output after both is not going to be corrected by a third
// reminder, and the failure is more useful surfaced to the replan machinery
// than spun on here.
const maxStreamRuleRetries = 2

// streamFlushInterval batches streamed model tokens to the editor at most this
// often. Per-token sendUpdate is fine at ~45 tg/s, but at higher rates (e.g. 450
// tg/s) it floods: each call takes the one conn write lock, and a slow editor
// would backpressure the SSE read and stall the LLM call (the same failure run_command
// had). Batching keeps editor updates at ~1/interval regardless of generation speed.
const streamFlushInterval = 200 * time.Millisecond

// throttledStream returns a per-token sink that accumulates tokens and emits an
// EMITTED chunk at most once per streamFlushInterval (driven by token arrivals;
// the first token emits immediately), plus a flush that emits whatever is left
// (call it when the stream ends). Single-goroutine by construction: the SSE loop
// drives the sink, the caller drives flush after llmStream returns, so no lock.
func throttledStream(emit func(string)) (sink func(string), flush func()) {
	var buf strings.Builder
	var lastSent time.Time
	send := func() {
		if buf.Len() == 0 {
			return
		}
		emit(buf.String())
		buf.Reset()
		lastSent = time.Now()
	}
	sink = func(token string) {
		buf.WriteString(token)
		if time.Since(lastSent) >= streamFlushInterval {
			send()
		}
	}
	return sink, send
}

// executeFailCap is the per-subtask budget of FAILED rounds. Successful work
// is uncounted, so a long productive subtask runs freely and only one that
// keeps failing burns the budget. A healthy fix cycle costs 1-2 (red, fix,
// green); past this the loop is stuck and bounces to a replan, where web tools
// and a fresh decomposition are available. Small here, generous maxReplans.
const executeFailCap = 8

// stuckEscalateRounds / stuckBailRounds drive the repetition ladder. A stuck
// round is one whose every tool call reproduced output it already produced.
// Consecutive stuck rounds first warm the sampler (execute→thinking, same
// server, so the prefix stays warm), then bail. Escalate strictly below bail,
// so the warmer sampler always gets a turn.
const (
	stuckEscalateRounds = 3
	stuckBailRounds     = 5
)

// noCallNudges is how many times a phase that ends on a terminal tool re-asks
// a model that answered in prose, with escalating wording: a reminder, an
// instruction, then one imperative line naming the tools. One polite nudge
// left weaker models answering in prose again; past three the model will not
// call it, and spinning costs more than taking the prose.
const noCallNudges = 3

// toolLoopResult is what an agentic tool loop (runToolLoop) returns.
type toolLoopResult struct {
	Text string
	// Content is the model's accumulated free-text (assistant content) across
	// the loop, separate from Text. For a terminal exit these diverge: Text is
	// the terminal tool's output (respond's message, or submit_plan's plan
	// JSON), while Content is whatever prose the model wrote alongside the tool
	// call. runPlanPhase relies on this split — submit_plan's args land in Text,
	// the planner's user-facing answer lands here — so the two never mix.
	Content  string
	ToolUses []ToolUse
	// RespondCalled is true when the loop exited because the model invoked
	// the registered terminal tool (typically `respond`). False when the
	// loop exited via the legacy empty-tool-calls path, hit the failed-round
	// soft cap, or returned an error. Subtask runners use this as the primary
	// success signal — a loop that ran out of turns without calling
	// `respond` is a failed subtask regardless of what's in res.Text.
	RespondCalled bool
	// Terminal is which terminal tool ended the loop ("" if none — legacy exit,
	// fail cap, or error). A phase can expose several terminals (execute exits on
	// respond OR submit_plan), so callers branch on this to tell a finished
	// subtask (respond) from a plan-upsert (submit_plan). RespondCalled stays
	// true for ANY terminal — it's the generic "loop reached a terminal" signal.
	Terminal string
	// StartedAt is when the first llmStream call of this loop began.
	// StartedAt/DurationMs are the loop's own LLM-call timing (DurationMs is
	// cumulative across iterations and excludes tool execution). runToolLoop
	// stamps them onto the trailing assistant message via MarkLastAssistantTiming
	// before returning — callers don't read them.
	StartedAt  time.Time
	DurationMs int64
}

// runToolLoop is the agentic loop every phase uses: call the model, run its
// tool calls, repeat. stream forwards tokens to the UI (execute, document) or
// keeps them silent (the planner's JSON pass). failSoftCap > 0 ends the loop
// after that many FAILED rounds with RespondCalled=false, so the subtask
// bounces to a replan; 0 leaves only maxToolLoopIterations.
//
// The context is built FRESH from the session each call, never from a caller's
// snapshot, which is stale the moment the loop stores anything. corrective is a
// caller's retry turn, STORED before the rebuild (see addCorrective).
// runToolLoopSeeded is the loop itself, taking the context explicitly.
func (a *agent) runToolLoop(ctx context.Context, sid string, conn *LLMConnection, policy phasePolicy, phase string, stream bool, failSoftCap int, corrective ...string) (toolLoopResult, error) {
	var messages []llmMessage
	if sess := a.getSession(sid); sess != nil {
		for _, c := range corrective {
			sess.AddUser(c)
		}
		if len(corrective) > 0 {
			sess.saveOrLog()
		}
		messages = a.buildLLMContext(sess)
	} else {
		// No session to store into (probe paths): nothing will ever rebuild this
		// context, so the wire is the only place the corrective can live.
		for _, c := range corrective {
			messages = append(messages, llmMessage{Role: "user", Content: c})
		}
	}
	return a.runToolLoopSeeded(ctx, sid, conn, messages, policy, phase, stream, failSoftCap)
}

// addCorrective puts a corrective turn on the wire AND stores it. One rule, no
// exceptions: what goes on the wire is what is stored.
//
// A wire-only turn looks like a free suffix, but runToolLoop rebuilds the
// context from the session on every entry, and an unstored turn then vanishes
// from the MIDDLE of history, shifting everything after it. Measured: a nudge
// that vanished at index 8 of 23 re-evaluated 9998 of 15346 tokens, 19s. The
// accepted cost is a few hundred bytes of context, replayed on session/load.
func (a *agent) addCorrective(sid string, messages []llmMessage, text string) []llmMessage {
	if sess := a.getSession(sid); sess != nil {
		sess.AddUser(text)
		sess.saveOrLog()
	}
	return append(messages, llmMessage{Role: "user", Content: text})
}

// startToolMeter shows "(running run_command go build ./...… 12s)" while a
// tool runs, so a slow tool reads as busy rather than frozen. The identifying
// argument rides along: "run_command" at 77s says something is slow but not
// WHAT, and the arguments have scrolled away by then.
func (a *agent) startToolMeter(ctx context.Context, sid string, tc toolCall) (stop func()) {
	label := tc.Function.Name
	// Tried in priority order: a tool can carry several of these (a search has
	// both a query and a path) and only one fits the row. Whitespace runs collapse
	// to single spaces because a heredoc or a multi-line command would otherwise
	// break the row apart, and the cut is on runes so it can't split one in half.
	var args map[string]any
	if json.Unmarshal([]byte(tc.Function.Arguments), &args) == nil {
		for _, key := range []string{"command", "query", "path", "id"} {
			v, _ := args[key].(string)
			if v = strings.Join(strings.Fields(v), " "); v == "" {
				continue
			}
			if r := []rune(v); len(r) > toolMeterArgRunes {
				v = string(r[:toolMeterArgRunes])
			}
			label += " " + v
			break
		}
	}
	start := time.Now()
	a.setStatus(ctx, sid, " (running "+label+"…)") // immediate, before the first tick
	return a.startStatusMeter(ctx, sid, func() string {
		return fmt.Sprintf(" (running %s… %ds)", label, int(time.Since(start).Seconds()))
	})
}

// toolMeterArgRunes caps the argument shown in the status row. The row is one
// line in a plan entry: past this the phase name and the seconds counter get
// pushed out of view, which costs more than the tail of a long command is worth.
// The trailing "…" the meter already prints doubles as the cut marker.
const toolMeterArgRunes = 48

// toolLoopCaller holds everything a tool loop needs to make ONE round's LLM
// call, so the per-round entry point takes only what changes between rounds.
// Built once per runToolLoopSeeded; conn is swapped in place by the repetition
// ladder's sampler escalation, and stalled latches in place by the <think>
// recovery below.
type toolLoopCaller struct {
	a     *agent
	sid   string
	phase string
	conn  *LLMConnection
	tools []map[string]any
	// on/think are the UI sinks (nil for a silent internal pass); flush emits
	// whatever the throttled sinks still hold.
	on, think func(string)
	flush     func()
	// stalled latches once the model burns a whole budget looping in <think>:
	// every later round then starts with thinking off, so a chronically-stalling
	// model can't re-waste ~max_tokens of reasoning each round. Scoped to one
	// tool loop — the next phase/subtask re-enables thinking.
	stalled bool
}

// round makes one LLM call with the whole recovery ladder around it: the
// stuck-<think> swap, the stream-rule re-ask, the two max_tokens rungs, the
// transient-drop retries and the context-full fold. It returns the messages
// too, since a corrective turn or a fold rewrites them. Its own function
// because the ladder is a state machine over one call with five independent
// latches, which inline buried the loop's shape (call, run tools, repeat).
func (c *toolLoopCaller) round(ctx context.Context, messages []llmMessage) (string, []toolCall, string, []llmMessage, error) {
	a, sid := c.a, c.sid
	var text, reasoning string
	var calls []toolCall
	var err error
	// On a context-overflow 400 (ground truth from the server), escalate the fold
	// and retry: step 1 keeps the unfinished small turn plus the most recent
	// ~keepSmallTurnTokens of completed small turns (folding everything older,
	// including the rest of the in-flight large turn); if that still 400s, step 2
	// keeps ONLY the unfinished small turn. Each step strictly shrinks the
	// context, so it terminates: out of steps, the 400 surfaces.
	recoverStep := 0
	recoverKeepFrom := []func(*Session) int{
		func(s *Session) int { return s.keepWindowStart(keepSmallTurnTokens) },
		(*Session).lastAssistantIndex,
	}
	// Arm the stream-rule check for this round. The tool loop is the only
	// caller that does: it owns the retry ladder below, which is what makes a
	// mid-generation abort recoverable rather than just a failed call.
	callConn := c.conn.forToolLoop() // a <think> stall retries (and latches) on a thinking-off copy
	if c.stalled {
		callConn = callConn.withThinkingDisabled()
	}
	thinkingRetried := false // at most one such retry per round
	capNudged := false       // cap ladder rung 1: one be-concise nudge retry per round
	capDoubled := false      // cap ladder rung 2: one doubled-max_tokens retry per round
	transientRetries := 0    // mid-response drops retried up to maxTransientStreamRetries
	ruleRetries := 0         // stream-rule aborts re-asked up to maxStreamRuleRetries
	for {
		// The plan-table sink is built fresh per attempt: a retry re-sends the
		// request from scratch, so the aborted attempt's partial arguments must
		// not carry into the next one. Rows already on screen stay there (the
		// same convention as an aborted stream-rule response) and the retry
		// simply renders a second table.
		text, calls, reasoning, err = a.llmStream(ctx, sid, callConn, messages, c.tools, c.on, c.think, a.planTableSink(ctx, sid))
		c.flush() // emit any batched tail of this call's tokens to the UI
		if err == nil {
			return text, calls, reasoning, messages, nil
		}
		// Stuck in <think>: the model burned the whole budget on reasoning with
		// no content and no tool calls. Retry once on a thinking-off copy so it
		// must answer directly, and latch it for the rest of the run so it can't
		// re-burn the budget next round. Any phase/depth — swaps the conn, no
		// history fold needed.
		if errors.Is(err, errStuckThinking) && !thinkingRetried {
			thinkingRetried = true
			c.stalled = true
			// withThinkingDisabled appends a closed <think></think> rather than
			// re-rendering, so the prefix cache survives the retry and the return to
			// normal, and no cache-lineage reset is needed. The kwargs route cost
			// one session 99614 tokens re-evaluated across the two switches.
			callConn = callConn.withThinkingDisabled()
			a.logSession(sid, "RECOVER", "model stuck in <think>: continuing a closed <think></think> for the rest of this tool loop")
			continue
		}
		// A stream rule fired and the generation was abandoned, so there is no
		// partial to salvage. Re-ask with the rule's reminder as a stored user turn
		// (addCorrective). Capped: a model that ignores the reminder twice will not
		// be talked out of it, and the replan machinery beats spinning here.
		if sr := asStreamRule(err); sr != nil {
			if ruleRetries >= maxStreamRuleRetries {
				a.logSession(sid, "RECOVER", "stream rule %q fired %d times — giving up on the nudge, letting the turn fail", sr.Rule, ruleRetries+1)
				break
			}
			ruleRetries++
			a.logSession(sid, "RECOVER", "stream rule %q fired — aborted mid-generation, re-asking with the reminder (%d/%d). Matched: %s",
				sr.Rule, ruleRetries, maxStreamRuleRetries, truncate(sr.Matched, 200))
			if sid != "" {
				// The partial is already on screen (llmStream streams before it
				// checks), so say what happened to it — otherwise the retry reads as
				// the model repeating itself.
				a.say(ctx, sid, "\n⟲ Response went off-format and was discarded; re-asking.\n")
			}
			// The partial generation was discarded, so the model must be told
			// that — otherwise a model that had already written half an answer
			// tends to continue from where it thinks it left off.
			messages = a.addCorrective(sid, messages, strings.TrimSpace(sr.Reminder)+
				"\n\nYour previous response was cut off at that point and DISCARDED — none of it was applied and it is not part of this conversation. Start the response over.")
			continue
		}
		// Cap ladder: the generation died AT the max_tokens cap. The partial is
		// discarded (truncated tool-call JSON cannot be resumed). Rung 1: retry with
		// a be-concise nudge, the cheapest. Rung 2: retry once on a doubled cap, a
		// sampler-side change, so the prefix survives. A third hit is an ordinary
		// failure for the replan machinery.
		if ce := asCapHit(err); ce != nil {
			if !capNudged {
				capNudged = true
				a.logSession(sid, "RECOVER", "generation hit the max_tokens cap (%d) — retrying with a be-concise nudge", ce.Cap)
				if sid != "" {
					a.say(ctx, sid, "⚠ Reply hit the output-token cap; retrying with a be-concise instruction.\n")
				}
				messages = a.addCorrective(sid, messages, fmt.Sprintf(
					"Your previous response was cut off at the %d-token output limit and was DISCARDED — nothing of it was applied. Respond again, keeping the output well under that limit: be concise. If you are writing a large file, write it in parts: write_file with the first part, then extend it with edit_file.", ce.Cap))
				continue
			}
			if !capDoubled {
				capDoubled = true
				base := ce.Cap
				if base <= 0 {
					base = defaultMaxTokens
				}
				callConn = callConn.withMaxTokens(base * 2)
				a.logSession(sid, "RECOVER", "still at the cap after the nudge — one retry with max_tokens=%d", base*2)
				if sid != "" {
					a.say(ctx, sid, fmt.Sprintf("⚠ Still at the cap; retrying once with max_tokens=%d.\n", base*2))
				}
				continue
			}
			break // both rungs spent — surface as a normal failure (replan)
		}
		// Mid-response connection drop (EOF/reset — server or router closed the
		// stream), distinct from a deliberate cancel. Usually momentary: re-send
		// a few times, then surface a clear, actionable message instead of the
		// raw EOF so the user knows it's transient.
		if isTransientStreamError(err) {
			if transientRetries >= maxTransientStreamRetries {
				err = fmt.Errorf("lost the connection to the LLM mid-response %d times (the server or router dropped the stream); this is usually transient — try again in a moment", transientRetries+1)
				break
			}
			transientRetries++
			a.logSession(sid, "RECOVER", "stream dropped mid-response (%v) — retry %d/%d", err, transientRetries, maxTransientStreamRetries)
			if sid != "" {
				// The dropped attempt's partial tokens are already on screen and the
				// retry re-streams from the top; flag it so the repeated prefix reads
				// as a reconnect, not a glitch (append-only streaming can't rewind).
				a.say(ctx, sid, "\n⟲ Connection dropped mid-response; reconnecting.\n")
			}
			select {
			case <-time.After(transientStreamBackoff):
				continue
			case <-ctx.Done():
				err = ctx.Err() // cancelled during backoff → falls through to the cancel path
			}
		}
		if !isContextFull(err) || // 400 reject OR n_ctx-ceiling truncation
			(c.phase != "plan" && c.phase != "execute") || sid == "" {
			break
		}
		s := a.getSession(sid)
		if s == nil {
			break
		}
		// Advance through fold steps until one frees something.
		folded := false
		for recoverStep < len(recoverKeepFrom) {
			keepFrom := recoverKeepFrom[recoverStep](s)
			recoverStep++
			if a.foldHistory(ctx, s, keepFrom) {
				folded = true
				break
			}
		}
		if !folded {
			break
		}
		messages = a.buildLLMContext(s)
	}
	return text, calls, reasoning, messages, err
}

// repetitionTracker answers one question per tool call: did this exact
// (name,args) call reproduce output it already produced in this loop? An
// interleaved revisit counts, not just a consecutive one, which is why it is a
// map and not a last-call comparison.
type repetitionTracker struct {
	// hash is the last output hash of each call, bag the same output as a word
	// bag: a re-issued call whose output is ≥ stuckOutputSimilarity
	// Jaccard-similar to its previous output also counts as reproduced. That
	// catches the loops the hash misses — a re-run failing build with a
	// timestamp in its output.
	hash map[string]uint64
	bag  map[string]map[string]bool
	// writes counts the file-changing calls so far; writeAt is that count
	// when each call last ran. A successful command re-run is a legitimate
	// re-verify only when something was written since its last run; the same
	// command again with nothing changed, alone or alternating with another
	// probe, is a spin.
	writes  int
	writeAt map[string]int
}

// changesFiles reports a call that may have changed the tree: the file tools,
// and a shell line with an in-place writer in it.
func changesFiles(tc toolCall) bool {
	switch tc.Function.Name {
	case "edit_file", "write_file":
		return true
	case "run_command":
		return inPlaceWriterRe.MatchString(tc.Function.Arguments)
	}
	return false
}

// inPlaceWriterRe: shell commands that rewrite files, so a green re-run after
// one of them is a re-verify, not a repeat.
var inPlaceWriterRe = regexp.MustCompile(`sed -i|-i\b.*\bsed|cargo fmt|gofmt -w|prettier --write|go mod tidy|git (checkout|stash|apply|revert|reset)|cargo add|npm i|apk add|apt-get install|>>? *[^&\s]`)

// sawAgain records this call's output and reports whether it made no progress.
func (rt *repetitionTracker) sawAgain(tc toolCall, tu ToolUse) bool {
	key := tc.Function.Name + "\x00" + tc.Function.Arguments
	h := fnvHash(tu.Output)
	bag := issueBag([]string{tu.Output})
	repeated := false
	if prev, ok := rt.hash[key]; ok && prev == h {
		repeated = true
	} else if prevBag, ok := rt.bag[key]; ok && jaccard(prevBag, bag) >= stuckOutputSimilarity {
		// Not byte-identical but near-identical in content — a re-run whose
		// output only differs in noise (timestamp, duration, pid) made no more
		// progress than an exact repeat.
		repeated = true
	}
	rt.hash[key] = h
	rt.bag[key] = bag
	// A SUCCESSFUL run_command re-run with identical output is a
	// legitimate re-verify after an edit (re-running just:build / just:test to
	// confirm a change held), NOT spinning — don't count it as no-progress. A
	// FAILED re-run still counts: the model IS stuck on a red build/test. And
	// so does a successful one with no file changed since its last run:
	// it verified nothing. Without this an executor ran one `ls; grep` line
	// 75 times in a row, then two `grep` lines in alternation 73 times, and
	// hit the iteration cap both times, because every one of them exited 0.
	if rt.writeAt == nil {
		rt.writeAt = map[string]int{}
	}
	if changesFiles(tc) {
		rt.writes++
	}
	changedSince := rt.writeAt[key] < rt.writes
	rt.writeAt[key] = rt.writes
	if repeated && !tu.Failed && tc.Function.Name == "run_command" && changedSince {
		repeated = false
	}
	// read_file/continue_read also honour the content-dedup marker —
	// serveRead prepends a note on a repeat, which would otherwise defeat the
	// hash on the first re-read.
	if (tc.Function.Name == "read_file" || tc.Function.Name == "continue_read") &&
		strings.Contains(tu.Output, readUnchangedMarker) {
		repeated = true
	}
	return repeated
}

// announceToolCall shows what the tool was actually asked to do. Only its NAME
// reaches the status ticker ("running run_command… 94s"), so without this a long
// tool ran with nothing on screen saying which command it was, and no ACP
// tool-call update carries it either.
func (a *agent) announceToolCall(ctx context.Context, sid string, tc toolCall) {
	// Most tools already show their arguments: the card is titled with them
	// ("Run: <command>", "Reading: <path>"), file tools add the diff, submit_plan
	// streams a table. The JSON would repeat that, escaped onto one line. It is
	// kept only where nothing else shows them: MCP tools and the card-less few.
	switch name := tc.Function.Name; {
	case strings.Contains(name, "__"),
		name == "view_image", name == "session_insights":
	default:
		return
	}
	shown := tc.Function.Arguments
	var pretty bytes.Buffer
	if json.Indent(&pretty, []byte(shown), "", "  ") == nil {
		shown = pretty.String()
	} // arguments too malformed to indent are shown raw, never dropped
	if shown == "" || shown == "{}" {
		a.say(ctx, sid, "\n**"+tc.Function.Name+"**\n")
		return
	}
	// A fence longer than any backtick run inside it: an argument carrying ```
	// would otherwise close the block early and spill the rest of the JSON into
	// the transcript as prose.
	fence := "```"
	for strings.Contains(shown, fence) {
		fence += "`"
	}
	a.say(ctx, sid, fmt.Sprintf("\n**%s**\n%sjson\n%s\n%s\n", tc.Function.Name, fence, shown, fence))
}

// runToolLoopSeeded runs the agentic loop with an EXPLICIT initial context.
// runToolLoop is its one production caller (tests drive it directly with a
// hand-built context). The failSoftCap / loop-exit semantics in the doc above
// apply here too.
func (a *agent) runToolLoopSeeded(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, policy phasePolicy, phase string, stream bool, failSoftCap int) (toolLoopResult, error) {
	// `stream` gates the TEXT channel only. Reasoning always streams when there
	// is a session to stream to: it is the sole live signal during a long call,
	// and it is never the machinery `stream=false` exists to hide — the planner's
	// output arrives as submit_plan arguments and text, not on the reasoning
	// channel. Silencing both left the planner showing a ↓ counter climbing past
	// 8k tokens with nothing on screen. nil callbacks are no-ops in the loop below.
	var on, think func(string)
	flushStream := func() {} // no-op unless streaming; flushes the batched tail
	if sid != "" {
		var flushOn, flushThink func()
		if stream {
			on, flushOn = throttledStream(func(chunk string) {
				// Straight to the client, not through say: the model's text
				// is logged whole in its RESPONSE block, not chunk by chunk.
				a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: chunk}})
			})
		}
		think, flushThink = throttledStream(func(chunk string) {
			// The reasoning channel, which clients render collapsed/dimmed
			// rather than as an answer.
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentThought, Content: ContentBlock{Type: "text", Text: chunk}})
		})
		flushStream = func() {
			if flushOn != nil { // nil when stream=false: no text channel to flush
				flushOn()
			}
			flushThink()
		}
	}
	tools := a.tools.defs()

	// Terminals come from the phase policy, not the tool array (which is the full
	// superset now). When the phase has any terminal, the empty-tool-calls branch
	// below stops meaning "model finished" and starts meaning "model dropped out
	// of tool-calling grammar" — see the nudge + fallback there.
	hasTerminal := len(policy.terminals) > 0
	termList := terminalList(policy)
	// loopStart bounds which background jobs can park this loop: only the
	// ones its own run_command calls handed over (see parkForJobs).
	loopStart := time.Now()

	var res toolLoopResult
	var allText strings.Builder
	var genElapsed time.Duration
	// stampTiming applies the accumulated start/duration/phase to the
	// trailing assistant message in session. Called on every exit path so
	// even error returns leave a recorded turn for postmortem analysis.
	stampTiming := func() {
		// Capture the accumulated free-text on every exit path so callers that
		// need the prose separate from a terminal tool's output (runPlanPhase)
		// always see it. Independent of the timing stamp below.
		res.Content = allText.String()
		if res.StartedAt.IsZero() {
			return
		}
		res.DurationMs = genElapsed.Milliseconds()
		if sess := a.getSession(sid); sess != nil {
			sess.MarkLastAssistantTiming(res.StartedAt, res.DurationMs, phase)
		}
	}

	// Repetition ladder state; repetitionTracker owns what counts as a repeat. A
	// round whose every call reproduced known output is "stuck"; consecutive
	// stuck rounds climb one ladder — corrective nudge each round, warm the
	// sampler at stuckEscalateRounds, bail at stuckBailRounds — and any
	// productive round resets the streak, so read-after-write and genuine fan-out
	// are never punished.
	repeats := &repetitionTracker{hash: map[string]uint64{}, bag: map[string]map[string]bool{}}
	var stuckRounds int
	var nudgedUI bool // the "repeating" UI warning fires only once
	var escalated bool
	// noCallNudged counts the empty tool-call lists this loop has re-asked (see
	// noCallNudges); once the budget is spent the next one falls through to the
	// text exit, so a model that refuses the terminal can't spin forever.
	var noCallNudged int
	// failedRounds counts iterations whose tool batch produced a failure; the
	// failSoftCap check below bounces the loop once it accumulates too many.
	var failedRounds int
	// Everything one round's LLM call needs, built once. Its conn is swapped in
	// place by the sampler escalation at the bottom of this loop.
	caller := &toolLoopCaller{a: a, sid: sid, phase: phase, conn: conn, tools: tools, on: on, think: think, flush: flushStream}
	for iter := 0; ; iter++ {
		if iter >= maxToolLoopIterations {
			res.Text = allText.String()
			stampTiming()
			return res, fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations)
		}
		// What the user typed while this turn was running, taken between rounds
		// and handed over as a plain user message: an append, so the prefix
		// cache is untouched and the model reads it as part of the conversation
		// rather than as a system correction.
		if sess := a.getSession(sid); sess != nil {
			if queued := sess.takeSteer(); len(queued) > 0 {
				joined := strings.Join(queued, "\n\n")
				messages = a.addCorrective(sid, messages, joined)
				a.say(ctx, sid, "\n↪ picked up: "+firstLine(joined)+"\n")
			}
			// A background job that finished during this turn is handed over
			// the same way, before the next model call. run_background promises
			// "do other work, the result comes back on its own"; until this it
			// only came back once the whole turn had ended, so a model that
			// needed the result to finish its subtask had no way to get it but
			// polling, and it polled with `sleep 150` in a foreground command,
			// worse than having run the job in the foreground to begin with.
			if notes := sess.takeBgNotes(); len(notes) > 0 {
				var full []string
				for _, n := range notes {
					a.say(ctx, sid, "\n🔔 "+n.line+"\n")
					full = append(full, n.full)
				}
				messages = a.addCorrective(sid, messages, strings.Join(full, "\n\n"))
			}
		}
		// Planning reasons for ~900 tokens a round, so a plan that keeps
		// exploring is the most expensive thing codehalter does: the median plan
		// took 16 rounds, the longest 32 rounds and 86k tokens. Past the nudge
		// the planner is told to submit with what it has; the executor reads
		// cheaply what the planner did not.
		if phase == "plan" && iter == planRoundNudge {
			messages = a.addCorrective(sid, messages, fmt.Sprintf("You have gathered for %d rounds. Call `submit_plan` NOW with what you have: name the files and the approach, and say in each subtask what you did not verify; the executor reads and checks cheaply.", iter))
			a.say(ctx, sid, "\n⏱ Planning past "+fmt.Sprint(planRoundNudge)+" rounds: asked to submit with what it has.\n")
		}

		streamStart := time.Now()
		if res.StartedAt.IsZero() {
			res.StartedAt = streamStart
		}
		// Call the model, with the whole recovery ladder around it (see
		// toolLoopCaller.round). messages comes back rewritten when the ladder
		// added a corrective turn or folded history to make the context fit.
		var text, reasoning string
		var calls []toolCall
		var err error
		text, calls, reasoning, messages, err = caller.round(ctx, messages)
		genElapsed += time.Since(streamStart)
		if err != nil {
			res.Text = allText.String()
			stampTiming()
			return res, err
		}
		allText.WriteString(text)

		// Persist this turn's assistant text verbatim (a fresh message per
		// iteration, not a merged turn) so a session replay reproduces the wire —
		// cache consistency, not readability. AppendToolUse attaches the tool uses.
		if s := a.getSession(sid); s != nil {
			s.AddAssistant(text)
			s.recordLastPromptTokens() // stamp the call's prompt_tokens for keepWindowStart
			s.saveOrLog()
		}

		if len(calls) == 0 {
			// No tool call. Re-ask, in escalating wording (noCallNudges), before
			// accepting the text exit, for either model slip:
			//   (a) reasoned a lot but emitted EMPTY content — the answer is stuck
			//       in the never-shown reasoning channel ("calculated a lot, then
			//       nothing"); tell it to write the answer as plain text.
			//   (b) terminal mode: it dropped out of tool-calling without finishing.
			reasonedButSilent := text == "" && len(reasoning) > reasoningNudgeBytes
			if noCallNudged < noCallNudges && (reasonedButSilent || hasTerminal) {
				noCallNudged++
				var nudge string
				switch {
				case reasonedButSilent:
					nudge = "You produced a lot of reasoning but no visible output — your reasoning/thinking is NEVER shown to the user. Write your result now as plain text; this message is what they read."
					if hasTerminal {
						nudge += fmt.Sprintf(" If you're done, put it in %s.", termList)
					}
				case noCallNudged == 1:
					nudge = fmt.Sprintf("Your last response was plain text with no tool call. "+
						"This turn ends only when you call a terminal tool (%s), or "+
						"another tool if you still have work to do. Do not reply in "+
						"prose — call a tool.", termList)
				case noCallNudged == 2:
					// Shorter and imperative: the first wording did not land, and a
					// longer explanation of the same thing reads as more prose to
					// answer in kind.
					nudge = fmt.Sprintf("Prose again. Nothing you write outside a tool call counts. "+
						"Call %s now if the work is done, or the tool that does the next step.", termList)
				default:
					nudge = fmt.Sprintf("STOP. You MUST call one of: %s. Emit the tool call and nothing else.", termList)
				}
				// The assistant turn above is already in the session (AddAssistant at
				// the top of this iteration); only the nudge needs storing.
				messages = append(messages, llmMessage{Role: "assistant", Content: text})
				messages = a.addCorrective(sid, messages, nudge)
				continue
			}
			res.Text = allText.String()
			stampTiming()
			return res, nil
		}

		messages = append(messages, llmMessage{
			Role:      "assistant",
			Content:   text,
			ToolCalls: calls,
		})

		// terminalCalled flips when this batch contains a Terminal tool call;
		// we finish processing the batch (so its tool result lands in history
		// for postmortem) and then exit with the message as res.Text.
		var terminalCalled bool
		var terminalName string
		var terminalMessage string
		// failedThisRound flips when any tool in this batch returns Failed=true,
		// feeding the failSoftCap counter once the batch is done.
		var failedThisRound bool
		// roundStuck stays true only if EVERY call this round reproduced output it
		// already produced this loop — i.e. the round made no progress (see the
		// repetition ladder below). A single new/productive call clears it.
		roundStuck := len(calls) > 0

		// Nothing calls the model while this batch runs, and a build or a test
		// suite is exactly the gap in which a server reclaims the slot holding
		// this conversation. Refresh it meanwhile: the messages so far are a
		// prefix of the next round's request, so the refresh keeps precisely
		// what that round will ask for.
		stopWarm := func() {}
		if sess := a.getSession(sid); sess != nil {
			sent := messages
			stopWarm = a.keepWarm(sess, caller.conn, func() []llmMessage { return sent })
		}

		for _, tc := range calls {
			// Terminal tools are skipped because their payload is already
			// rendered: submit_plan as the streamed table, respond as the turn's
			// own text.
			if !policy.terminals[tc.Function.Name] {
				a.announceToolCall(ctx, sid, tc)
			}

			// Live status while the tool runs (the tool-side counterpart of the LLM
			// meter in llmStream): "(running web_search… 12s)" ticks so a long tool or a
			// tool shows liveness instead of a frozen row.
			stopMeter := a.startToolMeter(ctx, sid, tc)

			// runToolCall (tools.go) executes the tool, caches its full output
			// in the session, and hands back the model-visible content (the
			// output truncated past truncateThreshold, or inline image parts).
			// A tool the phase policy forbids is rejected here WITHOUT executing —
			// the rejection is recorded so the model sees why and corrects.
			var tu ToolUse
			var content any
			denied := policy.deny[tc.Function.Name]
			switch {
			case denied:
				var msg string
				tu, msg = a.denyToolCall(ctx, sid, phase, tc)
				content = msg
			default:
				tu, content = a.runToolCall(ctx, sid, tc)
			}
			if stopMeter != nil {
				stopMeter() // stop the live ticker now the tool has returned
			}
			res.ToolUses = append(res.ToolUses, tu)
			// A denied call is the model's mistake, not a tool failure — don't feed
			// it to the fail cap (the repetition ladder still catches spamming it).
			if tu.Failed && !denied {
				failedThisRound = true
			}
			// A call that returns NEW output is progress and clears roundStuck.
			if !repeats.sawAgain(tc, tu) {
				roundStuck = false
			}
			if hasTerminal && policy.terminals[tc.Function.Name] && !terminalCalled {
				terminalCalled = true
				terminalName = tc.Function.Name
				terminalMessage = tu.Output
			}
			messages = append(messages, llmMessage{
				Role:       "tool",
				Content:    content,
				ToolCallID: tc.ID,
			})
		}
		stopWarm() // the model is about to be called again; the slot is busy from here

		// Terminal tool called: stream the message to the UI as one chunk
		// (the model emitted it as tool arguments, which never went through
		// the text-stream callback) and exit. The repetition ladder below is
		// skipped — this turn is over.
		if terminalCalled {
			// A respond while a command this loop handed to the background is
			// still running does not end the turn: the model has nothing else
			// to do until that job reports. The turn is parked here, the user
			// can interject meanwhile, and the job's note (its exit, or its
			// wake_after) resumes the loop where it stands. Inside /spec that
			// is the difference between a subtask waiting for its suite and a
			// subtask ending without a result.
			if terminalName == respondToolName {
				if jobs := a.parkableJobs(sid, loopStart); jobs != "" {
					if on != nil && terminalMessage != "" {
						on(terminalMessage)
						flushStream()
					}
					resume, perr := a.parkForJobs(ctx, sid, jobs)
					if perr != nil {
						res.Text = terminalMessage
						stampTiming()
						return res, perr
					}
					messages = a.addCorrective(sid, messages, resume)
					continue
				}
			}
			switch {
			case on == nil:
				// Silent internal pass (e.g. the planner): no UI emit.
			case terminalMessage != "":
				on(terminalMessage)
				flushStream() // the terminal message is the FINAL emit; never leave it batched in the throttled sink
			default:
				// An empty respond/terminal would end the turn wordlessly; emit a
				// minimal acknowledgement so a turn that ran is never silent.
				a.say(ctx, sid, "(done)\n")
			}
			res.Text = terminalMessage
			res.RespondCalled = true
			res.Terminal = terminalName
			stampTiming()
			return res, nil
		}

		// Failed-round soft cap: this non-terminal batch had a tool failure, so
		// count it. Once a subtask racks up failSoftCap failed rounds it is stuck
		// on something this web-blind execute loop can't clear; exit gracefully
		// (RespondCalled=false) so runExecutePhase records a failure to replan
		// against — replans are where web_search/web_read and a fresh
		// decomposition become available.
		if failedThisRound && failSoftCap > 0 {
			failedRounds++
			if failedRounds >= failSoftCap {
				res.Text = allText.String()
				stampTiming()
				return res, nil
			}
		}

		// Repetition ladder. A round where every call reproduced known output is
		// "stuck"; any productive round resets the streak. Consecutive stuck
		// rounds climb one ladder so the recovery step (warming the sampler)
		// always gets a turn before we give up.
		if !roundStuck {
			stuckRounds = 0
			continue
		}
		stuckRounds++
		if stuckRounds >= stuckBailRounds {
			// Give up gracefully (RespondCalled=false): in execute this surfaces
			// as a failed subtask → replan (where web tools + a fresh
			// decomposition open up); in plan, runPlanPhase salvages what's
			// there. A scary hard error for a re-read loop is worse than letting
			// the normal failure paths run.
			res.Text = allText.String()
			stampTiming()
			return res, nil
		}
		if !nudgedUI && sid != "" {
			nudgedUI = true
			a.say(ctx, sid, "⚠ Repeating with no new information — nudging the model to change course.\n")
		}
		// Corrective alongside the (unchanged) tool results, so the next round
		// sees the break-out instruction next to the output it just got back.
		messages = a.addCorrective(sid, messages,
			"Your last tool call(s) returned output you already have — that makes no progress. Do NOT repeat them. Instead:\n"+
				"1. If a read came back PARTIAL and you need more, call continue_read for the next chunk — never re-read the same window, never rewrite a whole file.\n"+
				"2. Act on what you already have: make a small targeted edit_file, run a DIFFERENT command, or finish by calling the terminal tool.\n"+
				"3. If you are stuck or the task is infeasible, say so and stop.")
		// Mid-ladder recovery: warm the sampler once before the bail. Same
		// server and model, and samplers do not enter the KV cache key, so the
		// prefix survives, provided the two roles differ in samplers alone (a
		// role carrying chat_template_kwargs would re-prefill here, deep into a
		// long context; see res/settings.toml). Skipped when already "thinking".
		if stuckRounds >= stuckEscalateRounds && !escalated && caller.conn != nil && caller.conn.Tag != "thinking" {
			if thinkConn := a.connFor("thinking"); thinkConn != nil {
				caller.conn = thinkConn
				escalated = true
				if sid != "" {
					a.say(ctx, sid, "⚠ Still repeating — switching to the thinking sampler to break out.\n")
				}
			}
		}
	}
}
