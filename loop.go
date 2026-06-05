package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// This file owns the plan + per-subtask machinery. The orchestrator
// (prompt.go) drives the larger plan → subtasks → replan cycle and the
// once-at-end document phase.
//
// The plan phase produces a list of subtasks; each subtask carries its
// own verify recipe. Every subtask runs as ONE tool-calling loop where
// the executor can read, edit, install, and self-verify before declaring
// done via `respond`. No separate verify-phase LLM call — the executor
// runs the recipe itself. The loop's only ceiling is the hard
// maxToolLoopIterations backstop in this file.

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
	// answer is the planner's user-facing prose when it answered a lookup
	// directly (report_only with no subtasks). It rides on the struct rather
	// than the JSON — runPlanPhase sets it from the loop's separate content
	// channel — so orchestrate can surface it instead of silently dropping a
	// finished answer. No json tag: it never comes from the model's arguments.
	answer string
}

// runPlanPhase appends PLAN.md (plus an optional replanContext on retries) as a
// fresh user message, runs the planning LLM, persists the JSON response as the
// trailing assistant turn, parses the plan, and resolves any clarification
// round-trip. It does NOT ask "Execute this plan?" — the orchestrator owns the
// single confirmation per plan / per replan so the user sees the full subtask
// list before deciding.
//
// Returns (nil, nil, err) when no LLM is configured. Returns (nil, ...) when
// there's no PLAN.md, the tool loop fails, or the response can't be parsed even
// after one corrective retry — callers treat any of those as "no plan, proceed
// without one". errUserCancelled fires when the user aborts on a clarification
// prompt.
//
// replanContext is "" on the first planning pass; on a replan the orchestrator
// supplies a short note (e.g. "REPLAN: prior attempt failed — see history.
// Same failure surfaced N times — try a structurally different approach.")
// The actual failure detail is already in history on the preceding executor
// response, so we don't repeat it here.
func (a *agent) runPlanPhase(ctx context.Context, sid string, replanContext string) (*planResult, []ToolUse, error) {
	thinking := a.connForSession(ctx, sid, "thinking")
	if thinking == nil {
		return nil, nil, fmt.Errorf("no [[llm]] in .codehalter/settings.toml")
	}
	planPrompt := a.loadPromptFile(sid, "PLAN.md")
	if planPrompt == "" {
		return nil, nil, nil
	}
	if replanContext != "" {
		planPrompt = planPrompt + "\n\n" + replanContext
	}

	sess := a.getSession(sid)
	if sess != nil {
		sess.AddUser(planPrompt)
		sess.saveOrLog()
	}

	var messages []llmMessage
	if sess != nil {
		messages = a.buildLLMContext(sess)
	}

	// Exclude respond: planning has its OWN terminal tool, submit_plan, whose
	// arguments are the structured plan. Keeping respond out forces the planner
	// onto submit_plan instead of escaping into execute's exit and emitting a
	// free-text answer with no plan attached.
	// Exclude write_file/edit_file: planning is information-gathering only.
	// When the planner edits files itself, those edits leak into history and
	// the executor either repeats them or assumes the work is already done.
	// run_command's `sed -i` is the other edit vector; PLAN.md forbids it
	// in prose since we can't block it at the tool layer without parsing.
	// Also exclude launch_subagent: otherwise the planner fans its forbidden
	// edits out to a leaf-worker subagent (which has the full edit toolkit)
	// and reports report_only=true, papering over the loophole.
	filter := toolFilter{exclude: map[string]bool{
		respondToolName:   true,
		"write_file":      true,
		"edit_file":       true,
		"launch_subagent": true,
	}}

	// Run the tool loop (stream=false: planning output is machinery, not shown
	// live — orchestrate renders the subtask list or surfaces a direct answer).
	// The planner ends by calling submit_plan, whose arguments ARE the plan, so
	// planRes.Text is clean JSON and planRes.Content is any user-facing answer
	// prose — already separated. A model that skips the tool and emits the plan
	// as free text falls through to the legacy parse + one corrective retry.
	// Merged ToolUses keep full visibility either way.
	var plan planResult
	planRes, err := a.runToolLoop(ctx, sid, thinking, messages, filter, "plan", false, 0)
	if err != nil {
		return nil, planRes.ToolUses, err
	}
	parseErr := json.Unmarshal([]byte(trimJSON(planRes.Text)), &plan)
	if parseErr != nil && !planRes.RespondCalled {
		slog.Info("planner skipped submit_plan and JSON parse failed; retrying with corrective", "snippet", truncate(planRes.Text, 200))
		fixMsgs := append([]llmMessage(nil), messages...)
		fixMsgs = append(fixMsgs,
			llmMessage{Role: "assistant", Content: planRes.Text},
			llmMessage{Role: "user", Content: "Call the `submit_plan` tool with your plan as its arguments. Do not reply in prose."},
		)
		retry, retryErr := a.runToolLoop(ctx, sid, thinking, fixMsgs, filter, "plan", false, 0)
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
		return nil, planRes.ToolUses, fmt.Errorf("plan not valid JSON: %w", parseErr)
	}

	// Direct-answer prose: when the planner called submit_plan it's in the clean
	// content channel; a free-text plan has it mashed with the JSON, so drop the
	// JSON object out. orchestrate shows this when the plan has no subtasks.
	if planRes.RespondCalled {
		plan.answer = strings.TrimSpace(planRes.Content)
	} else {
		plan.answer = strings.TrimSpace(strings.Replace(planRes.Text, trimJSON(planRes.Text), "", 1))
	}

	if sess != nil {
		// Persist the user-facing turn: the direct answer when there is one,
		// else the structured plan JSON (kept in history for replan context).
		msg := plan.answer
		if msg == "" {
			msg = planRes.Text
		}
		if msg != "" {
			sess.UpsertLastAssistant(msg)
			sess.saveOrLog()
		}
	}

	toolUses := planRes.ToolUses
	if !plan.Clear && len(plan.Choices) > 0 {
		question := plan.Question
		if question == "" {
			question = "I'm not sure what you mean. Which of these?"
		}
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: question}})

		tcId := a.StartToolCall(ctx, sid, "Clarification needed", "think", nil)
		choice, err := a.askChoiceAuto(ctx, sid, tcId, plan.Choices)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + choice)})

		if err != nil || choice == "abort" {
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
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Understood: " + choice + "\n"}})
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
}

// runExecutePhase runs one subtask as a single tool-calling loop. EXECUTE.md
// plus the subtask description and verify recipe open the loop; the
// executor runs with all execute tools (web tools excluded — those live in
// planning). The executor self-verifies via the recipe before calling
// respond. Bounded only by the hard maxToolLoopIterations backstop in this file.
func (a *agent) runExecutePhase(ctx context.Context, sid string, st subtask, idx, total int) subtaskOutcome {
	executeMD := a.loadPromptFile(sid, "EXECUTE.md")
	sess := a.getSession(sid)

	var prompt strings.Builder
	if executeMD != "" {
		prompt.WriteString(executeMD)
		prompt.WriteString("\n\n")
	}
	fmt.Fprintf(&prompt, "## Subtask %d/%d\n\n%s\n", idx+1, total, st.Description)
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

	var messages []llmMessage
	if sess != nil {
		messages = a.buildLLMContext(sess)
	}

	exclude := map[string]bool{
		"web_search":       true,
		"web_read":         true,
		"web_read_raw":     true,
		submitPlanToolName: true, // planning-only terminal; respond is execute's exit
	}
	res, err := a.runToolLoop(ctx, sid, a.connForSession(ctx, sid, "execute"), messages, toolFilter{exclude: exclude}, "execute", true, executeFailCap)
	if sess != nil && res.Text != "" {
		sess.UpsertLastAssistant(res.Text)
		sess.saveOrLog()
	}

	out := subtaskOutcome{Result: res}
	if err != nil {
		out.Reason = "executor error: " + err.Error()
		return out
	}
	if !res.RespondCalled {
		out.Reason = "executor exited without calling respond"
		return out
	}
	// Exit-code authority, last state only: typed Failed flags override
	// whatever the model claimed in its respond message — small models
	// routinely declare success while a command exited non-zero, so we trust
	// the tool stream. But we honour only the LAST invocation of each distinct
	// call (same tool name + same arguments): a verify command that failed, got
	// fixed, and then re-ran green leaves an early Failed=true in the history
	// that must NOT condemn the subtask. Keying on name+input means re-running
	// the exact same command updates its verdict, while a different command
	// keeps its own.
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
		// edit_file/write_file set Failed on a usage error to feed the loop's
		// fail cap, but they're NOT truth-bearing verdicts here: a failed edit the
		// model recovered from (a later edit with corrected text — a different
		// input, so a different key) must not condemn the subtask, and one it did
		// NOT recover from is caught by the verify recipe (the changed file won't
		// build). Only run_task/run_command exit codes are authoritative.
		if lastFailed[k] && k.name != "edit_file" && k.name != "write_file" {
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

// runDocumentPhase runs after every subtask in the prompt has succeeded. It
// routes to a non-foreground LLM entry (falling back to llm[0] on single-LLM
// setups) with a minimal stateless input: DOCUMENT.md as the instructions,
// plus the per-turn shadow notes and earlier-conversation summary as the
// only context. No system prompt, no conversation history — llm[0]'s prefix
// cache stays warm and the documenter sees a small focused message.
// DOCUMENT.md self-skips when no documentation update is warranted.
func (a *agent) runDocumentPhase(ctx context.Context, sid string, exec toolLoopResult) (toolLoopResult, error) {
	docPrompt := a.loadPromptFile(sid, "DOCUMENT.md")
	if docPrompt == "" {
		return exec, nil
	}

	sess := a.getSession(sid)

	// Prefer a non-foreground slot so llm[0]'s warm cache isn't evicted.
	// Falls back to MainLLM when only one entry is configured.
	conn := a.connForBackgroundLLM()
	if conn == nil {
		conn = a.settings.MainLLM("thinking")
	}
	if conn == nil {
		return exec, nil
	}

	var summaryParts []string
	if sess != nil {
		if sess.Summary != "" {
			summaryParts = append(summaryParts, "## Earlier conversation summary\n\n"+sess.Summary)
		}
		if shadow := sess.peekShadow(); shadow != "" {
			summaryParts = append(summaryParts, "## Per-turn notes\n\n"+shadow)
		}
	}
	summaryBlock := strings.Join(summaryParts, "\n\n")
	if summaryBlock == "" {
		summaryBlock = "(no turn summary available — assume nothing user-visible changed)"
	}

	userMsg := docPrompt + "\n\n---\n\n# Turn summary (your only context)\n\n" + summaryBlock

	// Mirror the document instruction into session history for display so
	// the user can see the documenter ran. The LLM call itself does not see
	// this message — it sees only `messages` below.
	if sess != nil {
		sess.AddUser(docPrompt)
		sess.saveOrLog()
	}

	// Blank line before the documenter streams, so its output (often just "No
	// documentation change needed.") starts a fresh markdown paragraph instead
	// of running into the executor's final sentence.
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "\n\n"}})
	messages := []llmMessage{{Role: "user", Content: userMsg}}
	docRes, err := a.runToolLoop(ctx, sid, conn, messages, toolFilter{exclude: map[string]bool{respondToolName: true, submitPlanToolName: true}}, "document", true, 0)
	if err != nil {
		slog.Warn("document phase failed", "err", err)
		return exec, nil
	}
	if sess != nil && docRes.Text != "" {
		sess.UpsertLastAssistant(docRes.Text)
		sess.saveOrLog()
	}
	exec.ToolUses = append(exec.ToolUses, docRes.ToolUses...)
	return exec, nil
}

// maxToolLoopIterations is runToolLoop's hard runaway backstop: a model
// emitting "different enough" tool calls forever can't spin past it. One
// iteration is one LLM round-trip; a complex execute pass is usually 10-20, so
// 100 leaves headroom while still bailing genuine runaways. The repetition
// ladder catches the common stuck patterns earlier.
const maxToolLoopIterations = 100

// executeFailCap is the per-subtask budget of FAILED rounds (iterations whose
// tool batch produced a failure). Successful work is uncounted, so a long but
// productive subtask runs unhindered; only one that keeps hitting failures it
// can't clear burns the budget. A healthy verify-fix cycle costs 1-2 failures
// (red build → fix → green), so this leaves room for a few bumps before
// concluding the loop is stuck and bouncing it to a replan — where
// web_search/web_read and a fresh decomposition are available, unlike this
// web-blind execute loop. maxToolLoopIterations stays above it as the absolute
// runaway backstop (and still solely governs the plan/subagent loops, which
// pass no cap). Small fail budget + larger maxReplans deliberately shifts work
// from one web-blind loop toward more web-capable replan rounds.
const executeFailCap = 8

// stuckEscalateRounds / stuckBailRounds drive runToolLoop's repetition ladder.
// A "stuck round" is an iteration whose every tool call reproduced output it
// already produced this loop (an unchanged re-read, a re-run command with
// identical output, a repeated failing call). Consecutive stuck rounds first
// warm the sampler (switch execute→thinking — same server/model, so the prefix
// cache stays warm), then bail. Escalate strictly below bail so the warmer
// sampler always gets a turn before we give up.
const (
	stuckEscalateRounds = 3
	stuckBailRounds     = 5
)

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
	// StartedAt is when the first llmStream call of this loop began.
	// DurationMs is the cumulative wall-clock time spent in llmStream calls
	// across all iterations (excludes tool execution). Phase is the pipeline
	// stage tag passed in by the caller ("plan", "execute", "document",
	// "subagent"). Callers that own the final assistant message (prompt.go,
	// runSubagent) use these to stamp the message they create after the
	// loop returns.
	StartedAt  time.Time
	DurationMs int64
	Phase      string
}

// runToolLoop is the core agentic tool loop: send to LLM, execute tool calls,
// repeat. When stream is true the model's text/reasoning tokens are forwarded to
// the UI live (execute / document phases); the planner's internal JSON pass
// passes false so its tokens stay silent. The phase tag ("plan"/"execute"/
// "document"/"subagent") and cumulative llmStream wall-clock are stamped onto
// the trailing assistant message via MarkLastAssistantTiming, so session.toml
// records who ran the turn and how much time was generation vs tool execution.
//
// failSoftCap is an optional soft ceiling on FAILED rounds — iterations whose
// tool batch produced at least one failed tool. When > 0 and that count is
// reached without the terminal tool firing, returns (res, nil) with
// RespondCalled=false instead of erroring, so a subtask runner bounces to a
// replan. Counting failures (not total iterations) lets a long-but-productive
// subtask run freely while a stuck one — one that keeps hitting red builds/tests
// this web-blind loop can't resolve — exits early. 0 means only the
// maxToolLoopIterations backstop applies.
func (a *agent) runToolLoop(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, filter toolFilter, phase string, stream bool, failSoftCap int) (toolLoopResult, error) {
	// Stream model text/reasoning to the UI unless this is a silent internal
	// pass (the planner's JSON). nil callbacks are no-ops in the loop below.
	var on, think func(string)
	if stream && sid != "" {
		on = func(token string) {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: token}})
		}
		think = func(token string) {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentThought, Content: ContentBlock{Type: "text", Text: token}})
		}
	}
	tools := llmToolDefinitionsFiltered(filter)

	// termName is the registered Terminal tool exposed in this phase ("" when
	// none — e.g. plan/document filter respond out). When non-empty,
	// the empty-tool-calls branch below stops meaning "model finished" and
	// starts meaning "model dropped out of tool-calling grammar" — see the
	// nudge + fallback there.
	termName := terminalToolName(filter)
	hasTerminal := termName != ""

	var res toolLoopResult
	res.Phase = phase
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
	// Repetition ladder state. callOutHash remembers the last output hash of each
	// (name,args) call this loop; a later call that reproduces it made no progress
	// (an interleaved revisit counts, not just a consecutive one). A round whose
	// every call reproduced known output is "stuck"; consecutive stuck rounds
	// climb one ladder — corrective nudge each round, warm the sampler at
	// stuckEscalateRounds, bail at stuckBailRounds — and any productive round
	// resets the streak, so read-after-write and genuine fan-out are never punished.
	callOutHash := map[string]uint64{}
	var stuckRounds int
	var nudgedUI bool // the "repeating" UI warning fires only once
	var escalated bool
	// respondNudged: with a terminal tool (hasTerminal), an empty tool-call list
	// gets one nudge to call it; a second empty list falls through to the legacy
	// text exit so a model that refuses the terminal can't spin forever.
	var respondNudged bool
	// failedRounds counts iterations whose tool batch produced a failure; the
	// failSoftCap check below bounces the loop once it accumulates too many.
	var failedRounds int
	for iter := 0; ; iter++ {
		if iter >= maxToolLoopIterations {
			res.Text = allText.String()
			stampTiming()
			return res, fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations)
		}
		streamStart := time.Now()
		if res.StartedAt.IsZero() {
			res.StartedAt = streamStart
		}
		text, calls, err := a.llmStream(ctx, sid, conn, messages, tools, on, think)
		genElapsed += time.Since(streamStart)
		if err != nil {
			res.Text = allText.String()
			stampTiming()
			return res, err
		}
		allText.WriteString(text)

		// Compaction check AFTER every LLM call: the stream just set the server's
		// ground-truth prompt size (LastCompletePromptTokens). The 64% trigger
		// leaves headroom below n_ctx, so the call we just made fit; compacting
		// now — for the growing phases on the main session — keeps the NEXT call
		// (this loop's next iteration, or the next subtask) from drifting over.
		// Runs after every call, including the terminal `respond` one, so the
		// session is already compacted when the next subtask starts. Rebuild the
		// in-flight slice from the now-smaller session; the current batch is
		// appended to it below.
		if (phase == "plan" || phase == "execute") && sid != "" {
			if s := a.getSession(sid); s != nil && s.Depth == 0 {
				if a.compressHistory(ctx, s) {
					messages = a.buildLLMContext(s)
				}
			}
		}

		if len(calls) == 0 {
			// Terminal-tool mode: empty tool_calls means the model dropped
			// out of tool-calling grammar instead of finishing. Nudge it to
			// either call the terminal tool or another tool, but only once —
			// a model that refuses twice gets the legacy text exit so we
			// don't loop indefinitely on the new constraint.
			if hasTerminal && !respondNudged {
				respondNudged = true
				messages = append(messages, llmMessage{Role: "assistant", Content: text})
				messages = append(messages, llmMessage{
					Role: "user",
					Content: fmt.Sprintf("Your last response was plain text with no tool call. "+
						"This turn ends only when you call `%s` with your final "+
						"user-facing message, or another tool if you still have work "+
						"to do. Do not reply in prose — call a tool.", termName),
				})
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
		var terminalMessage string
		// failedThisRound flips when any tool in this batch returns Failed=true,
		// feeding the failSoftCap counter once the batch is done.
		var failedThisRound bool
		// roundStuck stays true only if EVERY call this round reproduced output it
		// already produced this loop — i.e. the round made no progress (see the
		// repetition ladder below). A single new/productive call clears it.
		roundStuck := len(calls) > 0

		for _, tc := range calls {
			// Subagent sessions don't surface their own UI (Zed doesn't know
			// their sid), so forward a one-liner to the parent before each
			// tool call. Gives the user a live feed of what each subagent is
			// up to instead of just "Starting…" → 5 minutes → "Done".
			if sess := a.getSession(sid); sess != nil && sess.ParentID != "" && sess.DisplayLabel != "" {
				a.sendUpdate(ctx, sess.ParentID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("[%s] %s %s\n\n", sess.DisplayLabel, tc.Function.Name, truncate(tc.Function.Arguments, 80))}})
			}
			a.setStatus(ctx, sid, " (running "+tc.Function.Name+"…)")

			// runToolCall (tools.go) executes the tool, caches its full output
			// for view_output, and hands back the model-visible content (the
			// output truncated past truncateThreshold, or inline image parts).
			tu, content := a.runToolCall(ctx, sid, tc)
			res.ToolUses = append(res.ToolUses, tu)
			if tu.Failed {
				failedThisRound = true
			}
			// Did this exact (name,args) call reproduce output it already
			// produced this loop? read_file/continue_read also honour the
			// content-dedup marker — serveRead prepends a note on a repeat, which
			// would otherwise defeat the hash on the first re-read. A call that
			// returns NEW output is progress and clears roundStuck.
			key := tc.Function.Name + "\x00" + tc.Function.Arguments
			h := fnvHash(tu.Output)
			repeated := false
			if prev, ok := callOutHash[key]; ok && prev == h {
				repeated = true
			}
			callOutHash[key] = h
			if (tc.Function.Name == "read_file" || tc.Function.Name == "continue_read") &&
				strings.Contains(tu.Output, readUnchangedMarker) {
				repeated = true
			}
			if !repeated {
				roundStuck = false
			}
			if hasTerminal && tc.Function.Name == termName && !terminalCalled {
				terminalCalled = true
				terminalMessage = tu.Output
			}
			messages = append(messages, llmMessage{
				Role:       "tool",
				Content:    content,
				ToolCallID: tc.ID,
			})
		}

		// Per-LLM-call progress fan-out: fire the summariser after each
		// iteration's tool batch so a long tool loop keeps populating the shadow
		// buffer that mid-turn compaction draws on (without this, a planner that
		// spends 17 minutes in one loop produces zero shadow notes). git-commit
		// is deliberately NOT fired here: the commit message only matters once
		// the turn ends and the user is back in control, and the working tree
		// keeps changing mid-turn — so runTurn fires backgroundGitCommit once at
		// turn end, and per-iteration commits would only burn the background slot.
		// Subagents (Depth>0) skip — they don't own the shadow buffer.
		if sess := a.getSession(sid); sess != nil && sess.Depth == 0 {
			a.backgroundSummarise(sess)
		}

		// Terminal tool called: stream the message to the UI as one chunk
		// (the model emitted it as tool arguments, which never went through
		// the text-stream callback) and exit. The repetition ladder below is
		// skipped — this turn is over.
		if terminalCalled {
			if on != nil && terminalMessage != "" {
				on(terminalMessage)
			}
			res.Text = terminalMessage
			res.RespondCalled = true
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
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⚠ Repeating with no new information — nudging the model to change course.\n"}})
		}
		// Corrective alongside the (unchanged) tool results, so the next round
		// sees the break-out instruction next to the output it just got back.
		messages = append(messages, llmMessage{
			Role: "user",
			Content: "Your last tool call(s) returned output you already have — that makes no progress. Do NOT repeat them. Instead:\n" +
				"1. If a read came back PARTIAL and you need more, call continue_read for the next chunk — never re-read the same window, never rewrite a whole file.\n" +
				"2. Act on what you already have: make a small targeted edit_file, run a DIFFERENT command, or finish by calling the terminal tool.\n" +
				"3. If you are stuck or the task is infeasible, say so and stop.",
		})
		// Mid-ladder recovery: warm the sampler once before the bail. Same
		// server/model (sampler params don't enter the KV cache key, so the
		// prefix cache survives); the "thinking" role's params let it abandon the
		// stuck plan. Skip when already on "thinking" (plan phase) — a no-op swap.
		if stuckRounds >= stuckEscalateRounds && !escalated && conn != nil && conn.Tag != "thinking" {
			if thinkConn := a.connForSession(ctx, sid, "thinking"); thinkConn != nil {
				conn = thinkConn
				escalated = true
				if sid != "" {
					a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⚠ Still repeating — switching to the thinking sampler to break out.\n"}})
				}
			}
		}
	}
}
