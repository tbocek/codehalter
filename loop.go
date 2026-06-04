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
		if lastFailed[k] {
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
// 100 leaves headroom while still bailing genuine runaways. The signature nudge
// and per-name escalation catch the common stuck patterns earlier.
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

// toolNameEscalateThreshold is how many *redundant* calls to one tool name in a
// loop trigger a switch from the "execute" role to "thinking" — same server
// (sampler params don't enter the KV cache key, so the prefix cache stays warm),
// warmer sampler. A call is redundant only when its (name, arguments) pair was
// already seen this loop, so legitimate fan-out across distinct files doesn't
// count. Complements the signature nudge (byte-for-byte consecutive repeats);
// this catches interleaved revisits. One-shot per loop.
const toolNameEscalateThreshold = 5

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
	// Repetition state: track the signature of the previous iteration's tool
	// calls. Two consecutive identical signatures inject a corrective and
	// give the model one chance to break out; a third (post-nudge) bails.
	// A different signature resets the streak — natural variation isn't
	// punished, only tight loops.
	var lastSig string
	var sameSigCount int
	var nudged bool
	// Per-name redundant-call counts; crossing toolNameEscalateThreshold swaps
	// conn to the "thinking" role (see the const). toolArgSeen[name] tracks which
	// argument strings have already been seen, so only genuine revisits count as
	// redundant — fan-out across distinct files doesn't. One-shot per loop.
	toolNameCounts := map[string]int{}
	toolArgSeen := map[string]map[string]bool{}
	var escalated bool
	// respondNudged: with a terminal tool (hasTerminal), an empty tool-call list
	// gets one nudge to call it; a second empty list falls through to the legacy
	// text exit so a model that refuses the terminal can't spin forever.
	var respondNudged bool
	// failedRounds counts iterations whose tool batch produced a failure; the
	// failSoftCap check below bounces the loop once it accumulates too many.
	var failedRounds int
	// redundantFetches counts rounds where a tool re-served content the model
	// already has unchanged this turn (read_file dedup hit). Each one gets a
	// corrective; a third bails the loop as stuck. Catches the interleaved
	// re-read pattern the consecutive-repeat nudge can't.
	var redundantFetches int
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

		// Signature of this iteration's calls: byte-for-byte name(args), joined
		// by "|". Identical consecutive signatures are the tight-repetition
		// pathology (same grep / read_file re-run without acting on the result).
		var sigB strings.Builder
		for i, c := range calls {
			if i > 0 {
				sigB.WriteByte('|')
			}
			sigB.WriteString(c.Function.Name)
			sigB.WriteByte('(')
			sigB.WriteString(c.Function.Arguments)
			sigB.WriteByte(')')
		}
		sig := sigB.String()
		if sig != "" && sig == lastSig {
			sameSigCount++
		} else {
			sameSigCount = 1
			nudged = false
		}
		lastSig = sig

		// Third identical call after a nudge → give up: the model had its
		// recovery chance. A clean error beats waiting for the iteration cap.
		if sameSigCount >= 3 && nudged {
			res.Text = allText.String()
			stampTiming()
			return res, fmt.Errorf("tool loop stuck on identical call after nudge: %s", truncate(sig, 200))
		}
		// Second identical call → nudge once. We still EXECUTE the duplicate
		// (legitimate read-after-write needs the fresh read to land, even
		// when its sig matches the prior read), but tack on a user-role
		// corrective at the end of this iteration so the next LLM round-trip
		// sees a break-out instruction alongside the (possibly unchanged)
		// tool result.
		nudgeThisIter := sameSigCount == 2 && !nudged
		if nudgeThisIter {
			nudged = true
			if sid != "" {
				a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "⚠ Repeated tool call detected — nudging the model to try a different action.\n"}})
			}
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
		// redundantThisRound flips when a tool re-served unchanged content the
		// model already has (read_file dedup hit, flagged via readUnchangedMarker).
		var redundantThisRound bool

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
			if strings.Contains(tu.Output, readUnchangedMarker) {
				redundantThisRound = true
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

		// Per-LLM-call progress fan-out: fire summary + git_commit after each
		// iteration's tool batch. Without this, a planner that spends 17
		// minutes in one tool loop produces zero shadow notes and a stale
		// .codehalter/.git_commit — the existing Prompt() epilogue only runs
		// after the whole task finishes. Summariser enqueues every fire so a
		// 50-iteration loop produces 50 progress notes; git commit coalesces
		// via gitCommitJob (bgJob) since gitCommitLastHash already dedupes
		// identical snapshots. Subagents (Depth>0) skip — they already route
		// via their pinned slot and don't own the shadow buffer.
		if sess := a.getSession(sid); sess != nil && sess.Depth == 0 {
			a.backgroundSummarise(sess)
			a.backgroundGitCommit(sess)
		}

		// Terminal tool called: stream the message to the UI as one chunk
		// (the model emitted it as tool arguments, which never went through
		// the text-stream callback) and exit. The same-sig nudge and per-name
		// escalation below are skipped — this turn is over.
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

		// If this iteration was flagged as a repeat, append a corrective
		// user message after the tool results so the model sees the nudge
		// alongside the (often unchanged) output it just got back.
		if nudgeThisIter {
			messages = append(messages, llmMessage{
				Role: "user",
				Content: "You just repeated the same tool call with the same arguments. Re-running rarely produces new information. Either:\n" +
					"1. Act on the output you already have (edit a file, run a different command, or summarise your finding), or\n" +
					"2. If you are stuck or the task is infeasible, say so and stop.\n\n" +
					"Do not call the same tool with the same arguments again unless you have first changed state that the call observes (e.g. a file you just edited).",
			})
		}

		// Redundant-fetch corrective: a tool re-served content the model already
		// has unchanged this turn. Unlike nudgeThisIter (consecutive byte-identical
		// calls), this fires even when the re-read is interleaved with other calls
		// — the pattern that let a document phase re-read one README four times.
		// React on the FIRST redundant fetch; a third bails the loop as stuck.
		if redundantThisRound {
			redundantFetches++
			if redundantFetches >= 3 {
				res.Text = allText.String()
				stampTiming()
				return res, fmt.Errorf("tool loop stuck re-fetching content already in context")
			}
			messages = append(messages, llmMessage{
				Role:    "user",
				Content: "You just re-read a file you already have unchanged in this turn's tool history. The same bytes come back every time — it never makes progress. Use the copy you already have. Do not read or search the same unchanged file again; if you have what you need, act on it or finish.",
			})
		}

		// Per-name escalation: redundant calls to the same tool (same args
		// seen before) mean the sampler is too cold to abandon a stuck plan.
		// Distinct args don't count — surveying many files via read_file
		// doesn't trip this. Switch to the "thinking" role's params — same
		// URL/model so the prefix cache survives — and let the warmer
		// sampler pick a different action. One-shot per loop. Skip entirely
		// when conn.Tag is already "thinking" (plan phase): the swap would be
		// a no-op and the warning would mislead.
		if !escalated && conn != nil && conn.Tag != "thinking" {
			for _, c := range calls {
				seen := toolArgSeen[c.Function.Name]
				if seen == nil {
					seen = map[string]bool{}
					toolArgSeen[c.Function.Name] = seen
				}
				if seen[c.Function.Arguments] {
					toolNameCounts[c.Function.Name]++
				} else {
					seen[c.Function.Arguments] = true
				}
			}
			for name, n := range toolNameCounts {
				if n < toolNameEscalateThreshold {
					continue
				}
				thinkConn := a.connForSession(ctx, sid, "thinking")
				if thinkConn == nil {
					break
				}
				conn = thinkConn
				escalated = true
				if sid != "" {
					a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("⚠ Tool '%s' invoked %d× this loop — switching to thinking sampler to break out.\n", name, n)}})
				}
				break
			}
		}
	}
}
