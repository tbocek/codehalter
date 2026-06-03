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
}

// runPlanLLM appends PLAN.md (plus an optional replanContext on retries) as a
// fresh user message, runs the planning LLM, persists the JSON response as
// the trailing assistant turn, and returns the parsed plan. Returns
// (nil, nil, err) when no LLM is configured. Returns (nil, ...) when there's
// no PLAN.md, the tool loop fails, or the response can't be parsed even
// after one corrective retry — callers treat any of those as "no plan,
// proceed without one".
//
// replanContext is "" on the first planning pass; on a replan the orchestrator
// supplies a short note (e.g. "REPLAN: prior attempt failed — see history.
// Same failure surfaced N times — try a structurally different approach.")
// The actual failure detail is already in history on the preceding executor
// response, so we don't repeat it here.
func (a *agent) runPlanLLM(ctx context.Context, sid string, replanContext string) (*planResult, []ToolUse, error) {
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
		_ = sess.Save()
	}

	var messages []llmMessage
	if sess != nil {
		messages = a.buildLLMContext(sess)
	}

	// Exclude respond: planning emits a JSON object as plain text, so we don't
	// want the model to escape into the synthetic terminal-tool grammar that
	// execute uses.
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

	// Run the tool loop, then parse its JSON reply. Text never streams to the
	// UI (JSON is internal machinery — orchestrate renders the parsed plan via
	// renderSteps); tool calls still surface as cards so the user sees what the
	// planner probes. On a parse failure — small models routinely wrap JSON in
	// prose — send one corrective follow-up and retry before giving up. The
	// returned ToolUses always merge both passes so the caller keeps full
	// visibility.
	var plan planResult
	noop := func(string) {}
	planRes, err := a.runToolLoopOn(ctx, sid, thinking, messages, filter, "plan", noop, nil, 0)
	if err != nil {
		return nil, planRes.ToolUses, err
	}
	if err := json.Unmarshal([]byte(trimJSON(planRes.Text)), &plan); err != nil {
		slog.Info("planner JSON parse failed; retrying with corrective", "snippet", truncate(planRes.Text, 200))
		fixMsgs := append([]llmMessage(nil), messages...)
		fixMsgs = append(fixMsgs,
			llmMessage{Role: "assistant", Content: planRes.Text},
			llmMessage{Role: "user", Content: "Your previous reply was not valid JSON. Reply with ONLY the JSON object — first character `{`, last character `}`, nothing before or after. No prose, no markdown fences."},
		)
		retry, retryErr := a.runToolLoopOn(ctx, sid, thinking, fixMsgs, filter, "plan", noop, nil, 0)
		planRes.Text = retry.Text
		planRes.ToolUses = append(planRes.ToolUses, retry.ToolUses...)
		planRes.DurationMs += retry.DurationMs
		if retryErr != nil {
			return nil, planRes.ToolUses, retryErr
		}
		if err := json.Unmarshal([]byte(trimJSON(retry.Text)), &plan); err != nil {
			return nil, planRes.ToolUses, fmt.Errorf("non-JSON after retry: %w", err)
		}
	}

	if sess != nil && planRes.Text != "" {
		sess.UpsertLastAssistant(planRes.Text)
		_ = sess.Save()
	}
	return &plan, planRes.ToolUses, nil
}

// renderSubtasks shows the planned subtask list to the user and returns the
// rendered text so the caller can fold it into the session transcript on
// cancel. header is the section heading ("Plan:" for actionable plans,
// "Findings:" for report-only plans where every subtask just relays
// information).
func (a *agent) renderSubtasks(ctx context.Context, sid string, subs []subtask, header string) string {
	if len(subs) == 0 {
		return ""
	}
	var planText strings.Builder
	planText.WriteString(header + "\n")
	for i, st := range subs {
		fmt.Fprintf(&planText, "%d. %s\n", i+1, st.Description)
	}
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: planText.String()}})
	return planText.String()
}

// appendAssistantNote concatenates a short note (e.g. "Understood: A",
// "User declined execution") onto the trailing assistant message instead of
// creating a fresh assistant turn. Two consecutive assistant messages would
// break strict role alternation; UpsertLastAssistant overwrites in place,
// preserving the planner's JSON content plus any prior tool uses.
func appendAssistantNote(sess *Session, note string) {
	if sess == nil || note == "" {
		return
	}
	sess.mu.Lock()
	var existing string
	if len(sess.Messages) > 0 && sess.Messages[len(sess.Messages)-1].Role == "assistant" {
		existing = sess.Messages[len(sess.Messages)-1].Content
	}
	sess.mu.Unlock()
	if existing != "" {
		sess.UpsertLastAssistant(existing + "\n\n" + note)
	} else {
		sess.UpsertLastAssistant(note)
	}
}

// planAndAsk runs the planner and resolves any clarification round-trip. It
// does NOT ask "Execute this plan?" — the orchestrator owns the single
// confirmation per plan / per replan so the user sees the full subtask list
// before deciding.
//
// Returns the parsed plan (may be nil if parsing failed or there is no
// PLAN.md), the tool uses performed during planning, and an error.
// errUserCancelled fires when the user aborts on a clarification prompt.
func (a *agent) planAndAsk(ctx context.Context, sid string, replanContext string) (*planResult, []ToolUse, error) {
	plan, toolUses, err := a.runPlanLLM(ctx, sid, replanContext)
	if err != nil || plan == nil {
		return plan, toolUses, err
	}
	sess := a.getSession(sid)

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
				_ = sess.Save()
			}
			return nil, toolUses, errUserCancelled
		}

		appendAssistantNote(sess, "User chose: "+choice)
		if sess != nil {
			_ = sess.Save()
		}
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Understood: " + choice + "\n"}})
	}

	return plan, toolUses, nil
}

// ---------------------------------------------------------------------------
// Subtask loop: one bounded tool-calling pass that self-verifies
// ---------------------------------------------------------------------------

// subtaskOutcome captures what runSubtaskLoop produced. Success is true only
// when the model called respond AND no tool in the loop returned Failed=true.
// Reason summarises why a failed subtask failed — it feeds the orchestrator's
// replan context and the Jaccard duplicate-failure check.
type subtaskOutcome struct {
	Result  toolLoopResult
	Success bool
	Reason  string
}

// runSubtaskLoop runs one subtask as a single tool-calling loop. EXECUTE.md
// plus the subtask description and verify recipe open the loop; the
// executor runs with all execute tools (web tools excluded — those live in
// planning). The executor self-verifies via the recipe before calling
// respond. Bounded only by the hard maxToolLoopIterations backstop in this file.
func (a *agent) runSubtaskLoop(ctx context.Context, sid string, st subtask, idx, total int) subtaskOutcome {
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
		_ = sess.Save()
	}

	var messages []llmMessage
	if sess != nil {
		messages = a.buildLLMContext(sess)
	}

	exclude := map[string]bool{
		"web_search":   true,
		"web_read":     true,
		"web_read_raw": true,
	}
	res, err := a.runToolLoop(ctx, sid, a.connForSession(ctx, sid, "execute"), messages, toolFilter{exclude: exclude}, "execute", 0)
	if sess != nil && res.Text != "" {
		sess.UpsertLastAssistant(res.Text)
		_ = sess.Save()
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
	// Exit-code authority: typed Failed flags override whatever the model
	// said in its respond message. Small models routinely declare success
	// while a run_task call exited non-zero; we trust the tool stream.
	var failedNames []string
	for _, u := range res.ToolUses {
		if u.Failed {
			failedNames = append(failedNames, u.Name)
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

// document runs after every subtask in the prompt has succeeded. It routes
// to a non-foreground LLM entry (falling back to llm[0] on single-LLM
// setups) with a minimal stateless input: DOCUMENT.md as the instructions,
// plus the per-turn shadow notes and earlier-conversation summary as the
// only context. No system prompt, no conversation history — llm[0]'s prefix
// cache stays warm and the documenter sees a small focused message.
// DOCUMENT.md self-skips when no documentation update is warranted.
func (a *agent) document(ctx context.Context, sid string, exec toolLoopResult) (toolLoopResult, error) {
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
		_ = sess.Save()
	}

	messages := []llmMessage{{Role: "user", Content: userMsg}}
	docRes, err := a.runToolLoop(ctx, sid, conn, messages, toolFilter{exclude: map[string]bool{respondToolName: true}}, "document", 0)
	if err != nil {
		slog.Warn("document phase failed", "err", err)
		return exec, nil
	}
	if sess != nil && docRes.Text != "" {
		sess.UpsertLastAssistant(docRes.Text)
		_ = sess.Save()
	}
	exec.ToolUses = append(exec.ToolUses, docRes.ToolUses...)
	return exec, nil
}

// ---------------------------------------------------------------------------
// Jaccard duplicate-failure detection (orchestrator-side)
// ---------------------------------------------------------------------------

// failureSimilarityThreshold is the Jaccard ratio above which two failed-
// subtask reason bags are considered "the same problem." Tuned empirically
// for short LLM-generated strings: 0.6 catches "missing import" /
// "import is missing" (Jaccard 0.67) without collapsing genuinely different
// files. The orchestrator uses this to escalate the replan context — same
// failure recurring N times tells the planner the prior fix didn't work
// and a structurally different approach is needed.
const failureSimilarityThreshold = 0.6

// issueBag tokenises a list of issue strings into a single set of distinct
// lowercase alphanumeric words. Punctuation, casing and ordering are all
// discarded so two attempts reporting the same root cause in different
// phrasing collapse to comparable bags.
func issueBag(issues []string) map[string]bool {
	bag := make(map[string]bool)
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			bag[cur.String()] = true
			cur.Reset()
		}
	}
	for _, iss := range issues {
		for _, r := range strings.ToLower(iss) {
			switch {
			case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
				cur.WriteRune(r)
			default:
				flush()
			}
		}
		flush()
	}
	return bag
}

// jaccard returns |A ∩ B| / |A ∪ B| for two word sets. 1.0 = identical,
// 0.0 = disjoint. Two empty bags are treated as identical.
func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	inter := 0
	for w := range a {
		if b[w] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// trimJSON extracts a JSON object from an LLM response. Small models often
// wrap the JSON in prose ("Sure, here's the JSON: { … } Let me know!") or
// markdown fences; we just locate the first `{` and the matching `}` and
// keep that slice. Brace counting respects strings + escapes so braces inside
// string values don't confuse the scan. Returns the trimmed input unchanged
// if no balanced object is found — caller surfaces the parse error.
func trimJSON(s string) string {
	s = strings.TrimSpace(s)
	start := strings.IndexByte(s, '{')
	if start < 0 {
		return s
	}
	depth := 0
	inStr := false
	esc := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}
	return s
}

// maxToolLoopIterations bounds runToolLoop so a model that keeps producing
// "different enough" tool calls (e.g. path variations to dodge perceived
// repetition) can't spin forever. One iteration is one LLM round-trip; a
// complex execute pass is usually 10-20, so 100 leaves comfortable headroom
// for unusually long but legitimate runs while still bailing on genuine
// runaways. The signature nudge and per-name escalation above catch the
// common stuck patterns earlier, so this cap is the last-resort backstop.
const maxToolLoopIterations = 100

// toolNameEscalateThreshold is the number of *redundant* calls to a single
// tool name in one loop after which the connection switches from the
// "execute" role to "thinking". Same server (so the KV prefix cache stays
// warm — sampler params never enter the cache key), warmer sampler. A call
// is redundant when its (name, arguments) pair has already been seen this
// loop; legitimate fan-out across distinct files/queries (e.g. surveying
// every go.mod in a multi-module repo) does NOT count, so broad read
// passes no longer trip the escalation. Complementary to the signature-
// based nudge above: that one catches byte-for-byte consecutive repeats,
// this one catches interleaved revisits to the same args. One-shot per
// loop.
const toolNameEscalateThreshold = 5

// toolCallSig produces a stable signature for the tool calls emitted in one
// iteration: byte-for-byte name + arguments, joined when there's more than
// one. Used to spot tight repetition where the model keeps re-running the
// same call (same `grep`, same `read_file`) without doing anything with the
// result — the most common pathology when small models get stuck.
func toolCallSig(calls []toolCall) string {
	if len(calls) == 0 {
		return ""
	}
	var b strings.Builder
	for i, c := range calls {
		if i > 0 {
			b.WriteByte('|')
		}
		b.WriteString(c.Function.Name)
		b.WriteByte('(')
		b.WriteString(c.Function.Arguments)
		b.WriteByte(')')
	}
	return b.String()
}

// toolLoopResult is what an agentic tool loop (runToolLoopOn) returns.
type toolLoopResult struct {
	Text     string
	ToolUses []ToolUse
	// RespondCalled is true when the loop exited because the model invoked
	// the registered terminal tool (typically `respond`). False when the
	// loop exited via the legacy empty-tool-calls path, hit the soft cap,
	// or returned an error. Subtask runners use this as the primary
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

// runToolLoop runs the agentic tool loop with the default token-streaming
// callback so the user sees execute / document phases live. The internal
// JSON phases (planner) call runToolLoopOn directly with a no-op. softCap
// is the per-call soft iteration cap (0 = use the maxToolLoopIterations
// hard backstop only); when the soft cap is hit, the loop exits gracefully
// with RespondCalled=false so the caller can treat it as a failed turn.
func (a *agent) runToolLoop(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, filter toolFilter, phase string, softCap int) (toolLoopResult, error) {
	on := func(token string) {
		if sid != "" {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: token}})
		}
	}
	think := func(token string) {
		if sid != "" {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentThought, Content: ContentBlock{Type: "text", Text: token}})
		}
	}
	return a.runToolLoopOn(ctx, sid, conn, messages, filter, phase, on, think, softCap)
}

// runToolLoopOn is the core agentic tool loop: send to LLM, execute tool
// calls, repeat. Callers supply the per-token callback (default streaming
// for runToolLoop, no-op for the planner's JSON pass in runPlanLLM). The phase string ("plan",
// "execute", "document", "subagent") flows onto the trailing assistant
// message via MarkLastAssistantTiming together with the cumulative
// llmStream wall-clock — so session.toml records who ran the turn and how
// much of its time was model generation vs tool execution.
//
// softCap is an optional per-call soft iteration ceiling. When > 0 and the
// loop reaches it without the terminal tool firing, the function returns
// (res, nil) with RespondCalled=false instead of erroring. 0 means "no
// soft cap — only the maxToolLoopIterations runaway backstop applies".
func (a *agent) runToolLoopOn(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, filter toolFilter, phase string, on, think func(string), softCap int) (toolLoopResult, error) {
	tools := llmToolDefinitionsFiltered(filter)

	// termName is the registered Terminal tool exposed in this phase ("" when
	// none — e.g. plan/verify/document filter respond out). When non-empty,
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
	// Per-name cumulative counts across the whole loop. When any single tool
	// name crosses toolNameEscalateThreshold we swap conn to the "thinking"
	// role on the same server (sampler-only change; KV prefix cache key is
	// derived from prompt tokens, not sampler params, so the cache stays
	// warm). One-shot per loop — if the warmer sampler doesn't help, the
	// signature nudge / 50-iter cap will still bail us out.
	toolNameCounts := map[string]int{}
	// toolArgSeen[name] is the set of (already-seen) argument strings for a
	// given tool name. The per-name escalation only counts a call as
	// redundant when its args have been seen before — fan-out across
	// distinct files (e.g. read_file on go.mod, examples/go.mod, …) doesn't
	// trip the escalation; only genuine revisits do.
	toolArgSeen := map[string]map[string]bool{}
	var escalated bool
	// respondNudged: when respondEnabled and the model returns an empty tool
	// call list, we nudge once to call respond. If the next turn still has no
	// tool calls we fall through to the legacy text-only exit so the loop
	// can't spin forever on a model that refuses the synthetic terminal.
	var respondNudged bool
	for iter := 0; ; iter++ {
		if iter >= maxToolLoopIterations {
			res.Text = allText.String()
			stampTiming()
			return res, fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations)
		}
		// Soft cap: exit gracefully with RespondCalled=false so the caller
		// (e.g. the subtask runner) can treat "ran out of turns without
		// calling respond" as a failure to replan against, rather than as
		// a hard error.
		if softCap > 0 && iter >= softCap {
			res.Text = allText.String()
			stampTiming()
			return res, nil
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

		sig := toolCallSig(calls)
		if sig != "" && sig == lastSig {
			sameSigCount++
		} else {
			sameSigCount = 1
			nudged = false
		}
		lastSig = sig

		// Third identical call after a nudge → give up. The model had its
		// recovery chance and didn't take it; further iterations will burn
		// the same time. Surfaces as a clean error rather than waiting for
		// the 50-iter cap.
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

		for _, tc := range calls {
			// Subagent sessions don't surface their own UI (Zed doesn't know
			// their sid), so forward a one-liner to the parent before each
			// tool call. Gives the user a live feed of what each subagent is
			// up to instead of just "Starting…" → 5 minutes → "Done".
			if sess := a.getSession(sid); sess != nil && sess.ParentID != "" && sess.DisplayLabel != "" {
				a.sendUpdate(ctx, sess.ParentID, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("[%s] %s %s\n\n", sess.DisplayLabel, tc.Function.Name, truncate(tc.Function.Arguments, 80))}})
			}
			a.setStatus(ctx, sid, " (running "+tc.Function.Name+"…)")
			started := time.Now()

			// view_image short-circuit: when the server supports images, we
			// read the bytes off the content-addressed store and deliver them
			// as multimodal tool content in the SAME turn (so the next
			// llmStream call sees the image). The standard executeTool path
			// only returns text, which would defeat the point.
			var result string
			var failed bool
			var multimodalContent any
			if tc.Function.Name == "view_image" && a.imagesSupported {
				sess := a.getSession(sid)
				text, parts, ferr := dispatchViewImage(sess, tc.Function.Arguments)
				result = text
				failed = ferr
				if !ferr {
					multimodalContent = parts
				}
			} else {
				result, failed = a.executeTool(ctx, sid, tc)
			}

			if hasTerminal && tc.Function.Name == termName && !terminalCalled {
				terminalCalled = true
				terminalMessage = result
			}
			useID := nextToolUseID()
			tu := ToolUse{
				ID:         useID,
				Name:       tc.Function.Name,
				Input:      tc.Function.Arguments,
				Output:     result,
				Failed:     failed,
				StartedAt:  started,
				DurationMs: time.Since(started).Milliseconds(),
			}
			res.ToolUses = append(res.ToolUses, tu)

			// Save incrementally so tool results survive crashes.
			if sess := a.getSession(sid); sess != nil {
				sess.AppendToolUse(tu)
				_ = sess.Save()
			}
			// Shrink the model-visible tool result before it enters the
			// message stream: anything past truncateThreshold is replaced
			// with head/tail + a per-tool "to see more" hint. session.toml
			// still records the full output via tu.Output above — only the
			// in-flight messages[] that get re-sent every turn shrink. The
			// hint embeds useID so the model can `view_output id=useID …`
			// to retrieve any portion of the original without re-running.
			content := truncateForLLM(useID, tc.Function.Name, tc.Function.Arguments, result)
			var toolContent any = content
			if multimodalContent != nil {
				toolContent = multimodalContent
			}
			messages = append(messages, llmMessage{
				Role:       "tool",
				Content:    toolContent,
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
