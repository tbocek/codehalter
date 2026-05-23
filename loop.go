package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
)

// This file owns the agent's per-task loop. One iteration is plan → execute →
// verify, with an optional document phase after a successful verify. When
// verify fails with fix steps, the loop re-plans with the failure context and
// runs again (up to maxAttempts, with Jaccard duplicate-issue detection to
// bail early when the model is rephrasing the same problem).
//
// Each phase appends its prompt file (PLAN.md / EXECUTE.md / VERIFY.md /
// DOCUMENT.md) as a fresh user message on sess.Messages, runs the LLM with
// the full history as input, then persists the LLM's response as the
// trailing assistant message. The transcript IS the history — no synthetic
// handoff strings, no message rewriting, so the prefix cache stays warm
// across phases by simple append.

// ---------------------------------------------------------------------------
// Plan phase
// ---------------------------------------------------------------------------

type planResult struct {
	Clear    bool     `json:"clear"`
	Choices  []string `json:"choices"`
	Question string   `json:"question"`
	Steps    []string `json:"steps"`
	// Subtasks lists top-level pieces the planner wants to decompose the
	// request into. When non-empty, Prompt treats each entry as its own
	// plan→execute→verify cycle (see runSubtasks). Steps should be empty
	// in this case — each subtask's own planner pass produces its steps.
	Subtasks []string `json:"subtasks"`
	// Verify lists the concrete checks the verify phase should run for THIS
	// change. Decided by the planner because it has full project context
	// (justfile/package.json/etc.) and knows what the change actually
	// touched. The verify phase treats this as authoritative — see verify()
	// and VERIFY.md. Empty means "no explicit verification" → verify falls
	// back to the static verifyTargetHint or VERIFY.md's generic rule.
	// Examples: ["Run just:verify via run_task"], ["Run npm:ci via run_task",
	// "Confirm dist/bundle.js was regenerated"].
	Verify []string `json:"verify"`
	// ReportOnly is true when the planner already has the answer in hand
	// (pure-lookup tasks: "where is X?", "what version of Y?") and the
	// executor only needs to relay findings — no file edits, no commands.
	// When true, planAndRoute skips the "Execute this plan?" confirmation
	// because there is nothing destructive to gate. Default false → current
	// gating behavior preserved when the planner omits the field.
	ReportOnly bool `json:"report_only"`
}

// runPlanLLM appends PLAN.md (plus an optional replanContext on retries) as a
// fresh user message, runs the planning LLM, persists the JSON response as
// the trailing assistant turn, and returns the parsed plan. Returns
// (nil, nil, nil, err) when no LLM is configured. Returns (thinking, nil, ...)
// when there's no PLAN.md, the tool loop fails, or the response can't be
// parsed even after one corrective retry — callers treat any of those as
// "no plan, proceed without one".
func (a *agent) runPlanLLM(ctx context.Context, sid string, replanContext string) (*LLMConnection, *planResult, []ToolUse, error) {
	thinking := a.pickAvailable(ctx, sid, "thinking")
	if thinking == nil {
		return nil, nil, nil, fmt.Errorf("no [[llm]] in .codehalter/settings.toml")
	}
	planPrompt := a.loadPromptFile(sid, "PLAN.md")
	if planPrompt == "" {
		return thinking, nil, nil, nil
	}
	// Intentionally NOT appending verifyTargetHint here: when the planner saw
	// "this project declares just:verify as its verify-class target", it
	// pattern-matched that into running just:verify during planning AND
	// seeded the same pattern into history, which the executor then copied.
	// The verify phase is the only place that needs to know the target.
	// On retries the caller supplies a short "the previous attempt failed —
	// re-plan" note; the actual failure detail is already in history on the
	// preceding verify response, so we don't repeat it here.
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
		messages = a.buildLLMHistory(sess, -1)
	}

	var plan planResult
	// Exclude respond: planning emits a JSON object as plain text, so we don't
	// want the model to escape into the synthetic terminal-tool grammar that
	// execute uses. Same reasoning in verify() and document() below.
	// Exclude write_file/edit_file: planning is information-gathering only.
	// When the planner edits files itself, those edits leak into history and
	// the executor either repeats them or assumes the work is already done.
	// run_command's `sed -i` is the other edit vector; PLAN.md forbids it
	// in prose since we can't block it at the tool layer without parsing.
	// Also exclude launch_subagent: otherwise the planner fans its forbidden
	// edits out to a leaf-worker subagent (which has the full edit toolkit)
	// and reports report_only=true, papering over the loophole. Cost is real
	// — no parallel-lookup fan-out during gathering — but planning is short
	// enough that serial reads in one LLM turn are fine.
	planRes, err := a.runToolLoopJSON(ctx, sid, thinking, messages, toolFilter{exclude: map[string]bool{
		respondToolName:   true,
		"write_file":      true,
		"edit_file":       true,
		"launch_subagent": true,
	}}, "plan", &plan)
	if err != nil {
		return thinking, nil, planRes.ToolUses, nil
	}
	if sess != nil && planRes.Text != "" {
		sess.UpsertLastAssistant(planRes.Text)
		_ = sess.Save()
	}
	return thinking, &plan, planRes.ToolUses, nil
}

// renderSteps shows the plan's steps to the user and returns the rendered
// text so the caller can fold it into the session transcript on cancel.
// header is the section heading ("Plan:" for actionable plans, "Findings:"
// for report-only plans where the executor will just relay the answer).
func (a *agent) renderSteps(ctx context.Context, sid string, steps []string, header string) string {
	if len(steps) == 0 {
		return ""
	}
	var planText strings.Builder
	planText.WriteString(header + "\n")
	for i, step := range steps {
		fmt.Fprintf(&planText, "%d. %s\n", i+1, step)
	}
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: TextBlock(planText.String())})
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

// planAndRoute appends PLAN.md, runs the planner, asks for clarification if
// needed, shows the plan, and asks the user to confirm before execution.
//
// Returns the LLM connection, the parsed plan (may be nil if parsing failed
// or there is no PLAN.md), the tool uses performed during planning, and an
// error. When the plan has Subtasks, the "Execute this plan?" confirmation
// is skipped — the caller (Prompt → runSubtasks) handles a one-shot
// Interactive/Automatic/Cancel prompt for the whole batch instead.
//
// replanContext is "" on the first planning pass; on re-plan it's a short
// "the previous attempt failed verification — re-plan" note (the failure
// detail itself is already in history on the preceding verify response).
func (a *agent) planAndRoute(ctx context.Context, sid string, replanContext string) (*LLMConnection, *planResult, []ToolUse, error) {
	thinking, plan, toolUses, err := a.runPlanLLM(ctx, sid, replanContext)
	if err != nil || plan == nil {
		return thinking, plan, toolUses, err
	}
	sess := a.getSession(sid)

	// If the request is unclear, ask the user.
	if !plan.Clear && len(plan.Choices) > 0 {
		question := plan.Question
		if question == "" {
			question = "I'm not sure what you mean. Which of these?"
		}
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: TextBlock(question)})

		tcId := a.StartToolCall(ctx, sid, "Clarification needed", "think", nil)
		choice, err := a.askChoiceAuto(ctx, sid, tcId, plan.Choices)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + choice)})

		if err != nil || choice == "abort" {
			appendAssistantNote(sess, "User aborted on clarification.")
			if sess != nil {
				_ = sess.Save()
			}
			return nil, nil, toolUses, errUserCancelled
		}

		appendAssistantNote(sess, "User chose: "+choice)
		if sess != nil {
			_ = sess.Save()
		}
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: TextBlock("Understood: " + choice + "\n")})
	}

	// If the planner decomposed the request into subtasks, skip the per-plan
	// "Execute this plan?" prompt — the caller will show the subtask list and
	// ask Interactive/Automatic/Cancel for the whole batch instead.
	if len(plan.Subtasks) > 0 {
		return thinking, plan, toolUses, nil
	}

	// Show the plan and ask for confirmation.
	if len(plan.Steps) > 0 {
		header := "Plan:"
		if plan.ReportOnly {
			header = "Findings:"
		}
		a.renderSteps(ctx, sid, plan.Steps, header)

		// Pure-lookup plans need no execution gate — the executor just
		// relays what the planner already gathered. PLAN.md tells the
		// planner to set report_only=true for these so we skip the prompt.
		if plan.ReportOnly {
			return thinking, plan, toolUses, nil
		}

		tcId := a.StartToolCall(ctx, sid, "Execute this plan?", "think", nil)
		ok, err := a.askYesNoAuto(ctx, sid, tcId, "Execute", "Cancel")
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("User chose: %v", ok))})

		if err != nil || !ok {
			appendAssistantNote(sess, "User declined execution.")
			if sess != nil {
				_ = sess.Save()
			}
			return nil, nil, toolUses, errUserCancelled
		}
	}

	return thinking, plan, toolUses, nil
}

// ---------------------------------------------------------------------------
// Execute phase
// ---------------------------------------------------------------------------

// pickVerifyTarget returns runner:target for the most "comprehensive" task
// the project advertises (priority: verify > check > ci > all) or "" when
// none exists. Used by execute() and verify() to inject a concrete hint that
// the model should use the project's full verification contract via run_task
// rather than running its constituents directly via run_command. Generic
// across runners: any justfile/Makefile/npm-script target with one of these
// names gets picked up automatically.
func (a *agent) pickVerifyTarget() string {
	priority := []string{"verify", "check", "ci", "all"}
	a.mu.Lock()
	runners := a.runners
	a.mu.Unlock()
	for _, want := range priority {
		for _, r := range runners {
			for _, t := range r.Tasks {
				if strings.EqualFold(t, want) {
					return r.Name + ":" + t
				}
			}
		}
	}
	return ""
}

// verifyTargetHint formats the per-project hint pointing at the comprehensive
// verify-class target (see pickVerifyTarget). Empty string when there is no
// such target — caller treats that as "no hint, use the generic rule".
func (a *agent) verifyTargetHint() string {
	vt := a.pickVerifyTarget()
	if vt == "" {
		return ""
	}
	return fmt.Sprintf(
		"This project declares `%s` as its verify-class target. Run it via run_task — its body is the project's contract for \"verified\" and may chain tidy/build steps that piecemeal commands miss. Do NOT substitute `go test ./...` / `npm test` / equivalent constituents.",
		vt,
	)
}

// execute runs the execution phase. It appends EXECUTE.md (if present) as a
// fresh user message, excludes web tools (information retrieval belongs to
// the planning phase), and runs the agentic tool loop with write-enabled
// tools. The response is persisted as the trailing assistant turn.
//
// extraExclude lists additional tool names to exclude (e.g. launch_subagent
// when a subagent has reached its max nesting depth).
//
// Routed via pickAvailable(sid): the main session always uses LLM[0] for
// cache consistency — every phase (plan/execute/verify/document) hits the
// same slot so the prefix cache stays warm across the whole turn. Subagent
// sessions route to their pinned LLM[i] for the same reason.
func (a *agent) execute(ctx context.Context, sid string, extraExclude ...string) (toolLoopResult, error) {
	executeMD := a.loadPromptFile(sid, "EXECUTE.md")
	sess := a.getSession(sid)
	if sess != nil && executeMD != "" {
		sess.AddUser(executeMD)
		_ = sess.Save()
	}
	var messages []llmMessage
	if sess != nil {
		messages = a.buildLLMHistory(sess, -1)
	}
	exclude := map[string]bool{"web_search": true, "web_read": true, "web_read_raw": true}
	for _, name := range extraExclude {
		exclude[name] = true
	}
	res, err := a.runToolLoop(ctx, sid, a.pickAvailable(ctx, sid, "execute"), messages, toolFilter{
		exclude: exclude,
	}, "execute")
	if err != nil {
		return res, err
	}
	if sess != nil && res.Text != "" {
		sess.UpsertLastAssistant(res.Text)
		_ = sess.Save()
	}
	return res, nil
}

// ---------------------------------------------------------------------------
// Verify phase
// ---------------------------------------------------------------------------

type verifyResult struct {
	Success  bool     `json:"success"`
	Issues   []string `json:"issues"`
	FixSteps []string `json:"fix_steps,omitempty"`
	// SustainabilityConcerns lists fixes the executor performed that won't
	// survive a container rebuild / fresh clone — e.g. `apt-get install X`
	// run via run_command without a matching `.devcontainer/Dockerfile`
	// edit. Independent of pass/fail: even when exit codes are clean, a
	// non-empty list downgrades the verdict to failure so the re-plan can
	// persist the fix. See VERIFY.md.
	SustainabilityConcerns []string `json:"sustainability_concerns,omitempty"`
}

// verify appends VERIFY.md as a fresh user message and runs a self-check.
// It routes to the "execute" role (lower temperature, less thinking) rather
// than reusing the planner's connection — small thinking models tend to
// rationalize failures into success=true, while a colder execute model
// sticks to the rule. Result.FixSteps (when present) signals the caller to
// re-plan with the failure context. Returns the merged result and the verify
// outcome.
//
// When fallbackConn is non-nil, it's used only if no LLM is configured for
// the execute role (test path with synthetic agent and no settings).
//
// plan carries the verification recipe the planner specified for this
// change (plan.Verify). When non-empty it's injected as authoritative — the
// verify phase runs THOSE checks, not whatever it would have invented. nil
// or empty Verify falls back to the static verifyTargetHint.
func (a *agent) verify(ctx context.Context, sid string, fallbackConn *LLMConnection, plan *planResult, res toolLoopResult) (toolLoopResult, *verifyResult, error) {
	verifyPrompt := a.loadPromptFile(sid, "VERIFY.md")
	if verifyPrompt == "" {
		return res, &verifyResult{Success: true}, nil
	}

	// Same routing as execute(): main session uses LLM[0], subagents their
	// pinned LLM[i].
	conn := a.pickAvailable(ctx, sid, "execute")
	if conn == nil {
		conn = fallbackConn
	}

	prompt := verifyPrompt
	recipeInjected := plan != nil && len(plan.Verify) > 0
	hintInjected := false
	if recipeInjected {
		var recipe strings.Builder
		recipe.WriteString("\n\n## Verification recipe (from the planner)\n\n")
		recipe.WriteString("The planner specified the following checks for THIS change. Run each via the appropriate tool — they replace your own judgment about what to check:\n\n")
		for i, v := range plan.Verify {
			fmt.Fprintf(&recipe, "%d. %s\n", i+1, v)
		}
		prompt = prompt + recipe.String()
	} else if hint := a.verifyTargetHint(); hint != "" {
		prompt = prompt + "\n\n" + hint
		hintInjected = true
	}

	sess := a.getSession(sid)
	if sess != nil {
		sess.AddUser(prompt)
		_ = sess.Save()
	}

	// Capture executor's tool-use count before we append verify's own —
	// the Failed override only inspects executor tools.
	execToolCount := len(res.ToolUses)

	var messages []llmMessage
	if sess != nil {
		messages = a.buildLLMHistory(sess, -1)
	}
	var result verifyResult
	verifyRes, err := a.runToolLoopJSON(ctx, sid, conn, messages, toolFilter{exclude: map[string]bool{respondToolName: true}}, "verify", &result)
	verifyToolCount := len(verifyRes.ToolUses)
	res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
	if err != nil {
		slog.Error("verify: skipped, treating as success", "err", err)
		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: TextBlock("⚠ Verification skipped: " + err.Error() + "\n")})
		return res, &verifyResult{Success: true}, nil
	}
	if sess != nil && verifyRes.Text != "" {
		sess.UpsertLastAssistant(verifyRes.Text)
		_ = sess.Save()
	}

	// Exit-code authority: the executor's typed Failed flags are ground
	// truth for pass/fail. The LLM's `success` field is advisory — small
	// models hallucinate in BOTH directions (inventing failures on clean
	// runs, rationalizing real failures as success=true), so we override
	// either way from the raw tool stream.
	anyToolFailed := false
	for i := 0; i < execToolCount; i++ {
		if res.ToolUses[i].Failed {
			anyToolFailed = true
			break
		}
	}
	if anyToolFailed {
		if result.Success {
			result.Success = false
			if len(result.Issues) == 0 {
				for i := 0; i < execToolCount; i++ {
					u := res.ToolUses[i]
					if u.Failed {
						result.Issues = append(result.Issues,
							fmt.Sprintf("%s failed (exit-code override; verifier had reported success=true)", u.Name))
					}
				}
			}
		}
	} else if !result.Success {
		// LLM invented issues with no failed tool to back them up — discard.
		result.Success = true
		result.Issues = nil
		result.FixSteps = nil
	}

	// Skipped-verification override: when we injected a verification recipe
	// (planner's plan.Verify) or a fallback runner:target hint, the verifier
	// MUST have called at least one tool to actually run it. Zero verify-turn
	// tool calls + success=true is rationalization, not verification — caught
	// in a real run where the prompt told the verifier to run `just:verify`
	// and it returned success in 5s without a single tool dispatch. Triggers
	// re-plan so the next pass gets an explicit recipe.
	if (recipeInjected || hintInjected) && verifyToolCount == 0 && result.Success {
		result.Success = false
		result.Issues = append(result.Issues,
			"verifier returned success without running the prescribed verification (zero tool calls in verify turn)")
		result.FixSteps = append(result.FixSteps,
			"the verify phase must call the recipe/target before reporting success")
	}

	// Sustainability: a clean exit with non-sustainable fixes (e.g. an
	// `apt-get install X` run via run_command without a matching Dockerfile
	// edit) is still incomplete — the fix will vanish on container rebuild.
	// Treat as failure with concerns folded into fix_steps so re-plan can
	// persist them.
	if result.Success && len(result.SustainabilityConcerns) > 0 {
		result.Success = false
		for _, c := range result.SustainabilityConcerns {
			result.Issues = append(result.Issues, "not yet sustainable: "+c)
			result.FixSteps = append(result.FixSteps, c)
		}
	}

	return res, &result, nil
}

// ---------------------------------------------------------------------------
// Document phase
// ---------------------------------------------------------------------------

// document runs after a successful verify when the turn actually wrote files.
// It appends DOCUMENT.md as a fresh user message and runs the thinking LLM so
// it can update (or create) the project README when the change is
// user-visible.
func (a *agent) document(ctx context.Context, sid string, conn *LLMConnection, exec toolLoopResult) (toolLoopResult, error) {
	docPrompt := a.loadPromptFile(sid, "DOCUMENT.md")
	if docPrompt == "" {
		return exec, nil
	}

	sess := a.getSession(sid)
	if sess != nil {
		sess.AddUser(docPrompt)
		_ = sess.Save()
	}

	var messages []llmMessage
	if sess != nil {
		messages = a.buildLLMHistory(sess, -1)
	}
	docRes, err := a.runToolLoop(ctx, sid, conn, messages, toolFilter{exclude: map[string]bool{respondToolName: true}}, "document")
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
// Loop: plan → execute → verify (→ document), with re-plan on verify failure
// ---------------------------------------------------------------------------

// runTaskCycle runs plan → execute → verify with retries for a single task.
// On attempt 0, a pre-computed plan may be supplied (prePlan/preConn) to
// avoid re-planning after an upfront planAndRoute call; retries always
// re-plan, with a short replanContext nudging the planner toward the prior
// verify failure that's already in history.
func (a *agent) runTaskCycle(
	ctx context.Context, sid string,
	prePlan *planResult, preConn *LLMConnection,
) (toolLoopResult, error) {
	const maxAttempts = 20
	var result toolLoopResult
	var lastVR *verifyResult
	var seenBags []map[string]bool
	sess := a.getSession(sid)

	conn := preConn
	plan := prePlan
	var replanContext string

	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 || prePlan == nil {
			a.sendPhase(ctx, sid, 0, false)
			c, p, tu, err := a.planAndRoute(ctx, sid, replanContext)
			if err != nil {
				if errors.Is(err, errUserCancelled) {
					return result, err
				}
				if sess != nil && len(tu) > 0 {
					sess.AddAssistantWithTools("❌ "+err.Error(), tu)
					_ = sess.Save()
				}
				return result, err
			}
			conn = c
			plan = p
		}

		a.sendPhase(ctx, sid, 1, false)
		var err error
		result, err = a.execute(ctx, sid)
		if err != nil {
			return result, err
		}
		a.sendPhase(ctx, sid, 2, false)
		var vr *verifyResult
		result, vr, err = a.verify(ctx, sid, conn, plan, result)
		if err != nil {
			return result, err
		}
		lastVR = vr
		if vr == nil || vr.Success || len(vr.FixSteps) == 0 {
			break
		}

		// Loop-detection: if verify reports a near-duplicate of an earlier
		// failure (Jaccard ≥ threshold over the bag of issue words), the LLM
		// is stuck rephrasing the same problem. Bail rather than burn the
		// remaining retry budget. Order, casing and punctuation are ignored
		// so "missing import" and "import is missing" collapse together.
		currentBag := issueBag(vr.Issues)
		looped := false
		for _, prev := range seenBags {
			if jaccard(currentBag, prev) >= failureSimilarityThreshold {
				looped = true
				break
			}
		}
		if looped {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: TextBlock(
				fmt.Sprintf("⚠ Same verification failure as a prior attempt — retrying would loop. Giving up. Last issues:\n%s\n", strings.Join(vr.Issues, "\n")),
			)})
			break
		}
		seenBags = append(seenBags, currentBag)

		if attempt == maxAttempts-1 {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: TextBlock(
				fmt.Sprintf("⚠ Verification still failing after %d attempts — giving up. Last issues:\n%s\n", maxAttempts, strings.Join(vr.Issues, "\n")),
			)})
			break
		}

		a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: TextBlock(
			fmt.Sprintf("⚠ Verification failed (attempt %d/%d). Re-planning with the failure context:\n%s\n", attempt+1, maxAttempts, strings.Join(vr.FixSteps, "\n")),
		)})
		// The full failure detail is already in history on the preceding
		// verify response — point the planner at it without re-stitching.
		replanContext = "The previous attempt failed verification (see the verify response above). Re-plan to address the failure. If the fix steps conflict with the original request or the task is infeasible, say so instead of attempting it."
	}

	// Document phase: run whenever verify passed. DOCUMENT.md decides for
	// itself whether docs need updating ("No documentation change needed."
	// short-circuit at Step 1) — cheaper than tracking edit_file/write_file
	// calls here, which miss mutations via run_command/run_task anyway.
	// Failures with no fix_steps still reach here (we couldn't fix it but
	// didn't bail) — skip for those, the change isn't trustworthy enough to
	// advertise.
	lastPhase := 2
	if (lastVR == nil || lastVR.Success) && conn != nil {
		a.sendPhase(ctx, sid, 3, false)
		result, _ = a.document(ctx, sid, conn, result)
		lastPhase = 3
	}
	a.sendPhase(ctx, sid, lastPhase, true)
	return result, nil
}

// failureSimilarityThreshold is the Jaccard ratio above which two
// verify-failure bags are considered "the same problem." Tuned empirically
// for short LLM-generated issue strings: 0.6 catches "missing import" /
// "import is missing" (Jaccard 0.67) without collapsing genuinely different
// files (e.g. foo.go vs bar.go syntax errors land around 0.67-0.83 and
// would false-positive; the cost of an over-eager bail is recoverable by
// re-prompting, so we accept that for the upside of catching small-model
// rewordings).
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
