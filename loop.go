package main

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
)

// This file owns the plan + per-subtask machinery. The orchestrator
// (prompt.go) drives the larger plan → subtasks → replan cycle and the
// once-at-end document phase.
//
// The plan phase produces a list of subtasks; each subtask carries its
// own verify recipe. Every subtask runs as ONE bounded tool-calling
// loop (≤maxSubtaskTurns) where the executor can read, edit, install,
// and self-verify before declaring done via `respond`. No separate
// verify-phase LLM call — the executor runs the recipe itself.

// maxSubtaskTurns caps a single subtask loop. Hitting it without the
// model calling `respond` is a failed subtask — the orchestrator records
// the outcome and triggers a replan.
const maxSubtaskTurns = 10

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
	thinking := a.pickAvailable(ctx, sid, "thinking")
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
		messages = a.buildLLMHistory(sess, -1)
	}

	var plan planResult
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
	planRes, err := a.runToolLoopJSON(ctx, sid, thinking, messages, toolFilter{exclude: map[string]bool{
		respondToolName:   true,
		"write_file":      true,
		"edit_file":       true,
		"launch_subagent": true,
	}}, "plan", &plan)
	if err != nil {
		return nil, planRes.ToolUses, nil
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

// runSubtaskLoop runs one subtask as a single bounded tool-calling loop.
// EXECUTE.md plus the subtask description and verify recipe are appended as
// the user message that opens the loop; the executor runs with all execute
// tools (web tools excluded — those live in planning), up to maxSubtaskTurns.
// The executor self-verifies via the recipe before calling respond.
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
		messages = a.buildLLMHistory(sess, -1)
	}

	exclude := map[string]bool{
		"web_search":   true,
		"web_read":     true,
		"web_read_raw": true,
	}
	res, err := a.runToolLoop(ctx, sid, a.pickAvailable(ctx, sid, "execute"), messages, toolFilter{exclude: exclude}, "execute", maxSubtaskTurns)
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
		out.Reason = fmt.Sprintf("executor did not call respond within %d turns", maxSubtaskTurns)
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
	conn := a.pickBackgroundLLM(sid)
	if conn == nil {
		conn = a.settings.MainLLM("thinking")
	}
	if conn == nil {
		return exec, nil
	}

	// Wait for in-flight per-turn shadow notes so the summary we feed the
	// documenter reflects everything that happened on this turn — without
	// the wait, the most recent subtask's note may not have landed yet.
	if sess != nil {
		sess.summariseJob.Wait()
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
