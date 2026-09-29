package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"log/slog"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"
)

// Each subtask runs as ONE tool loop that verifies itself before `respond`;
// there is no separate verify call.

// An empty Verify is legal only for pure-lookup subtasks that edit no files.
type subtask struct {
	Description string   `json:"description"`
	Verify      []string `json:"verify,omitempty"`
}

// Also accepts `subtasks` serialised into a string, which planners send; a
// rejection would cost a whole planning call.
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
		raw.Subtasks = json.RawMessage(trimJSONArray(inner))
	}
	return json.Unmarshal(raw.Subtasks, &p.Subtasks)
}

type planResult struct {
	Clear    bool      `json:"clear"`
	Choices  []string  `json:"choices"`
	Question string    `json:"question"`
	Subtasks []subtask `json:"subtasks"`
	// A /spec round's question: the spec text that comes closest, and options with examples.
	SpecQuote string       `json:"spec_quote"`
	Options   []specOption `json:"options"`
	// Every subtask only relays findings, so the orchestrator skips the confirmation.
	ReportOnly bool `json:"report_only"`
	// The planner's third exit, for a request too big for one plan (see specFromPlan).
	Redo    []string   `json:"redo"`
	Spec    []specFile `json:"spec"`
	SpecDir string     `json:"spec_dir"`
	OutDir  string     `json:"out_dir"`
	Target  string     `json:"target"`
	// An argument because a server that forces the tool call (Halogen) returns no
	// message text beside it.
	Answer string `json:"answer"`
	// Answer when given, else the message text beside the call (set by planFrom).
	answer string
}

// planProblem names how a planner submission missed its contract, with the
// corrective for the one retry: invalid JSON, prose (after the loop's own
// nudge), or a plan with neither an answer, a question nor any work.
func planProblem(res toolLoopResult, p *planResult, parseErr error) (wrong, corrective string) {
	switch {
	case parseErr != nil && res.Terminal == "":
		return "the planner replied in prose instead of calling submit_plan",
			"Call the `submit_plan` tool with your plan as its arguments. Do not reply in prose."
	case parseErr != nil:
		return fmt.Sprintf("submit_plan's arguments were not valid JSON (%v)", parseErr),
			fmt.Sprintf("Your `submit_plan` arguments were not valid JSON (%v). Call it again, emitting the arguments as one well-formed JSON object.", parseErr)
	// Whether or not it set clear: an empty plan once ended six /spec rounds in a row.
	case len(p.Subtasks) == 0 && p.answer == "" && len(p.Redo) == 0 && len(p.Spec) == 0 && strings.TrimSpace(p.Question) == "":
		return "the planner submitted neither an answer nor any subtasks",
			"Your `submit_plan` had neither an answer nor subtasks, so nothing reaches the user. Call it again with EITHER the complete answer in `answer` (report_only=true, no subtasks) OR the subtasks that do the work. Put the fields at the top level of the arguments, `{\"clear\": true, \"subtasks\": [...]}`, not inside another key."
	}
	return "", ""
}

// unwrapPlan: a plan put inside one wrapper key, as an object or as JSON text
// (`{"plan": {...}}`, `{"plan": "[{...}]"}`), is the plan itself. Qwen3.8 did
// this six rounds in a row, copying its own call from the history, and each
// round read as an empty plan.
func unwrapPlan(raw string) string {
	var outer map[string]json.RawMessage
	if json.Unmarshal([]byte(raw), &outer) != nil || len(outer) != 1 {
		return raw
	}
	for _, inner := range outer {
		var text string
		if json.Unmarshal(inner, &text) == nil {
			inner = json.RawMessage(strings.TrimSpace(text))
		}
		var list []json.RawMessage
		if json.Unmarshal(inner, &list) == nil && len(list) == 1 {
			inner = list[0]
		}
		var plan map[string]json.RawMessage
		if json.Unmarshal(inner, &plan) == nil && (plan["subtasks"] != nil || plan["clear"] != nil || plan["answer"] != nil) {
			return string(inner)
		}
	}
	return raw
}

// respond as the plan terminal is a direct answer: a report_only plan with no subtasks.
func planFrom(res toolLoopResult) (*planResult, error) {
	if res.Terminal == respondToolName {
		return &planResult{Clear: true, ReportOnly: true, answer: strings.TrimSpace(res.Text)}, nil
	}
	var p planResult
	if err := json.Unmarshal([]byte(unwrapPlan(trimJSON(res.Text))), &p); err != nil {
		return nil, err
	}
	switch {
	case strings.TrimSpace(p.Answer) != "":
		p.answer = strings.TrimSpace(p.Answer)
	case res.Terminal != "":
		p.answer = strings.TrimSpace(res.Content)
	default:
		p.answer = strings.TrimSpace(strings.Replace(res.Text, trimJSON(res.Text), "", 1))
	}
	return &p, nil
}

// Does not ask "Execute this plan?": the orchestrator does, after the whole list
// is shown. errUserCancelled means the user aborted a clarification.
func (a *agent) runPlanPhase(ctx context.Context, sid string, replanContext string) (*planResult, error) {
	thinking := a.connFor("thinking")
	if thinking == nil {
		return nil, fmt.Errorf("no [[llm]] in .codehalter/settings.toml")
	}
	sess := a.getSession(sid)
	if sess == nil {
		return nil, fmt.Errorf("no session found")
	}
	// PLAN.md lives in the cached system prompt; repeating it here would stack a copy per replan.
	marker := "Begin the PLANNING phase — produce the plan now (planning guidance is in the system prompt)."
	if replanContext != "" {
		marker = "Begin the PLANNING phase again.\n\n" + replanContext
	}

	sess.AddUser(marker)
	sess.saveOrLog()
	// A plan phase that errored after streaming a row never reaches renderPlan.
	sess.phaseMu.Lock()
	sess.planTableShown = false
	sess.phaseMu.Unlock()

	// Planner edits would leak into history (`sed -i` cannot be blocked here).
	policy := phasePolicy{
		deny:      map[string]bool{"write_file": true, "edit_file": true},
		terminals: map[string]bool{submitPlanToolName: true, respondToolName: true},
	}

	// stream=false: planning output is machinery; orchestrate renders the result.
	planRes, err := a.runToolLoop(ctx, sid, thinking, policy, "plan", false, 0)
	if err != nil {
		return nil, err
	}
	plan, parseErr := planFrom(planRes)
	// Prose was already nudged in the loop; this retry covers what the loop cannot see.
	if wrong, corrective := planProblem(planRes, plan, parseErr); wrong != "" {
		// Rows already on screen must be named as not final.
		salvaged := ""
		var partial struct {
			Subtasks []subtask `json:"subtasks"`
		}
		if json.Unmarshal([]byte(repairJSON(planRes.Text)), &partial) == nil && len(partial.Subtasks) > 0 {
			salvaged = fmt.Sprintf(" %d subtask(s) already reached the table and are not final.", len(partial.Subtasks))
		}
		// Planning does not stream, so otherwise nothing shows during the extra round trip.
		a.say(ctx, sid, fmt.Sprintf("\n⚠ Planning went wrong! %s.%s Asking the planner to try again.\n", wrong, salvaged))
		slog.Info("planner submission missed its contract; retrying with corrective",
			"sid", sid, "calledSubmitPlan", planRes.Terminal != "", "err", parseErr, "snippet", truncate(planRes.Text, 200))
		retry, retryErr := a.runToolLoop(ctx, sid, thinking, policy, "plan", false, 0, corrective)
		if retryErr != nil {
			return nil, retryErr
		}
		plan, parseErr = planFrom(retry)
	}
	if parseErr != nil {
		a.say(ctx, sid, fmt.Sprintf("\n⚠ Planning failed! The planner could not produce a valid plan even after a corrective retry (%v). Nothing will run.\n", parseErr))
		return nil, fmt.Errorf("plan not valid JSON: %w", parseErr)
	}

	// A /spec question goes to the spec's QUESTIONS.md, once it is one the user can
	// answer from the spec alone: autopilot's first option would let the model settle
	// an open point of the spec by itself, and an answer given in chat is gone for
	// every later round of the item.
	if fence := sess.specFence(); fence != "" && !plan.Clear {
		q, wrong := specQuestionFrom(plan, fence)
		if wrong != "" {
			a.say(ctx, sid, fmt.Sprintf("\n⚠ The planner asked a question that is not answerable from the spec as asked: %s. Asking it to look again.\n", wrong))
			retry, retryErr := a.runToolLoop(ctx, sid, thinking, policy, "plan", false, 0, specQuestionCorrective(wrong))
			if retryErr != nil {
				return nil, retryErr
			}
			if plan, parseErr = planFrom(retry); parseErr != nil {
				a.say(ctx, sid, fmt.Sprintf("\n⚠ Planning failed! The planner's second answer was not a valid plan (%v). Nothing will run.\n", parseErr))
				return nil, fmt.Errorf("plan not valid JSON: %w", parseErr)
			}
			if plan.Clear {
				return plan, nil // it found the answer in the spec
			}
			q, wrong = specQuestionFrom(plan, fence)
		}
		a.say(ctx, sid, q.Question+"\n")
		sess.AddUser("Question parked for the user by the spec loop: " + q.Question)
		sess.saveOrLog()
		return nil, &specQuestionError{Q: q, Problem: wrong}
	}

	if !plan.Clear && len(plan.Choices) > 0 {
		question := plan.Question
		if question == "" {
			question = "I'm not sure what you mean. Which of these?"
		}
		a.say(ctx, sid, question)

		tcId := a.StartToolCall(ctx, sid, "Clarification needed", "think", nil)
		choice := plan.Choices[0]
		var err error
		if !a.autoAnswer(ctx, sid, choice) {
			choice, err = a.askChoice(ctx, sid, tcId, question, plan.Choices)
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + choice)})

		// A new user message, never an edit: the planner's stored reply is in the server's cache.
		note := "User chose: " + choice
		switch {
		case err != nil:
			note = "Clarification cancelled."
		case choice == "abort":
			note = "User aborted on clarification."
		}
		sess.AddUser(note)
		sess.saveOrLog()
		if err != nil {
			return nil, err
		}
		if choice == "abort" {
			return nil, errUserCancelled
		}
		a.say(ctx, sid, "Understood: "+choice+"\n")

		return a.runPlanPhase(ctx, sid, replanContext)
	}

	return plan, nil
}

// Reason feeds the replan context and the Jaccard duplicate-failure check.
type subtaskOutcome struct {
	Result  toolLoopResult
	Success bool
	Reason  string
	// The executor's revision of the remaining plan; Success is false and Reason empty.
	Upsert *planResult
}

func (a *agent) runExecutePhase(ctx context.Context, sid string, st subtask, idx, total int) subtaskOutcome {
	sess := a.getSession(sid)

	// EXECUTE.md lives in the system prompt; repeating it would stack a copy per subtask.
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

	// submit_plan revises the remaining plan in place (see subtaskOutcome.Upsert).
	policy := phasePolicy{terminals: map[string]bool{respondToolName: true, submitPlanToolName: true}}
	// tool_choice=required: this phase ends only on a terminal tool, so prose is always a slip.
	conn := a.connFor("execute").withThinkingDisabled().withBody("tool_choice", "required")
	res, err := a.runToolLoop(ctx, sid, conn, policy, "execute", true, executeFailCap)

	out := subtaskOutcome{Result: res}
	if err != nil {
		out.Reason = "executor error: " + err.Error()
		return out
	}
	if res.Terminal == submitPlanToolName {
		var up planResult
		if err := json.Unmarshal([]byte(unwrapPlan(trimJSON(res.Text))), &up); err == nil && len(up.Subtasks) > 0 {
			out.Upsert = &up
			return out
		}
		out.Reason = "executor called submit_plan with no usable subtasks"
		return out
	}
	if res.Terminal != respondToolName {
		out.Reason = "executor exited without calling respond"
		return out
	}
	// Exit codes override the model's verdict (small models declare success over a
	// non-zero exit), but only each call's LAST run counts: a fixed re-run is green.
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
		// Other tools set Failed only to feed the fail cap; they are not verdicts.
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

// A failed documentation pass is logged, never the turn's failure.
func (a *agent) runDocumentPhase(ctx context.Context, sid string, exec toolLoopResult) toolLoopResult {
	docPrompt := a.loadPromptFile(sid, "DOCUMENT.md")
	if docPrompt == "" {
		return exec
	}

	sess := a.getSession(sid)
	if sess == nil {
		return exec
	}

	// The execute connection, so it reuses execute's warm KV prefix.
	conn := a.connFor("execute").withThinkingDisabled()
	if conn == nil {
		return exec
	}

	sess.AddUser(docPrompt)
	sess.saveOrLog()

	a.say(ctx, sid, "\n\n")
	// respond must be a terminal, or the model calls it again and trips the repetition ladder.
	docPolicy := phasePolicy{
		deny:      map[string]bool{submitPlanToolName: true},
		terminals: map[string]bool{respondToolName: true},
		proseEnds: true, // DOCUMENT.md asks for a prose reply when nothing changes
	}
	docRes, err := a.runToolLoop(ctx, sid, conn, docPolicy, "document", true, 0)
	if err != nil {
		slog.Warn("document phase failed", "err", err)
		return exec
	}
	exec.ToolUses = append(exec.ToolUses, docRes.ToolUses...)
	return exec
}

// Above the median plan, below the runaways; the executor reads what is left cheaply.
const planRoundNudge = 20

// Backstop for "different enough" calls forever; the repetition ladder catches the rest earlier.
const maxToolLoopIterations = 100

// An execute step at the cap with at least two edits and a build or test run in
// its last productiveWindow calls gets productiveExtension more, once.
const (
	productiveWindow    = 40
	productiveExtension = 50
)

// An execute subtask that only reads for this long is lost: in the logs, no
// successful one read more than about 32 calls in a row, and most capped ones did.
const (
	readStreakEscalate = 20
	readStreakBail     = 40
)

// The completed small turns 400 recovery keeps verbatim (see Session.keepWindowStart).
const keepSmallTurnTokens = 10_000

// A mid-response drop (router model swap, a blip) is usually momentary. A server
// that refuses connections is restarting, which took two minutes in the logs, so
// it gets doubling waits for up to serverDownPatience. Vars so tests can shorten them.
const maxTransientStreamRetries = 5

var (
	transientStreamBackoff = 3 * time.Second
	serverDownPatience     = 5 * time.Minute
)

// Per-token updates at high rates contend for the one conn write lock, and a slow
// editor would backpressure the SSE read and stall the LLM call.
const streamFlushInterval = 200 * time.Millisecond

// No lock: the SSE loop drives sink, and the caller calls flush after llmStream returns.
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

// Counts FAILED rounds only; past it the subtask bounces to a replan.
const executeFailCap = 8

// Escalate strictly below bail, so the warmer sampler always gets a turn.
const (
	stuckEscalateRounds = 3
	stuckBailRounds     = 5
)

type toolLoopResult struct {
	Text string
	// The prose beside the calls. On a terminal exit Text is the terminal's output
	// instead; runPlanPhase relies on the split.
	Content  string
	ToolUses []ToolUse
	// "" when no terminal ended the loop: a failed subtask whatever Text says.
	Terminal string
}

// failSoftCap > 0 ends the loop after that many FAILED rounds, 0 leaves only
// maxToolLoopIterations. The context is rebuilt from the session; corrective is stored first.
func (a *agent) runToolLoop(ctx context.Context, sid string, conn *LLMConnection, policy phasePolicy, phase string, stream bool, failSoftCap int, corrective ...string) (toolLoopResult, error) {
	sess := a.getSession(sid)
	if sess == nil {
		return toolLoopResult{}, fmt.Errorf("no session found")
	}
	for _, c := range corrective {
		sess.AddUser(c)
	}
	if len(corrective) > 0 {
		sess.saveOrLog()
	}
	return a.runToolLoopSeeded(ctx, sid, conn, a.buildLLMContext(sess), policy, phase, stream, failSoftCap)
}

// What goes on the wire is stored: runToolLoop rebuilds from the session, and an
// unstored turn would vanish from the MIDDLE of history and bust the cache.
func (a *agent) addCorrective(sid string, messages []llmMessage, text string) []llmMessage {
	if sess := a.getSession(sid); sess != nil {
		sess.AddUser(text)
		sess.saveOrLog()
	}
	return append(messages, llmMessage{Role: "user", Content: text})
}

func (a *agent) startToolMeter(ctx context.Context, sid string, tc toolCall) (stop func()) {
	label := tc.Function.Name
	// Only one argument fits the row; collapsing whitespace keeps a heredoc on one line.
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

// Past this the phase name and the seconds counter get pushed out of view.
const toolMeterArgRunes = 48

// An interleaved revisit counts too, hence maps rather than a last-call comparison.
// Repeats are keyed by what the model got back, not by what it typed: a model that
// varies the call (a new echo label, a counter in a log name: one step rendered the
// same screen 98 times, each into a new log) still gets the same answer.
type repetitionTracker struct {
	// A change to the project makes a command's answer new again (a re-check), but
	// not a read's: re-reading a file the edit did not touch still says nothing new.
	reads, runs repeatMemory
	// stuck: outputs that ended an earlier step of this turn by repeating.
	stuck map[uint64]bool
	// cwd and sig: the project and its fingerprint after the last observed call.
	cwd, sig string
	changes  int  // calls that changed the project in this step
	hitStuck bool // the last sawAgain returned an output that ended an earlier step
}

// repeatMemory: the outputs the model got, from any call, and per exact call its
// last output, whose bag catches near-identical output the hash misses (a
// timestamp in a failing build).
type repeatMemory struct {
	seen map[uint64]bool
	last map[string]uint64
	bag  map[string]map[string]bool
}

func newRepeatMemory() repeatMemory {
	return repeatMemory{seen: map[uint64]bool{}, last: map[string]uint64{}, bag: map[string]map[string]bool{}}
}

// readerTools cannot change the project.
var readerTools = map[string]bool{"read_file": true, "continue_read": true, "screenshot": true, "view_image": true,
	"web_search": true, "web_read": true, "ask_user": true, respondToolName: true, submitPlanToolName: true}

func newRepetitionTracker(cwd string, stuck map[uint64]bool) *repetitionTracker {
	rt := &repetitionTracker{reads: newRepeatMemory(), runs: newRepeatMemory(), stuck: maps.Clone(stuck), cwd: cwd}
	if cwd != "" {
		rt.sig = projectSig(cwd)
	}
	return rt
}

// From the same call again, a timestamp, a pid or a counter in the output does not
// make it new. Between different calls a number can be the news ("3 passed", "5
// passed"), so there only a clock time or a date is folded.
var (
	digitRunRe = regexp.MustCompile(`\d+`)
	clockRe    = regexp.MustCompile(`\d{4}-\d\d-\d\d|\d{1,2}:\d\d(?::\d\d(?:\.\d+)?)?`)
)

// repeatMinOutput: shorter answers ("exit 0", "no matches found") come from many
// different probes alike, so across calls they say nothing about a loop; from the
// same call again they do.
const repeatMinOutput = 24

// callKey is exactly the call: the tool and its arguments.
func callKey(tc toolCall) string { return tc.Function.Name + "\x00" + tc.Function.Arguments }

// repeatText is an output as a repeat between different calls is judged: without
// the batching note (it depends on the call before), without the lines the call
// spelled out itself (an echo label is the model's own text, not an answer), clock
// times and dates folded.
func repeatText(args, output string) string {
	out, _, _ := strings.Cut(output, batchNoteLead)
	var own strings.Builder
	var walk func(v any)
	walk = func(v any) {
		switch v := v.(type) {
		case string:
			own.WriteString(v)
			own.WriteByte('\n')
		case []any:
			for _, e := range v {
				walk(e)
			}
		case map[string]any:
			for _, e := range v {
				walk(e)
			}
		}
	}
	var v any
	if json.Unmarshal([]byte(args), &v) == nil {
		walk(v)
	}
	spelled := own.String()
	lines := strings.Split(out, "\n")
	lines = slices.DeleteFunc(lines, func(l string) bool {
		t := strings.TrimSpace(l)
		return len(t) >= 3 && strings.Contains(spelled, t)
	})
	return strings.TrimSpace(clockRe.ReplaceAllString(strings.Join(lines, "\n"), "#"))
}

// changedBy: whether a call changed the project. A write tool says so; a tool that
// cannot write did not; anything else (a shell command, an MCP tool) is observed,
// not guessed from its text: a redirect, `sed -i` or a Python heredoc all show up
// the same way.
func (rt *repetitionTracker) changedBy(tc toolCall, tu ToolUse) bool {
	if readerTools[tc.Function.Name] {
		return false
	}
	// Without a project to look at, a write tool's own word is all there is. With
	// one, the look decides: an edit that wrote the same bytes changed nothing.
	if rt.cwd == "" {
		return (tc.Function.Name == "edit_file" || tc.Function.Name == "write_file") && strings.HasPrefix(tu.Output, "file written")
	}
	sig := projectSig(rt.cwd)
	changed := sig != rt.sig
	rt.sig = sig
	if changed {
		rt.changes++
	}
	return changed
}

// sawAgain: the call brought nothing new, since the model already got this output
// with the project as it is.
func (rt *repetitionTracker) sawAgain(tc toolCall, tu ToolUse, changed bool) bool {
	rt.hitStuck = false
	if changed {
		// Every command may answer differently now: a re-run is a re-check.
		rt.runs = newRepeatMemory()
		clear(rt.stuck)
		return false
	}
	mem := &rt.runs
	if readerTools[tc.Function.Name] {
		mem = &rt.reads
	}
	out := repeatText(tc.Function.Arguments, tu.Output)
	h := fnvHash(out)
	folded := digitRunRe.ReplaceAllString(out, "#")
	bag := issueBag([]string{folded})
	key := callKey(tc)
	last, again := mem.last[key]
	repeated := again && last == fnvHash(folded)
	if prev, ok := mem.bag[key]; ok && jaccard(prev, bag) >= stuckOutputSimilarity {
		repeated = true
	}
	// A write's error and a job's launch note read alike for any call: only the
	// same call again repeats them.
	// The length is the answer's, without run_command's own exit line.
	answer := strings.TrimSpace(strings.TrimPrefix(folded, "exit #"))
	if name := tc.Function.Name; len(answer) >= repeatMinOutput && name != "edit_file" && name != "write_file" && name != "run_background" {
		repeated = repeated || mem.seen[h]
		if rt.stuck[h] {
			repeated, rt.hitStuck = true, true
		}
		mem.seen[h] = true
	}
	mem.last[key] = fnvHash(folded)
	mem.bag[key] = bag
	return repeated
}

// projectSig fingerprints what a call could change in the project: the files git
// reports as changed or new, by content. Content, not mtime: a render that writes
// the same picture again changed nothing. Outside git, each file's size and mtime.
func projectSig(cwd string) string {
	h := fnv.New64a()
	out, err := exec.Command("git", "-C", cwd, "status", "--porcelain=v1", "-z", "--untracked-files=all").Output()
	if err != nil {
		n := 0
		walkErr := filepath.WalkDir(cwd, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return nil // a file gone mid-walk is a change the next look sees
			}
			if d.IsDir() {
				if path != cwd && skipWalkDir(d.Name()) {
					return filepath.SkipDir
				}
				return nil
			}
			if n++; n > projectSigMaxFiles {
				return filepath.SkipAll
			}
			if info, err := d.Info(); err == nil {
				fmt.Fprintf(h, "%s\x00%d\x00%d\x00", path, info.Size(), info.ModTime().UnixNano())
			}
			return nil
		})
		if walkErr != nil {
			slog.Debug("projectSig: walk", "cwd", cwd, "err", walkErr)
		}
		return strconv.FormatUint(h.Sum64(), 16)
	}
	h.Write(out)
	hashed := 0
	for _, entry := range strings.Split(string(out), "\x00") {
		// "XY path"; a rename's second field is the old path alone, with no status.
		if len(entry) < 4 || entry[2] != ' ' {
			continue
		}
		path := filepath.Join(cwd, entry[3:])
		info, err := os.Stat(path)
		if err != nil || info.IsDir() {
			continue
		}
		if hashed++; hashed > projectSigMaxFiles || info.Size() > projectSigMaxBytes {
			fmt.Fprintf(h, "%d\x00%d\x00", info.Size(), info.ModTime().UnixNano())
			continue
		}
		// An unreadable file goes in as its error: seen the same way twice, it is no change.
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(h, "\x00%v\x00", err)
			continue
		}
		if _, err := io.Copy(h, f); err != nil {
			fmt.Fprintf(h, "\x00%v\x00", err)
		}
		f.Close()
	}
	return strconv.FormatUint(h.Sum64(), 16)
}

// Bounds on projectSig's cost per call: content is hashed for this many changed
// files up to this size; past them size and mtime stand in.
const (
	projectSigMaxFiles = 500
	projectSigMaxBytes = 8 << 20
)

func (a *agent) announceToolCall(ctx context.Context, sid string, tc toolCall) {
	// Only MCP tools and the card-less few lack a card that already shows the arguments.
	switch name := tc.Function.Name; {
	case strings.Contains(name, "__"), name == "view_image":
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
	// Longer than any backtick run inside, so an argument cannot close the block early.
	fence := "```"
	for strings.Contains(shown, fence) {
		fence += "`"
	}
	a.say(ctx, sid, fmt.Sprintf("\n**%s**\n%sjson\n%s\n%s\n", tc.Function.Name, fence, shown, fence))
}

func (a *agent) runToolLoopSeeded(ctx context.Context, sid string, conn *LLMConnection, messages []llmMessage, policy phasePolicy, phase string, stream bool, failSoftCap int) (toolLoopResult, error) {
	// `stream` gates the TEXT channel only: reasoning is the sole live signal during a
	// long call, and never the machinery stream=false hides.
	var on, think func(string)
	flushStream := func() {} // no-op unless streaming; flushes the batched tail
	if sid != "" {
		var flushOn, flushThink func()
		if stream {
			on, flushOn = throttledStream(func(chunk string) {
				// Not through say: the text is logged whole in its RESPONSE block.
				a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: chunk}})
			})
		}
		think, flushThink = throttledStream(func(chunk string) {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentThought, Content: ContentBlock{Type: "text", Text: chunk}})
		})
		flushStream = func() {
			if flushOn != nil {
				flushOn()
			}
			flushThink()
		}
	}
	tools := a.tools.defs()

	// Only jobs this loop's own run_command calls handed over can park it.
	loopStart := time.Now()

	var res toolLoopResult
	var allText strings.Builder
	nudged := false // the last reply was prose and got its contract nudge
	finish := func(err error) (toolLoopResult, error) {
		res.Text = allText.String()
		res.Content = res.Text
		return res, err
	}

	// Returns the messages too: a corrective turn or a fold rewrites them.
	callModel := func(sess *Session, messages []llmMessage) (string, []toolCall, []llmMessage, error) {
		// Each fold step strictly shrinks the context, so recovery terminates.
		recoverStep := 0
		recoverKeepFrom := []func(*Session) int{
			func(s *Session) int { return s.keepWindowStart(keepSmallTurnTokens) },
			(*Session).lastAssistantIndex,
		}
		// Tool-loop calls append to each other, which the rewind check relies on.
		callConn := *conn
		callConn.cacheLineage = true
		capNudged, capDoubled := false, false // per round: one be-concise nudge, then one doubled cap
		transientRetries, refusals := 0, 0
		var downFor time.Duration
		for {
			// Fresh sink per attempt: an aborted attempt's partial arguments must not carry over.
			text, calls, _, err := a.llmStream(ctx, sid, &callConn, messages, tools, on, think, a.planTableSink(ctx, sid))
			flushStream()
			if err == nil {
				return text, calls, messages, nil
			}
			var ce *capHitError
			if errors.As(err, &ce) {
				switch {
				case !capNudged:
					capNudged = true
					a.logSession(sid, "RECOVER", "generation hit the max_tokens cap (%d) — retrying with a be-concise nudge", ce.Cap)
					if sid != "" {
						a.say(ctx, sid, "⚠ Reply hit the output-token cap; retrying with a be-concise instruction.\n")
					}
					messages = a.addCorrective(sid, messages, fmt.Sprintf(
						"Your previous response was cut off at the %d-token output limit and was DISCARDED — nothing of it was applied. Respond again, keeping the output well under that limit: be concise. If you are writing a large file, write it in parts: write_file with the first part, then extend it with edit_file.", ce.Cap))
					continue
				case !capDoubled:
					// A file that needs more than the cap cannot be made concise. The cap is
					// a sampler setting, so the doubled retry keeps the prefix cache.
					capDoubled = true
					callConn = *callConn.withBody("max_tokens", 2*ce.Cap)
					a.logSession(sid, "RECOVER", "still at the cap after the nudge: one retry with max_tokens=%d", 2*ce.Cap)
					if sid != "" {
						a.say(ctx, sid, fmt.Sprintf("⚠ Still at the cap; retrying once with max_tokens=%d.\n", 2*ce.Cap))
					}
					continue
				}
				return "", nil, messages, err
			}
			if isTransientStreamError(err) {
				wait := transientStreamBackoff
				switch refused := strings.Contains(err.Error(), "connection refused"); {
				case refused && downFor >= serverDownPatience:
					return "", nil, messages, fmt.Errorf("the LLM server refused connections for %s; start it, then send the message again: %w", humanDuration(downFor.Milliseconds()), err)
				case refused:
					wait = min(transientStreamBackoff<<refusals, time.Minute)
					refusals++
					downFor += wait
					a.logSession(sid, "RECOVER", "server refused the connection (%v): retry in %s", err, wait)
					if sid != "" && refusals == 1 {
						a.say(ctx, sid, fmt.Sprintf("\n⟲ The LLM server refuses connections (restarting?); retrying for up to %s.\n", humanDuration(serverDownPatience.Milliseconds())))
					}
				case transientRetries >= maxTransientStreamRetries:
					return "", nil, messages, fmt.Errorf("lost the connection to the LLM mid-response %d times (the server or router dropped the stream); this is usually transient — try again in a moment", transientRetries+1)
				default:
					transientRetries++
					a.logSession(sid, "RECOVER", "stream dropped mid-response (%v) — retry %d/%d", err, transientRetries, maxTransientStreamRetries)
					if sid != "" {
						// Streaming cannot rewind, so the retry's repeated prefix needs this flag.
						a.say(ctx, sid, "\n⟲ Connection dropped mid-response; reconnecting.\n")
					}
				}
				select {
				case <-time.After(wait):
					continue
				case <-ctx.Done():
					return "", nil, messages, ctx.Err()
				}
			}
			if !isContextFull(err) || // 400 reject OR n_ctx-ceiling truncation
				(phase != "plan" && phase != "execute") || sess == nil {
				return "", nil, messages, err
			}
			folded := false
			for recoverStep < len(recoverKeepFrom) {
				keepFrom := recoverKeepFrom[recoverStep](sess)
				recoverStep++
				if a.foldHistory(ctx, sess, keepFrom) {
					folded = true
					break
				}
			}
			if !folded {
				return "", nil, messages, err
			}
			messages = a.buildLLMContext(sess)
		}
	}

	// Consecutive stuck rounds climb one ladder (nudge, warm the sampler, bail); any
	// productive round resets it, so read-after-write and fan-out are never punished.
	// An earlier step of this turn was ended repeating these calls; its successor
	// copied the same call from the history, five steps in a row on one /spec item.
	var stuckBefore map[string]string
	var stuckOutputs map[uint64]bool
	cwd := ""
	loopSess := a.getSession(sid) // nil for a sessionless internal pass
	if loopSess != nil {
		cwd = loopSess.Cwd
		loopSess.rt.mu.Lock()
		stuckBefore, stuckOutputs = maps.Clone(loopSess.rt.stuckCalls), maps.Clone(loopSess.rt.stuckOutputs)
		loopSess.rt.mu.Unlock()
	}
	repeats := newRepetitionTracker(cwd, stuckOutputs)
	type repeatedCall struct {
		key, output string
		hash        uint64
	}
	var roundRepeats []repeatedCall // this round's calls that repeated
	var hitStuck bool               // one of them was an output that ended an earlier step
	var stuckRounds int
	var nudgedUI bool // the "repeating" UI warning fires only once
	var escalated bool
	var failedRounds int
	var readStreak int // execute calls since the last edit, test or build
	limit, extended := maxToolLoopIterations, false
	readNudged := false
	// Samplers do not enter the KV cache key, so the prefix survives unless a role
	// carries chat_template_kwargs (see res/settings.toml).
	escalate := func() bool {
		if escalated || conn == nil || conn.Tag == "thinking" {
			return false
		}
		thinkConn := a.connFor("thinking")
		if thinkConn == nil {
			return false
		}
		// Without it the thinking role may answer in prose, ending an execute loop. It
		// is a grammar, not a rendering, on llama.cpp and Halogen alike.
		if tc, ok := conn.ExtraBody["tool_choice"]; ok {
			thinkConn = thinkConn.withBody("tool_choice", tc)
		}
		conn = thinkConn
		escalated = true
		return true
	}
	for iter := 0; ; iter++ {
		if iter >= limit {
			// A step still editing and building at the cap is big work, not a spiral (the
			// read tripwire ends those): two such steps building a page hit it in one morning,
			// and each cost a replan that had to find out again where things stood.
			edits, runs := 0, 0
			for _, u := range res.ToolUses[max(0, len(res.ToolUses)-productiveWindow):] {
				switch {
				case (u.Name == "edit_file" || u.Name == "write_file") && strings.HasPrefix(u.Output, "file written"):
					edits++
				case u.Name == "run_command" && !onlyReads(parseArgs(u.Input).str("command")):
					runs++
				}
			}
			if extended || phase != "execute" || edits < 2 || runs < 1 {
				return finish(fmt.Errorf("tool loop exceeded %d iterations", limit))
			}
			extended = true
			limit += productiveExtension
			messages = a.addCorrective(sid, messages, fmt.Sprintf(
				"You have used %d calls on this step. You are still making progress, so you get %d more, and no more after that: finish the step, run its check, and call `respond`. What does not fit goes into `respond` as the next step.", iter, productiveExtension))
			a.say(ctx, sid, fmt.Sprintf("\n⏱ Step at %d calls and still editing and building: %d more to finish it.\n", iter, productiveExtension))
		}
		sess := a.getSession(sid)
		// Steering and finished jobs go in mid-turn, or the model polls for jobs, as one
		// plain user message: an append, so the prefix cache holds.
		if sess != nil {
			if items := sess.takePending(); len(items) > 0 {
				text, _ := a.sayPending(ctx, sid, items, true)
				messages = a.addCorrective(sid, messages, text)
			}
		}
		// Planning rounds are the most expensive thing codehalter does.
		if phase == "plan" && iter == planRoundNudge {
			messages = a.addCorrective(sid, messages, fmt.Sprintf("You have gathered for %d rounds. Call `submit_plan` NOW with what you have: name the files and the approach, and say in each subtask what you did not verify; the executor reads and checks cheaply.", iter))
			a.say(ctx, sid, "\n⏱ Planning past "+fmt.Sprint(planRoundNudge)+" rounds: asked to submit with what it has.\n")
		}

		text, calls, rewritten, err := callModel(sess, messages)
		messages = rewritten
		if err != nil {
			if ctx.Err() == nil {
				err = &llmCallError{err}
			}
			return finish(err)
		}
		allText.WriteString(text)

		// One message per iteration, not a merged turn, so a replay reproduces the wire.
		if sess != nil {
			sess.AddAssistant(text)
			sess.recordLastPromptTokens() // stamp the call's prompt_tokens for keepWindowStart
			sess.saveOrLog()
		}

		// Prose ends the loop where the phase allows it; elsewhere one nudge per slip.
		if len(calls) == 0 {
			if policy.proseEnds || len(policy.terminals) == 0 || nudged {
				return finish(nil)
			}
			nudged = true
			messages = append(messages, llmMessage{Role: "assistant", Content: text})
			messages = a.addCorrective(sid, messages, contractNudge(policy, text))
			a.say(ctx, sid, "\n⚠ The model replied without the tool call this step needs; asking it once more.\n")
			continue
		}
		nudged = false

		messages = append(messages, llmMessage{
			Role:      "assistant",
			Content:   text,
			ToolCalls: calls,
		})

		// The rest of the batch still runs, so its results land in history.
		var terminalCalled bool
		var terminalName string
		var terminalMessage string
		var failedThisRound bool
		roundRepeats, hitStuck = nil, false
		// True only if EVERY call this round reproduced known output.
		roundStuck := len(calls) > 0

		// A build or test run is the gap in which a server reclaims the idle slot.
		stopWarm := func() {}
		if sess != nil {
			sent := messages
			stopWarm = a.keepWarm(sess, conn, func() []llmMessage { return sent })
			sess.markReplyStart()
		}
		for _, tc := range calls {
			// Terminal payloads are already rendered: submit_plan's table, respond's text.
			if !policy.terminals[tc.Function.Name] {
				a.announceToolCall(ctx, sid, tc)
			}

			stopMeter := a.startToolMeter(ctx, sid, tc)

			var tu ToolUse
			var content any
			denied := policy.deny[tc.Function.Name]
			key := callKey(tc)
			before, wasStuck := stuckBefore[key]
			switch {
			case denied:
				var msg string
				tu, msg = a.denyToolCall(ctx, sid, phase, tc)
				content = msg
			// Once the step wrote something, the same call may answer differently: a re-check.
			case wasStuck && repeats.changes == 0:
				msg := "not run: an earlier attempt at this task ran exactly this call again and again until codehalter ended it, so it answers nothing new. Its output, which you already have:\n\n" +
					before + "\n\nDo what the task asks with a different call; if the task asks you to look at a picture, call `screenshot` on it."
				tcId := a.StartToolCall(ctx, sid, tc.Function.Name+" (repeated from an ended attempt)", "tool", nil)
				a.FailToolCall(ctx, sid, tcId, "not run: this call ended an earlier attempt by repeating")
				tu, content = a.recordToolUse(sid, tc, ToolUse{Output: msg, Failed: true, StartedAt: time.Now()}), msg
			default:
				tu, content = a.runToolCall(ctx, sid, tc)
			}
			stopMeter()
			res.ToolUses = append(res.ToolUses, tu)
			// A denied call is the model's mistake, not a tool failure.
			if tu.Failed && !denied {
				failedThisRound = true
			}
			changed := repeats.changedBy(tc, tu)
			res.ToolUses[len(res.ToolUses)-1].Changed = changed
			if !repeats.sawAgain(tc, tu, changed) {
				roundStuck = false
			} else {
				roundRepeats = append(roundRepeats, repeatedCall{key, tu.Output, fnvHash(repeatText(tc.Function.Arguments, tu.Output))})
				hitStuck = hitStuck || repeats.hitStuck
			}
			if phase == "execute" && !denied {
				switch name := tc.Function.Name; {
				case name == "read_file", name == "web_search", name == "web_read",
					name == "run_command" && onlyReads(parseArgs(tc.Function.Arguments).str("command")):
					readStreak++
				case name == "edit_file", name == "write_file", name == "run_command", name == "run_background":
					readStreak, readNudged = 0, false
				}
			}
			if policy.terminals[tc.Function.Name] && !terminalCalled {
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

		// The terminal message came as tool arguments, never through the text stream.
		if terminalCalled {
			// A respond while this loop's background job still runs parks the turn until
			// the job's note resumes it.
			if terminalName == respondToolName {
				if jobs := a.parkableJobs(sid, loopStart); jobs != "" {
					if on != nil && terminalMessage != "" {
						on(terminalMessage)
						flushStream()
					}
					resume, perr := a.parkForJobs(ctx, sid, jobs)
					if perr != nil {
						res.Text, res.Content = terminalMessage, allText.String()
						return res, perr
					}
					messages = a.addCorrective(sid, messages, resume)
					continue
				}
			}
			switch {
			case on == nil:
				// Silent internal pass, such as the planner.
			case terminalMessage != "":
				on(terminalMessage)
				flushStream() // the FINAL emit: never leave it batched
			default:
				a.say(ctx, sid, "(done)\n")
			}
			res.Text, res.Content, res.Terminal = terminalMessage, allText.String(), terminalName
			return res, nil
		}

		// No Terminal, so runExecutePhase records a failure to replan against.
		if failedThisRound && failSoftCap > 0 {
			failedRounds++
			if failedRounds >= failSoftCap {
				return finish(nil)
			}
		}

		// The nudge always comes first, even when one reply of parallel reads jumps past both marks.
		if readStreak >= readStreakBail && readNudged {
			return finish(fmt.Errorf("read %d calls in a row without an edit, a test or a build", readStreak))
		}
		if readStreak >= readStreakEscalate && !readNudged {
			readNudged = true
			messages = a.addCorrective(sid, messages, fmt.Sprintf(
				"You have read for %d calls in a row without changing a file or running a test. Stop gathering: make the edit, run the test, or call `respond` with what blocks you. At %d reads in a row this subtask ends and is planned again.", readStreak, readStreakBail))
			if escalate() {
				a.say(ctx, sid, "⚠ Reading without acting: nudged, and switched to the thinking sampler.\n")
			} else {
				a.say(ctx, sid, "⚠ Reading without acting: nudged the model to act.\n")
			}
		}

		if !roundStuck {
			stuckRounds = 0
			continue
		}
		stuckRounds++
		if stuckRounds >= stuckBailRounds {
			if loopSess != nil {
				loopSess.rt.mu.Lock()
				if loopSess.rt.stuckCalls == nil {
					loopSess.rt.stuckCalls, loopSess.rt.stuckOutputs = map[string]string{}, map[uint64]bool{}
				}
				for _, c := range roundRepeats {
					loopSess.rt.stuckCalls[c.key] = c.output
					loopSess.rt.stuckOutputs[c.hash] = true
				}
				loopSess.rt.mu.Unlock()
			}
			// No error: the normal failure paths (replan, plan salvage) beat a hard error.
			return finish(nil)
		}
		if !nudgedUI && sid != "" {
			nudgedUI = true
			a.say(ctx, sid, "⚠ Repeating with no new information — nudging the model to change course.\n")
		}
		if hitStuck {
			// The call differs from the one that ended the earlier attempt; its answer does not.
			messages = a.addCorrective(sid, messages,
				"That output is the one an earlier attempt at this task got again and again until codehalter ended it. Changing the command does not change the answer. Do what the task asks with a different call; if it asks you to look at a picture, call `screenshot` on it.")
			continue
		}
		messages = a.addCorrective(sid, messages,
			"Your last tool call(s) returned output you already have — that makes no progress. Do NOT repeat them. Instead:\n"+
				"1. If a read came back PARTIAL and you need more, make the read_file call its note names for the next part; never re-read the same window, never rewrite a whole file.\n"+
				"2. Act on what you already have: make a small targeted edit_file, run a DIFFERENT command, or finish by calling the terminal tool.\n"+
				"3. If you are stuck or the task is infeasible, say so and stop.")
		if stuckRounds >= stuckEscalateRounds && escalate() && sid != "" {
			a.say(ctx, sid, "⚠ Still repeating — switching to the thinking sampler to break out.\n")
		}
	}
}
