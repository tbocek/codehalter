package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"slices"
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

	if !plan.Clear && len(plan.Choices) > 0 {
		question := plan.Question
		if question == "" {
			question = "I'm not sure what you mean. Which of these?"
		}
		a.say(ctx, sid, question)

		// Under /spec autopilot, taking the first option would let the model settle an
		// open spec point by itself, so the question is parked instead.
		if sess.specFence() != "" && a.isAutopilot() {
			sess.AddUser("Question parked for the user by the spec loop: " + question)
			sess.saveOrLog()
			return nil, &specQuestionError{Question: question, Choices: plan.Choices}
		}

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

// An interleaved revisit counts too, hence a map rather than a last-call comparison.
type repetitionTracker struct {
	// bag catches near-identical output the hash misses, like a timestamp in a failing build.
	hash map[string]uint64
	bag  map[string]map[string]bool
	// writeAt is the writes count when each call last ran.
	writes  int
	writeAt map[string]int
}

// A timestamp or a counter in the output does not make it new.
var digitRunRe = regexp.MustCompile(`\d+`)

// Shell commands that rewrite source files. A redirect is NOT one: it writes a
// log, not the tree.
var inPlaceWriterRe = regexp.MustCompile(`sed -i|-i\b.*\bsed|cargo fmt|gofmt -w|prettier --write|go mod tidy|git (checkout|stash|apply|revert|reset)|cargo add|npm i|apk add|apt-get install`)

// commandWrites: a command that rewrites the tree, including a script that opens a
// file for writing and a heredoc redirected into a file outside /tmp.
func commandWrites(cmd string) bool {
	if inPlaceWriterRe.MatchString(cmd) || scriptWriteRe.MatchString(cmd) {
		return true
	}
	if strings.Contains(cmd, "<<") {
		for _, f := range redirectTargets(cmd, "/") {
			if !strings.HasPrefix(f, "/tmp/") && !strings.HasPrefix(f, "/dev/") {
				return true
			}
		}
	}
	return false
}

// commandKey drops a command that only prints a label (an echo or printf that is
// neither piped nor redirected), and labels is what those echoes print: a new
// label on the same grep is the same grep, in the key and in the output.
func commandKey(cmd string) (key string, labels map[string]bool) {
	var keep []string
	labels = map[string]bool{}
	for _, seg := range shellSegments(cmd, false) {
		first := strings.Fields(seg)[0]
		if bare := unquoted(seg); strings.ContainsAny(bare, "|>") || first != "echo" && first != "printf" && first != "true" && first != ":" {
			keep = append(keep, seg)
			continue
		}
		if first == "echo" {
			text := strings.TrimSpace(strings.TrimPrefix(seg, "echo"))
			text = strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(text, "-e "), "-n "))
			if len(text) >= 2 && (text[0] == '"' || text[0] == '\'') && text[len(text)-1] == text[0] {
				text = text[1 : len(text)-1]
			}
			labels[text] = true
		}
	}
	return strings.Join(keep, " ; "), labels
}

func (rt *repetitionTracker) sawAgain(tc toolCall, tu ToolUse) bool {
	args := tc.Function.Arguments
	out, _, _ := strings.Cut(tu.Output, batchNoteLead) // the note depends on the call before, not on this one
	if tc.Function.Name == "run_command" {
		var labels map[string]bool
		args, labels = commandKey(parseArgs(args).str("command"))
		lines := strings.Split(out, "\n")
		out = strings.Join(slices.DeleteFunc(lines, func(l string) bool { return labels[strings.TrimSpace(l)] }), "\n")
	}
	key := tc.Function.Name + "\x00" + args
	out = digitRunRe.ReplaceAllString(out, "#")
	h := fnvHash(out)
	bag := issueBag([]string{out})
	repeated := false
	if prev, ok := rt.hash[key]; ok && prev == h {
		repeated = true
	} else if prevBag, ok := rt.bag[key]; ok && jaccard(prevBag, bag) >= stuckOutputSimilarity {
		repeated = true
	}
	rt.hash[key] = h
	rt.bag[key] = bag
	// A green run_command re-run after a write is a re-verify. A red one, or a green
	// one with nothing written since its last run, is a spin.
	if rt.writeAt == nil {
		rt.writeAt = map[string]int{}
	}
	// Before this call's own write: a writer re-run with nothing changed since is a spin.
	changedSince := rt.writeAt[key] < rt.writes
	switch tc.Function.Name {
	case "edit_file", "write_file":
		rt.writes++
	case "run_command":
		// The command, not the JSON arguments: there a script's "w" is \"w\".
		if commandWrites(parseArgs(tc.Function.Arguments).str("command")) {
			rt.writes++
		}
	}
	rt.writeAt[key] = rt.writes
	if repeated && !tu.Failed && tc.Function.Name == "run_command" && changedSince {
		repeated = false
	}
	return repeated
}

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
	repeats := &repetitionTracker{hash: map[string]uint64{}, bag: map[string]map[string]bool{}}
	var stuckRounds int
	var nudgedUI bool // the "repeating" UI warning fires only once
	var escalated bool
	var failedRounds int
	var readStreak int // execute calls since the last edit, test or build
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
		if iter >= maxToolLoopIterations {
			return finish(fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations))
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
			switch {
			case denied:
				var msg string
				tu, msg = a.denyToolCall(ctx, sid, phase, tc)
				content = msg
			default:
				tu, content = a.runToolCall(ctx, sid, tc)
			}
			stopMeter()
			res.ToolUses = append(res.ToolUses, tu)
			// A denied call is the model's mistake, not a tool failure.
			if tu.Failed && !denied {
				failedThisRound = true
			}
			if !repeats.sawAgain(tc, tu) {
				roundStuck = false
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
			// No error: the normal failure paths (replan, plan salvage) beat a hard error.
			return finish(nil)
		}
		if !nudgedUI && sid != "" {
			nudgedUI = true
			a.say(ctx, sid, "⚠ Repeating with no new information — nudging the model to change course.\n")
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
