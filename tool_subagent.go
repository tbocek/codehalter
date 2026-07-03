package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// subagentTaskHash is a stable digest of an instructions+context pair, used
// to dedupe launch_subagent calls. We normalise aggressively (lowercase,
// strip code fences, collapse whitespace) because small models love to
// re-issue the same task with cosmetically different framing (e.g. wrapping
// the context in ```...``` the second time).
func subagentTaskHash(t subagentTask) string {
	h := sha256.New()
	h.Write([]byte(normalizeForHash(t.Instructions)))
	h.Write([]byte{0})
	h.Write([]byte(normalizeForHash(t.Context)))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

func normalizeForHash(s string) string {
	s = strings.ToLower(s)
	s = strings.ReplaceAll(s, "`", "")
	var b strings.Builder
	b.Grow(len(s))
	prevSpace := true
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
			continue
		}
		b.WriteRune(r)
		prevSpace = false
	}
	return strings.TrimSpace(b.String())
}

const maxSubagentDepth = 2

// subagentFanoutBudget is the per-concurrent-subagent token allowance used to
// derive the context-based fan-out cap: reserve half the context window for
// the foreground conversation (and its summariser), then give each concurrent
// subagent ~20k tokens of prompt + generation headroom in what remains. A
// 160k window caps the fan-out at 4; at the 32k minimum the cap floors at 1.
// This keeps concurrent subagents from crowding the foreground's KV cache out
// of the server's pool (the eviction is what breaks prefix-cache consistency);
// the per-conn semaphores still bound per-server load underneath, and tasks
// past the cap queue rather than drop.
const subagentFanoutBudget = 20_000

type subagentTask struct {
	Instructions string `json:"instructions"`
	Context      string `json:"context,omitempty"`
	// Task selects the subagent's flow:
	//   "execute" (default) — runs the tool loop only. Use for surgical,
	//     pre-specified work the parent has already planned: one edit, one
	//     lookup, one bounded command. The subagent must NOT plan or verify.
	//   "thinking" — runs plan → execute → verify on the subagent's own
	//     session. Use when the subtask is itself a small task that needs
	//     its own planning and self-check (e.g. "investigate X and apply
	//     the fix").
	Task string `json:"task,omitempty"`
}

type subagentResult struct {
	Index   int    `json:"index"`
	Success bool   `json:"success"`
	Result  string `json:"result"`
	Error   string `json:"error,omitempty"`
}

// registerSubagentTool registers the launch_subagent tool if not already registered.
func (a *agent) registerSubagentTool() {
	if hasToolPrefix("launch_subagent") {
		return
	}

	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "launch_subagent",
			"description": "Run 2+ short, independent tasks in parallel. Each task has a `task` field selecting its flow: `execute` (default) runs ONLY the tool loop — use for surgical edits, one lookup, one bounded command the caller has already planned; `thinking` runs plan → execute → verify on the subagent itself — use when the subtask needs its own planning and self-check. Subagents cannot talk to the user. The first subagent in a batch inherits this conversation's full history (it runs on the main LLM, its prefix cache is already warm); the rest start fresh on extra LLM slots and see only their own `instructions` + `context`. Parallelism is bounded by the sum of `parallel` across configured [[llm]] entries and by a context-window budget (concurrent subagents share the server's KV space with this conversation); excess tasks queue.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"tasks"},
				"properties": map[string]any{
					"tasks": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"instructions": map[string]any{"type": "string", "description": "The exact task this subagent should perform. Phrase it as a self-contained order — the subagent has no other source of intent."},
								"context":      map[string]any{"type": "string", "description": "EVERY fact the subagent needs to start working immediately: exact file paths, the literal text to find/replace, version numbers, error strings, relevant snippets from prior tool output. The subagent does NOT see this conversation or earlier tool results — anything it has to discover on its own is wasted time."},
								"task":         map[string]any{"type": "string", "enum": []string{"execute", "thinking"}, "description": "`execute` (default): subagent runs the tool loop only, no plan or verify — the caller already planned this. `thinking`: subagent runs plan → execute → verify on its own session — only for subtasks that need independent planning."},
							},
							"required": []string{"instructions"},
						},
					},
				},
			},
		},
	}, Execute: func(ctx context.Context, ag *agent, sid string, rawArgs string) (string, bool) {
		// Parse tasks from raw JSON arguments.
		var parsed struct {
			Tasks []subagentTask `json:"tasks"`
		}
		if err := json.Unmarshal([]byte(rawArgs), &parsed); err != nil {
			return "error: invalid JSON: " + err.Error(), false
		}
		tasks := parsed.Tasks
		if len(tasks) == 0 {
			return "error: no tasks provided", false
		}

		// Check depth.
		sess := ag.getSession(sid)
		if sess == nil {
			return "error: no session", false
		}
		if sess.Depth >= maxSubagentDepth {
			return "error: maximum subagent nesting depth reached", false
		}

		tcId := ag.StartToolCall(ctx, sid, fmt.Sprintf("Launching: %d subagent(s)", len(tasks)), "execute", nil)

		// Dedup tasks. Small models frequently re-issue identical
		// launch_subagent calls (within one tasks[] array, or across
		// successive turns). We hash (instructions+context) and:
		//   - within this call: collapse duplicates to one launch, then fan
		//     the same result out to every original index.
		//   - across calls in this session: return the prior result instead
		//     of relaunching. Cache lives on the parent Session (in-memory
		//     only — see Session.launchedSubagents docs).
		results := make([]subagentResult, len(tasks))
		var mu sync.Mutex

		type pending struct {
			task    subagentTask
			indices []int   // original task[] indices sharing this hash
			pin     PinSlot // (conn, slot) assigned at first dispatch; stable across relaunches
			ctxFull bool    // last attempt failed on a context-full error (fold-and-relaunch candidate)
		}
		uniq := make(map[string]*pending)
		var order []string // stable launch order: first-seen indices win

		for i, task := range tasks {
			h := subagentTaskHash(task)
			if cached, ok := sess.recallSubagent(h); ok {
				results[i] = subagentResult{Index: i, Success: true, Result: cached}
				ag.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("[subagent %d] Reusing prior result for identical task\n\n", i+1)}})
				continue
			}
			if p, ok := uniq[h]; ok {
				p.indices = append(p.indices, i)
				continue
			}
			uniq[h] = &pending{task: task, indices: []int{i}}
			order = append(order, h)
		}

		// Parallelism is bounded by the sum of `parallel` across configured
		// [[llm]] entries. SubagentPinOrder returns a breadth-first interleave
		// of (conn, slot) pairs: for caps [1, 3] it yields
		// [{0,0}, {1,0}, {1,1}, {1,2}] — task 0 pins to LLM[0] (warm parent
		// cache benefit), tasks 1..3 to LLM[1] slots 0..2. We dispatch
		// len(order) tasks but cap concurrency at len(pinOrder), and within
		// each task the per-call semaphore in llmStream throttles further if
		// the conn is already saturated by another subagent's nested calls.
		pinOrder := ag.settings.SubagentPinOrder()
		if len(pinOrder) == 0 {
			pinOrder = []PinSlot{{Conn: 0, Slot: 0}}
		}
		fanout := len(pinOrder)
		if fanout > len(order) {
			fanout = len(order)
		}
		// Context-derived fan-out cap: each concurrent subagent occupies KV
		// pool next to the foreground conversation, and pool overflow is what
		// evicts the foreground's cached prefix. Reserve half the window for
		// the foreground, budget subagentFanoutBudget tokens per concurrent
		// subagent for the rest — 160k → 4. Purely a concurrency cap: tasks
		// past it queue inside parallel(), nothing is dropped.
		if mst := ag.getMainSlotTokens(); mst > 0 {
			ctxCap := (mst / 2) / subagentFanoutBudget
			if ctxCap < 1 {
				ctxCap = 1
			}
			if fanout > ctxCap {
				fanout = ctxCap
			}
		}

		// Pin every unique task up-front in breadth-first order. The pin
		// survives the whole subagent run so the conn's prefix cache stays
		// warm across the tool loop, and it's stored on the pending so a
		// fold-and-relaunch keeps the task on the same conn/slot. Tasks past
		// pinOrder's length wrap modulo — the per-conn semaphore in llmStream
		// is what actually enforces the cap, so wrapping just means "extra
		// tasks queue on already-assigned conns".
		for k, h := range order {
			uniq[h].pin = pinOrder[k%len(pinOrder)]
		}

		runOne := func(h string) {
			p := uniq[h]
			primary := p.indices[0]
			task := p.task
			pin := p.pin
			subSess := newSubagentSession(sess.Cwd, sid, primary, sess.Depth+1, pin.Conn)
			subSess.DisplayLabel = fmt.Sprintf("llm%d@%d/%d", primary+1, pin.Conn, pin.Slot)
			ag.putSession(subSess)
			defer ag.deleteSession(subSess.ID)

			ag.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("[%s] Starting: %s\n\n", subSess.DisplayLabel, truncate(task.Instructions, 100))}})

			// All internal plan/execute/verify calls use the subagent's own
			// session id. That keeps plan steps, tool uses, and assistant
			// output in subSess.toml — and because Zed does not know about
			// that session id, the chat/message notifications the tool loop
			// emits are silently dropped (sendUpdate). The subagent therefore
			// runs without polluting the parent conversation; we surface only
			// the final result (the tool return value below) plus a live
			// progress feed: per-tool breadcrumbs (loop.go) and the folded
			// status meter (setStatus -> setSubagentStatus) on the parent row.
			result, err := ag.runSubagent(ctx, subSess, task)

			mu.Lock()
			p.ctxFull = err != nil && isContextFull(err)
			if err != nil {
				for _, i := range p.indices {
					results[i] = subagentResult{Index: i, Success: false, Error: err.Error()}
				}
			} else {
				sess.rememberSubagent(h, result)
				for _, i := range p.indices {
					results[i] = subagentResult{Index: i, Success: true, Result: result}
				}
			}
			mu.Unlock()

			if len(p.indices) > 1 {
				ag.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("[%s] Done (shared with %d duplicate task(s))\n\n", subSess.DisplayLabel, len(p.indices)-1)}})
			} else {
				ag.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("[%s] Done\n\n", subSess.DisplayLabel)}})
			}
		}
		runBatch := func(hs []string) {
			f := fanout
			if f > len(hs) {
				f = len(hs)
			}
			parallel(len(hs), f, func(k int) { runOne(hs[k]) })
		}
		runBatch(order)

		// Context-full recovery: a subagent that overflowed (its own inherited
		// history, or the server's shared KV pool crowded by the foreground)
		// is relaunched after folding the FOREGROUND session — the parent's
		// history is what both the pool and the history-inheriting subagent
		// are full of, and the parent was heading for the same fold on its own
		// next overflow anyway. Same escalation steps as the tool loop's 400
		// recovery; one relaunch per fold step, then the failure stands and
		// the orchestrator replans. Depth-gated like that recovery: only the
		// real foreground folds — a depth-1 parent just fails upward.
		if sess.Depth == 0 {
			recoverKeepFrom := []func(*Session) int{
				func(s *Session) int { return s.keepWindowStart(keepSmallTurnTokens) },
				(*Session).lastAssistantIndex,
			}
			for _, keepFrom := range recoverKeepFrom {
				var retry []string
				for _, h := range order {
					if p := uniq[h]; p.ctxFull {
						retry = append(retry, h)
					}
				}
				if len(retry) == 0 {
					break
				}
				if !ag.foldHistory(ctx, sess, keepFrom(sess)) {
					continue // nothing freed at this step — escalate to the next
				}
				ag.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: fmt.Sprintf("⚠ %d subagent(s) hit the context limit — compacted the conversation, relaunching.\n", len(retry))}})
				runBatch(retry)
			}
		}

		// Format results. Surface failure to the orchestrator's verdict: when EVERY
		// subagent failed the whole fan-out is a real failure (failed=true, so it can
		// override an LLM "success=true" and trigger a replan). A partial failure
		// stays non-failing — some work landed, and the FAILED markers below let the
		// model react without discarding the successes.
		var out strings.Builder
		failures := 0
		for _, r := range results {
			if r.Success {
				fmt.Fprintf(&out, "=== Subagent %d ===\n%s\n\n", r.Index+1, r.Result)
			} else {
				failures++
				fmt.Fprintf(&out, "=== Subagent %d (FAILED) ===\n%s\n\n", r.Index+1, r.Error)
			}
		}

		ag.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("%d subagent(s) completed", len(tasks)))})
		return out.String(), len(results) > 0 && failures == len(results)
	}})
}

// subagentRules is the execute-mode subagent's entire phase prompt.
// Deliberately tiny: execute-mode subagents are leaf workers, not
// investigators. They must do exactly what `instructions` says — no extra
// `run_task` "to verify", no re-reading files they just wrote, no chasing
// failures that aren't their job. The parent already planned the work and
// the parent's executor runs its own self-verify recipe on the overall
// result; a subagent that goes broader wastes the parallelism it was
// forked for. Thinking-mode subagents do NOT see these rules — they run
// their own plan + per-subtask cycle.
const subagentRules = `You are a subagent in EXECUTE mode. Do exactly the task in "Task:" — nothing more.

Hard rules:
- Do NOT run any verify-class command (` + "`" + `just:verify` + "`" + `, ` + "`" + `npm:ci` + "`" + `,
  ` + "`" + `make:check` + "`" + `, ` + "`" + `cargo:test` + "`" + `, etc.). The parent will verify.
- Do NOT investigate or report on failures unrelated to your task.
- Do NOT re-read a file after editing it — your edit succeeded if
  edit_file/write_file returned success. Trust your own tool successes.
- Do NOT explain what you did at length. End with a single short line
  stating the outcome (e.g. "Edited <file>: <old> → <new>").
- If your task is impossible or ambiguous, return one sentence
  explaining why. Do NOT improvise a different task.

Use only the tools needed for "Task:". Typical subagents make 1-3 tool
calls and then stop.`

// runSubagent dispatches to the execute- or thinking-mode flow based on
// task.Task. Default is "execute" — leaf worker, tool loop only. "thinking"
// is for subtasks that need their own plan → execute → verify cycle.
func (a *agent) runSubagent(ctx context.Context, subSess *Session, task subagentTask) (string, error) {
	switch strings.ToLower(strings.TrimSpace(task.Task)) {
	case "thinking":
		return a.runSubagentThinking(ctx, subSess, task)
	case "", "execute":
		return a.runSubagentExecute(ctx, subSess, task)
	default:
		return "", fmt.Errorf("invalid task type %q (use \"execute\" or \"thinking\")", task.Task)
	}
}

// runSubagentExecute runs the tool loop only against subSess — no plan, no
// verify. The parent already planned the work and will verify the overall
// result, so this is just a short, scoped tool-loop runner. All internal
// calls use subSess.ID so tool uses and assistant text are saved there (not
// in the parent). UI updates emitted by the tool loop target subSess.ID too
// and are dropped by the client since it has never been announced — only the
// returned result string flows back to the parent.
//
// Bypasses agent.execute on purpose: that path injects the full EXECUTE.md
// (which encourages investigating failures, reading project files, etc.) and
// the verify-target hint. Execute-mode subagents need the opposite — a tight
// "do only what's asked" prompt. We build the message and tool exclusions
// directly.
//
// History inheritance: when pinned to LLM[0] we prepend the parent's full
// message history so the byte-identical prefix keeps that conn's KV cache
// warm. Subagents on LLM[1+] start fresh — those conns hold no relevant
// prefix and seeding parent history would just bloat their context for no
// cache benefit. This matters because LLM[0] is the dispatch target for the
// first task in every breadth-first fan-out.
// framed renders the subagent's prompt: the Task, prefixed with Context when set.
func (t subagentTask) framed() string {
	if t.Context != "" {
		return "Context:\n" + t.Context + "\n\nTask:\n" + t.Instructions
	}
	return "Task:\n" + t.Instructions
}

func (a *agent) runSubagentExecute(ctx context.Context, subSess *Session, task subagentTask) (string, error) {
	sid := subSess.ID

	instructions := task.framed()

	subSess.AddUser(instructions)
	subSess.saveOrLog()

	var b strings.Builder
	if sysPrompt, err := a.systemPrompt(sid); err == nil && sysPrompt != "" {
		b.WriteString(sysPrompt)
		b.WriteString("\n---\n")
	}
	b.WriteString(subagentRules)
	b.WriteString("\n---\n")
	b.WriteString(instructions)

	var messages []llmMessage
	if subSess.PinnedLLMIdx == 0 {
		// Inherit parent's history so LLM[0]'s prefix cache stays warm. The
		// new instructions land as a fresh user turn at the end. The parent
		// session's lock is acquired briefly to copy; tool dispatch keeps
		// running on the parent in parallel, but we only read the snapshot.
		if parent := a.getSession(subSess.ParentID); parent != nil {
			messages = a.buildLLMContext(parent)
		}
	}
	messages = append(messages, llmMessage{Role: "user", Content: b.String()})

	// Subagents can't interact with the user (UI is dropped), so ask_user
	// would hang. Web tools were planner-phase only — execute-tier work has
	// no business going to the web. launch_subagent is excluded at max depth
	// to bound recursion.
	// Subagent restrictions are enforced at dispatch (the tools array stays the
	// full superset): no ask_user / web / submit_plan, and no further fan-out at
	// max depth. respond is the exit.
	deny := map[string]bool{
		"ask_user":         true,
		"web_search":       true,
		"web_read":         true,
		"web_read_raw":     true,
		submitPlanToolName: true,
	}
	if subSess.Depth >= maxSubagentDepth {
		deny["launch_subagent"] = true
	}
	policy := phasePolicy{deny: deny, terminals: map[string]bool{respondToolName: true}}

	result, err := a.runToolLoopSeeded(ctx, sid, a.connForSession(ctx, sid, "execute"), messages, policy, "subagent", true, 0)
	if err != nil {
		return result.Text, err
	}

	// The tool loop incrementally appends tool uses to the last assistant
	// message via AppendToolUse; here we just set that message's Content to
	// the final text (or add a new assistant message if none exists).
	if result.Text != "" {
		if n := len(subSess.Messages); n > 0 && subSess.Messages[n-1].Role == "assistant" {
			subSess.Messages[n-1].Content = result.Text
		} else {
			subSess.AddAssistantWithTools(result.Text, result.ToolUses)
		}
		subSess.saveOrLog()
	}

	return result.Text, nil
}

// runSubagentThinking runs a plan + per-subtask cycle on subSess. Unlike the
// parent's orchestrate, this is a single pass: plan, then each subtask loop
// in order, then return whatever the last subtask produced. Replan-on-failure
// and DOCUMENT.md belong to the parent — a thinking subagent surfaces its
// result and lets the parent decide what to do with a subtask failure.
//
// The subagent's session is seeded like the parent's: SystemPrompt is stored
// on subSess so buildLLMContext folds it into LLM[0]'s first message. We do
// NOT inherit the parent's message history here — thinking subagents own
// their cycle and shouldn't get confused by the parent's open task. Cache
// warmth on LLM[0] is sacrificed for clarity; if it becomes a problem we can
// revisit by copying parent messages.
//
// User-facing prompts inside runPlanPhase (clarification choices) auto-confirm
// because shouldAutoAnswer returns true for any session with Depth > 0 (see
// tool_ask.go). The orchestrator's "Execute / Abort" gate is
// skipped here — subagents are dispatched by the parent and never gate on
// the user.
func (a *agent) runSubagentThinking(ctx context.Context, subSess *Session, task subagentTask) (string, error) {
	sid := subSess.ID

	if subSess.SystemPrompt == "" {
		if sysPrompt, err := a.systemPrompt(sid); err == nil {
			subSess.SystemPrompt = sysPrompt
		}
	}

	instructions := task.framed()
	subSess.AddUser(instructions)
	subSess.saveOrLog()

	plan, _, err := a.runPlanPhase(ctx, sid, "")
	if err != nil {
		return "", err
	}
	if plan == nil || len(plan.Subtasks) == 0 {
		return "", nil
	}

	var last toolLoopResult
	for i, st := range plan.Subtasks {
		outcome := a.runExecutePhase(ctx, sid, st, i, len(plan.Subtasks))
		last = outcome.Result
		if !outcome.Success {
			return last.Text, fmt.Errorf("subtask %d/%d failed: %s", i+1, len(plan.Subtasks), outcome.Reason)
		}
	}
	return last.Text, nil
}
