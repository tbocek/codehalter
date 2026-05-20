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

type subagentTask struct {
	Instructions string `json:"instructions"`
	Context      string `json:"context,omitempty"`
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
			"description": "Run 2+ short, independent leaf tasks in parallel. Each subagent runs ONLY the tool loop (no separate plan/verify phase) and reports back — use it for unambiguous single-file edits, one focused lookup, or one bounded command, NOT for multi-step work that needs its own planning. Subagents cannot talk to the user. The first subagent in a batch inherits this conversation's full history (it runs on the main LLM, its prefix cache is already warm); the rest start fresh on extra LLM slots and see only their own `instructions` + `context`. Parallelism is bounded by the sum of `parallel` across configured [[llm]] entries; excess tasks queue.",
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
							},
							"required": []string{"instructions"},
						},
					},
				},
			},
		},
	}, Execute: func(ctx context.Context, ag *agent, sid SessionId, rawArgs string) (string, bool) {
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
			indices []int // original task[] indices sharing this hash
		}
		uniq := make(map[string]*pending)
		var order []string // stable launch order: first-seen indices win

		for i, task := range tasks {
			h := subagentTaskHash(task)
			if cached, ok := sess.recallSubagent(h); ok {
				results[i] = subagentResult{Index: i, Success: true, Result: cached}
				ag.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("[subagent %d] Reusing prior result for identical task\n\n", i+1))))
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
		// of conn indices: for caps [1, 3] it yields [0, 1, 1, 1] — task 0
		// pins to LLM[0] (warm parent cache benefit), tasks 1..3 to LLM[1].
		// We dispatch len(order) tasks but cap concurrency at len(pinOrder),
		// and within each task the per-call semaphore in llmStream throttles
		// further if the conn is already saturated by another subagent's
		// nested calls.
		pinOrder := ag.settings.SubagentPinOrder()
		if len(pinOrder) == 0 {
			pinOrder = []int{0}
		}
		fanout := len(pinOrder)
		if fanout > len(order) {
			fanout = len(order)
		}

		// Launch unique tasks in parallel, each pinned to one [[llm]] entry
		// in breadth-first order. The pin survives the whole subagent run so
		// the conn's prefix cache stays warm across the tool loop. Tasks past
		// pinOrder's length wrap modulo — the per-conn semaphore in llmStream
		// is what actually enforces the cap, so wrapping just means "extra
		// tasks queue on already-assigned conns".
		parallel(len(order), fanout, func(k int) {
			h := order[k]
			p := uniq[h]
			primary := p.indices[0]
			task := p.task
			pinnedIdx := pinOrder[k%len(pinOrder)]
			subSess := newSubagentSession(sess.Cwd, sid, primary, sess.Depth+1, pinnedIdx)
			subSess.DisplayLabel = fmt.Sprintf("subagent %d", primary+1)
			ag.putSession(subSess)
			defer ag.deleteSession(subSess.ID)

			ag.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("[subagent %d] Starting: %s\n\n", primary+1, truncate(task.Instructions, 100)))))

			// All internal plan/execute/verify calls use the subagent's own
			// session id. That keeps plan steps, tool uses, and assistant
			// output in subSess.toml — and because Zed does not know about
			// that session id, every session/update notification the tool
			// loop emits is silently dropped. The subagent therefore runs
			// without polluting the parent conversation; we surface only the
			// final result (via the tool return value below) to the parent.
			result, err := ag.runSubagent(ctx, subSess, task)

			mu.Lock()
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
				ag.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("[subagent %d] Done (shared with %d duplicate task(s))\n\n", primary+1, len(p.indices)-1))))
			} else {
				ag.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf("[subagent %d] Done\n\n", primary+1))))
			}
		})

		// Format results.
		var out strings.Builder
		for _, r := range results {
			if r.Success {
				fmt.Fprintf(&out, "=== Subagent %d ===\n%s\n\n", r.Index+1, r.Result)
			} else {
				fmt.Fprintf(&out, "=== Subagent %d (FAILED) ===\n%s\n\n", r.Index+1, r.Error)
			}
		}

		ag.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(fmt.Sprintf("%d subagent(s) completed", len(tasks)))})
		return out.String(), false
	}})
}

// subagentRules is the subagent's entire phase prompt. Deliberately tiny:
// subagents are leaf workers, not investigators. They must do exactly what
// `instructions` says — no extra `run_task` "to verify", no re-reading files
// they just wrote, no chasing failures that aren't their job. The parent
// already planned the work and runs the verify phase on the overall result;
// a subagent that goes broader wastes the parallelism it was forked for.
const subagentRules = `You are a subagent. Do exactly the task in "Task:" — nothing more.

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

// runSubagent runs the tool loop only against subSess — no plan, no verify.
// The parent already planned the work and will verify the overall result, so
// the subagent is just a short, scoped tool-loop runner. All internal calls
// use subSess.ID so tool uses and assistant text are saved there (not in the
// parent). UI updates emitted by the tool loop target subSess.ID too and are
// dropped by the client since it has never been announced — only the returned
// result string flows back to the parent.
//
// Bypasses agent.execute on purpose: that path injects the full EXECUTE.md
// (which encourages investigating failures, reading project files, etc.) and
// the verify-target hint. Subagents need the opposite — a tight "do only
// what's asked" prompt. We build the message and tool exclusions directly.
//
// History inheritance: when pinned to LLM[0] we prepend the parent's full
// message history so the byte-identical prefix keeps that conn's KV cache
// warm. Subagents on LLM[1+] start fresh — those conns hold no relevant
// prefix and seeding parent history would just bloat their context for no
// cache benefit. This matters because LLM[0] is the dispatch target for the
// first task in every breadth-first fan-out.
func (a *agent) runSubagent(ctx context.Context, subSess *Session, task subagentTask) (string, error) {
	sid := subSess.ID

	instructions := task.Instructions
	if task.Context != "" {
		instructions = "Context:\n" + task.Context + "\n\nTask:\n" + task.Instructions
	} else {
		instructions = "Task:\n" + task.Instructions
	}

	subSess.AddUser(instructions)
	_ = subSess.Save()

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
			messages = a.buildLLMHistory(parent, -1)
		}
	}
	messages = append(messages, llmMessage{Role: "user", Content: b.String()})

	// Subagents can't interact with the user (UI is dropped), so ask_user
	// would hang. Web tools were planner-phase only — execute-tier work has
	// no business going to the web. launch_subagent is excluded at max depth
	// to bound recursion.
	exclude := map[string]bool{
		"ask_user":     true,
		"web_search":   true,
		"web_read":     true,
		"web_read_raw": true,
	}
	if subSess.Depth >= maxSubagentDepth {
		exclude["launch_subagent"] = true
	}

	result, err := a.runToolLoop(ctx, sid, a.pickAvailable(ctx, sid, "execute"), messages, toolFilter{exclude: exclude}, "subagent")
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
		_ = subSess.Save()
	}

	return result.Text, nil
}
