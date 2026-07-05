package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

//go:embed res/AUTHOR.md
var authorPrompt string

//go:embed res/JUDGE.md
var judgePrompt string

// Question is the judge-authored probe for one claim: a coding task that makes
// the claim's behavior relevant without naming it, plus the rubric an
// independent judge applies to two answers. Authoring depends only on the claim
// and the judge (not on any target model), so it is cached per skill and reused
// across every target.
//
// Tools lists probe-tool names (from probeToolCatalog) offered to the target
// during the A/B runs. The judge sets it for claims whose behavior IS tool
// usage (run web_search before claiming unavailable, read before edit, …) —
// without real tools in the request such claims are only testable via proxies.
type Question struct {
	Question string   `json:"question"`
	Rubric   string   `json:"rubric"`
	Tools    []string `json:"tools,omitempty"`
}

// probeToolCatalog mirrors codehalter's own agent tools with minimal schemas —
// enough for a target model to express WHICH tool it would call and with what
// arguments. Calls are never executed; the call itself is the observation.
var probeToolCatalog = map[string]map[string]any{
	"run_command": probeTool("run_command", "Run a shell command in the project workspace and return its output.",
		map[string]any{"command": map[string]any{"type": "string", "description": "the shell command to run"}}, "command"),
	"read_file": probeTool("read_file", "Read a file from the project and return its content.",
		map[string]any{"path": map[string]any{"type": "string", "description": "project-relative file path"}}, "path"),
	"edit_file": probeTool("edit_file", "Replace text in a project file.",
		map[string]any{
			"path": map[string]any{"type": "string", "description": "project-relative file path"},
			"old":  map[string]any{"type": "string", "description": "exact text to replace"},
			"new":  map[string]any{"type": "string", "description": "replacement text"},
		}, "path", "old", "new"),
	"search_text": probeTool("search_text", "Search the project files for a pattern and return matching lines.",
		map[string]any{"pattern": map[string]any{"type": "string", "description": "text or regex to search for"}}, "pattern"),
	"web_search": probeTool("web_search", "Search the web and return result snippets with URLs.",
		map[string]any{"query": map[string]any{"type": "string", "description": "the search query"}}, "query"),
}

// probeTool assembles one OpenAI function-tool definition.
func probeTool(name, desc string, props map[string]any, required ...string) map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        name,
			"description": desc,
			"parameters": map[string]any{
				"type":       "object",
				"properties": props,
				"required":   required,
			},
		},
	}
}

// buildProbeTools resolves the judge-chosen tool names against the catalog.
// Unknown names are warned and skipped, not fatal — the probe still runs with
// whatever resolved (or as a plain probe when nothing did).
func buildProbeTools(names []string) []map[string]any {
	var out []map[string]any
	for _, n := range names {
		t, ok := probeToolCatalog[n]
		if !ok {
			fmt.Fprintf(os.Stderr, "warn: authored probe requests unknown tool %q, skipping it\n", n)
			continue
		}
		out = append(out, t)
	}
	return out
}

// Sample is one A/B/judge round for a claim on one target model. AnswerA is the
// model's natural reply (no instruction); AnswerB is its reply with the claim
// injected as a system instruction; ACalls/BCalls are the tool calls each run
// emitted (tool probes only); the rest is the judge's verdict for this round.
type Sample struct {
	AnswerA string   `json:"answer_a"`
	AnswerB string   `json:"answer_b"`
	ACalls  []string `json:"a_calls,omitempty"`
	BCalls  []string `json:"b_calls,omitempty"`
	ASat    bool     `json:"a_satisfies"`
	BSat    bool     `json:"b_satisfies"`
	Similar bool     `json:"similar"`
	Verdict string   `json:"verdict"` // "keep" | "drop"
	Reason  string   `json:"reason"`
}

// ProbeResult is the persisted outcome of probing one claim against one target
// model: the majority verdict plus every sample, so results.jsonl is both the
// resume ledger and the raw material for the report. Err is set (and Keep left
// false) when the probe could not complete — a failed probe is recorded, not
// silently dropped, so resume doesn't retry it forever without the operator
// seeing it.
type ProbeResult struct {
	Model    string   `json:"model"`
	Skill    string   `json:"skill"`
	ClaimID  string   `json:"claim_id"`
	Text     string   `json:"text"`
	Question string   `json:"question"`
	Rubric   string   `json:"rubric"`
	Verdict  string   `json:"verdict"` // "keep" | "drop"
	Keep     bool     `json:"keep"`
	Unstable bool     `json:"unstable,omitempty"` // samples disagreed (not unanimous) — borderline, worth a look
	Samples  []Sample `json:"samples"`
	Err      string   `json:"error,omitempty"`

	// EndedAt and DurationMs record when this probe finished and how long its
	// LLM calls took, so results.jsonl doubles as a durable timeline — you can
	// reconstruct pace and spot slow/stalled probes long after the stderr log
	// is gone. Stamped by the caller (main), not probeClaim.
	EndedAt    string `json:"ended_at,omitempty"`
	DurationMs int64  `json:"duration_ms,omitempty"`
}

// authorQuestion authors the probe (question + rubric) for one claim with the
// judge. It is one stage of the scheduler pipeline, peer to generateSamples and
// judgeSamples. ok is false when the judge errored (even after chat's repetition
// nudge) or returned an empty question/rubric — the claim then stays untested
// this run rather than aborting the pass (a flaky judge costs one claim, not the
// whole run). Callers persist the question and warn on !ok.
func authorQuestion(ctx context.Context, judge ModelSpec, c Claim) (Question, bool) {
	var q Question
	if err := chatJSON(ctx, judge, authorPrompt, c.Text, &q); err != nil {
		fmt.Fprintf(os.Stderr, "warn: author probe for %s failed, claim stays untested: %v\n", c.ID, err)
		return q, false
	}
	if strings.TrimSpace(q.Question) == "" || strings.TrimSpace(q.Rubric) == "" {
		fmt.Fprintf(os.Stderr, "warn: author probe for %s: judge returned empty question or rubric, claim stays untested\n", c.ID)
		return q, false
	}
	return q, true
}

// samplePair is one A/B generation round on the target: the natural answer
// and the claim-instructed answer (plus any tool calls each emitted), not yet
// judged. The split from judging lets the target generate claim N+1's pairs
// while the judge scores claim N's — they run on different servers, so the
// overlap is free wall-clock.
type samplePair struct {
	AnswerA string   `json:"answer_a"`
	AnswerB string   `json:"answer_b"`
	ACalls  []string `json:"a_calls,omitempty"`
	BCalls  []string `json:"b_calls,omitempty"`
}

// newProbeResult seeds the persisted result for one claim×target with its
// identifying fields; the caller fills verdict/samples or Err.
func newProbeResult(target ModelSpec, claim Claim, q Question) ProbeResult {
	return ProbeResult{
		Model:    target.Name,
		Skill:    claim.Skill,
		ClaimID:  claim.ID,
		Text:     claim.Text,
		Question: q.Question,
		Rubric:   q.Rubric,
	}
}

// generateSamples runs the target-only half of a probe: `samples` A/B rounds.
// Arm A gets no system prompt (the model's natural behavior); arm B gets
// bSystem — the whole clean skill — so the claim's behavior is tested in the
// realistic context the model actually receives, not as one line out of
// context. A single failed generation aborts the claim — partial sampling would
// bias the majority.
func generateSamples(ctx context.Context, target ModelSpec, claim Claim, q Question, samples int, bSystem string) ([]samplePair, error) {
	// Tool probes offer the judge-chosen tools so the target can actually CALL
	// them; the calls are captured, never executed. nil = plain text probe.
	tools := buildProbeTools(q.Tools)

	// Runs are grouped by ARM (all A, then all B) instead of alternating
	// A,B,A,B: consecutive calls share their full prompt prefix, so the
	// server's prefix cache prefills the question once per arm instead of
	// re-evaluating it on every alternation. With parallel >= 2 the two arms
	// run concurrently on separate slots — each slot keeps its own cache, and
	// wall time halves.
	runArm := func(system string) ([]string, [][]string, error) {
		answers := make([]string, samples)
		calls := make([][]string, samples)
		for i := 0; i < samples; i++ {
			a, c, err := chatWithTools(ctx, target, system, q.Question, tools)
			if err != nil {
				return nil, nil, fmt.Errorf("sample %d: %w", i, err)
			}
			answers[i] = a
			calls[i] = renderCalls(c)
		}
		return answers, calls, nil
	}

	var aAns, bAns []string
	var aCalls, bCalls [][]string
	var errA, errB error
	if target.Parallel >= 2 {
		var wg sync.WaitGroup
		wg.Add(2)
		go func() { defer wg.Done(); aAns, aCalls, errA = runArm("") }()
		go func() { defer wg.Done(); bAns, bCalls, errB = runArm(bSystem) }()
		wg.Wait()
	} else {
		aAns, aCalls, errA = runArm("")
		if errA == nil {
			bAns, bCalls, errB = runArm(bSystem)
		}
	}
	if errA != nil {
		return nil, fmt.Errorf("answer A: %w", errA)
	}
	if errB != nil {
		return nil, fmt.Errorf("answer B: %w", errB)
	}

	pairs := make([]samplePair, 0, samples)
	for i := 0; i < samples; i++ {
		pairs = append(pairs, samplePair{AnswerA: aAns[i], AnswerB: bAns[i], ACalls: aCalls[i], BCalls: bCalls[i]})
	}
	return pairs, nil
}

// renderCalls formats accumulated tool calls for the judge and the ledger.
func renderCalls(calls []toolCallRec) []string {
	var out []string
	for _, c := range calls {
		out = append(out, c.render())
	}
	return out
}

// judgeSamples runs the judge-only half: score each generated pair against the
// rubric and fold the verdicts into a majority. A failed judge call records
// the error on the result (retried on resume). progress (nil = silent) fires
// after each sample verdict — three verdicts take the judge several minutes.
func judgeSamples(ctx context.Context, judge, target ModelSpec, claim Claim, q Question, pairs []samplePair, keepThreshold int, progress func(i, n int, verdict string)) ProbeResult {
	res := newProbeResult(target, claim, q)

	judgeOne := func(p samplePair) (Sample, error) {
		var s Sample
		s.AnswerA = p.AnswerA
		s.AnswerB = p.AnswerB
		s.ACalls = p.ACalls
		s.BCalls = p.BCalls
		judgeUser := fmt.Sprintf("RUBRIC:\n%s\n\nANSWER A (without the skill):\n%s\n\nANSWER B (with the skill loaded):\n%s",
			q.Rubric, p.AnswerA, p.AnswerB)
		if len(q.Tools) > 0 {
			// Tool probe: show the judge what each run actually called — for
			// tool-usage rubrics the calls ARE the evidence.
			judgeUser = fmt.Sprintf("RUBRIC:\n%s\n\nANSWER A (without the skill):\n%s\nTOOL CALLS A:\n%s\n\nANSWER B (with the skill loaded):\n%s\nTOOL CALLS B:\n%s",
				q.Rubric, orNone(p.AnswerA), callsBlock(p.ACalls), orNone(p.AnswerB), callsBlock(p.BCalls))
		}
		if err := chatJSON(ctx, judge, judgePrompt, judgeUser, &s); err != nil {
			return s, err
		}
		if s.Verdict != "keep" && s.Verdict != "drop" {
			// Normalize an off-spec verdict from the booleans: keep only when B
			// added the behavior A lacked.
			if s.BSat && !s.ASat {
				s.Verdict = "keep"
			} else {
				s.Verdict = "drop"
			}
		}
		return s, nil
	}

	// The per-sample verdicts are independent — with a multi-slot judge they
	// run concurrently (the endpoint semaphore gates real concurrency), with a
	// single slot the serial path keeps deterministic early-abort behavior.
	verdicts := make([]Sample, len(pairs))
	errs := make([]error, len(pairs))
	if judge.Parallel >= 2 {
		var wg sync.WaitGroup
		for i, p := range pairs {
			wg.Add(1)
			go func(i int, p samplePair) {
				defer wg.Done()
				verdicts[i], errs[i] = judgeOne(p)
				if errs[i] == nil && progress != nil {
					progress(i+1, len(pairs), verdicts[i].Verdict)
				}
			}(i, p)
		}
		wg.Wait()
	} else {
		for i, p := range pairs {
			verdicts[i], errs[i] = judgeOne(p)
			if errs[i] != nil {
				break // abort remaining samples — partial majorities are recorded as errors anyway
			}
			if progress != nil {
				progress(i+1, len(pairs), verdicts[i].Verdict)
			}
		}
	}
	for i, err := range errs {
		if err != nil {
			res.Err = fmt.Sprintf("sample %d judge: %v", i, err)
			return res
		}
	}

	keep := 0
	for _, s := range verdicts {
		if s.Verdict == "keep" {
			keep++
		}
		res.Samples = append(res.Samples, s)
	}
	// Keep the statement when at least keepThreshold of the samples vote keep.
	// Threshold 1 (the default) keeps unless the samples unanimously say drop —
	// the asymmetry is deliberate: a wrong drop silently regresses a behavior,
	// a wrong keep costs a little prefill. Guard against a caller passing 0/neg.
	if keepThreshold < 1 {
		keepThreshold = 1
	}
	res.Keep = keep >= keepThreshold
	if res.Keep {
		res.Verdict = "keep"
	} else {
		res.Verdict = "drop"
	}
	// A non-unanimous vote means the model is inconsistent on this claim — the
	// verdict went one way but it was a judgment call, so flag it for review.
	res.Unstable = keep != 0 && keep != len(verdicts)
	return res
}

// callsBlock renders a tool-call list for the judge prompt.
func callsBlock(calls []string) string {
	if len(calls) == 0 {
		return "(none)"
	}
	return strings.Join(calls, "\n")
}

// orNone substitutes a placeholder for an empty visible answer (a tool-probe
// run may legitimately answer only with calls).
func orNone(s string) string {
	if strings.TrimSpace(s) == "" {
		return "(no text — see tool calls)"
	}
	return s
}

// questionCache is the on-disk authored-probe cache. Hash covers the AUTHOR
// prompt, so editing AUTHOR.md invalidates cached probes — content-only keying
// would silently keep questions authored under the old rules (same latent bug
// the segment cache had).
type questionCache struct {
	Hash      string              `json:"hash"`
	Questions map[string]Question `json:"questions"`
}

func readQuestionCache(path string) map[string]Question {
	empty := map[string]Question{}
	data, err := os.ReadFile(path)
	if err != nil {
		return empty // absent cache is the normal first-run case
	}
	var c questionCache
	if err := json.Unmarshal(data, &c); err != nil {
		// Corrupt cache: reset and re-author rather than run on a partial map.
		fmt.Fprintf(os.Stderr, "warn: %s: unreadable authored-probe cache, re-authoring: %v\n", path, err)
		return empty
	}
	// A pre-wrapper cache (bare map) or a different AUTHOR.md both land here.
	if c.Hash != hashOf([]byte(authorPrompt)) || c.Questions == nil {
		fmt.Fprintf(os.Stderr, "warn: %s: authored-probe cache stale (authoring prompt changed), re-authoring\n", path)
		return empty
	}
	return c.Questions
}

func writeQuestionCache(path string, cache map[string]Question) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(questionCache{Hash: hashOf([]byte(authorPrompt)), Questions: cache}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
