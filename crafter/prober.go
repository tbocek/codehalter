package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
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
type Question struct {
	Question string `json:"question"`
	Rubric   string `json:"rubric"`
}

// Sample is one A/B/judge round for a claim on one target model. AnswerA is the
// model's natural reply (no instruction); AnswerB is its reply with the claim
// injected as a system instruction; the rest is the judge's verdict for this
// round.
type Sample struct {
	AnswerA string `json:"answer_a"`
	AnswerB string `json:"answer_b"`
	ASat    bool   `json:"a_satisfies"`
	BSat    bool   `json:"b_satisfies"`
	Similar bool   `json:"similar"`
	Verdict string `json:"verdict"` // "keep" | "drop"
	Reason  string `json:"reason"`
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
	Samples  []Sample `json:"samples"`
	Err      string   `json:"error,omitempty"`

	// EndedAt and DurationMs record when this probe finished and how long its
	// LLM calls took, so results.jsonl doubles as a durable timeline — you can
	// reconstruct pace and spot slow/stalled probes long after the stderr log
	// is gone. Stamped by the caller (main), not probeClaim.
	EndedAt    string `json:"ended_at,omitempty"`
	DurationMs int64  `json:"duration_ms,omitempty"`
}

// authorClaims produces a Question for every claim using the judge model,
// caching the map (keyed by claim ID) under cachePath so a re-run or a second
// target model reuses the authored probes instead of paying the judge again.
//
// A claim whose authoring fails (judge error even after chat's repetition
// nudge, or an off-spec reply) is warned about and skipped, not fatal: the
// claim ends up with no Question, is never probed, and counts as errored in
// the stats — a flaky judge costs one claim, not the run. A cache-write
// failure is still an error: losing authored probes silently would re-pay the
// judge every run.
// progress (nil = silent) is called after each freshly authored claim —
// authoring a whole skill takes the judge tens of minutes (measured: 42 min
// for 16 claims), and without per-claim lines that window reads as a hang.
func authorClaims(ctx context.Context, judge ModelSpec, claims []Claim, cachePath string, progress func(done, total int, id string, d time.Duration)) (map[string]Question, error) {
	cache := readQuestionCache(cachePath)
	dirty := false
	for _, c := range claims {
		if _, ok := cache[c.ID]; ok {
			continue
		}
		if ctx.Err() != nil {
			break // interrupted — keep what we have, cache it below
		}
		cStart := time.Now()
		var q Question
		if err := chatJSON(ctx, judge, authorPrompt, c.Text, &q); err != nil {
			fmt.Fprintf(os.Stderr, "warn: author probe for %s failed, claim stays untested: %v\n", c.ID, err)
			continue
		}
		if strings.TrimSpace(q.Question) == "" || strings.TrimSpace(q.Rubric) == "" {
			fmt.Fprintf(os.Stderr, "warn: author probe for %s: judge returned empty question or rubric, claim stays untested\n", c.ID)
			continue
		}
		cache[c.ID] = q
		dirty = true
		if progress != nil {
			progress(len(cache), len(claims), c.ID, time.Since(cStart))
		}
	}
	if dirty {
		if err := writeQuestionCache(cachePath, cache); err != nil {
			return nil, fmt.Errorf("cache authored probes: %w", err)
		}
	}
	return cache, nil
}

// samplePair is one A/B generation round on the target: the natural answer
// and the claim-instructed answer, not yet judged. The split from judging lets
// the target generate claim N+1's pairs while the judge scores claim N's —
// they run on different servers, so the overlap is free wall-clock.
type samplePair struct {
	AnswerA string
	AnswerB string
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
// A single failed generation aborts the claim — partial sampling would bias
// the majority.
func generateSamples(ctx context.Context, target ModelSpec, claim Claim, q Question, samples int) ([]samplePair, error) {
	// The claim as a system instruction for the treated (B) run. Text is the
	// self-contained rewrite, so it reads as a standalone directive even when
	// the verbatim source bullet would not.
	instruction := claim.Text

	pairs := make([]samplePair, 0, samples)
	for i := 0; i < samples; i++ {
		// A: natural behavior, no instruction. B: same task, claim injected.
		answerA, err := chat(ctx, target, "", q.Question)
		if err != nil {
			return nil, fmt.Errorf("sample %d answer A: %w", i, err)
		}
		answerB, err := chat(ctx, target, instruction, q.Question)
		if err != nil {
			return nil, fmt.Errorf("sample %d answer B: %w", i, err)
		}
		pairs = append(pairs, samplePair{AnswerA: answerA, AnswerB: answerB})
	}
	return pairs, nil
}

// judgeSamples runs the judge-only half: score each generated pair against the
// rubric and fold the verdicts into a majority. A failed judge call records
// the error on the result (retried on resume). progress (nil = silent) fires
// after each sample verdict — three verdicts take the judge several minutes.
func judgeSamples(ctx context.Context, judge, target ModelSpec, claim Claim, q Question, pairs []samplePair, progress func(i, n int, verdict string)) ProbeResult {
	res := newProbeResult(target, claim, q)
	keep := 0
	for i, p := range pairs {
		var s Sample
		s.AnswerA = p.AnswerA
		s.AnswerB = p.AnswerB
		judgeUser := fmt.Sprintf("RUBRIC:\n%s\n\nANSWER A (no instruction):\n%s\n\nANSWER B (with instruction):\n%s",
			q.Rubric, p.AnswerA, p.AnswerB)
		if err := chatJSON(ctx, judge, judgePrompt, judgeUser, &s); err != nil {
			res.Err = fmt.Sprintf("sample %d judge: %v", i, err)
			return res
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
		if s.Verdict == "keep" {
			keep++
		}
		res.Samples = append(res.Samples, s)
		if progress != nil {
			progress(i+1, len(pairs), s.Verdict)
		}
	}

	// Majority: keep the statement only if a strict majority of samples say so.
	res.Keep = keep*2 > len(pairs)
	if res.Keep {
		res.Verdict = "keep"
	} else {
		res.Verdict = "drop"
	}
	return res
}

func readQuestionCache(path string) map[string]Question {
	out := map[string]Question{}
	data, err := os.ReadFile(path)
	if err != nil {
		return out // absent cache is the normal first-run case
	}
	if err := json.Unmarshal(data, &out); err != nil {
		// Corrupt cache: reset and re-author rather than run on a partial map.
		fmt.Fprintf(os.Stderr, "warn: %s: unreadable authored-probe cache, re-authoring: %v\n", path, err)
		return map[string]Question{}
	}
	return out
}

func writeQuestionCache(path string, cache map[string]Question) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cache, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
