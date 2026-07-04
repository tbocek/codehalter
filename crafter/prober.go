package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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
func authorClaims(ctx context.Context, judge ModelSpec, claims []Claim, cachePath string) (map[string]Question, error) {
	cache := readQuestionCache(cachePath)
	dirty := false
	for _, c := range claims {
		if _, ok := cache[c.ID]; ok {
			continue
		}
		var q Question
		if err := chatJSON(ctx, judge, authorPrompt, c.Text, &q); err != nil {
			return nil, fmt.Errorf("author probe for %s: %w", c.ID, err)
		}
		if strings.TrimSpace(q.Question) == "" || strings.TrimSpace(q.Rubric) == "" {
			return nil, fmt.Errorf("author probe for %s: judge returned empty question or rubric", c.ID)
		}
		cache[c.ID] = q
		dirty = true
	}
	if dirty {
		if err := writeQuestionCache(cachePath, cache); err != nil {
			return nil, fmt.Errorf("cache authored probes: %w", err)
		}
	}
	return cache, nil
}

// probeClaim runs the A/B/judge round `samples` times for one claim on one
// target model and folds them into a majority verdict. A single failed sample
// aborts the probe and returns a ProbeResult carrying the error — partial
// sampling would bias the majority.
func probeClaim(ctx context.Context, judge, target ModelSpec, claim Claim, q Question, samples int) ProbeResult {
	res := ProbeResult{
		Model:    target.Name,
		Skill:    claim.Skill,
		ClaimID:  claim.ID,
		Text:     claim.Text,
		Question: q.Question,
		Rubric:   q.Rubric,
	}

	// The claim as a system instruction for the treated (B) run. Text is the
	// self-contained rewrite, so it reads as a standalone directive even when
	// the verbatim source bullet would not.
	instruction := claim.Text

	keep := 0
	for i := 0; i < samples; i++ {
		// A: natural behavior, no instruction. B: same task, claim injected.
		answerA, err := chat(ctx, target, "", q.Question)
		if err != nil {
			res.Err = fmt.Sprintf("sample %d answer A: %v", i, err)
			return res
		}
		answerB, err := chat(ctx, target, instruction, q.Question)
		if err != nil {
			res.Err = fmt.Sprintf("sample %d answer B: %v", i, err)
			return res
		}

		var s Sample
		s.AnswerA = answerA
		s.AnswerB = answerB
		judgeUser := fmt.Sprintf("RUBRIC:\n%s\n\nANSWER A (no instruction):\n%s\n\nANSWER B (with instruction):\n%s",
			q.Rubric, answerA, answerB)
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
	}

	// Majority: keep the statement only if a strict majority of samples say so.
	res.Keep = keep*2 > samples
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
