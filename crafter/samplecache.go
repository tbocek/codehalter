package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// sampleRecord is one claim's generated A/B pairs, persisted to a model's
// samples.jsonl the moment the target finishes generating them. Persisting here
// is what decouples generation from judging: the target can run far ahead of
// the (bottleneck) judge and then be freed — its work is durable — and a
// restart resumes from the saved pairs instead of regenerating. results.jsonl
// still wins on resume; once a claim has a verdict its samples are dead weight,
// kept only for provenance.
type sampleRecord struct {
	ClaimID string `json:"claim_id"`
	Skill   string `json:"skill"`
	// QuestionHash ties the pairs to the exact question that produced them
	// (question + rubric + tools). Resume reuses saved pairs only when it still
	// matches the claim's current question — otherwise the answers belong to a
	// question that is no longer being asked (e.g. AUTHOR.md changed and the
	// probe re-authored) and judging them against the new rubric would be
	// silently wrong. Absent on legacy rows → treated as a mismatch.
	QuestionHash string       `json:"question_hash,omitempty"`
	Pairs        []samplePair `json:"pairs"`
}

// questionHash fingerprints the probe a set of pairs answered.
func questionHash(q Question) string {
	return hashOf([]byte(q.Question + "\x00" + q.Rubric + "\x00" + strings.Join(q.Tools, ",")))[:16]
}

// matches reports whether these saved pairs answered exactly this question.
func (r sampleRecord) matches(q Question) bool {
	return r.QuestionHash != "" && r.QuestionHash == questionHash(q)
}

// readSamples loads a model's samples.jsonl into a map keyed by claim ID, last
// line winning (a regenerated claim overrides the older pairs). A missing file
// is the normal first-run case. Mirrors readResults so the two ledgers behave
// identically on malformed/truncated input.
func readSamples(path string) map[string]sampleRecord {
	out := map[string]sampleRecord{}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)
	n := 0
	for sc.Scan() {
		n++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var r sampleRecord
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			fmt.Fprintf(os.Stderr, "warn: %s line %d: malformed sample record, skipping (claim will be regenerated): %v\n", path, n, err)
			continue
		}
		out[r.ClaimID] = r
	}
	if err := sc.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "warn: %s: read stopped at line %d: %v (some samples may be regenerated)\n", path, n, err)
	}
	return out
}

// appendSamples appends one claim's generated pairs as a JSON line, checking the
// close so a failed write can't be reported as success — a lost write would
// silently regenerate the (expensive) samples next run.
func appendSamples(path string, r sampleRecord) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	row, err := json.Marshal(r)
	if err != nil {
		f.Close()
		return err
	}
	if _, err := fmt.Fprintln(f, string(row)); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
