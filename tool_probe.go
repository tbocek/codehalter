package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// probeStatementToolName is /improve's measurement tool: it A/B-tests skill
// statements on the MAIN model (LLM[0]) — the model the skills actually serve —
// while the /improve analysis itself may run on a stronger model (purpose =
// "improve"). Same idea as the crafter's probe, scaled down to the handful of
// statements /improve suspects: answer a question with and without the
// statement in the skill, and let the (strong) improve model judge whether the
// statement changed anything. A statement that changes nothing is removable
// evidence-first, instead of "this looks verbose".
const probeStatementToolName = "probe_statement"

// probeSpec is one requested A/B probe.
type probeSpec struct {
	// File is the bare skill filename the statement lives in (or should join),
	// e.g. "SKILL-go.md". Empty = context-free probe: WITHOUT arm has no system
	// prompt at all, WITH arm gets the statement alone.
	File string `json:"file"`
	// Statement is the exact statement under test. When it already occurs in
	// File this is a removal check (WITH = current skill, WITHOUT = skill minus
	// the statement); when it doesn't, an addition check (WITHOUT = current
	// skill, WITHOUT+statement appended = WITH).
	Statement string `json:"statement"`
	// Question is a task that makes the statement's behavior observable without
	// naming it — the model should be able to express the behavior in plain
	// text, since probe answers offer no tools.
	Question string `json:"question"`
	// Replaces optionally names the exact current statement that Statement is a
	// REWRITE of. Both arms then drop Replaces from the skill body and the WITH
	// arm appends Statement — so the rewrite is measured alone, not sitting
	// next to the original it is meant to supersede (the file on disk still
	// carries the original until submit_improvement applies the edit).
	Replaces string `json:"replaces,omitempty"`
}

const (
	// maxProbesPerCall bounds one batch. Each probe costs 2×samples main-model
	// calls; ten of them is a full overnight-scale batch already.
	maxProbesPerCall = 10
	// probeSamplesDefault / probeSamplesMax: repetitions per arm. Sampling at
	// execute temperature is stochastic, so a single pair can mislead; two
	// pairs is the screening default. The cap of five exists for the DECIDING
	// probe of a rewrite (validate with more power than you screened with) —
	// an overnight run has the wall-clock for it.
	probeSamplesDefault = 2
	probeSamplesMax     = 5
	// probeCallTimeout bounds one main-model generation. Generous, because on a
	// routed server the first probe call may block while the router swaps the
	// main model back in (it queues, then answers 200 — no error to react to).
	probeCallTimeout = 10 * time.Minute
)

// probeArms derives the two system prompts for a spec against the on-disk
// (variant-resolved) skill. Returned mode names the check for the report.
// Placeholders are expanded AFTER the with/without split, so both arms match
// what loadSkills would actually feed the main model.
func probeArms(cwd, variant string, p probeSpec) (without, with, mode string, err error) {
	if strings.TrimSpace(p.Statement) == "" {
		return "", "", "", fmt.Errorf("statement is required")
	}
	name := strings.TrimSpace(p.File)
	if name == "" {
		return "", expandCmdPlaceholders(p.Statement), "context-free check (no skill file given)", nil
	}
	if !skillFileNameRe.MatchString(name) {
		return "", "", "", fmt.Errorf("file %q must be a bare SKILL-<topic>.md filename (or empty for a context-free probe)", name)
	}
	raw, err := os.ReadFile(skillPath(cwd, variant, name))
	if err != nil {
		return "", "", "", fmt.Errorf("read %s: %w", name, err)
	}
	body := string(raw)
	if p.Replaces != "" {
		// Rewrite check: strip the original from BOTH arms, append the rewrite
		// to WITH. The stripped WITHOUT arm is byte-identical to the original
		// statement's removal-check WITHOUT arm, so on the same question the
		// probe cache (probeGenerate) serves it for free.
		if !strings.Contains(body, p.Replaces) {
			return "", "", "", fmt.Errorf("`replaces` text not found in %s — copy it byte-exactly from the loaded copy", name)
		}
		stripped := strings.Replace(body, p.Replaces, "", 1)
		return expandCmdPlaceholders(stripped),
			expandCmdPlaceholders(strings.TrimRight(stripped, "\n") + "\n" + p.Statement + "\n"),
			"rewrite check (replaced statement stripped from both arms, rewrite appended to WITH)", nil
	}
	if strings.Contains(body, p.Statement) {
		// Removal check: the exact bytes are in the loaded copy.
		return expandCmdPlaceholders(strings.Replace(body, p.Statement, "", 1)),
			expandCmdPlaceholders(body),
			"removal check (statement is in the loaded skill)", nil
	}
	// Addition check: statement not present yet — test what adding it changes.
	return expandCmdPlaceholders(body),
		expandCmdPlaceholders(strings.TrimRight(body, "\n") + "\n" + p.Statement + "\n"),
		"addition check (statement is NOT in the loaded skill — probing it as an append)", nil
}

// probeGenerate runs `samples` generations of one arm on the main model.
// Sequential within the arm, arms grouped by the caller — consecutive calls
// share their prompt prefix in the server's cache (same trick as the crafter).
// Arms are cached per (system, question) for the run: a validation probe whose
// WITHOUT arm matches an earlier screening probe reuses those answers instead
// of re-burning main-model calls, and only generates any shortfall.
func (a *agent) probeGenerate(ctx context.Context, sid string, conn *LLMConnection, system, question string, samples int) ([]string, error) {
	sess := a.getSession(sid)
	key := system + "\x00" + question
	var out []string
	if sess != nil {
		if out = sess.probeCached(key); len(out) >= samples {
			return out[:samples], nil
		}
	}
	var msgs []llmMessage
	if system != "" {
		msgs = append(msgs, llmMessage{Role: "system", Content: system})
	}
	// Same reasoning switch as the execute phase: the probe mimics how the main
	// model runs during real work, and it keeps overnight probes cheap.
	msgs = append(msgs, llmMessage{Role: "user", Content: question + noThinkSwitch})
	for i := len(out); i < samples; i++ {
		cctx, cancel := context.WithTimeout(ctx, probeCallTimeout)
		text, _, _, err := a.llmStream(cctx, sid, conn, msgs, nil, nil, nil)
		cancel()
		if err != nil {
			return nil, fmt.Errorf("sample %d: %w", i+1, err)
		}
		out = append(out, text)
	}
	if sess != nil {
		sess.probeStore(key, out)
	}
	return out, nil
}

// probeRecord is one line of .codehalter/improve-probes.jsonl — the durable
// evidence log of every A/B probe. Written deliberately to .codehalter/ (NOT
// the /improve scratch dir): the answers are measurements worth keeping after
// the throwaway analysis turn is gone — the morning review reads them, and
// they cross-check what the offline crafter later decides about the same
// statements.
type probeRecord struct {
	Time      string   `json:"time"`
	File      string   `json:"file,omitempty"`
	Mode      string   `json:"mode,omitempty"`
	Statement string   `json:"statement"`
	Replaces  string   `json:"replaces,omitempty"`
	Question  string   `json:"question,omitempty"`
	Samples   int      `json:"samples,omitempty"`
	Without   []string `json:"without,omitempty"`
	With      []string `json:"with,omitempty"`
	Err       string   `json:"err,omitempty"`
}

// recordProbe appends one probe outcome to improve-probes.jsonl. Best-effort:
// a write failure costs the durable record, never the probe result.
func recordProbe(cwd string, rec probeRecord) {
	rec.Time = time.Now().UTC().Format(time.RFC3339)
	line, err := json.Marshal(rec)
	if err != nil {
		return
	}
	path := filepath.Join(cwd, ".codehalter", "improve-probes.jsonl")
	if err := appendFile(path, string(line)+"\n"); err != nil {
		slog.Warn("improve: recording probe", "path", path, "err", err)
	}
}

func init() {
	RegisterTool(Tool{Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        probeStatementToolName,
			"description": "/improve only: A/B-test skill statements on the MAIN model (llm[0], the model the skills serve) before proposing to remove them. `probes` is a JSON array; each object: file (bare skill filename the statement lives in, e.g. \"SKILL-go.md\"; empty for a context-free probe), statement (the EXACT statement text as it appears in the loaded skill file), question (a task that makes the statement's behavior observable in plain text, WITHOUT naming or hinting at the statement), replaces (optional: when statement is a REWRITE of an existing line, the exact current text it supersedes — both arms then drop it, so the rewrite is measured alone). Each probe answers the question on the main model both WITH and WITHOUT the statement in the skill and returns all answers verbatim for YOU to judge: same behavior in both arms means the statement is removable; different behavior means it earns its place. Batch your probes into as few calls as possible (screen in one call, validate rewrites in a second) — the main model may have to be reloaded by a routing server, and one batch loads it once. Costs up to 2×samples main-model calls per probe (identical arms are cached within the run); max 10 probes per call.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"probes"},
				"properties": map[string]any{
					"probes": map[string]any{
						"type":        "string",
						"description": "JSON array of probe objects (see above).",
					},
					"samples": map[string]any{
						"type":        "integer",
						"description": "Answer pairs per probe, 1-5 (default 2). Screen at the default; use 5 for the deciding validation of a rewrite — more samples smooth out sampling noise.",
					},
				},
			},
		},
	}, Execute: probeStatementExecute})
}

// probeStatementExecute runs one batch of A/B probes against LLM[0] and
// returns every answer pair verbatim — judging is deliberately left to the
// calling (strong) model, which authored the questions and knows the evidence.
func probeStatementExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	sess := a.getSession(sid)
	if sess == nil {
		return "error: no session", true
	}
	// Gated to /improve: probes burn real main-model calls, and outside an
	// /improve turn nothing consumes their output.
	if !sess.improving.Load() {
		return "error: probe_statement is only available during an /improve run", true
	}
	args := parseArgs(rawArgs)
	probesJSON := args.str("probes")
	if probesJSON == "" {
		return "error: probes is required (a JSON array of {file, statement, question})", true
	}
	var probes []probeSpec
	if err := json.Unmarshal([]byte(probesJSON), &probes); err != nil {
		return fmt.Sprintf("error: invalid probes JSON: %v", err), true
	}
	if len(probes) == 0 {
		return "error: probes is empty — nothing to test", true
	}
	dropped := 0
	if len(probes) > maxProbesPerCall {
		dropped = len(probes) - maxProbesPerCall
		probes = probes[:maxProbesPerCall]
	}
	samples := probeSamplesDefault
	if n, ok := args.num("samples"); ok && n >= 1 {
		samples = min(n, probeSamplesMax)
	}

	// The probe target is ALWAYS LLM[0] — deliberately not connForSession, which
	// routes this /improve turn to the improve-purposed conn. The statements are
	// measured on the model they serve, not on the analyst.
	a.cfgMu.RLock()
	conn := a.settings.MainLLM("execute")
	a.cfgMu.RUnlock()
	if conn == nil {
		return "error: no main LLM configured to probe", true
	}
	variant := a.skillVariant()

	var b strings.Builder
	if dropped > 0 {
		fmt.Fprintf(&b, "(%d probe(s) beyond the first %d were dropped — batch them into a follow-up call if still needed)\n\n", dropped, maxProbesPerCall)
	}
	failures := 0
	for i, p := range probes {
		rec := probeRecord{File: strings.TrimSpace(p.File), Statement: p.Statement, Replaces: p.Replaces, Question: p.Question, Samples: samples}
		fmt.Fprintf(&b, "## Probe %d/%d — %s\nStatement: %s\n", i+1, len(probes), orGeneric(p.File), p.Statement)
		without, with, mode, err := probeArms(sess.Cwd, variant, p)
		if err != nil {
			failures++
			fmt.Fprintf(&b, "ERROR: %v\n\n", err)
			rec.Err = err.Error()
			recordProbe(sess.Cwd, rec)
			continue
		}
		rec.Mode = mode
		fmt.Fprintf(&b, "Mode: %s\n\n", mode)
		if strings.TrimSpace(p.Question) == "" {
			failures++
			b.WriteString("ERROR: question is required\n\n")
			rec.Err = "question is required"
			recordProbe(sess.Cwd, rec)
			continue
		}
		// Arms grouped (all WITHOUT, then all WITH) for prefix-cache reuse.
		ansWithout, err := a.probeGenerate(ctx, sid, conn, without, p.Question, samples)
		if err == nil {
			var ansWith []string
			ansWith, err = a.probeGenerate(ctx, sid, conn, with, p.Question, samples)
			if err == nil {
				for s := range samples {
					fmt.Fprintf(&b, "### Sample %d WITHOUT the statement\n%s\n\n### Sample %d WITH the statement\n%s\n\n", s+1, ansWithout[s], s+1, ansWith[s])
				}
				rec.Without, rec.With = ansWithout, ansWith
				recordProbe(sess.Cwd, rec)
				continue
			}
		}
		failures++
		fmt.Fprintf(&b, "ERROR: main-model call failed: %v\n\n", err)
		rec.Err = err.Error()
		recordProbe(sess.Cwd, rec)
	}
	b.WriteString("Judge each probe yourself: if the WITH and WITHOUT answers show the SAME behavior, the statement changed nothing on the main model (removal is safe / the addition is pointless); if they differ in the probed behavior, the statement earns its place. Sampling is stochastic — weigh all samples, not the prettiest one.")
	return b.String(), failures == len(probes)
}

// orGeneric labels a probe's target for the report header.
func orGeneric(file string) string {
	if strings.TrimSpace(file) == "" {
		return "context-free"
	}
	return strings.TrimSpace(file)
}
