package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

// crafter probes coding-agent SKILL files against target models to decide, per
// model, which behavioral statements the model actually needs. For each atomic
// claim it authors a test question with the judge model, answers it on the
// target both without and with the claim, and lets the judge decide whether the
// claim changed anything. Statements that change nothing are pruned from that
// model's copy of the skill.
//
// Runs are long (claims × models × samples × 3 LLM calls) and resumable: every
// probe is appended to models/<name>/results.jsonl and a re-run skips claims
// already decided there. Segmentation and question authoring are cached under
// .cache/ so only the target A/B calls repeat.
//
// Usage:
//
//	./crafter                       # probe every SKILL in ground-skills/
//	./crafter -config crafter.toml -ground ground-skills
func main() {
	configFlag := flag.String("config", "crafter.toml", "crafter config: [settings], [judge], [[model]]")
	groundFlag := flag.String("ground", "ground-skills", "directory of SKILL-*.md files to probe (copied here manually)")
	modelsFlag := flag.String("models", "models", "output directory; one subfolder per target model")
	cacheFlag := flag.String("cache", ".cache", "directory for cached segmentation and authored probes")
	docsFlag := flag.String("docs", filepath.Join("..", "docs", "skill-crafter-report.html"), "generated results report path (empty disables)")
	skipPreflightFlag := flag.Bool("skip-preflight", false, "skip the reachability check of judge + model endpoints before starting")
	llmLogFlag := flag.String("llmlog", "llm.jsonl", "wire-level LLM trace: one JSON line per call with full messages + response (empty disables)")
	flag.Parse()

	if err := mustExist(*configFlag, "config"); err != nil {
		log.Fatal(err)
	}
	if err := mustExist(*groundFlag, "ground-skills directory"); err != nil {
		log.Fatal(err)
	}
	cfg, err := loadConfig(*configFlag)
	if err != nil {
		log.Fatal(err)
	}
	if err := openLLMLog(*llmLogFlag); err != nil {
		log.Fatal(err)
	}

	skills, err := discoverSkills(*groundFlag, cfg.Settings)
	if err != nil {
		log.Fatal(err)
	}
	if len(skills) == 0 {
		log.Fatalf("no SKILL-*.md found in %s (copy the skills you want to probe there)", *groundFlag)
	}

	// SIGINT/SIGTERM cancel the context so an in-flight LLM call unwinds and the
	// run stops at a clean boundary; results already appended are the resume
	// point.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	start := time.Now()
	// logf prefixes every operational line with elapsed wall time since start,
	// so a teed log answers "how far in / when did it stall" after the fact.
	logf := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "[%s] %s\n", fmtDuration(time.Since(start)), fmt.Sprintf(format, a...))
	}
	logf("skills: %s | models: %d | samples: %d", strings.Join(skillNames(skills), ", "), len(cfg.Models), cfg.Settings.Samples)
	if *llmLogFlag != "" {
		logf("llm trace: %s (full prompts + replies, one JSON line per call)", *llmLogFlag)
	}

	// Preflight: fail fast if the judge or any target endpoint is unreachable,
	// unauthorized, or not serving its configured model — cheaper than
	// discovering a bad key an hour into a run.
	if !*skipPreflightFlag {
		if err := preflight(ctx, logf, cfg); err != nil {
			log.Fatalf("%v (use -skip-preflight to bypass)", err)
		}
	}

	// Per-model state, loaded once up front: the resume ledger (claim IDs are
	// stable content hashes, so ordering doesn't matter) and the stats that
	// accumulate skill by skill.
	done := map[string]map[string]ProbeResult{}
	statsByModel := map[string]*ModelStats{}
	for _, m := range cfg.Models {
		modelDir := filepath.Join(*modelsFlag, m.Name)
		if err := os.MkdirAll(modelDir, 0o755); err != nil {
			log.Fatalf("create %s: %v", modelDir, err)
		}
		done[m.Name] = readResults(filepath.Join(modelDir, "results.jsonl"))
		statsByModel[m.Name] = &ModelStats{Model: m.Name}
	}
	// finalizeAll writes the docs report from whatever has accumulated so far —
	// called on normal completion AND on interrupt, so a partial run still
	// reports the skills it finished.
	finalizeAll := func() {
		var report []ModelStats
		for _, m := range cfg.Models {
			if len(statsByModel[m.Name].Skills) > 0 {
				report = append(report, *statsByModel[m.Name])
			}
		}
		finalize(report, cfg, *modelsFlag, *docsFlag)
	}

	// Pipeline runs skill by skill: segment + author ONE skill (judge), then
	// probe it on every target and prune it (per model), then move to the next
	// skill. Each skill's full result (pruned SKILL-*.md, updated stats.json)
	// lands before the next skill starts, so a long run produces usable output
	// incrementally instead of all at the end.
	//
	// prepare is the judge-side work for one skill: segment into claims,
	// author a question per claim. Both are disk-cached, so a prepared skill
	// that never gets probed (crash, interrupt) costs nothing next run.
	type preparedSkill struct {
		src       skillSource
		orig      []byte
		claims    []Claim
		questions map[string]Question
		err       error
	}
	prepare := func(sk skillSource) preparedSkill {
		orig, err := os.ReadFile(sk.path)
		if err != nil {
			return preparedSkill{src: sk, err: fmt.Errorf("read %s: %w", sk.path, err)}
		}
		logf("segmenting %s ...", sk.stack)
		segStart := time.Now()
		claims, err := segmentSkill(ctx, cfg.Judge, sk.stack, sk.path, filepath.Join(*cacheFlag, "segments"))
		if err != nil {
			return preparedSkill{src: sk, err: err}
		}
		logf("segmented %s: %d claims (%s)", sk.stack, len(claims), cachedOr(time.Since(segStart)))
		logf("authoring probes for %s ...", sk.stack)
		authStart := time.Now()
		questions, err := authorClaims(ctx, cfg.Judge, claims, filepath.Join(*cacheFlag, "authored", sk.stack+".json"),
			func(done, total int, id string, d time.Duration) {
				logf("authored %s (%d/%d, %s)", id, done, total, fmtDuration(d))
			})
		if err != nil {
			return preparedSkill{src: sk, err: err}
		}
		logf("authored %s: %d/%d probes (%s)", sk.stack, len(questions), len(claims), cachedOr(time.Since(authStart)))
		return preparedSkill{src: sk, orig: orig, claims: claims, questions: questions}
	}

	// One-step lookahead: while skill N is probed on the TARGET servers, skill
	// N+1 is segmented+authored on the JUDGE server in the background — they
	// live on different endpoints, so this overlaps otherwise-idle judge time.
	// (Probing also sends the judge one verdict call per sample; those simply
	// queue with the prefetch calls on the judge server.) Buffered chan of 1 =
	// exactly one skill in flight ahead.
	nextCh := make(chan preparedSkill, 1)
	go func() { nextCh <- prepare(skills[0]) }()

	for i := range skills {
		p := <-nextCh
		if i+1 < len(skills) {
			next := skills[i+1]
			go func() { nextCh <- prepare(next) }()
		}
		if p.err != nil {
			// A judge failure on one skill (loop even after the nudge, bad
			// JSON) must not abort the run — skip the skill, the next run
			// retries it.
			logf("warn: skipping skill %s: %v", p.src.stack, p.err)
			continue
		}
		sk, orig, claims, questions := p.src, p.orig, p.claims, p.questions
		if len(questions) < len(claims) {
			logf("%s: %d/%d claims authored — the rest stay untested this run", sk.stack, len(questions), len(claims))
		}

		for _, m := range cfg.Models {
			modelDir := filepath.Join(*modelsFlag, m.Name)
			resultsPath := filepath.Join(modelDir, "results.jsonl")
			dm := done[m.Name]

			// Pending list computed up front: the generator goroutine below
			// must not read dm while the consumer writes it.
			type pendingClaim struct {
				idx   int
				claim Claim
				q     Question
			}
			var pend []pendingClaim
			for idx, c := range claims {
				q, authored := questions[c.ID]
				if !authored {
					continue // no question to probe with (warned above); counts as errored in stats
				}
				if r, ok := dm[c.ID]; ok && r.Err == "" {
					continue // already decided; resume skips it
				}
				pend = append(pend, pendingClaim{idx: idx, claim: c, q: q})
			}
			logf("== %s / %s: %d claims, %d pending ==", sk.stack, m.Name, len(claims), len(pend))

			// Explicit model-switch request: a routed endpoint loads the model
			// named in the request, so ping it once up front under a generous
			// timeout instead of letting the switch inflate (or time out) the
			// first probe. Harmless one-token call on a dedicated server.
			if len(pend) > 0 {
				logf("switching endpoint to %s (a router may load the model now) ...", m.Name)
				switchStart := time.Now()
				wctx, wcancel := context.WithTimeout(ctx, 10*time.Minute)
				if err := ping(wctx, m); err != nil {
					logf("warn: model switch/warm-up for %s failed: %v — probing anyway", m.Name, err)
				} else {
					logf("%s ready (%s)", m.Name, cachedOr(time.Since(switchStart)))
				}
				wcancel()
			}

			// Generate/judge pipeline: the TARGET produces claim N+1's A/B
			// answers while the JUDGE scores claim N's — different servers,
			// so the overlap is free wall-clock. Buffer 1 = at most one claim
			// generated ahead.
			type genItem struct {
				pc    pendingClaim
				pairs []samplePair
				genMs int64
				err   string
			}
			genCh := make(chan genItem, 1)
			go func() {
				defer close(genCh)
				for _, pc := range pend {
					if ctx.Err() != nil {
						return
					}
					// Announce before and after generating: a claim takes
					// minutes of target time before its verdict line appears,
					// and without these the resume start looks hung.
					logf("[%s %d/%d] %s %s generating %d samples ...", sk.stack, pc.idx+1, len(claims), m.Name, pc.claim.ID, cfg.Settings.Samples)
					genStart := time.Now()
					pairs, err := generateSamples(ctx, m, pc.claim, pc.q, cfg.Settings.Samples)
					it := genItem{pc: pc, pairs: pairs, genMs: time.Since(genStart).Milliseconds()}
					if err != nil {
						it.err = err.Error()
					} else {
						logf("[%s %d/%d] %s %s generated (%dms) → judging", sk.stack, pc.idx+1, len(claims), m.Name, pc.claim.ID, it.genMs)
					}
					genCh <- it
				}
			}()

			for it := range genCh {
				judgeStart := time.Now()
				var res ProbeResult
				if it.err != "" {
					res = newProbeResult(m, it.pc.claim, it.pc.q)
					res.Err = it.err
				} else {
					res = judgeSamples(ctx, cfg.Judge, m, it.pc.claim, it.pc.q, it.pairs,
						func(i, n int, verdict string) {
							logf("[%s %d/%d] %s %s judged sample %d/%d: %s", sk.stack, it.pc.idx+1, len(claims), m.Name, it.pc.claim.ID, i, n, verdict)
						})
				}
				res.DurationMs = it.genMs + time.Since(judgeStart).Milliseconds()
				res.EndedAt = time.Now().Format(time.RFC3339)
				if err := appendResult(resultsPath, res); err != nil {
					log.Fatalf("write result: %v", err)
				}
				dm[res.ClaimID] = res
				if res.Err != "" {
					logf("[%s %d/%d] %s %s ERROR (%dms): %s", sk.stack, it.pc.idx+1, len(claims), m.Name, res.ClaimID, res.DurationMs, res.Err)
				} else {
					logf("[%s %d/%d] %s %s → %s (%dms: gen %dms + judge)", sk.stack, it.pc.idx+1, len(claims), m.Name, res.ClaimID, strings.ToUpper(res.Verdict), res.DurationMs, it.genMs)
				}
			}
			if ctx.Err() != nil {
				logf("interrupted during %s/%s — progress saved to %s, re-run to resume", sk.stack, m.Name, resultsPath)
				finalizeAll()
				return
			}

			// Prune this skill for this model right away and refresh the
			// accumulated stats.json, so results are on disk per skill, not
			// only at run end.
			pruned := pruneSkill(string(orig), droppedClaims(claims, dm))
			outPath := filepath.Join(modelDir, "SKILL-"+sk.stack+".md")
			if err := os.WriteFile(outPath, []byte(pruned), 0o644); err != nil {
				log.Fatalf("write %s: %v", outPath, err)
			}
			st := skillStats(sk.stack, string(orig), pruned, claims, dm)
			ms := statsByModel[m.Name]
			ms.Skills = append(ms.Skills, st)
			ms.OrigBytes += st.OrigBytes
			ms.PrunedBytes += st.PrunedBytes
			if err := writeJSON(filepath.Join(modelDir, "stats.json"), *ms); err != nil {
				log.Fatalf("write stats: %v", err)
			}
			logf("%s / %s: %d kept, %d dropped, %d errored | %d → %d bytes (%s)",
				sk.stack, m.Name, st.Kept, st.Dropped, st.Errored, st.OrigBytes, st.PrunedBytes, pct(st.OrigBytes, st.PrunedBytes))
		}

		// Refresh the docs report after every completed skill, not only at run
		// end — a half-done overnight run should still have a current report.
		finalizeAll()
	}

	for _, m := range cfg.Models {
		ms := statsByModel[m.Name]
		kept, dropped, errored := tally(*ms)
		logf("%s done: %d kept, %d dropped, %d errored | %d → %d bytes (%s)",
			m.Name, kept, dropped, errored, ms.OrigBytes, ms.PrunedBytes, pct(ms.OrigBytes, ms.PrunedBytes))
	}
	logf("run complete in %s — %d skill(s) × %d model(s)", fmtDuration(time.Since(start)), len(skills), len(cfg.Models))
	finalizeAll()
}

// finalize writes the docs report from the accumulated per-model stats. Split
// out so an interrupted run still emits a report of whatever completed.
func finalize(report []ModelStats, cfg *Config, modelsDir, docsPath string) {
	if len(report) == 0 || docsPath == "" {
		return
	}
	if err := writeReport(docsPath, report, cfg, modelsDir); err != nil {
		fmt.Fprintf(os.Stderr, "warn: write report: %v\n", err)
		return
	}
	fmt.Fprintf(os.Stderr, "report updated: %s\n", docsPath)
}

// preflight pings the judge and every target endpoint before any real work, so
// an unreachable host, wrong key, or unknown model id stops the run up front
// instead of failing every probe. Each check is bounded so a dead host can't
// hang the whole preflight. Returns an error naming every endpoint that failed.
func preflight(ctx context.Context, logf func(string, ...any), cfg *Config) error {
	type check struct {
		label string
		m     ModelSpec
	}
	checks := []check{{"judge", cfg.Judge}}
	for _, m := range cfg.Models {
		checks = append(checks, check{m.Name, m})
	}
	logf("preflight: checking %d endpoint(s) — a router may need to load each model, this can take minutes ...", len(checks))

	var failed []string
	for _, c := range checks {
		logf("  checking %s (%s) ...", c.label, c.m.Model)
		// Generous per-endpoint budget: on a routed server the ping itself
		// triggers the model load. A dead host still fails fast (dial error).
		cctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
		err := ping(cctx, c.m)
		cancel()
		if err != nil {
			logf("  ✗ %s (%s): %v", c.label, c.m.Model, err)
			failed = append(failed, c.label)
			continue
		}
		logf("  ✓ %s (%s)", c.label, c.m.Model)
	}
	if len(failed) > 0 {
		return fmt.Errorf("preflight failed: %d/%d endpoint(s) unreachable: %s", len(failed), len(checks), strings.Join(failed, ", "))
	}
	logf("preflight OK — all %d endpoint(s) reachable", len(checks))
	return nil
}

type skillSource struct {
	stack string
	path  string
}

// discoverSkills lists ground-skills/SKILL-*.md, filtered by the settings skill
// allowlist, sorted for reproducible runs.
func discoverSkills(dir string, s Settings) ([]skillSource, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "SKILL-*.md"))
	if err != nil {
		return nil, err
	}
	sort.Strings(matches)
	var out []skillSource
	for _, p := range matches {
		base := filepath.Base(p)
		stack := strings.TrimSuffix(strings.TrimPrefix(base, "SKILL-"), ".md")
		if !s.wantSkill(stack) {
			continue
		}
		out = append(out, skillSource{stack: stack, path: p})
	}
	return out, nil
}

func skillNames(s []skillSource) []string {
	out := make([]string, len(s))
	for i, sk := range s {
		out[i] = sk.stack
	}
	return out
}

// readResults loads results.jsonl into a map keyed by claim ID, last line
// winning — a retried claim's newer verdict overrides the older one.
func readResults(path string) map[string]ProbeResult {
	out := map[string]ProbeResult{}
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
		var r ProbeResult
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			// A malformed line (e.g. a half-written row from a crash) means that
			// claim simply gets re-probed — but say so, don't hide it.
			fmt.Fprintf(os.Stderr, "warn: %s line %d: malformed result, skipping (claim will be re-probed): %v\n", path, n, err)
			continue
		}
		out[r.ClaimID] = r
	}
	// A scanner error (I/O fault, or a row longer than the 16MB cap) stops
	// Scan early and would otherwise silently truncate the resume state,
	// re-probing everything after it. Surface it loudly.
	if err := sc.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "warn: %s: read stopped at line %d: %v (resume state may be incomplete — later results will be re-probed)\n", path, n, err)
	}
	return out
}

// appendResult appends one probe result as a JSON line, checking the flush on
// close so a failed write to the durable resume ledger can't be reported as
// success (which would lose the result and re-probe the claim next run).
func appendResult(path string, r ProbeResult) error {
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

func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// tally sums keep/drop/error counts across a model's skills for the summary line.
func tally(ms ModelStats) (kept, dropped, errored int) {
	for _, s := range ms.Skills {
		kept += s.Kept
		dropped += s.Dropped
		errored += s.Errored
	}
	return
}

// cachedOr renders a step duration, marking near-instant completions as cache
// hits — "segmented bash: 20 claims (cached)" reads very differently from
// "(00:04:55)", and the distinction is what tells a watching user whether the
// judge actually worked or the run replayed disk state.
func cachedOr(d time.Duration) string {
	if d < 2*time.Second {
		return "cached"
	}
	return fmtDuration(d)
}

// fmtDuration renders a duration as HH:MM:SS for compact, sortable log prefixes.
func fmtDuration(d time.Duration) string {
	d = d.Round(time.Second)
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
}

// pct renders the size change from orig to pruned as a signed percentage, e.g.
// "-34%". Guards against a zero original.
func pct(orig, pruned int) string {
	if orig == 0 {
		return "0%"
	}
	d := float64(pruned-orig) / float64(orig) * 100
	return fmt.Sprintf("%+.0f%%", d)
}
