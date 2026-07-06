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
	"sync"
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
	cleanFlag := flag.String("clean", "clean-skills", "directory for the streamlined (clean) skills the pipeline probes; derived from ground-skills, cached")
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

	// Resolve each endpoint's concurrency: an unset `parallel` is auto-detected
	// from the server's slot count (llama.cpp -np), same as the main codehalter
	// binary. Runs after preflight so a router has loaded the model /props
	// reports on.
	logf("resolving endpoint parallelism ...")
	cfg.resolveParallelism(func(m ModelSpec) int {
		dctx, dcancel := context.WithTimeout(ctx, 30*time.Second)
		n := detectSlots(dctx, m)
		dcancel()
		return n
	}, logf)

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

	// The whole run streams per model: PREP, AUTHOR, GENERATE and SCORE are
	// scheduled together, and a free judge slot always does the task that refills
	// the emptiest upstream queue (AUTHOR > PREP > SCORE) so the target never idles
	// while a later skill is being streamlined. Prep + authoring are cached and
	// model-independent, so model A does the real work and model B reuses it.
	//
	// prepSkill streamlines + segments one skill (both cached) and, once the
	// claims are known, GCs any authored-probe entries orphaned by a segmentation
	// change from that skill's shared question cache. Segmentation runs the first
	// pass on judge A and the repair (re-quote) pass on judge B — a different
	// endpoint when the pool has more than one, else the same judge.
	segJudgeA := cfg.Judges[0]
	segJudgeB := cfg.Judges[len(cfg.Judges)-1]
	prepSkill := func(judge ModelSpec, sk skillSource, store *probeStore) ([]byte, []Claim, error) {
		cleanPath, err := ensureCleanSkill(ctx, judge, sk.stack, sk.path, *cleanFlag)
		if err != nil {
			return nil, nil, err
		}
		orig, err := os.ReadFile(cleanPath)
		if err != nil {
			return nil, nil, fmt.Errorf("read clean skill: %w", err)
		}
		claims, err := segmentSkill(ctx, segJudgeA, segJudgeB, sk.stack, cleanPath, filepath.Join(*cacheFlag, "segments"))
		if err != nil {
			return nil, nil, err
		}
		store.mu.Lock()
		if qm := store.questions[sk.stack]; qm != nil {
			ids := make(map[string]bool, len(claims))
			for _, c := range claims {
				ids[c.ID] = true
			}
			for id := range qm {
				if !ids[id] {
					delete(qm, id)
				}
			}
		}
		store.mu.Unlock()
		return orig, claims, nil
	}

	// Model-outer ordering: probe ALL skills on model A, then all on model B. The
	// targets share one router (ai.jos.li) that swaps models on demand, so this
	// loads each target ONCE for its whole pass instead of thrashing A<->B.
	for mi, m := range cfg.Models {
		modelDir := filepath.Join(*modelsFlag, m.Name)
		dm := done[m.Name]
		logf("==== model %d/%d: %s ====", mi+1, len(cfg.Models), m.Name)

		// Resume state, loaded up front. Authored questions are shared across models
		// and cached per skill (raw here; prepSkill GCs orphans once claims exist);
		// generated samples are this model's ledger.
		questions := map[string]map[string]Question{}
		questionPath := map[string]string{}
		for _, sk := range skills {
			path := filepath.Join(*cacheFlag, "authored", sk.stack+".json")
			questionPath[sk.stack] = path
			questions[sk.stack] = readQuestionCache(path)
		}
		samplesPath := filepath.Join(modelDir, "samples.jsonl")
		savedSamples := readSamples(samplesPath)
		store := &probeStore{
			questions:    questions,
			questionPath: questionPath,
			samplesPath:  samplesPath,
			resultsPath:  filepath.Join(modelDir, "results.jsonl"),
		}

		// dm is written by judge workers (onResult) and read by resume + pruneOne;
		// dmMu serializes them. Model-outer means only this model's dm mutates now,
		// so finalizeAll's read of the other models' (final) stats stays race-free.
		var dmMu sync.Mutex
		prunedSkills := map[string]bool{}
		pruneOne := func(stack string, orig []byte, claims []Claim) {
			dmMu.Lock()
			defer dmMu.Unlock()
			if prunedSkills[stack] {
				return
			}
			prunedSkills[stack] = true
			out := pruneSkill(string(orig), droppedClaims(claims, dm), strengthenedClaims(claims, dm))
			outPath := filepath.Join(modelDir, "SKILL-"+stack+".md")
			// Failures here warn, never exit: this runs inside a scheduler worker
			// (onSkillDone), and the pruned skill + stats are pure derivatives of
			// results.jsonl — a resume re-prunes them for free, while an exit here
			// would kill probing that is persisting results just fine.
			if err := os.WriteFile(outPath, []byte(out), 0o644); err != nil {
				logf("warn: write %s: %v — pruned skill lost this run, a re-run regenerates it", outPath, err)
				return
			}
			st := skillStats(stack, string(orig), out, claims, dm)
			ms := statsByModel[m.Name]
			ms.Skills = append(ms.Skills, st)
			ms.OrigBytes += st.OrigBytes
			ms.PrunedBytes += st.PrunedBytes
			if err := writeJSON(filepath.Join(modelDir, "stats.json"), *ms); err != nil {
				logf("warn: write stats: %v", err)
			}
			logf("%s / %s: %d kept, %d dropped, %d errored | %d → %d bytes (%s)",
				stack, m.Name, st.Kept, st.Dropped, st.Errored, st.OrigBytes, st.PrunedBytes, pct(st.OrigBytes, st.PrunedBytes))
			finalizeAll()
		}

		// resume resolves a claim's disk state for this model.
		resume := func(c Claim) (Question, bool, []samplePair, bool, bool) {
			dmMu.Lock()
			r, ok := dm[c.ID]
			dmMu.Unlock()
			if ok && r.Err == "" {
				return Question{}, false, nil, false, true // already decided
			}
			store.mu.Lock()
			q, haveQ := store.questions[c.Skill][c.ID]
			store.mu.Unlock()
			var pairs []samplePair
			havePairs := false
			if haveQ {
				// Saved pairs are reused only when they answered THIS question
				// (hash match) — pairs generated for a since-re-authored question
				// would otherwise be judged against a rubric they never saw.
				if p, ok := savedSamples[c.ID]; ok && p.matches(q) {
					pairs, havePairs = p.Pairs, true
				}
			}
			return q, haveQ, pairs, havePairs, false
		}
		prep := func(judge ModelSpec, sk skillSource) ([]byte, []Claim, error) {
			return prepSkill(judge, sk, store)
		}

		// Warm the router to this model LAZILY, on the first claim that actually
		// needs generation — a fully-resumed pass (or a judge-only resume from
		// saved samples) never loads the model. The Once also gates concurrent
		// gen workers: they block until the switch completes, so a router loads
		// the model exactly once instead of racing several first calls into it.
		var warm sync.Once
		warmup := func() {
			warm.Do(func() {
				logf("switching endpoint to %s (a router may load the model now) ...", m.Name)
				switchStart := time.Now()
				wctx, wcancel := context.WithTimeout(ctx, 10*time.Minute)
				if err := ping(wctx, m); err != nil {
					logf("warn: model switch/warm-up for %s failed: %v — probing anyway", m.Name, err)
				} else {
					logf("%s ready (%s)", m.Name, cachedOr(time.Since(switchStart)))
				}
				wcancel()
			})
		}

		ev := schedEvents{
			onPrepped: func(j ModelSpec, stack string, claims, pending int, d time.Duration) {
				logf("prepped %s: %d claims, %d pending (%s)%s", stack, claims, pending, cachedOr(d), judgeTag(j))
			},
			onAuthored: func(j ModelSpec, it *probeItem, d time.Duration) {
				logf("authored %s (%s)%s", it.claim.ID, cachedOr(d), judgeTag(j))
			},
			onAuthorFail: func(j ModelSpec, it *probeItem) {
				logf("warn: %s unauthored — untested this run%s", it.claim.ID, judgeTag(j))
			},
			onGenerating: func(it *probeItem) {
				warmup() // lazy: first generation loads/switches the model, exactly once
				if it.strengthened != "" {
					logf("[%s %d/%d] %s %s regenerating arm B with strengthened wording ...", it.claim.Skill, it.idx+1, it.nClaims, m.Name, it.claim.ID)
					return
				}
				logf("[%s %d/%d] %s %s generating %d samples ...", it.claim.Skill, it.idx+1, it.nClaims, m.Name, it.claim.ID, cfg.Settings.Samples)
			},
			onGenerated: func(it *probeItem, genMs int64) {
				logf("[%s %d/%d] %s %s generated (%dms) → judging", it.claim.Skill, it.idx+1, it.nClaims, m.Name, it.claim.ID, genMs)
			},
			onJudged: func(j ModelSpec, it *probeItem, i, n int, verdict string) {
				logf("[%s %d/%d] %s %s judged sample %d/%d: %s%s", it.claim.Skill, it.idx+1, it.nClaims, m.Name, it.claim.ID, i, n, verdict, judgeTag(j))
			},
			onResult: func(j ModelSpec, it *probeItem, res ProbeResult, genMs int64) {
				dmMu.Lock()
				dm[res.ClaimID] = res
				dmMu.Unlock()
				if res.Err != "" {
					logf("[%s %d/%d] %s %s ERROR (%dms): %s", it.claim.Skill, it.idx+1, it.nClaims, m.Name, res.ClaimID, res.DurationMs, res.Err)
					return
				}
				// Verdict qualifiers: UNSTABLE = samples disagreed; INEFFECTIVE =
				// failed the rubric even with the skill loaded; STRENGTHENED =
				// this verdict came from the rewritten-wording retry.
				mark := ""
				if res.Unstable {
					mark += " UNSTABLE"
				}
				if res.Ineffective {
					mark += " INEFFECTIVE"
				}
				if res.StrengthenedText != "" {
					mark += " STRENGTHENED"
				}
				logf("[%s %d/%d] %s %s → %s%s (%dms: gen %dms + judge)%s", it.claim.Skill, it.idx+1, it.nClaims, m.Name, res.ClaimID, strings.ToUpper(res.Verdict), mark, res.DurationMs, genMs, judgeTag(j))
			},
			onStrengthen: func(j ModelSpec, it *probeItem, text string) {
				logf("[%s %d/%d] %s %s ignored the statement — retrying with strengthened wording: %q%s", it.claim.Skill, it.idx+1, it.nClaims, m.Name, it.claim.ID, truncate(text, 100), judgeTag(j))
			},
			onSkillDone: func(stack string, orig []byte, claims []Claim) { pruneOne(stack, orig, claims) },
		}
		if err := runProbePass(ctx, cfg.Judges, m, skills, prep, resume, cfg.Settings.Samples, cfg.Settings.KeepThreshold, store, ev); err != nil {
			log.Fatalf("%v", err)
		}
		if ctx.Err() != nil {
			logf("interrupted during %s — progress saved, re-run to resume", m.Name)
			finalizeAll()
			return
		}
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
	var checks []check
	for _, j := range cfg.Judges {
		checks = append(checks, check{j.Name, j})
	}
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

// judgeTag renders " [name @ url]" for the judge endpoint that ran a task, so a
// multi-judge pool's logs show which judge did what. Empty for the zero ModelSpec
// (e.g. a generation error, where no judge ran).
func judgeTag(m ModelSpec) string {
	if m.Name == "" && m.Server == "" {
		return ""
	}
	return fmt.Sprintf(" [%s @ %s]", m.Name, m.Server)
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
