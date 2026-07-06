package main

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"
)

// probeItem is one claim scheduled for probing on the current target, plus the
// resume state the caller resolved from disk. haveQ means its question is
// already authored (the question cache is shared across models); havePairs means
// its A/B samples are already generated (this model's samples.jsonl). idx/nClaims
// drive the "3/13" progress display. The worker pools fill question/pairs/genMs
// in place as the item moves through the stages.
type probeItem struct {
	claim   Claim
	idx     int
	nClaims int
	skillMD string // the whole clean skill, used as the arm-B system prompt

	question  Question
	haveQ     bool
	pairs     []samplePair
	havePairs bool
	genMs     int64 // generation wall time, carried into the result's duration split

	// Strengthen-and-retry state: after an ineffective drop (model ignored the
	// statement even with the skill loaded), the judge rewrites it once and the
	// claim re-runs arm B with `strengthened` spliced over its span. retried
	// bounds the loop to a single attempt.
	strengthened string
	retried      bool
}

// probeStore persists the three durable artifacts of a pass, each write
// mutex-guarded so the judge and target worker pools can't corrupt a file:
//   - questions: the shared authored-probe cache (one file per skill), rewritten
//     after each author so an interrupt keeps the (expensive) authored probes.
//     Only current claims are ever in the in-memory map, so the rewrite also GCs
//     entries orphaned by a segmentation change.
//   - samples: this model's samples.jsonl, appended the instant the target
//     finishes a claim — the durability that lets the target run ahead / be freed.
//   - results: this model's results.jsonl resume ledger, appended per verdict.
//
// The store does file I/O only; updating the caller's in-memory result map is
// the caller's job (via schedEvents.onResult), keeping the store lock-simple.
type probeStore struct {
	mu sync.Mutex

	questions    map[string]map[string]Question // skill -> claim ID -> question (current claims only)
	questionPath map[string]string              // skill -> cache file
	samplesPath  string
	resultsPath  string
}

func (s *probeStore) saveQuestion(skill, id string, q Question) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	m := s.questions[skill]
	if m == nil {
		m = map[string]Question{}
		s.questions[skill] = m
	}
	m[id] = q
	return writeQuestionCache(s.questionPath[skill], m)
}

func (s *probeStore) saveSamples(r sampleRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return appendSamples(s.samplesPath, r)
}

func (s *probeStore) saveResult(r ProbeResult) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return appendResult(s.resultsPath, r)
}

// schedEvents are runProbePass's progress callbacks; any may be nil. onResult
// fires after a verdict (or errored result) is persisted. onSkillDone fires once,
// when a skill's last claim reaches a terminal state (clean finish only, never
// mid-cancel), carrying the clean skill bytes + claims so the caller can prune.
// The judge-side callbacks carry the judge ModelSpec that ran the task, so the
// caller can show which endpoint (and its URL) did the work — useful with a
// multi-judge pool. onResult's judge is the empty ModelSpec for a generation
// error (no judge ran).
type schedEvents struct {
	onPrepped    func(judge ModelSpec, skill string, claims, pending int, d time.Duration)
	onAuthored   func(judge ModelSpec, it *probeItem, d time.Duration)
	onAuthorFail func(judge ModelSpec, it *probeItem)
	onGenerating func(it *probeItem)
	onGenerated  func(it *probeItem, genMs int64)
	onJudged     func(judge ModelSpec, it *probeItem, i, n int, verdict string)
	onResult     func(judge ModelSpec, it *probeItem, res ProbeResult, genMs int64)
	// onStrengthen fires when an ineffective drop earned a strengthened retry:
	// the judge rewrote the statement and the claim went back to the target.
	onStrengthen func(judge ModelSpec, it *probeItem, text string)
	onSkillDone  func(skill string, orig []byte, claims []Claim)
}

// prepFunc streamlines + segments one skill on the given judge (both cached),
// returning the clean skill bytes and its claims.
type prepFunc func(judge ModelSpec, sk skillSource) (orig []byte, claims []Claim, err error)

// resumeFunc resolves one claim's disk state for the current model: its cached
// question (if authored), its generated pairs (if any), and whether it already
// has a final verdict (done → skip probing, just count toward the skill).
type resumeFunc func(c Claim) (q Question, haveQ bool, pairs []samplePair, havePairs bool, done bool)

// preferAuthor reports whether a free judge slot should feed the target (author)
// before doing anything else: true while fewer than one authored claim per target
// slot is queued in genQ, i.e. the target is about to starve.
func preferAuthor(genQueued, targetSlots int) bool {
	if targetSlots < 1 {
		targetSlots = 1
	}
	return genQueued < targetSlots
}

// runProbePass probes every claim of every skill on one target as a single
// streaming, demand-driven schedule. Four task types flow through four queues:
//
//	prepQ ⟨PREP judge⟩→ authorQ ⟨AUTHOR judge⟩→ genQ ⟨GENERATE target⟩→ judgeQ ⟨SCORE judge⟩→ result
//
// A free judge slot always does the task that refills the emptiest upstream queue,
// priority AUTHOR > PREP > SCORE: feed the target first (author a claim into
// genQ); if the target's fed but the claim buffer is low, make more (prep a skill
// → authorQ); score (drain judgeQ) only when everything upstream is healthy. So
// the judge never streamlines the next skill while this skill's claims sit
// un-authored and the target idles. The target only generates (genQ → judgeQ).
//
// Streaming: claims don't exist until a skill is PREP'd, so all four queues are
// unbounded (a push never blocks — a run is a few hundred tiny jobs) and
// termination is dynamic: the pass ends when every skill is prepped AND every
// emitted claim is terminal. judges is the interchangeable pool; any judge does
// any judge task. Returns the first persistence error (fatal). A cancelled ctx
// stops at task boundaries with all persisted progress intact.
func runProbePass(ctx context.Context, judges []ModelSpec, target ModelSpec, skills []skillSource, prep prepFunc, resume resumeFunc, samples, keepThreshold int, store *probeStore, ev schedEvents) error {
	if len(skills) == 0 {
		return nil
	}
	// Guard, not just validation: with zero judges the worker accounting below
	// would Add(1) for a floored nJudge but spawn no judge worker — Wait() would
	// then hang forever once the (never-served) queues drain.
	if len(judges) == 0 {
		return fmt.Errorf("runProbePass: no judge endpoints")
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	nJudge := 0
	for _, j := range judges {
		if j.Parallel < 1 {
			nJudge++
		} else {
			nJudge += j.Parallel
		}
	}
	if nJudge < 1 {
		nJudge = 1
	}
	nGen := target.Parallel
	if nGen < 1 {
		nGen = 1
	}
	// prepBuffer: keep at least this many claims queued for AUTHOR so a target that
	// drains genQ doesn't starve while an (~minutes-long) PREP runs.
	prepBuffer := nGen

	// All queues are unbounded (sized past any real run's job count) so a push
	// never blocks — the deadlock-free invariant and the whole point: put every
	// job in a queue, then just pick the right queue to pull from.
	big := len(skills)*1024 + 4096
	authorQ := make(chan *probeItem, big)
	genQ := make(chan *probeItem, big)
	judgeQ := make(chan *probeItem, big)
	prepQ := make(chan skillSource, len(skills))
	for _, sk := range skills {
		prepQ <- sk
	}
	// prepQ is NOT closed: workers take skills with non-blocking or blocking
	// receives, and a drained-but-open channel simply blocks (no busy-spin).

	type skillInfo struct {
		orig   []byte
		claims []Claim
	}
	var mu sync.Mutex
	prepsLeft := len(skills)
	inFlight := 0 // emitted, non-terminal claims
	skillLeft := map[string]int{}
	skillMeta := map[string]skillInfo{}

	done := make(chan struct{})
	var closeDone sync.Once
	finish := func() { closeDone.Do(func() { close(done) }) }
	maybeFinish := func() { // caller holds mu
		if prepsLeft == 0 && inFlight == 0 {
			finish()
		}
	}

	var errMu sync.Mutex
	var firstErr error
	fail := func(e error) {
		errMu.Lock()
		if firstErr == nil {
			firstErr = e
		}
		errMu.Unlock()
		cancel()
		finish()
	}

	go func() {
		select {
		case <-ctx.Done():
			finish()
		case <-done:
		}
	}()

	// terminal marks one claim finished; fires onSkillDone for a skill's last claim
	// (clean finish only) and ends the pass when the last claim lands.
	terminal := func(it *probeItem) {
		s := it.claim.Skill
		mu.Lock()
		inFlight--
		skillLeft[s]--
		last := skillLeft[s] == 0
		info := skillMeta[s]
		maybeFinish()
		mu.Unlock()
		if last && ctx.Err() == nil && ev.onSkillDone != nil {
			ev.onSkillDone(s, info.orig, info.claims)
		}
	}

	score := func(judge ModelSpec, it *probeItem) bool {
		jStart := time.Now()
		var prog func(i, n int, verdict string)
		if ev.onJudged != nil {
			pit := it
			prog = func(i, n int, verdict string) { ev.onJudged(judge, pit, i, n, verdict) }
		}
		res := judgeSamples(ctx, judge, target, it.claim, it.question, it.pairs, keepThreshold, prog)
		res.StrengthenedText = it.strengthened // "" on the first round
		res.DurationMs = it.genMs + time.Since(jStart).Milliseconds()
		res.EndedAt = time.Now().Format(time.RFC3339)
		if err := store.saveResult(res); err != nil {
			fail(fmt.Errorf("write result %s: %w", it.claim.ID, err))
			return false
		}
		if ev.onResult != nil {
			ev.onResult(judge, it, res, it.genMs)
		}
		// Strengthen-and-retry: an INEFFECTIVE drop (model failed the rubric even
		// with the skill loaded — it read the statement and ignored it) gets one
		// second chance with a judge-strengthened wording. The judge rewrites the
		// statement from the failing answers; the item goes back to the target to
		// regenerate arm B only (arm A has no skill, so it's reused). The
		// ineffective-drop row above stays in the ledger for the audit trail; the
		// retry's row lands after it and wins on resume (last-line-wins). A crash
		// between the two leaves the drop — a valid, if unretried, final state.
		if res.Err == "" && res.Ineffective && !it.retried && ctx.Err() == nil {
			if text, ok := strengthenClaim(ctx, judge, it.claim, it.question, it.pairs); ok {
				it.retried = true
				it.strengthened = text
				if ev.onStrengthen != nil {
					ev.onStrengthen(judge, it, text)
				}
				genQ <- it // regenerate B with the strengthened skill; NOT terminal yet
				return true
			}
		}
		terminal(it)
		return true
	}

	authorItem := func(judge ModelSpec, it *probeItem) bool {
		aStart := time.Now()
		q, ok := authorQuestion(ctx, judge, it.claim)
		if !ok {
			if ev.onAuthorFail != nil {
				ev.onAuthorFail(judge, it)
			}
			terminal(it) // unauthored → untested this run
			return true
		}
		if err := store.saveQuestion(it.claim.Skill, it.claim.ID, q); err != nil {
			fail(fmt.Errorf("cache question %s: %w", it.claim.ID, err))
			return false
		}
		it.question, it.haveQ = q, true
		if ev.onAuthored != nil {
			ev.onAuthored(judge, it, time.Since(aStart))
		}
		genQ <- it // generous buffer → never blocks
		return true
	}

	// doPrep streamlines+segments one skill, registers it, and emits its claims at
	// their resume entry point. A prep error skips the skill (untested this run) —
	// not fatal.
	doPrep := func(judge ModelSpec, sk skillSource) {
		pStart := time.Now()
		orig, claims, err := prep(judge, sk)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warn: skipping skill %s: %v\n", sk.stack, err)
			mu.Lock()
			prepsLeft--
			maybeFinish()
			mu.Unlock()
			return
		}
		n := len(claims)
		var emit []*probeItem
		doneCount := 0
		for idx, c := range claims {
			q, haveQ, pairs, havePairs, isDone := resume(c)
			if isDone {
				doneCount++
				continue
			}
			emit = append(emit, &probeItem{
				claim: c, idx: idx, nClaims: n, skillMD: string(orig),
				question: q, haveQ: haveQ, pairs: pairs, havePairs: havePairs,
			})
		}
		mu.Lock()
		skillMeta[sk.stack] = skillInfo{orig: orig, claims: claims}
		skillLeft[sk.stack] = n - doneCount // already-done claims are already terminal
		inFlight += len(emit)               // reserve the full count before emitting
		prepsLeft--
		last := skillLeft[sk.stack] == 0
		maybeFinish()
		mu.Unlock()
		if ev.onPrepped != nil {
			ev.onPrepped(judge, sk.stack, n, len(emit), time.Since(pStart))
		}
		for _, it := range emit { // emit outside the lock; unbounded queues never block
			switch {
			case it.havePairs:
				judgeQ <- it
			case it.haveQ:
				genQ <- it
			default:
				authorQ <- it
			}
		}
		if last && ctx.Err() == nil && ev.onSkillDone != nil {
			ev.onSkillDone(sk.stack, orig, claims) // fully-resumed skill → prune now
		}
	}

	judgeWorker := func(judge ModelSpec) {
		for {
			// AUTHOR > PREP > SCORE, each a non-blocking try gated on the emptiest
			// upstream queue; fall through to a blocking select on all sources.
			if preferAuthor(len(genQ), nGen) { // target hungry → feed it
				select {
				case it := <-authorQ:
					if !authorItem(judge, it) {
						return
					}
					continue
				case <-done:
					return
				default:
				}
			}
			if len(authorQ) < prepBuffer { // claim buffer low → make more
				select {
				case sk := <-prepQ:
					doPrep(judge, sk)
					continue
				case <-done:
					return
				default:
				}
			}
			select { // target fed + claims buffered → score
			case it := <-judgeQ:
				if !score(judge, it) {
					return
				}
				continue
			case <-done:
				return
			default:
			}
			// Nothing preferred was ready → block on anything (incl. prep, so an
			// otherwise-idle judge preps ahead). No busy-spin.
			select {
			case <-done:
				return
			case it := <-authorQ:
				if !authorItem(judge, it) {
					return
				}
			case it := <-judgeQ:
				if !score(judge, it) {
					return
				}
			case sk := <-prepQ:
				doPrep(judge, sk)
			}
		}
	}

	genWorker := func() {
		for {
			select {
			case <-done:
				return
			case it := <-genQ:
				if ev.onGenerating != nil {
					ev.onGenerating(it)
				}
				gStart := time.Now()
				var pairs []samplePair
				var err error
				if it.strengthened != "" {
					// Strengthened retry: splice the rewrite over the claim's span
					// and regenerate ONLY arm B — arm A saw no skill, so the first
					// round's A answers are reused as-is.
					bSkill := pruneSkill(it.skillMD, nil, []replacement{{Claim: it.claim, Text: it.strengthened}})
					pairs, err = regenerateB(ctx, target, it.question, it.pairs, bSkill)
				} else {
					pairs, err = generateSamples(ctx, target, it.claim, it.question, samples, it.skillMD)
				}
				it.genMs = time.Since(gStart).Milliseconds()
				if err != nil {
					res := newProbeResult(target, it.claim, it.question)
					res.Err = err.Error()
					res.StrengthenedText = it.strengthened
					res.DurationMs = it.genMs
					res.EndedAt = time.Now().Format(time.RFC3339)
					if serr := store.saveResult(res); serr != nil {
						fail(fmt.Errorf("write result %s: %w", it.claim.ID, serr))
						return
					}
					if ev.onResult != nil {
						ev.onResult(ModelSpec{}, it, res, it.genMs) // no judge ran (generation error)
					}
					terminal(it)
					continue
				}
				it.pairs = pairs
				if it.strengthened == "" {
					// Retry pairs are deliberately NOT persisted: samples.jsonl is
					// the resume ledger for the ORIGINAL skill wording; a resumed
					// run re-judges from those and re-strengthens if still needed.
					if err := store.saveSamples(sampleRecord{ClaimID: it.claim.ID, Skill: it.claim.Skill, QuestionHash: questionHash(it.question), Pairs: pairs}); err != nil {
						fail(fmt.Errorf("persist samples %s: %w", it.claim.ID, err))
						return
					}
				}
				if ev.onGenerated != nil {
					ev.onGenerated(it, it.genMs)
				}
				judgeQ <- it // unbounded → never blocks
			}
		}
	}

	var workers sync.WaitGroup
	workers.Add(nJudge + nGen)
	for _, j := range judges {
		slots := j.Parallel
		if slots < 1 {
			slots = 1
		}
		for i := 0; i < slots; i++ {
			jj := j
			go func() { defer workers.Done(); judgeWorker(jj) }()
		}
	}
	for i := 0; i < nGen; i++ {
		go func() { defer workers.Done(); genWorker() }()
	}

	<-done
	// No cancel() here: on a clean finish every claim is already terminal, so
	// workers are idle and exit via <-done. Cancelling would race the last skill's
	// onSkillDone (guarded by ctx.Err()==nil) and skip its pruning. In-flight calls
	// are unblocked only where it's actually needed — fail() and a real
	// parent-context cancel — and defer cancel() cleans up on return.
	workers.Wait()
	errMu.Lock()
	defer errMu.Unlock()
	return firstErr
}
