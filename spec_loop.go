package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"
)

// The whole /spec loop is one Prompt: Cancel stops it anywhere and the next /spec resumes from
// the recomputed ledger. Every passing item is committed, so nothing depends on the session.

// specTestTimeout is generous: a GUI project's first test build compiles its whole dependency tree.
const specTestTimeout = 30 * time.Minute

const specTestTailBytes = 4000

const specSpecDiffBytes = 4000

// specQuestionError stops the item until the spec's QUESTIONS.md answers it (see runPlanPhase).
// Problem: the question still missed what makes it answerable after one corrective, so it
// is not written to QUESTIONS.md.
type specQuestionError struct {
	Q       specQuestion
	Problem string
}

func (e *specQuestionError) Error() string { return "the planner needs an answer: " + e.Q.Question }

// setSpecFence takes an absolute dir; "" lifts the fence.
func (s *Session) setSpecFence(dir string) {
	s.rt.mu.Lock()
	s.rt.specFenceDir = dir
	s.rt.mu.Unlock()
}

// requestSpecStop ends the running loop at its next round boundary, not now.
func (s *Session) requestSpecStop() {
	s.rt.mu.Lock()
	s.rt.specStop = true
	s.rt.mu.Unlock()
}

// setSpecHandoff records the /spec command the planner handed back instead of a plan.
func (s *Session) setSpecHandoff(cmd string) {
	s.rt.mu.Lock()
	s.rt.specHandoff = cmd
	s.rt.mu.Unlock()
}

func (s *Session) specHandoffPending() bool {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	return s.rt.specHandoff != ""
}

func (s *Session) takeSpecHandoff() string {
	s.rt.mu.Lock()
	cmd := s.rt.specHandoff
	s.rt.specHandoff = ""
	s.rt.mu.Unlock()
	return cmd
}

func (s *Session) takeSpecStop() bool {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	stop := s.rt.specStop
	s.rt.specStop = false
	return stop
}

func (s *Session) specFence() string {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	return s.rt.specFenceDir
}

// specFenceRefusal is enforced in code, not in a prompt: deleting an id from the spec is the one
// way a round could make an item "done" without implementing it.
func (a *agent) specFenceRefusal(sid, absPath string) string {
	sess := a.getSession(sid)
	if sess == nil {
		return ""
	}
	fence := sess.specFence()
	if fence == "" || (absPath != fence && !strings.HasPrefix(absPath, fence+string(filepath.Separator))) {
		return ""
	}
	return "error: the spec is read-only while /spec runs, so " + absPath + " cannot be written. The spec is the source of truth: implement what it says in the output directory. If the spec itself is wrong, say so in your reply instead of changing it."
}

type specMode int

const (
	specModeItem     specMode = iota // implement an item the spec added
	specModeSetup                    // the one-off project skeleton round
	specModeChange                   // redo an item whose spec text moved
	specModeRemove                   // delete an item the spec no longer has
	specModeRefactor                 // every few items: the code's shape, no item
)

// specRefactorID is the refactor round's pseudo-item.
const specRefactorID = "refactor"

type specRoundResult struct {
	Mode      specMode
	Redo      bool // the item was sent back with /spec redo
	Committed bool
	Question  string
	// UIUnseen lists UI source files edited without a single screenshot.
	UIUnseen  []string
	TurnErr   string
	TestsPass bool
	TestTail  string
	Covered   bool // setup: a test source exists
	// Gates on what the round wrote: functions only tests reach, lint findings in
	// its own lines, files grown past the size budget.
	Unreachable []string
	StandIns    []string // program lines that stand in for the work: a for-test helper asked, "not implemented"
	Lint        []string
	Oversize    []string
	// Refactor: the debt shrank, and how it moved.
	Improved bool
	Debt     string
}

// specDecide counts the attempt in cfg.Attempts, so a cancelled and resumed /spec gives no fresh budget.
// prev is what the item's last attempt failed; fails is what this one failed.
// A second failure of another kind earns one more attempt: the item moved.
func specDecide(cfg *specConfig, item string, r specRoundResult, prev string) (done, block bool, reason, fails string) {
	if r.Question != "" {
		return false, true, "the planner asked a question instead of guessing", ""
	}
	switch {
	case r.Mode == specModeRemove:
		if r.TurnErr == "" && r.TestsPass && !r.Covered {
			return true, false, "", ""
		}
	case r.Mode == specModeRefactor:
		if r.TurnErr == "" && r.TestsPass && r.Committed && r.Improved && len(r.Lint)+len(r.Oversize) == 0 {
			return true, false, "", ""
		}
	case r.Mode == specModeChange:
		// The test for the OLD text still covers a changed item; the commit is the evidence of work.
		if r.TurnErr == "" && r.Covered && r.TestsPass && r.Committed && len(r.Unreachable)+len(r.StandIns)+len(r.Lint)+len(r.Oversize) == 0 {
			return true, false, "", ""
		}
	default:
		// A reopened item's old test still names it: like a change, it needs a commit.
		if r.TurnErr == "" && r.Covered && r.TestsPass && len(r.UIUnseen) == 0 && (r.Committed || !r.Redo) &&
			len(r.Unreachable)+len(r.StandIns)+len(r.Lint)+len(r.Oversize) == 0 {
			return true, false, "", ""
		}
	}
	var why, kinds []string
	add := func(kind, text string) {
		why = append(why, text)
		if !slices.Contains(kinds, kind) {
			kinds = append(kinds, kind)
		}
	}
	if len(r.UIUnseen) > 0 && r.TurnErr == "" {
		add("ui", fmt.Sprintf("you changed the UI (%s) and never looked at it: render the screen and look at it with `screenshot`", strings.Join(r.UIUnseen, ", ")))
	}
	if r.TurnErr != "" {
		add("error", "the round ended with an error: "+r.TurnErr)
	}
	if r.Mode == specModeChange && !r.Committed && r.TurnErr == "" {
		add("nochange", "the spec text changed but no code did: update the implementation AND its test to match the new text")
	}
	if r.Redo && !r.Committed && r.TurnErr == "" {
		add("nochange", "the item was sent back with /spec redo but no code changed: rebuild what falls short of the spec text")
	}
	if r.Mode == specModeRemove && r.Covered {
		add("stillnamed", fmt.Sprintf("a test in the output directory still names %s", item))
	}
	switch {
	case r.Covered, r.Mode == specModeRemove, r.Mode == specModeRefactor:
	case r.Mode == specModeSetup:
		add("notests", "no test source exists in the output directory yet")
	default:
		add("unnamed", fmt.Sprintf("no test this round added or changed is named after %s: put `%s` in a test function's name or in a test call's title; a comment or a string that mentions it does not count", item, specTestToken(item)))
	}
	if len(r.Unreachable) > 0 {
		add("unreachable", "the program does not call these functions the round added: "+strings.Join(r.Unreachable, "; ")+
			". Call each from the path the spec describes (a UI handler, main, the flow that owns it) or delete it; a helper that exists only for tests says so in its name (`rows_for_test`)")
	}
	if len(r.StandIns) > 0 {
		add("standin", "the program stands in for the work instead of doing it: "+strings.Join(r.StandIns, "; ")+
			". Build the real thing the spec describes (call the service, write the format, drive the device); a test replaces it through a seam the program itself uses, such as a server URL pointed at a local fake, never through a slot only tests fill. An item whose work is a refusal or a scripted answer is not done")
	}
	if len(r.Lint) > 0 {
		add("lint", "the linter reports problems in lines this round wrote:\n\n```\n"+strings.Join(r.Lint, "\n")+"\n```")
	}
	for _, o := range r.Oversize {
		add("oversize", o)
	}
	if r.Mode == specModeRefactor && !r.Improved && r.TurnErr == "" {
		add("noprogress", "the refactoring made no measured progress ("+r.Debt+"): one of the numbers must go down and none up")
	}
	if !r.TestsPass {
		add("tests", "the test command did not pass. The end of its output:\n\n```\n"+strings.TrimSpace(r.TestTail)+"\n```")
	}
	if len(why) == 0 {
		add("unfinished", "the round did not finish the item")
	}
	if cfg.Attempts == nil {
		cfg.Attempts = map[string]int{}
	}
	cfg.Attempts[item]++
	reason, fails = strings.Join(why, "; "), strings.Join(kinds, ",")
	limit := specMaxAttempts
	if cfg.Attempts[item] == specMaxAttempts && prev != "" && fails != prev {
		limit++
	}
	if cfg.Attempts[item] >= limit {
		return false, true, reason, fails
	}
	return false, false, reason, fails
}

func specPaths(cwd, specArg, outArg string) (specRel, outRel string, err error) {
	rel := func(p string) (string, error) {
		abs := p
		if !filepath.IsAbs(abs) {
			abs = filepath.Join(cwd, p)
		}
		abs = filepath.Clean(abs)
		r, err := filepath.Rel(cwd, abs)
		if err != nil || r == ".." || strings.HasPrefix(r, ".."+string(filepath.Separator)) {
			return "", fmt.Errorf("%s is outside the project", p)
		}
		return filepath.ToSlash(r), nil
	}
	if specRel, err = rel(specArg); err != nil {
		return "", "", err
	}
	if outRel, err = rel(outArg); err != nil {
		return "", "", err
	}
	if st, err := os.Stat(filepath.Join(cwd, specRel)); err != nil || !st.IsDir() {
		return "", "", fmt.Errorf("spec directory %s does not exist", specArg)
	}
	switch {
	case specRel == ".":
		return "", "", errors.New("the spec directory cannot be the project root: it is fenced read-only while the loop runs")
	case outRel == specRel,
		strings.HasPrefix(outRel+"/", specRel+"/"),
		outRel != "." && strings.HasPrefix(specRel+"/", outRel+"/"):
		return "", "", errors.New("the spec directory and the output directory must not contain each other")
	}
	return specRel, outRel, nil
}

// specIgnoredProbes finds unanchored ignore rules from the old code (`cut/`, `*.json`) that would
// silently drop the rewrite's files from every per-item commit.
func specIgnoredProbes(ctx context.Context, cwd, outRel string, idx *specIndex) []string {
	if _, err := specGit(ctx, cwd, "rev-parse", "--is-inside-work-tree"); err != nil {
		return nil
	}
	probes := []string{"Cargo.toml", "package.json", "src/lib.rs", "src/main.rs", "tests/smoke.rs", "tests/fixture.json", "test_data/x", "fixtures/x.json"}
	// A rewrite names its modules after the chapters: "09-llm-and-tools.md" → "llm_and_tools", "llm".
	var names []string
	for _, d := range idx.docs {
		if strings.Contains(d.rel, "/") {
			continue
		}
		stem := strings.ToLower(strings.TrimSuffix(d.rel, filepath.Ext(d.rel)))
		stem = strings.TrimLeft(stem, "0123456789-_ ")
		words := strings.FieldsFunc(stem, func(r rune) bool { return r == '-' || r == '_' || r == ' ' })
		if len(words) == 0 || stem == "readme" {
			continue
		}
		for _, n := range []string{strings.Join(words, "_"), words[0]} {
			if !slices.Contains(names, n) {
				names = append(names, n)
				probes = append(probes, "src/"+n+"/mod.rs", n+"/x")
			}
		}
	}
	var in strings.Builder
	for _, p := range probes {
		in.WriteString(filepath.ToSlash(filepath.Join(outRel, p)))
		in.WriteByte('\n')
	}
	c := exec.CommandContext(ctx, "git", "-C", cwd, "check-ignore", "-v", "--stdin")
	c.Stdin = strings.NewReader(in.String())
	out, err := c.Output()
	if err != nil && len(out) == 0 {
		return nil // exit 1 with no output: nothing is ignored
	}
	var lines []string
	seen := map[string]bool{}
	for _, ln := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		src, path, ok := strings.Cut(ln, "\t") // "<file>:<line>:<pattern>\t<path>"
		if !ok || seen[src] {
			continue
		}
		seen[src] = true
		lines = append(lines, fmt.Sprintf("`%s` (%s) hides %s", src[strings.LastIndex(src, ":")+1:], src[:strings.LastIndex(src, ":")], path))
	}
	return lines
}

// specRun.idx is replaced every round, since the loop re-reads the spec.
type specRun struct {
	a      *agent
	sid    string
	sess   *Session
	cfg    *specConfig
	idx    *specIndex
	outAbs string

	reasons     map[string]string // why the last round on an item did not count
	lastDelta   string            // the change report already shown
	vanished    bool              // the last scan lost most of the ledger at once
	lintMissing string            // why the linter did not run: the next round installs it
	// blocked: items given up on in this run, with why. Not saved: the next /spec
	// tries them afresh, since most blocks outlive their cause (a bug since fixed,
	// a red build another item repaired).
	blocked map[string]string
	// completionChecked: the done items were checked against the spec in this run.
	completionChecked bool
	fails             map[string]string // which checks an item's last attempt failed (see specDecide)
}

type specWork struct {
	Item string
	Mode specMode
	// Answered: the item's answered questions from QUESTIONS.md, as the file words them.
	Answered string
	Reason   string
	Note     string // change: the spec's diff; removal: the text that was deleted
	Redo     bool
	Debt     specDebt // refactor: measured when the round was picked
	// LintMissing: why the linter did not run last round, so this one installs it.
	LintMissing string
}

func (r *specRun) say(ctx context.Context, s string) { r.a.say(ctx, r.sid, s) }

func (r *specRun) save(ctx context.Context) {
	if err := saveSpecConfig(r.sess.Cwd, r.cfg); err != nil {
		r.say(ctx, "⚠ /spec: "+err.Error()+"\n")
	}
}

func (r *specRun) scan() error {
	idx, err := scanSpec(filepath.Join(r.sess.Cwd, r.cfg.SpecDir), r.cfg.idPatterns(), r.cfg.Context, r.cfg.Skip)
	if err == nil {
		r.idx = idx
	}
	return err
}

func (r *specRun) testCmd() string {
	if r.cfg.TestCmd != "" {
		return r.cfg.TestCmd
	}
	return detectSpecTestCmd(r.outAbs)
}

func (a *agent) runSpec(ctx context.Context, sid string, sess *Session, args string, pendingFixes []fixProblem) (PromptResponse, error) {
	end := PromptResponse{StopReason: "end_turn"}
	// Taken first so an early return below cannot leave it for a later turn.
	handoff := sess.takeSpecHandoff()
	cfg, cmd, ok := a.specResolve(ctx, sid, sess, args)
	if !ok {
		return end, nil
	}
	r := &specRun{a: a, sid: sid, sess: sess, cfg: cfg, outAbs: filepath.Join(sess.Cwd, cfg.OutDir),
		reasons: map[string]string{}, blocked: map[string]string{}}
	if err := r.scan(); err != nil {
		r.say(ctx, "⚠ /spec: reading the spec: "+err.Error()+"\n")
		return end, nil
	}
	if len(r.idx.order) == 0 {
		r.say(ctx, fmt.Sprintf("⚠ /spec: found no requirement ids and no sections in `%s/`. The id patterns are %s; set `id_patterns` in .codehalter/spec.toml if this spec names its requirements differently.\n", cfg.SpecDir, strings.Join(cfg.idPatterns(), ", ")))
		return end, nil
	}
	if cmd == "resume" && strings.HasPrefix(handoff, "redo ") {
		cmd = handoff
	}
	if cmd == "redo" {
		// The audit's submit_plan `redo` goes through specFromPlan, which leaves "redo <ids>" as a handoff.
		if !r.preflight(ctx) {
			return end, nil
		}
		r.say(ctx, "\n## /spec redo · finding what falls short\n\n")
		sess.rt.mu.Lock()
		sess.rt.specAudit = true
		sess.rt.mu.Unlock()
		turnErr := a.runPromptTurn(ctx, sess, a.specAuditPrompt(sid, cfg, r.idx, r.testCmd()))
		sess.rt.mu.Lock()
		sess.rt.specAudit = false
		sess.rt.mu.Unlock()
		if isCancelled(turnErr) {
			return PromptResponse{StopReason: "cancelled"}, nil
		}
		handoff = sess.takeSpecHandoff()
		if !strings.HasPrefix(handoff, "redo ") {
			r.say(ctx, "Nothing was reopened: the audit found every finished item delivering.\n")
			return end, nil
		}
		cmd = handoff
	}
	if strings.HasPrefix(cmd, "redo ") {
		// The ids come from the model: one misremembered id must not discard the rest.
		ids, unknown := specRedoTargets(cfg, r.idx, strings.Fields(strings.TrimPrefix(cmd, "redo ")))
		if len(unknown) > 0 {
			r.say(ctx, fmt.Sprintf("⚠ not in `%s/`, skipped: %s\n", cfg.SpecDir, strings.Join(unknown, ", ")))
		}
		if len(ids) == 0 {
			r.say(ctx, "Nothing to reopen.\n")
			return end, nil
		}
		cfg.reopen(ids)
		r.save(ctx)
		r.say(ctx, fmt.Sprintf("↩ reopened %d item(s): %s. The loop rebuilds them now, under the current rules.\n\n", len(ids), strings.Join(ids, ", ")))
		// Fall through: reopening is the first half of "do these again".
	}
	if cmd == "status" || cmd == "stop" {
		covered, testFiles, err := specCoverage(r.outAbs, r.idx.order)
		if err != nil {
			r.say(ctx, "⚠ /spec: scanning "+cfg.OutDir+": "+err.Error()+"\n")
			return end, nil
		}
		if cmd == "stop" {
			// A running loop's stop is intercepted in Prompt; reaching here means none runs.
			r.say(ctx, "No /spec loop is running. Where it stands:\n\n")
		}
		// Reconcile as a run would, so status names the work it picks up first.
		delta := specReconcile(cfg, r.idx, covered)
		r.say(ctx, renderSpecStatus(cfg, r.idx, testFiles, r.testCmd())+"\n"+renderSpecDelta(delta))
		r.save(ctx)
		return end, nil
	}
	if !r.preflight(ctx) {
		return end, nil
	}

	sess.setSpecFence(filepath.Join(sess.Cwd, cfg.SpecDir))
	defer sess.setSpecFence("")
	sess.takeSpecStop() // a request left over from an earlier loop is not this one's

	// The pre-turn checks run every round: dedupe so an unfixed problem gets one card.
	var fixes []fixProblem
	addFixes := func(fs []fixProblem) {
		for _, f := range fs {
			if !slices.ContainsFunc(fixes, func(g fixProblem) bool { return g.desc == f.desc }) {
				fixes = append(fixes, f)
			}
		}
	}
	addFixes(pendingFixes)
	stopped := func(err error) (PromptResponse, error) {
		r.save(context.Background())
		msg := "⏹ Spec loop stopped. `/spec` resumes it where the ledger says it is.\n"
		if !errors.Is(err, errUserCancelled) {
			msg = "⏹ Spec loop cancelled (" + cancelReason(err) + "). `/spec` resumes it where the ledger says it is.\n"
		}
		a.say(context.Background(), sid, msg)
		return PromptResponse{StopReason: "cancelled"}, nil
	}

	maxBlocked := cfg.MaxBlocked
	if maxBlocked <= 0 {
		maxBlocked = defaultSpecMaxBlocked
	}
	for consecutive, round := 0, 1; ; round++ {
		if err := ctx.Err(); err != nil {
			return stopped(err)
		}
		// Re-read every round: the user may edit the spec or the code in between.
		if err := r.scan(); err != nil {
			r.say(ctx, "⚠ /spec: reading the spec: "+err.Error()+"\n")
			break
		}
		covered, testFiles, err := specCoverage(r.outAbs, r.idx.order)
		if err != nil {
			r.say(ctx, "⚠ /spec: scanning "+cfg.OutDir+": "+err.Error()+"\n")
			break
		}
		// The last round has committed and is in the ledger: the clean place to honour a stop.
		if sess.takeSpecStop() {
			r.say(ctx, fmt.Sprintf("\n⏹ **/spec stopped** as asked, after %d round(s). Nothing is half done: every finished item is committed and in the ledger. `/spec` resumes with the next item.\n\n", round-1))
			r.say(ctx, renderSpecStatus(cfg, r.idx, testFiles, r.testCmd())+"\n")
			r.save(ctx)
			break
		}
		w := r.pickWork(ctx, covered, testFiles)
		if w.Item == "" {
			if r.vanished {
				break // the change report said why; "finished" would vouch for a spec that is not there
			}
			if !r.completionChecked {
				r.completionChecked = true
				reopened, err := r.completionCheck(ctx)
				if err != nil {
					return stopped(err)
				}
				if len(reopened) > 0 {
					r.say(ctx, fmt.Sprintf("\n🔍 %d item(s) do not do all their spec says yet and are open again: %s\n\n", len(reopened), strings.Join(reopened, ", ")))
					continue
				}
			}
			if f := cfg.Final; f == nil || f.Items != len(cfg.Items) || f.Blocked != r.notDone() {
				if err := r.finalPass(ctx); err != nil {
					return stopped(err)
				}
			}
			r.say(ctx, fmt.Sprintf("\n✅ **/spec finished**: every item in `%s/` is covered by a passing test, blocked in this run, or waiting on a question.", cfg.SpecDir))
			if open := specOpenQuestions(r.idx); len(open) > 0 {
				r.say(ctx, fmt.Sprintf(" %d question(s) wait on your answer in `%s/%s`; answer them there and run /spec.", len(open), cfg.SpecDir, specQuestionsFile))
			}
			if len(r.blocked) > 0 {
				r.say(ctx, fmt.Sprintf(" %d item(s) blocked in this run; the next /spec tries them again.", len(r.blocked)))
			}
			if cfg.Final != nil {
				r.say(ctx, fmt.Sprintf(" `%s/README.md` says how to build, run and test it.", cfg.OutDir))
			}
			r.say(ctx, "\n\n")
			break
		}
		addFixes(a.prepareChecks(ctx, sess, sid))

		w.LintMissing = r.lintMissing
		prompt, head := a.specRoundPrompt(sid, cfg, r.idx, w, r.testCmd(), r.lintCmd())
		r.say(ctx, fmt.Sprintf("\n## /spec round %d · %s\n\n", round, head))
		since, started := len(sess.Messages), time.Now() // the round's own tool calls start here
		turnErr := a.runPromptTurn(ctx, sess, prompt)
		if isCancelled(turnErr) {
			return stopped(turnErr)
		}
		// The checks run once before the round is judged, so what they find costs a
		// turn, not an attempt: one uncalled function once blocked a two-round item.
		if turnErr == nil && (w.Mode == specModeItem || w.Mode == specModeChange) {
			var uses []ToolUse
			for i := since; i < len(sess.Messages); i++ {
				uses = append(uses, sess.Messages[i].ToolUses...)
			}
			res := specRoundResult{Mode: w.Mode, Redo: w.Redo, Committed: true}
			if _, _, err := r.inspect(ctx, w, uses, started, false, &res); err == nil {
				if done, _, findings, _ := specDecide(&specConfig{}, w.Item, res, ""); !done {
					r.say(ctx, "🔎 Before this round counts: "+firstLine(findings)+"\n")
					turnErr = a.runPromptTurn(ctx, sess, a.renderSpecPrompt(sid, "SPEC-CHECK.md", []string{
						"{{id}}", w.Item, "{{findings}}", findings, "{{out_dir}}", cfg.OutDir, "{{test_cmd}}", r.testCmd()}))
					if isCancelled(turnErr) {
						return stopped(turnErr)
					}
				}
			}
		}
		// The model call failed, not the item: nothing counts, and the next round would
		// meet the same server. Three items became questions about an image limit.
		var lce *llmCallError
		if errors.As(turnErr, &lce) {
			r.save(ctx)
			r.say(ctx, fmt.Sprintf("\n⏹ **/spec paused**: the LLM request failed, so this round does not count against %s and nothing was recorded: %s\nWhen the server answers again, `/spec` resumes with the same item.\n", w.Item, firstLine(lce.Error())))
			break
		}
		done, block, err := r.finishRound(ctx, w, turnErr, since, started)
		if isCancelled(err) {
			return stopped(err)
		}
		if err != nil {
			r.save(ctx) // keeps what reconcile adopted and renamed
			break       // finishRound said why
		}
		switch {
		case done:
			consecutive = 0
		case block:
			consecutive++
		}
		if block && w.Mode == specModeSetup {
			r.say(ctx, "The project setup could not be finished, and every item depends on it. Fix what the block says, then run /spec again.\n")
			break
		}
		if consecutive >= maxBlocked {
			r.say(ctx, fmt.Sprintf("Stopping: %d items in a row did not pass their attempts, which usually means one shared problem (the build, the test command, a missing toolchain). Each has a question in `%s/%s` with what stopped it; `/spec status` lists them.\n", consecutive, cfg.SpecDir, specQuestionsFile))
			break
		}
	}

	a.drainSteer(ctx, sess)
	a.drainFixes(ctx, sid, fixes)
	return end, nil
}

func (a *agent) specResolve(ctx context.Context, sid string, sess *Session, args string) (cfg *specConfig, cmd string, ok bool) {
	say := func(s string) { a.say(ctx, sid, s) }
	cmd, err := parseSpecArgs(args)
	if err != nil {
		say(err.Error() + "\n")
		return nil, "", false
	}
	if cfg, err = loadSpecConfig(sess.Cwd); err != nil {
		say("⚠ " + err.Error() + "\n")
		return nil, "", false
	}
	if cfg != nil {
		return cfg, cmd, true
	}
	specArg, outArg, target, ok := a.specSetupDialog(ctx, sid, sess)
	if !ok {
		return nil, "", false
	}
	specRel, outRel, err := specPaths(sess.Cwd, specArg, outArg)
	if err != nil {
		say("⚠ /spec: " + err.Error() + "\n")
		return nil, "", false
	}
	if cfg == nil || cfg.SpecDir != specRel {
		cfg = &specConfig{} // a different spec: its attempts and blocks don't carry over
	}
	cfg.SpecDir, cfg.OutDir, cfg.Target = specRel, outRel, target
	if err := os.MkdirAll(filepath.Join(sess.Cwd, outRel), 0o755); err != nil {
		say("⚠ /spec: creating " + outRel + ": " + err.Error() + "\n")
		return nil, "", false
	}
	if err := saveSpecConfig(sess.Cwd, cfg); err != nil {
		say("⚠ /spec: " + err.Error() + "\n")
		return nil, "", false
	}
	line := fmt.Sprintf("**/spec** `%s/` → `%s/`", cfg.SpecDir, cfg.OutDir)
	if cfg.Target != "" {
		line += " · target: " + cfg.Target
	}
	say(line + "\n\n")
	return cfg, cmd, true
}

func (r *specRun) preflight(ctx context.Context) bool {
	cfg := r.cfg
	var problems []string
	markers := r.idx.openMarkers(cfg.openMarkers(), cfg.SpecDir)
	if len(markers) > 0 && !cfg.AcceptOpenMarkers {
		shown := markers
		if len(shown) > 10 {
			shown = append(shown[:10:10], fmt.Sprintf("and %d more", len(markers)-10))
		}
		problems = append(problems, fmt.Sprintf("The spec still has %d open-decision marker(s) (%s): %s. Every round would settle those by itself.",
			len(markers), strings.Join(cfg.openMarkers(), "/"), strings.Join(shown, ", ")))
	}
	if ign := specIgnoredProbes(ctx, r.sess.Cwd, cfg.OutDir, r.idx); len(ign) > 0 {
		problems = append(problems, fmt.Sprintf("git would ignore files the rewrite creates, so the per-item commits would silently leave them out: %s. Anchor those rules to where the old code writes (a leading `/`, e.g. `/cut/`), or scope them to its directory.",
			strings.Join(ign, "; ")))
	}
	if len(problems) == 0 {
		return true
	}
	r.say(ctx, "**/spec pre-flight**\n\n- "+strings.Join(problems, "\n- ")+"\n\n")
	if r.a.isAutopilot() {
		r.say(ctx, "Autopilot does not start over these. Resolve them, or set `accept_open_markers = true` in `.codehalter/spec.toml` to take the open points as written, then run /spec again.\n")
		return false
	}
	ok, tcId, err := r.a.askYesNoWithCard(ctx, r.sid, "Start the spec loop anyway?", "think", "Start anyway", "Stop")
	if err != nil {
		r.a.FailToolCall(ctx, r.sid, tcId, err.Error())
		return false
	}
	r.a.CompleteToolCall(ctx, r.sid, tcId, []ToolCallContent{TextContent(map[bool]string{true: "Starting", false: "Stopped"}[ok])})
	if ok && len(markers) > 0 {
		cfg.AcceptOpenMarkers = true
		r.save(ctx)
	}
	return ok
}

// pickWork returns an empty Item when nothing is left. Deletions go first so no later round
// builds on code the spec dropped.
func (r *specRun) pickWork(ctx context.Context, covered map[string]string, testFiles int) specWork {
	cfg := r.cfg
	delta := specReconcile(cfg, r.idx, covered)
	if rep := renderSpecDelta(delta); rep != "" && rep != r.lastDelta {
		r.say(ctx, rep)
		r.lastDelta = rep
	}
	r.vanished = delta.Vanished

	skip := func(id string) bool { return r.blocked[id] != "" || r.idx.asked(id) }
	firstOpen := func(ids []string) string {
		for _, id := range ids {
			if !skip(id) {
				return id
			}
		}
		return ""
	}
	removed := ""
	if !delta.Vanished {
		removed = firstOpen(delta.Removed)
	}
	changed := firstOpen(delta.Changed)

	var w specWork
	switch {
	case r.testCmd() == "" || testFiles == 0:
		w = specWork{Item: specSetupID, Mode: specModeSetup}
	case removed != "":
		w = specWork{Item: removed, Mode: specModeRemove}
		w.Note = specRemovedText(ctx, r.sess.Cwd, cfg, w.Item)
	case changed != "":
		w = specWork{Item: changed, Mode: specModeChange}
		w.Note = specSpecDiff(ctx, r.sess.Cwd, cfg.Items[w.Item].Commit, cfg.SpecDir+"/"+r.idx.docs[r.idx.items[w.Item].Doc].rel)
	default:
		// Every refactor_every finished items one round goes to the code's shape instead of an item.
		if every := cfg.refactorEvery(); every > 0 && len(cfg.Items)-cfg.RefactorAt >= every {
			if d := measureSpecDebt(r.outAbs, cfg); d.overLines+d.dead+d.copies > 0 {
				w = specWork{Item: specRefactorID, Mode: specModeRefactor, Debt: d}
				break
			}
			cfg.RefactorAt = len(cfg.Items) // nothing to clean up
			delete(cfg.Attempts, specRefactorID)
		}
		w.Item = nextSpecItem(r.idx, cfg, skip)
	}
	w.Answered = r.idx.answered(w.Item)
	w.Redo = cfg.Redo[w.Item] != ""
	w.Reason = r.reasons[w.Item]
	// The differences stay in the reason on every attempt; a failed attempt adds why it failed.
	if found, ok := strings.CutPrefix(cfg.Redo[w.Item], specCheckMark); ok {
		w.Reason = strings.TrimSpace("The completion check found that the program does not do all this item's spec text says yet, though its tests pass. Build what is missing, in the program, not in a test; where the spec pictures a screen, look at yours next to the picture. What is missing:\n\n" + found + "\n\n" + w.Reason)
	} else if w.Reason == "" && w.Redo {
		w.Reason = specRedoReason
	}
	return w
}

// specBaseVars: noTarget stands in for an empty target; contextLead/contextTail wrap the rules list.
func specBaseVars(cfg *specConfig, idx *specIndex, testCmd, noTarget, contextLead, contextTail string) []string {
	target := cfg.Target
	if target == "" {
		target = noTarget
	}
	context := ""
	if names := cfg.context(idx); len(names) > 0 {
		// A new slice: cfg.context may hand back cfg.Context, which is saved.
		files := make([]string, len(names))
		for i, f := range names {
			files[i] = "`" + cfg.SpecDir + "/" + f + "`"
		}
		context = contextLead + strings.Join(files, ", ") + contextTail
	}
	var docs []string
	for _, d := range idx.docs {
		docs = append(docs, "`"+cfg.SpecDir+"/"+d.rel+"`")
	}
	return []string{"{{spec_dir}}", cfg.SpecDir, "{{out_dir}}", cfg.OutDir, "{{target}}", target,
		"{{test_cmd}}", testCmd, "{{context}}", context, "{{files}}", strings.Join(docs, ", ")}
}

const specNoTargetBuilt = "No technology was given with /spec; the project in the output directory is what it is."

func (a *agent) renderSpecPrompt(sid, file string, vars []string) string {
	s := strings.NewReplacer(vars...).Replace(a.loadPromptFile(sid, file))
	// Empty optional placeholders leave blank-line runs.
	for strings.Contains(s, "\n\n\n") {
		s = strings.ReplaceAll(s, "\n\n\n", "\n\n")
	}
	return s
}

// specFinalPrompt: blocked is the run's blocked items with why; the open questions come from idx.
func (a *agent) specFinalPrompt(sid string, cfg *specConfig, idx *specIndex, testCmd string, blocked map[string]string) string {
	var lines []string
	for _, id := range idx.order {
		if why := blocked[id]; why != "" {
			lines = append(lines, "- "+id+": "+why)
		}
	}
	for _, q := range specOpenQuestions(idx) {
		lines = append(lines, "- "+q.ID+": waits on the user's answer to \""+q.Question+"\" in "+cfg.SpecDir+"/"+specQuestionsFile)
	}
	list := "none"
	if len(lines) > 0 {
		list = strings.Join(lines, "\n")
	}
	return a.renderSpecPrompt(sid, "SPEC-FINAL.md", append(specBaseVars(cfg, idx, testCmd, specNoTargetBuilt, "Standing rules: ", "."),
		"{{items}}", strconv.Itoa(len(cfg.Items)), "{{blocked_count}}", strconv.Itoa(len(lines)), "{{blocked}}", list))
}

// notDone counts what the final pass lists as not done, so it runs again when that moves.
func (r *specRun) notDone() int { return len(r.blocked) + len(specOpenQuestions(r.idx)) }

// roundRanGreen trusts the terminal's exit code, not the model, so a wrapped run
// (`just test > log; echo exit=$?`) does not count: its exit code is the echo's.
// The last run counts, in the round's calls or as a background job that exited
// since the round began; an edit inside the output directory after it undoes it.
func (r *specRun) roundRanGreen(uses []ToolUse, cmd string, since time.Time) bool {
	type event struct {
		at    time.Time
		green bool
		since time.Time // nothing may be written after this: a job ran on the code as it started
	}
	var evs []event
	for _, u := range uses {
		args := parseArgs(u.Input)
		switch {
		case u.StartedAt.IsZero():
		case u.Name == "edit_file" || u.Name == "write_file":
			p := args.str("path")
			if p != "" && !filepath.IsAbs(p) {
				p = filepath.Join(r.sess.Cwd, p)
			}
			if p != "" && realInside(p, r.outAbs) {
				evs = append(evs, event{at: u.StartedAt})
			}
		// A handed-over run reports its exit as a job run.
		case u.Name == "run_command" && strings.HasPrefix(u.Output, "exit ") && testRunMatches(args.str("command"), cmd, r.sess.Cwd, r.outAbs):
			end := u.StartedAt.Add(time.Duration(u.DurationMs) * time.Millisecond)
			evs = append(evs, event{at: end, green: strings.HasPrefix(u.Output, "exit 0\n"), since: end})
		}
	}
	r.sess.rt.mu.Lock()
	for _, j := range r.sess.rt.jobRuns {
		if j.ended.After(since) && testRunMatches(j.cmd, cmd, r.sess.Cwd, r.outAbs) {
			evs = append(evs, event{at: j.ended, green: j.code == 0, since: j.started})
		}
	}
	r.sess.rt.mu.Unlock()
	slices.SortStableFunc(evs, func(x, y event) int { return x.at.Compare(y.at) })
	if len(evs) == 0 || !evs[len(evs)-1].green {
		return false
	}
	return !writtenSince(r.outAbs, evs[len(evs)-1].since)
}

// testRunMatches: the terminal's cwd is the project root, so a run without `cd <out> &&`
// only matches when the output directory is the root.
func testRunMatches(line, cmd, cwd, outAbs string) bool {
	line = strings.TrimSpace(line)
	dir := cwd
	if strings.HasPrefix(line, "cd ") {
		rest := strings.TrimSpace(line[3:])
		i := strings.Index(rest, "&&")
		if i < 0 {
			return false
		}
		d := strings.Trim(strings.TrimSpace(rest[:i]), "'\"")
		if !filepath.IsAbs(d) {
			d = filepath.Join(cwd, d)
		}
		dir = d
		line = strings.TrimSpace(rest[i+2:])
	}
	if filepath.Clean(dir) != filepath.Clean(outAbs) {
		return false
	}
	if !strings.HasPrefix(line, cmd) {
		return false
	}
	tail := strings.TrimSpace(line[len(cmd):])
	// Only redirections may follow: `> f`, `>> f`, `2>&1`, `2> f`.
	for _, tok := range strings.Fields(tail) {
		switch {
		case tok == "2>&1", strings.HasPrefix(tok, ">"), strings.HasPrefix(tok, "2>"):
		case strings.ContainsAny(tok, ";&|(`$"):
			return false
		default:
			// a redirection's file name, which must have followed a `>`
			if !strings.Contains(tail, ">") {
				return false
			}
		}
	}
	return true
}

func writtenSince(root string, t time.Time) bool {
	found := false
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || found {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if path != root && (skipWalkDir(name) || name == "shots") {
				return filepath.SkipDir
			}
			return nil
		}
		if info, err := d.Info(); err == nil && info.ModTime().After(t) {
			found = true
		}
		return nil
	})
	return found
}

// uiSourceExt keeps a README that mentions libadwaita from counting as a screen.
var uiSourceExt = map[string]bool{".rs": true, ".py": true, ".go": true, ".c": true, ".cc": true, ".cpp": true, ".h": true, ".hpp": true, ".ts": true, ".tsx": true, ".js": true, ".jsx": true, ".vue": true, ".svelte": true, ".swift": true, ".kt": true, ".ui": true, ".blp": true, ".qml": true, ".slint": true}

var uiMarkers = []string{"gtk::", "adw::", "use gtk", "use adw", "gtk4::", "libadwaita", "QtWidgets", "QWidget", "#include <Q", "from PyQt", "from PySide", "import tkinter", "egui::", "iced::", "slint::", "fltk::"}

// uiEditedUnseen ignores test code: a widget test imports the toolkit too.
func uiEditedUnseen(uses []ToolUse, cwd string) []string {
	seen := map[string]bool{}
	var files []string
	for _, u := range uses {
		if u.ImageID != "" {
			return nil // a screenshot, or a render codehalter attached (attachRenderedScreen)
		}
		switch u.Name {
		case "screenshot":
			return nil
		case "edit_file", "write_file":
			path := parseArgs(u.Input).str("path")
			if path == "" || seen[path] || !uiSourceExt[strings.ToLower(filepath.Ext(path))] {
				continue
			}
			seen[path] = true
			abs := path
			if !filepath.IsAbs(abs) {
				abs = filepath.Join(cwd, abs)
			}
			data, err := os.ReadFile(abs)
			if err != nil {
				continue
			}
			text := string(data)
			text = strings.TrimSuffix(text, testSourceText(strings.TrimPrefix(path, cwd+string(filepath.Separator)), text))
			for _, m := range uiMarkers {
				if strings.Contains(text, m) {
					files = append(files, path)
					break
				}
			}
		}
	}
	return files
}

type specFile struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// writeSpecFiles never overwrites: the user may already have edited that spec.
func writeSpecFiles(cwd, specDir string, files []specFile) ([]string, error) {
	dir := filepath.Clean(filepath.Join(cwd, specDir))
	var written []string
	for _, f := range files {
		rel := filepath.Clean(strings.TrimPrefix(f.Path, specDir+"/"))
		if rel == "." || rel == "" || filepath.IsAbs(rel) || strings.HasPrefix(rel, "..") {
			return written, fmt.Errorf("spec file path %q is not inside %s/", f.Path, specDir)
		}
		abs := filepath.Join(dir, rel)
		if _, err := os.Stat(abs); err == nil {
			return written, fmt.Errorf("%s exists already; the planner writes a spec only into empty space", filepath.Join(specDir, rel))
		}
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			return written, err
		}
		if err := os.WriteFile(abs, []byte(strings.TrimRight(f.Content, "\n")+"\n"), 0o644); err != nil {
			return written, err
		}
		written = append(written, filepath.ToSlash(filepath.Join(specDir, rel)))
	}
	return written, nil
}

// specFromPlan asks the user first: a spec the model wrote is its reading of the request,
// and dozens of rounds should not be tested against it unread.
func (a *agent) specFromPlan(ctx context.Context, sid string, sess *Session, p *planResult) (toolLoopResult, error) {
	if sess.specFence() != "" {
		a.say(ctx, sid, "⚠ The planner handed this request to /spec, but a /spec loop is already running: nothing was reopened or written.\n")
		return toolLoopResult{}, nil
	}
	cfg, err := loadSpecConfig(sess.Cwd)
	if err != nil {
		a.say(ctx, sid, "⚠ "+err.Error()+"\n")
		return toolLoopResult{}, nil
	}
	if len(p.Redo) > 0 {
		if cfg == nil {
			a.say(ctx, sid, "⚠ The planner named spec items to redo, but this project has no /spec ledger. Run /spec first.\n")
			return toolLoopResult{}, nil
		}
		ids := strings.Join(p.Redo, " ")
		a.say(ctx, sid, fmt.Sprintf("The planner reads this request as %d item(s) of the spec that are recorded as done but do not deliver: %s\n\n", len(p.Redo), strings.Join(p.Redo, ", ")))
		sess.rt.mu.Lock()
		audit := sess.rt.specAudit
		sess.rt.mu.Unlock()
		if audit {
			// A bare /spec redo asked for exactly this: no second question.
			sess.setSpecHandoff("redo " + ids)
			return toolLoopResult{Text: "handed over to /spec"}, nil
		}
		ok, tcId, err := a.askYesNoWithCard(ctx, sid, fmt.Sprintf("Reopen %d spec item(s) and rebuild them with /spec, one per round?", len(p.Redo)), "think", "Run /spec", "Not now")
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return toolLoopResult{}, nil
		}
		if !ok {
			a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Nothing reopened")})
			a.say(ctx, sid, "Nothing reopened. `/spec redo` finds and rebuilds them later.\n")
			return toolLoopResult{Text: "not now"}, nil
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Handing over to /spec")})
		sess.setSpecHandoff("redo " + ids)
		return toolLoopResult{Text: "handed over to /spec"}, nil
	}
	if cfg != nil {
		a.say(ctx, sid, fmt.Sprintf("⚠ The planner wrote a new spec, but this project already has one in `%s/`. A request this size maps onto that spec's items (`redo`), it does not get a second spec. Nothing was written.\n", cfg.SpecDir))
		return toolLoopResult{}, nil
	}
	specDir := strings.Trim(filepath.ToSlash(filepath.Clean(p.SpecDir)), "/")
	if specDir == "" || specDir == "." {
		specDir = "spec"
	}
	written, err := writeSpecFiles(sess.Cwd, specDir, p.Spec)
	if err != nil {
		a.say(ctx, sid, "⚠ writing the spec: "+err.Error()+"\n")
		return toolLoopResult{}, nil
	}
	idx, err := scanSpec(filepath.Join(sess.Cwd, specDir), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		a.say(ctx, sid, "⚠ reading the spec back: "+err.Error()+"\n")
		return toolLoopResult{}, nil
	}
	a.say(ctx, sid, fmt.Sprintf("This request is more than one round of work, so the planner wrote it down as a spec: %s, %d item(s). Read it; it is the model's reading of what you asked, and the loop will build and test exactly that.\n\n", strings.Join(written, ", "), len(idx.order)))
	if p.OutDir != "" {
		specRel, outRel, err := specPaths(sess.Cwd, specDir, p.OutDir)
		if err == nil {
			err = saveSpecConfig(sess.Cwd, &specConfig{SpecDir: specRel, OutDir: outRel, Target: p.Target})
		}
		if err != nil {
			a.say(ctx, sid, "⚠ /spec: "+err.Error()+"\n")
		}
	}
	ok, tcId, err := a.askYesNoWithCard(ctx, sid, fmt.Sprintf("Run /spec on %s now (%d items, one per round)?", specDir+"/", len(idx.order)), "think", "Run /spec", "Let me read it first")
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return toolLoopResult{}, nil
	}
	if !ok {
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Spec written, loop not started")})
		a.say(ctx, sid, "The spec is in `"+specDir+"/`. Edit it as you like, then `/spec` builds it.\n")
		return toolLoopResult{Text: "spec written"}, nil
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Handing over to /spec")})
	sess.setSpecHandoff("resume")
	return toolLoopResult{Text: "handed over to /spec"}, nil
}

const specRedoReason = "The user sent this item back with /spec redo. It was implemented and its test passes, but what was built does not do what the spec says: a screen without its widgets, a button without its wire, a flow that cannot be reached from the UI. Rebuild it against the spec text below under the rules above. Keep and extend the existing code and tests where they are right. Where AGENT.md describes the old state as the design, correct that line, in a line; do not add notes about what this round built."

func (a *agent) specAuditPrompt(sid string, cfg *specConfig, idx *specIndex, testCmd string) string {
	// Every id verbatim: named from memory, the model invents them.
	var ids strings.Builder
	for d, doc := range idx.docs {
		var in []string
		for _, id := range idx.order {
			if idx.items[id].Doc == d {
				in = append(in, "`"+id+"`")
			}
		}
		if len(in) > 0 {
			fmt.Fprintf(&ids, "- `%s`: %s\n", doc.rel, strings.Join(in, ", "))
		}
	}
	return a.renderSpecPrompt(sid, "SPEC-REDO.md", append(specBaseVars(cfg, idx, testCmd, specNoTargetBuilt, "Standing rules: ", "."),
		"{{ids}}", strings.TrimRight(ids.String(), "\n")))
}

func (r *specRun) finalPass(ctx context.Context) error {
	cfg := r.cfg
	prompt := r.a.specFinalPrompt(r.sid, cfg, r.idx, r.testCmd(), r.blocked)
	r.say(ctx, "\n## /spec final pass · build, run and write the README\n\n")
	turnErr := r.a.runPromptTurn(ctx, r.sess, prompt)
	if isCancelled(turnErr) {
		return turnErr
	}
	if turnErr != nil {
		r.say(ctx, "⚠ final pass: the round ended with an error: "+firstLine(turnErr.Error())+". It runs again on the next /spec.\n")
		return nil
	}
	pass, tail := r.a.runSpecTests(ctx, r.sid, r.outAbs, cfg.OutDir, r.testCmd())
	if err := ctx.Err(); err != nil {
		return err
	}
	if !pass {
		r.say(ctx, "⚠ final pass: the test command did not pass afterwards, so it is not recorded and runs again on the next /spec. The end of its output:\n"+tail+"\n")
		return nil
	}
	sha := r.a.specCommit(ctx, r.sid, r.sess.Cwd, cfg, r.idx, "final pass", specModeItem)
	cfg.Final = &specFinal{Items: len(cfg.Items), Blocked: r.notDone(), Commit: sha, At: time.Now().UTC(), Version: versionStamp()}
	r.save(ctx)
	r.say(ctx, "✅ final pass done.\n")
	return nil
}

// finishRound's err means cancelled, or a scan failure it already reported.
func (r *specRun) finishRound(ctx context.Context, w specWork, turnErr error, since int, started time.Time) (done, block bool, err error) {
	cfg, item := r.cfg, w.Item
	var uses []ToolUse
	for i := since; i < len(r.sess.Messages); i++ {
		uses = append(uses, r.sess.Messages[i].ToolUses...)
	}
	res := specRoundResult{Mode: w.Mode, Redo: w.Redo}
	var covered map[string]string
	var coveredBy string
	var q *specQuestionError
	if errors.As(turnErr, &q) {
		res.Question = q.Q.Question
	} else {
		if turnErr != nil {
			res.TurnErr = turnErr.Error()
		}
		if covered, coveredBy, err = r.inspect(ctx, w, uses, started, true, &res); err != nil {
			return false, false, err
		}
	}

	// Commit before deciding: whether anything changed is the verdict for a change round.
	sha := ""
	if res.TurnErr == "" {
		sha = r.a.specCommit(ctx, r.sid, r.sess.Cwd, cfg, r.idx, item, w.Mode)
		res.Committed = sha != ""
	}
	done, block, reason, fails := specDecide(cfg, item, res, r.fails[item])
	if r.fails == nil {
		r.fails = map[string]string{}
	}
	r.fails[item] = fails
	if done || block {
		delete(cfg.Bases, item)
		delete(r.fails, item)
	}
	switch {
	case block && w.Mode == specModeRefactor:
		// A refactor is not an item: it never blocks the loop, it waits for the next interval.
		cfg.RefactorAt = len(cfg.Items)
		delete(cfg.Attempts, item)
		delete(r.reasons, item)
		block = false
		r.say(ctx, fmt.Sprintf("↷ refactor round skipped: %s. The next one comes after %d more items.\n", firstLine(reason), cfg.refactorEvery()))
	case done:
		delete(cfg.Attempts, item)
		delete(cfg.Redo, item)
		delete(r.reasons, item)
		switch w.Mode {
		case specModeRemove:
			delete(cfg.Items, item)
			r.say(ctx, fmt.Sprintf("🗑 %s removed: the spec no longer has it.\n", item))
		case specModeSetup:
			r.say(ctx, "✅ project setup is done.\n")
		case specModeRefactor:
			cfg.RefactorAt = len(cfg.Items)
			r.say(ctx, "✅ refactor done: "+res.Debt+"\n")
		default:
			if coveredBy == "" {
				coveredBy = covered[item]
			}
			cfg.Items[item] = specLedger{
				Hash:      specItemHash(r.idx, item),
				Title:     r.idx.items[item].Title,
				File:      r.idx.docs[r.idx.items[item].Doc].rel,
				CoveredBy: coveredBy,
				Commit:    sha,
				At:        time.Now().UTC(),
				Version:   versionStamp(),
				Named:     true,
			}
			r.say(ctx, fmt.Sprintf("✅ %s is covered.\n", item))
		}
	case block && q != nil && q.Problem == "":
		// Not a block: the item waits in QUESTIONS.md, where the answer becomes spec.
		// Questions in a row are no shared failure, so they do not stop the loop.
		block = w.Mode == specModeSetup
		delete(cfg.Attempts, item)
		delete(r.reasons, item)
		q.Q.ID = item
		title := item
		if it := r.idx.items[item]; it != nil && it.Title != "" {
			title = it.Title
		}
		if err := appendSpecQuestion(filepath.Join(r.sess.Cwd, cfg.SpecDir), title, q.Q); err != nil {
			r.blocked[item] = "the planner asked a question, and writing it to " + specQuestionsFile + " failed: " + err.Error()
			r.say(ctx, fmt.Sprintf("⛔ %s blocked for this run: %s\nThe question: %s\n", item, r.blocked[item], q.Q.Question))
			break
		}
		r.say(ctx, fmt.Sprintf("❓ %s waits on your answer: %s\nIt is in `%s/%s` with the spec text it rests on and the options. Write the answer there, then run /spec.\n", item, q.Q.Question, cfg.SpecDir, specQuestionsFile))
	case block && w.Mode != specModeSetup:
		// A stuck item becomes a question too: whether its cause is gone, or what
		// gets past it, is the user's to say, and the item waits in the spec.
		if q != nil {
			reason = "the planner needed a decision the spec does not make, but could not ask it answerably (" + q.Problem + "). Its question: " + q.Q.Question
		}
		delete(cfg.Attempts, item)
		delete(r.reasons, item)
		title := item
		if it := r.idx.items[item]; it != nil && it.Title != "" {
			title = it.Title
		} else if led, ok := cfg.Items[item]; ok && led.Title != "" {
			title = led.Title
		}
		if err := appendSpecQuestion(filepath.Join(r.sess.Cwd, cfg.SpecDir), title, specStuckQuestion(r.idx, cfg.SpecDir, item, reason)); err != nil {
			r.blocked[item] = firstLine(reason)
			r.say(ctx, fmt.Sprintf("⛔ %s blocked for this run: %s\nWriting its question to %s failed (%v); the next /spec tries it again.\n", item, firstLine(reason), specQuestionsFile, err))
			break
		}
		r.say(ctx, fmt.Sprintf("❓ %s did not pass its attempts: %s\nHow it goes on is a question in `%s/%s`, with what stopped it and the options. It waits there; answer it and run /spec.\n", item, firstLine(reason), cfg.SpecDir, specQuestionsFile))
	case block:
		r.blocked[item] = firstLine(reason)
		delete(cfg.Attempts, item)
		delete(r.reasons, item)
		r.say(ctx, fmt.Sprintf("⛔ %s blocked for this run: %s\nThe next /spec tries it again.\n", item, firstLine(reason)))
	default:
		r.reasons[item] = reason
		r.say(ctx, fmt.Sprintf("↻ %s is not done yet, one more round: %s\n", item, firstLine(reason)))
	}
	r.save(ctx)
	return done, block, nil
}

// inspect runs the checks that decide a round into res: the suite (unless
// runTests is off, for the in-round check), the UI look, the test naming this
// item in the round's own files, and the gates on what the round wrote.
func (r *specRun) inspect(ctx context.Context, w specWork, uses []ToolUse, started time.Time, runTests bool, res *specRoundResult) (covered map[string]string, coveredBy string, err error) {
	cfg, item := r.cfg, w.Item
	switch cmd := r.testCmd(); {
	case !runTests:
		res.TestsPass = true // not run yet: the other checks still say what to fix
	case cmd == "":
		res.TestTail = "no test command found: `" + cfg.OutDir + "/` has no justfile with a test recipe, no Cargo.toml, package.json or go.mod"
	case r.roundRanGreen(uses, cmd, started):
		res.TestsPass = true
		r.say(ctx, fmt.Sprintf("🧪 `%s` passed in the round, after its last change; not run again\n", cmd))
	default:
		res.TestsPass, res.TestTail = r.a.runSpecTests(ctx, r.sid, r.outAbs, cfg.OutDir, cmd)
		if err := ctx.Err(); err != nil {
			return nil, "", err
		}
	}
	// A snapshot recipe makes looking possible; only then is an unseen UI change blind.
	if w.Mode != specModeRemove && justRecipe(r.outAbs, "snapshot") {
		res.UIUnseen = uiEditedUnseen(uses, r.sess.Cwd)
	}
	covered, testFiles, err := specCoverage(r.outAbs, r.idx.order)
	if err != nil {
		r.say(ctx, "⚠ /spec: scanning "+cfg.OutDir+": "+err.Error()+"\n")
		return nil, "", err
	}
	base := cfg.Bases[item]
	if base == "" || cfg.Attempts[item] == 0 {
		base = "HEAD"
		if sha, err := specGit(ctx, r.sess.Cwd, "rev-parse", "HEAD"); err == nil {
			base = strings.TrimSpace(sha)
			if cfg.Bases == nil {
				cfg.Bases = map[string]string{}
			}
			cfg.Bases[item] = base
		}
	}
	ch := specRoundChanges(ctx, r.sess.Cwd, cfg.OutDir, base)
	switch {
	case w.Mode == specModeSetup:
		res.Covered = testFiles > 0
	case w.Mode == specModeRemove || !ch.ok:
		_, res.Covered = covered[item]
	default:
		// An older test that happens to carry the token proves nothing about this round.
		coveredBy = specNamedIn(r.outAbs, item, ch.files())
		res.Covered = coveredBy != ""
	}
	if ch.ok && res.TurnErr == "" {
		r.gates(ctx, w, ch, res)
	}
	return covered, coveredBy, nil
}

func (a *agent) specRoundPrompt(sid string, cfg *specConfig, idx *specIndex, w specWork, testCmd, lintCmd string) (prompt, head string) {
	item, mode, note := w.Item, w.Mode, w.Note
	base := specBaseVars(cfg, idx, testCmd,
		"No technology was given with /spec. Use what the spec implies; where it leaves the choice open, pick a mainstream option and state it in the project README.",
		"Standing rules for every item: ", ". Read them if they are not already in this conversation.")
	lint := "none is set up for this project, so this one is free"
	if lintCmd != "" {
		lint = fmt.Sprintf("`%s`, run from `%s/`", lintCmd, cfg.OutDir)
	}
	if lintCmd != "" && w.LintMissing != "" {
		lint += fmt.Sprintf(". It does not run in this container yet (%s): install it first, the way SKILL-base.md installs tools, and persist the install in `.devcontainer/Dockerfile`", w.LintMissing)
	}
	maxLines := strconv.Itoa(cfg.maxFileLines())
	if cfg.maxFileLines() < 0 {
		maxLines = "any number of" // the budget is off
	}
	base = append(base, "{{lint_cmd}}", lint, "{{max_lines}}", maxLines, "{{growth_slack}}", strconv.Itoa(specFileGrowthSlack))
	previous := ""
	switch {
	case w.Answered != "" && w.Reason != "":
		previous = "## Answered in the spec's " + specQuestionsFile + "\n\n" + w.Answered + "\n\n## The previous round on this item did not count\n\n" + w.Reason + "\n\nFix that first."
	case w.Answered != "":
		previous = "## Answered in the spec's " + specQuestionsFile + "\n\nThe user answered this item's question. The answer is spec: build to it.\n\n" + w.Answered
	case w.Reason != "":
		previous = "## The previous round on this item did not count\n\n" + w.Reason + "\n\nFix that first."
	}

	if mode == specModeRemove {
		led := cfg.Items[item]
		title := led.Title
		if title == "" {
			title = item
		}
		coveredBy := led.CoveredBy
		if coveredBy == "" {
			coveredBy = "not recorded; search the output directory for the id"
		}
		old := note
		if old == "" {
			old = "(the old text is not recoverable from git; go by the id and the test that names it)"
		}
		return a.renderSpecPrompt(sid, "SPEC-REMOVE.md", append(base,
			"{{id}}", item, "{{title}}", title, "{{covered_by}}", coveredBy, "{{old_text}}", old,
			"{{previous}}", previous,
		)), "remove " + item + " (gone from the spec)"
	}

	if mode == specModeSetup {
		return a.renderSpecPrompt(sid, "SPEC-SETUP.md", append(base,
			"{{previous}}", previous, "{{items}}", fmt.Sprintf("%d requirement items", len(idx.order)),
		)), "project setup in `" + cfg.OutDir + "/`"
	}

	if mode == specModeRefactor {
		return a.renderSpecPrompt(sid, "SPEC-REFACTOR.md", append(base,
			"{{previous}}", previous, "{{debt}}", w.Debt.String()+".", "{{targets}}", w.Debt.targets,
		)), "refactor `" + cfg.OutDir + "/` · " + w.Debt.String()
	}

	it := idx.items[item]
	sl := idx.slice(item, cfg.SpecDir)
	title := strings.TrimSpace(strings.TrimPrefix(it.Title, item))
	if title == "" {
		title = item
	}
	k, _ := specTallies(cfg, idx)
	done := 0
	for _, t := range k {
		done += t.done
	}
	progress := fmt.Sprintf("%d of %d items covered (flows %d/%d, sections %d/%d, parameters and tools %d/%d)",
		done, len(idx.order), k[specDefHeading].done, k[specDefHeading].total, k[specDefSection].done, k[specDefSection].total, k[specDefTableRow].done, k[specDefTableRow].total)
	screens := ""
	if len(sl.Images) > 0 {
		screens = "Screens this text shows (look at them with `screenshot path=...`): " + strings.Join(sl.Images, ", ") + "\n"
	}
	related := ""
	if len(sl.Related) > 0 {
		related = "Also relevant, read if you need it: " + strings.Join(sl.Related, ", ") + "\n"
	}
	if note != "" {
		change := "## This item was implemented before, and the spec has changed since\n\n" +
			"The code and its test match the OLD text. Update both to the text below; where the diff removes something, remove it from the code too.\n\n```diff\n" +
			note + "\n```"
		if previous != "" {
			change += "\n\n" + previous
		}
		previous = change
	}
	var rows []string
	for _, id := range idx.idsIn(sl.Text) {
		if cited := idx.items[id]; cited != nil && cited.Kind == specDefTableRow && id != item && !cfg.done(id) {
			rows = append(rows, fmt.Sprintf("`%s` → `%s`", id, specTestToken(id)))
		}
	}
	rowTokens := "none open"
	if len(rows) > 0 {
		rowTokens = strings.Join(rows, ", ")
	}
	return a.renderSpecPrompt(sid, "SPEC.md", append(base,
		"{{id}}", item, "{{title}}", title, "{{file}}", cfg.SpecDir+"/"+idx.docs[it.Doc].rel,
		"{{progress}}", progress, "{{token}}", specTestToken(item),
		"{{slice}}", strings.TrimRight(sl.Text, "\n"), "{{screens}}", screens, "{{related}}", related,
		"{{previous}}", previous, "{{row_tokens}}", rowTokens,
	)), fmt.Sprintf("%s %s · %s", item, title, progress)
}

// gates checks what the round wrote, beyond its tests: functions only tests
// reach, lint findings in its own lines, files grown past the size budget, and
// for a refactor whether the debt shrank.
func (r *specRun) gates(ctx context.Context, w specWork, ch specChanges, res *specRoundResult) {
	if w.Mode == specModeItem || w.Mode == specModeChange {
		res.Unreachable = specUnreachable(r.outAbs, ch, loadSpecProgram(r.outAbs))
		res.StandIns = specStandIns(r.outAbs, ch)
	}
	res.Oversize = specOversize(r.outAbs, r.cfg, ch)
	if cmd := r.lintCmd(); cmd != "" && res.TestsPass {
		r.say(ctx, fmt.Sprintf("🔎 linting: `%s` in `%s/`\n", cmd, r.cfg.OutDir))
		findings, missing := specLint(ctx, r.outAbs, cmd, ch)
		if missing != "" && r.lintMissing == "" {
			r.say(ctx, fmt.Sprintf("⚠ /spec: `%s` does not run here (%s); the next round installs it. `lint_cmd` in .codehalter/spec.toml sets another linter, `lint_cmd = \"off\"` none.\n", cmd, missing))
		}
		r.lintMissing = missing
		switch {
		case len(findings) > 0:
			r.say(ctx, fmt.Sprintf("🔎 %d lint finding(s) in lines this round wrote\n", len(findings)))
		}
		res.Lint = findings
	}
	if w.Mode == specModeRefactor {
		after := measureSpecDebt(r.outAbs, r.cfg)
		res.Improved = after.better(w.Debt)
		res.Debt = w.Debt.String() + " → " + after.String()
	}
}

// specCheckVerdictRe reads one line of the completion check's answer: "F1.2: DONE",
// "- `F1.2` - MISSING: the Save button writes nothing" (a dash of any width, or a colon).
var specCheckVerdictRe = regexp.MustCompile("(?i)^[\\s*>-]*`?([^\\s:*`]+)`?\\s*[:\u2014\u2013-]\\s*\\**(DONE|MISSING)\\b\\**[\\s:\u2014\u2013.-]*(.*)$")

// specCheckMark starts a Redo entry the completion check wrote; the rest is what is missing.
const specCheckMark = "completion check: "

// specCheckBatch is how many items of one spec file a completion round checks.
const specCheckBatch = 8

// completionCheck has every done item that was not checked since it was built
// looked at against its spec text: the program, not a test, must do what the text
// says, and a screen the spec pictures must look like the picture. A test named
// after the item and a green suite said neither: naivepost counted its LLM calls
// done while they only ever answered "no llm server here", and its Prepare screen
// with the panels swapped. The rounds go by spec file, a few items each; what is
// missing reopens the item with the list, what is done is marked checked in the
// ledger, so the next run looks only at what was built since.
func (r *specRun) completionCheck(ctx context.Context) (reopened []string, err error) {
	cfg, idx := r.cfg, r.idx
	type batch struct {
		file string
		ids  []string
	}
	var batches []batch
	total := 0
	for _, id := range idx.order {
		// A parameter or tool row is checked through the flows that cite it.
		if it := idx.items[id]; !cfg.done(id) || cfg.Items[id].Checked != "" || it.Kind == specDefTableRow {
			continue
		}
		file := cfg.SpecDir + "/" + idx.docs[idx.items[id].Doc].rel
		if n := len(batches); n == 0 || batches[n-1].file != file || len(batches[n-1].ids) == specCheckBatch {
			batches = append(batches, batch{file: file})
		}
		batches[len(batches)-1].ids = append(batches[len(batches)-1].ids, id)
		total++
	}
	if total == 0 {
		return nil, nil
	}
	r.say(ctx, fmt.Sprintf("\n🔍 Every item is done, blocked in this run or waiting on a question. Checking the %d done item(s) not checked yet against the spec, in %d round(s).\n", total, len(batches)))
	base := specBaseVars(cfg, idx, r.testCmd(), specNoTargetBuilt, "", "")
	for i, b := range batches {
		var list []string
		for _, id := range b.ids {
			it := idx.items[id]
			line := fmt.Sprintf("- `%s` %s (`%s:%d`)", id, strings.TrimSpace(strings.TrimPrefix(it.Title, id)), b.file, it.Line+1)
			if pics := specOwnPictures(idx.slice(id, cfg.SpecDir)); len(pics) > 0 {
				line += "; the spec pictures it: " + strings.Join(pics, ", ")
			}
			list = append(list, line)
		}
		r.say(ctx, fmt.Sprintf("\n## /spec completion check %d of %d · `%s` · %s\n\n", i+1, len(batches), b.file, strings.Join(b.ids, ", ")))
		since := len(r.sess.Messages)
		turnErr := r.a.runPromptTurn(ctx, r.sess, r.a.renderSpecPrompt(r.sid, "SPEC-COMPLETE.md", append(base,
			"{{file}}", b.file, "{{items}}", strings.Join(list, "\n"))))
		if isCancelled(turnErr) {
			return reopened, turnErr
		}
		if turnErr != nil {
			r.say(ctx, fmt.Sprintf("⚠ The completion check of %s failed: %v\n", strings.Join(b.ids, ", "), turnErr))
		}
		// The last verdict per item counts: the executor's respond, or the planner's
		// answer when it looked and answered directly; a closing summary may follow.
		verdicts, found := map[string]string{}, map[string]string{}
		for _, m := range r.sess.Messages[since:] {
			for _, u := range m.ToolUses {
				text := u.Output
				if u.Name == submitPlanToolName {
					text = parseArgs(u.Input).str("answer")
				} else if u.Name != respondToolName {
					continue
				}
				for _, ln := range strings.Split(text, "\n") {
					if v := specCheckVerdictRe.FindStringSubmatch(strings.TrimSpace(ln)); v != nil && slices.Contains(b.ids, v[1]) {
						verdicts[v[1]], found[v[1]] = strings.ToUpper(v[2]), strings.TrimSpace(v[3])
					}
				}
			}
		}
		for _, id := range b.ids {
			switch {
			case verdicts[id] == "MISSING" && found[id] != "":
				cfg.reopen([]string{id})
				cfg.Redo[id] = specCheckMark + found[id]
				reopened = append(reopened, id)
				r.say(ctx, "↩ "+id+" is not done: "+found[id]+"\n")
			case verdicts[id] == "DONE":
				led := cfg.Items[id]
				led.Checked = time.Now().UTC().Format("2006-01-02") + " " + versionStamp()
				cfg.Items[id] = led
				r.say(ctx, "✅ "+id+" does what the spec says.\n")
			default:
				r.say(ctx, "⚠ "+id+": the check gave no verdict; it stays done and is checked on the next run.\n")
			}
		}
		r.save(ctx)
	}
	return reopened, nil
}

// specOwnPictures: the screenshots in an item's own section. A linked section's
// picture is another item's, and an .svg is a drawing (a proposed state, a diagram).
func specOwnPictures(sl specSlice) []string {
	own, _, _ := strings.Cut(sl.Text, "\n\n--- ")
	var pics []string
	for _, img := range sl.Images {
		if ext := strings.ToLower(filepath.Ext(img)); (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp") && strings.Contains(own, filepath.Base(img)) {
			pics = append(pics, "`"+img+"`")
		}
	}
	return pics
}

// runSpecTests is codehalter's own check, so it runs directly, not in a visible terminal.
func (a *agent) runSpecTests(ctx context.Context, sid, outAbs, outRel, cmd string) (bool, string) {
	a.say(ctx, sid, fmt.Sprintf("🧪 checking: `%s` in `%s/`\n", cmd, outRel))
	tctx, cancel := context.WithTimeout(ctx, specTestTimeout)
	defer cancel()
	c := exec.CommandContext(tctx, specShell(), "-lc", cmd)
	c.Dir = outAbs
	start := time.Now()
	out, err := c.CombinedOutput()
	tail := string(out)
	if len(tail) > specTestTailBytes {
		tail = "…" + tailUTF8(tail, specTestTailBytes)
	}
	took := humanDuration(time.Since(start).Milliseconds())
	if err != nil {
		// Always append the exit status: a command that dies before printing leaves an empty tail.
		switch {
		case tctx.Err() == context.DeadlineExceeded:
			tail += fmt.Sprintf("\n[codehalter stopped the test command after %s]", specTestTimeout)
		case strings.TrimSpace(tail) == "":
			tail = fmt.Sprintf("[the command printed nothing and ended with: %v]", err)
		default:
			tail += fmt.Sprintf("\n[%v]", err)
		}
		a.say(ctx, sid, fmt.Sprintf("🧪 tests failed after %s\n", took))
		return false, tail
	}
	a.say(ctx, sid, fmt.Sprintf("🧪 tests passed in %s\n", took))
	return true, tail
}

// specShell runs with -lc: a login shell finds toolchains on the profile PATH
// (rustup's ~/.cargo/bin). Alpine has no bash.
func specShell() string {
	if _, err := exec.LookPath("bash"); err == nil {
		return "bash"
	}
	return "sh"
}

func specGit(ctx context.Context, cwd string, args ...string) (string, error) {
	out, err := exec.CommandContext(ctx, "git", append([]string{"-C", cwd}, args...)...).CombinedOutput()
	return string(out), err
}

// specCommit stages only the output dir and .devcontainer so a user's unrelated change does not
// ride along. "" means nothing to commit, which is how a change round detects no code change.
func (a *agent) specCommit(ctx context.Context, sid, cwd string, cfg *specConfig, idx *specIndex, item string, mode specMode) string {
	git := func(args ...string) (string, error) { return specGit(ctx, cwd, args...) }
	if _, err := git("rev-parse", "--is-inside-work-tree"); err != nil {
		return ""
	}
	paths := []string{cfg.OutDir}
	if fileExists(cwd, ".devcontainer") {
		paths = append(paths, ".devcontainer")
	}
	if out, err := git(append([]string{"add", "-A", "--"}, paths...)...); err != nil {
		a.say(ctx, sid, "⚠ /spec: git add failed: "+firstLine(out)+"\n")
		return ""
	}
	if _, err := git("diff", "--cached", "--quiet"); err == nil {
		return "" // nothing staged
	}
	msg := "spec: " + item
	switch {
	case item == specSetupID:
		msg = "spec: set up " + cfg.OutDir + "/"
	case mode == specModeRefactor:
		msg = "spec: refactor " + cfg.OutDir + "/"
	case mode == specModeRemove:
		msg = "spec: remove " + item
	case idx.items[item] != nil && idx.items[item].Title != "":
		msg = "spec: " + item + " " + strings.TrimPrefix(idx.items[item].Title, item+" ")
	}
	if mode == specModeChange {
		msg = strings.Replace(msg, "spec: ", "spec: update ", 1)
	}
	if out, err := git("commit", "-q", "-m", msg); err != nil {
		a.say(ctx, sid, "⚠ /spec: git commit failed, the work stays uncommitted: "+firstLine(out)+"\n")
		return ""
	}
	a.say(ctx, sid, "📌 committed: "+msg+"\n")
	sha, err := git("rev-parse", "HEAD")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(sha)
}

const specSuggestTimeout = 2 * time.Minute

// suggestSpecTarget asks the model first: a page says things a keyword scan cannot.
func (a *agent) suggestSpecTarget(ctx context.Context, sid, entry string) (outDir, target string) {
	if entry == "" {
		return "", ""
	}
	if conn, _ := a.connForBackgroundLLM(); conn != nil {
		askCtx, cancel := context.WithTimeout(ctx, specSuggestTimeout)
		defer cancel()
		prompt := "Below is the entry page of a software specification that is about to be implemented from scratch.\n\n" +
			"Answer in exactly two lines, nothing else:\n" +
			"out_dir=<one directory name for the new implementation>\n" +
			"target=<the language and toolkit the spec asks for, a few words>\n\n" +
			"Write out_dir=unknown and target=unknown if the page does not say.\n\n---\n" +
			clipBytes(entry, maxLLMInputBytes)
		out, _, _, err := a.llmStream(askCtx, sid, conn.withThinkingDisabled(),
			[]llmMessage{{Role: "user", Content: prompt}}, nil, nil, nil, nil)
		if err != nil {
			slog.Debug("spec setup: the suggestion call failed", "err", err)
		}
		for _, line := range strings.Split(out, "\n") {
			line = strings.TrimSpace(strings.Trim(strings.TrimSpace(line), "`*"))
			v, ok := strings.CutPrefix(line, "out_dir=")
			if ok {
				outDir = strings.Trim(strings.TrimSpace(v), "`\"/")
			}
			if v, ok := strings.CutPrefix(line, "target="); ok {
				target = strings.Trim(strings.TrimSpace(v), "`\"")
			}
		}
		if strings.EqualFold(outDir, "unknown") {
			outDir = ""
		}
		if strings.EqualFold(target, "unknown") {
			target = ""
		}
	}
	if outDir == "" || target == "" {
		gOut, gTarget := specGuessTarget(entry)
		if outDir == "" {
			outDir = gOut
		}
		if target == "" {
			target = gTarget
		}
	}
	return outDir, target
}

// specSetupDialog uses three cards, not one form: each question's options depend on the previous answer.
func (a *agent) specSetupDialog(ctx context.Context, sid string, sess *Session) (specDir, outDir, target string, ok bool) {
	say := func(s string) { a.say(ctx, sid, s) }
	cands := specDirCandidates(sess.Cwd)
	if len(cands) == 0 {
		say("No specification found: `/spec` looks for a directory holding at least two markdown files. Put the spec in one and run `/spec` again.\n")
		return "", "", "", false
	}
	if len(cands) > 3 {
		cands = cands[:3]
	}
	tcId := a.StartToolCall(ctx, sid, "Which directory holds the specification?", "think", nil)
	answer, err := a.askFormAuto(ctx, sid, tcId, "Which directory holds the specification? Pick one, or type another path.", cands, true)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return "", "", "", false
	}
	specDir = strings.TrimSpace(answer)
	if specDir == "" {
		specDir = cands[0]
	}
	specDir = filepath.Clean(strings.Trim(specDir, "`\""))
	if st, err := os.Stat(filepath.Join(sess.Cwd, specDir)); err != nil || !st.IsDir() {
		a.FailToolCall(ctx, sid, tcId, specDir+" is not a directory in this project")
		say("⚠ /spec: `" + specDir + "` is not a directory in this project.\n")
		return "", "", "", false
	}
	entryRel, entry := specEntryPage(filepath.Join(sess.Cwd, specDir))
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Spec: " + specDir + "/, entry page " + entryRel)})
	say("Reading `" + specDir + "/" + entryRel + "` for what it asks to be built…\n")
	gOut, gTarget := a.suggestSpecTarget(ctx, sid, entry)

	outOpts := specOutDirOptions(sess.Cwd, specDir, gOut)
	var described []string
	for _, d := range outOpts {
		desc := "new"
		if lang, deps := manifestStack(filepath.Join(sess.Cwd, d)); lang != "" {
			desc = "existing " + lang
			if len(deps) > 0 {
				desc += " project: " + strings.Join(deps, ", ")
			}
		}
		described = append(described, "`"+d+"/` ("+desc+")")
	}
	q := "Where should the implementation go? Pick a directory, or type one."
	if len(described) > 0 {
		q += " " + strings.Join(described, "; ") + "."
	}
	tcId2 := a.StartToolCall(ctx, sid, "Where to build?", "think", nil)
	answer2, err := a.askFormAuto(ctx, sid, tcId2, q, outOpts, true)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId2, err.Error())
		return "", "", "", false
	}
	outDir = filepath.Clean(strings.Trim(strings.TrimSpace(answer2), "`\"/ "))
	if outDir == "" || outDir == "." {
		a.FailToolCall(ctx, sid, tcId2, "no directory given")
		say("⚠ /spec: nothing to build into. Run `/spec` again when you know where it should go.\n")
		return "", "", "", false
	}
	a.CompleteToolCall(ctx, sid, tcId2, []ToolCallContent{TextContent("Building into " + outDir + "/")})

	targetOpts := specTargetOptions(sess.Cwd, outDir, gTarget, entry)
	tcId3 := a.StartToolCall(ctx, sid, "Build it with what?", "think", nil)
	answer3, err := a.askFormAuto(ctx, sid, tcId3, "Build `"+outDir+"/` with what? Pick one, or type the language and toolkit.", targetOpts, true)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId3, err.Error())
		return "", "", "", false
	}
	target = strings.Trim(strings.TrimSpace(answer3), "`\"")
	done := "Target: " + target
	if target == "" {
		done = "No target stated"
	}
	a.CompleteToolCall(ctx, sid, tcId3, []ToolCallContent{TextContent(done)})
	return specDir, outDir, target, true
}

func specRemovedText(ctx context.Context, cwd string, cfg *specConfig, id string) string {
	led := cfg.Items[id]
	if led.Commit == "" || led.File == "" {
		return ""
	}
	out, err := specGit(ctx, cwd, "show", led.Commit+":"+cfg.SpecDir+"/"+led.File)
	if err != nil {
		return ""
	}
	return clipBytes(specSectionFromText(out, id), specSpecDiffBytes)
}

func specSpecDiff(ctx context.Context, cwd, commit, relPath string) string {
	if commit == "" {
		return ""
	}
	out, err := specGit(ctx, cwd, "diff", commit+"..HEAD", "--", relPath)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(clipBytes(out, specSpecDiffBytes))
}
