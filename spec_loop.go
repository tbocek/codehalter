package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// /spec drives the implementation of a specification as a loop of rounds, one
// item per round, until every requirement id in the spec is named by a passing
// test. spec.go decides what is covered and what comes next; this file runs the
// rounds through the ordinary turn pipeline (runTurn: plan, execute, verify,
// document) and checks each one afterwards.
//
//	/spec <spec-dir> <out-dir> [technology prompt]   set up, then loop
//	/spec                                            resume the loop
//	/spec status                                     coverage report, no turn
//
// The whole loop is one Prompt: Zed's Cancel stops it between or inside a
// round, and the next /spec resumes from the recomputed ledger. Each item that
// passes is committed, so nothing depends on the session surviving.

// specTestTimeout bounds one run of the project's test command. A GUI project's
// first test build compiles its whole dependency tree (gtk4-rs is minutes on a
// cold cache); a hung test must still not stall the loop forever.
const specTestTimeout = 30 * time.Minute

// specTestTailBytes is how much test output a failed round hands to its retry.
// The end of a test run is where the failures and the summary are.
const specTestTailBytes = 4000

// specSpecDiffBytes caps the spec diff a change round is shown. An edit to one
// section is small; a cap keeps a reorganised file from crowding out the
// section text itself.
const specSpecDiffBytes = 4000

// specQuestionError is how a planner question travels up out of a round under
// autopilot. Answering it with the first option, as autopilot does elsewhere,
// would let the model settle an open point of the spec by itself; instead the
// item is blocked with the question recorded, and the loop moves on.
type specQuestionError struct {
	Question string
	Choices  []string
}

func (e *specQuestionError) Error() string { return "the planner needs an answer: " + e.Question }

// setSpecFence marks dir (absolute) read-only for write_file/edit_file while a
// spec loop runs in this session; "" lifts it.
func (s *Session) setSpecFence(dir string) {
	s.rt.mu.Lock()
	s.rt.specFenceDir = dir
	s.rt.mu.Unlock()
}

// specFence returns the fenced spec dir, or "" when no loop is running.
func (s *Session) specFence() string {
	s.rt.mu.Lock()
	defer s.rt.mu.Unlock()
	return s.rt.specFenceDir
}

// specFenceRefusal is the tool-level guard on the spec: while a loop runs, the
// file tools refuse to write under the spec dir. Editing the spec is the one way
// a round could make an item "done" without implementing it (delete the id and
// it is no longer missing), so this is a refusal in code, not a line in a
// prompt. run_command can still reach the files; the file tools are what a
// model uses to edit, and the refusal tells it why.
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

// specRoundResult is what a finished round looked like to the checker.
// specMode is what a round is for. The item rounds (plain and change) share
// the done rule — a test names the item and the suite passes — while setup and
// removal each need their own, so the mode travels with the result.
type specMode int

const (
	specModeItem   specMode = iota // implement an item the spec added
	specModeSetup                  // the one-off project skeleton round
	specModeChange                 // redo an item whose spec text moved
	specModeRemove                 // delete an item the spec no longer has
)

type specRoundResult struct {
	Mode      specMode
	Committed bool   // the round left something to commit
	Question  string // the planner asked instead of working (autopilot)
	TurnErr   string // the round itself failed
	TestsPass bool
	TestTail  string
	Covered   bool // the item's id is named by a test source (setup: a test source exists)
}

// specDecide turns a round's result into the item's fate: done, one more round,
// or blocked. It counts the attempt in cfg.Attempts, so a /spec cancelled and
// resumed does not hand an item a fresh budget.
func specDecide(cfg *specConfig, item string, r specRoundResult) (done, block bool, reason string) {
	if r.Question != "" {
		return false, true, "the planner asked a question instead of guessing"
	}
	switch {
	case r.Mode == specModeRemove:
		// Inverted: the item is done when NO test names it any more and the
		// rest of the suite still passes.
		if r.TurnErr == "" && r.TestsPass && !r.Covered {
			return true, false, ""
		}
	case r.Mode == specModeChange:
		// A changed item is still covered by the test written for the OLD text,
		// so coverage alone would call it done before the model touched
		// anything. The commit is the evidence that it did.
		if r.TurnErr == "" && r.Covered && r.TestsPass && r.Committed {
			return true, false, ""
		}
	default:
		if r.TurnErr == "" && r.Covered && r.TestsPass {
			return true, false, ""
		}
	}
	var why []string
	if r.TurnErr != "" {
		why = append(why, "the round ended with an error: "+r.TurnErr)
	}
	if r.Mode == specModeChange && !r.Committed && r.TurnErr == "" {
		why = append(why, "the spec text changed but no code did: update the implementation AND its test to match the new text")
	}
	if r.Mode == specModeRemove && r.Covered {
		why = append(why, fmt.Sprintf("a test in the output directory still names %s", item))
	}
	if !r.Covered && r.Mode != specModeRemove {
		if r.Mode == specModeSetup {
			why = append(why, "no test source exists in the output directory yet")
		} else {
			why = append(why, fmt.Sprintf("no test in the output directory names %s (a test name containing `%s`, or `%s` in a comment or string inside a test)", item, specTestToken(item), item))
		}
	}
	if !r.TestsPass {
		why = append(why, "the test command did not pass. The end of its output:\n\n```\n"+strings.TrimSpace(r.TestTail)+"\n```")
	}
	if len(why) == 0 {
		why = append(why, "the round did not finish the item")
	}
	if cfg.Attempts == nil {
		cfg.Attempts = map[string]int{}
	}
	cfg.Attempts[item]++
	reason = strings.Join(why, "; ")
	if cfg.Attempts[item] >= specMaxAttempts {
		return false, true, reason
	}
	return false, false, reason
}

// specPaths validates the two directories of a /spec setup and returns them
// relative to the project root.
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

// specIgnoredProbes reports the git ignore rules that would hide files the
// rewrite creates under the output directory, one line per rule with an
// example path. A rewrite commits its manifests, sources and tests there, and
// an unanchored rule left over from the old code matches at any depth: `*.json`
// hides a manifest, `test*` a tests/ directory, and a rule named after a part
// of the app (`cut/`, for the prototype's cut/ output folder) hides the
// rewrite's module of the same name. Git skips such files without a word, so
// every per-item commit would be partial and a fresh clone would not build.
//
// Module names are guessed from the spec's chapter files, because a rewrite
// that follows the spec names its modules after them ("05-cut.md" → cut/).
func specIgnoredProbes(ctx context.Context, cwd, outRel string, idx *specIndex) []string {
	if err := exec.CommandContext(ctx, "git", "-C", cwd, "rev-parse", "--is-inside-work-tree").Run(); err != nil {
		return nil
	}
	probes := []string{"Cargo.toml", "package.json", "src/lib.rs", "src/main.rs", "tests/smoke.rs", "tests/fixture.json", "test_data/x", "fixtures/x.json"}
	for _, name := range specModuleNames(idx) {
		probes = append(probes, "src/"+name+"/mod.rs", name+"/x")
	}
	var in strings.Builder
	for _, p := range probes {
		in.WriteString(filepath.ToSlash(filepath.Join(outRel, p)))
		in.WriteByte('\n')
	}
	// One call for every probe. -v names the rule; --stdin reads the paths.
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

// specModuleNames derives likely module names from the spec's root chapter
// files: the stem without its number prefix, in snake_case ("05-cut.md" →
// "cut", "09-llm-and-tools.md" → "llm_and_tools"), plus the stem's first word
// ("llm"), which is what a module is often called instead.
func specModuleNames(idx *specIndex) []string {
	var names []string
	add := func(n string) {
		if n != "" && !slices.Contains(names, n) {
			names = append(names, n)
		}
	}
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
		add(strings.Join(words, "_"))
		add(words[0])
	}
	return names
}

// runSpec is the /spec command: status, or setup/resume followed by the loop.
func (a *agent) runSpec(ctx context.Context, sid string, sess *Session, args string, pendingFixes []fixProblem) (PromptResponse, error) {
	end := PromptResponse{StopReason: "end_turn"}
	say := func(s string) { a.say(ctx, sid, s) }

	cmd, specArg, outArg, target, err := parseSpecArgs(args)
	if err != nil {
		say(err.Error() + "\n")
		return end, nil
	}
	cfg, err := loadSpecConfig(sess.Cwd)
	if err != nil {
		say("⚠ " + err.Error() + "\n")
		return end, nil
	}
	// applySetup records where the spec is, what is built from it and where,
	// whether that came from the command line or from the setup dialog.
	applySetup := func(specArg, outArg, target string) bool {
		specRel, outRel, err := specPaths(sess.Cwd, specArg, outArg)
		if err != nil {
			say("⚠ /spec: " + err.Error() + "\n")
			return false
		}
		if cfg == nil || cfg.SpecDir != specRel {
			cfg = &specConfig{} // a different spec: its attempts and blocks don't carry over
		}
		cfg.SpecDir, cfg.OutDir, cfg.Target = specRel, outRel, target
		if err := os.MkdirAll(filepath.Join(sess.Cwd, outRel), 0o755); err != nil {
			say("⚠ /spec: creating " + outRel + ": " + err.Error() + "\n")
			return false
		}
		if err := saveSpecConfig(sess.Cwd, cfg); err != nil {
			say("⚠ /spec: " + err.Error() + "\n")
			return false
		}
		say(fmt.Sprintf("**/spec** `%s/` → `%s/`%s\n\n", cfg.SpecDir, cfg.OutDir,
			map[bool]string{true: "", false: " · target: " + cfg.Target}[cfg.Target == ""]))
		return true
	}
	switch cmd {
	case "setup":
		if !applySetup(specArg, outArg, target) {
			return end, nil
		}
	default:
		if cfg == nil {
			// First /spec in this project: ask rather than print a usage line.
			specDir, outDir, tgt, ok := a.specSetupDialog(ctx, sid, sess)
			if !ok || !applySetup(specDir, outDir, tgt) {
				return end, nil
			}
		}
	}

	specAbs := filepath.Join(sess.Cwd, cfg.SpecDir)
	outAbs := filepath.Join(sess.Cwd, cfg.OutDir)
	scan := func() (*specIndex, error) {
		var context []string
		if len(cfg.Context) > 0 {
			context = cfg.Context
		}
		return scanSpec(specAbs, cfg.idPatterns(), context, cfg.Skip)
	}
	idx, err := scan()
	if err != nil {
		say("⚠ /spec: reading the spec: " + err.Error() + "\n")
		return end, nil
	}
	if len(idx.order) == 0 {
		say(fmt.Sprintf("⚠ /spec: found no requirement ids and no sections in `%s/`. The id patterns are %s; set `id_patterns` in .codehalter/spec.toml if this spec names its requirements differently.\n", cfg.SpecDir, strings.Join(cfg.idPatterns(), ", ")))
		return end, nil
	}
	testCmd := func() string {
		if cfg.TestCmd != "" {
			return cfg.TestCmd
		}
		return detectSpecTestCmd(outAbs)
	}

	if cmd == "status" {
		covered, testFiles, err := specCoverage(outAbs, idx.order)
		if err != nil {
			say("⚠ /spec: scanning " + cfg.OutDir + ": " + err.Error() + "\n")
			return end, nil
		}
		// The same comparison a run would do, so `status` reports the work the
		// next `/spec` would pick up before anything else.
		delta := specReconcile(cfg, idx, covered)
		say(renderSpecStatus(cfg, idx, covered, testFiles, testCmd()) + "\n" + renderSpecDelta(delta))
		if err := saveSpecConfig(sess.Cwd, cfg); err != nil {
			say("⚠ /spec: " + err.Error() + "\n")
		}
		return end, nil
	}

	// Pre-flight: what would make an unattended run go wrong from the start.
	var problems []string
	markers := idx.openMarkers(cfg.openMarkers(), cfg.SpecDir)
	if len(markers) > 0 && !cfg.AcceptOpenMarkers {
		shown := markers
		if len(shown) > 10 {
			shown = append(shown[:10:10], fmt.Sprintf("and %d more", len(markers)-10))
		}
		problems = append(problems, fmt.Sprintf("The spec still has %d open-decision marker(s) (%s): %s. Every round would settle those by itself.",
			len(markers), strings.Join(cfg.openMarkers(), "/"), strings.Join(shown, ", ")))
	}
	if ign := specIgnoredProbes(ctx, sess.Cwd, cfg.OutDir, idx); len(ign) > 0 {
		problems = append(problems, fmt.Sprintf("git would ignore files the rewrite creates, so the per-item commits would silently leave them out: %s. Anchor those rules to where the old code writes (a leading `/`, e.g. `/cut/`), or scope them to its directory.",
			strings.Join(ign, "; ")))
	}
	if len(problems) > 0 {
		say("**/spec pre-flight**\n\n- " + strings.Join(problems, "\n- ") + "\n\n")
		if a.isAutopilot() {
			say("Autopilot does not start over these. Resolve them, or set `accept_open_markers = true` in `.codehalter/spec.toml` to take the open points as written, then run /spec again.\n")
			return end, nil
		}
		ok, tcId, err := a.askYesNoWithCard(ctx, sid, "Start the spec loop anyway?", "think", "Start anyway", "Stop")
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return end, nil
		}
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(map[bool]string{true: "Starting", false: "Stopped"}[ok])})
		if !ok {
			return end, nil
		}
		if len(markers) > 0 {
			cfg.AcceptOpenMarkers = true
			if err := saveSpecConfig(sess.Cwd, cfg); err != nil {
				say("⚠ /spec: " + err.Error() + "\n")
			}
		}
	}

	sess.setSpecFence(specAbs)
	defer sess.setSpecFence("")

	reasons := map[string]string{} // why the last round on an item did not count, for its retry
	consecutive, round := 0, 0
	// Removals are confirmed once per id, then worked one round each. Declining
	// forgets the item instead: it is gone from the spec and the user chose to
	// keep its code, so there is nothing left to track or to ask about again.
	removeOK := map[string]bool{}
	lastDelta := ""
	fixes := pendingFixes
	stopped := func(err error) (PromptResponse, error) {
		if serr := saveSpecConfig(sess.Cwd, cfg); serr != nil {
			a.say(context.Background(), sid, "⚠ /spec: "+serr.Error()+"\n")
		}
		msg := "⏹ Spec loop stopped. `/spec` resumes it where the ledger says it is.\n"
		if !errors.Is(err, errUserCancelled) && !sess.superseded() {
			msg = "⏹ Spec loop cancelled (" + cancelReason(err) + "). `/spec` resumes it where the ledger says it is.\n"
		}
		if !sess.superseded() {
			a.say(context.Background(), sid, msg)
		}
		return PromptResponse{StopReason: "cancelled"}, nil
	}

	for {
		if err := ctx.Err(); err != nil {
			return stopped(err)
		}
		// Re-read everything each round: the user may edit the spec or the code
		// between rounds, and coverage is never trusted from memory.
		if idx, err = scan(); err != nil {
			say("⚠ /spec: reading the spec: " + err.Error() + "\n")
			break
		}
		covered, testFiles, err := specCoverage(outAbs, idx.order)
		if err != nil {
			say("⚠ /spec: scanning " + cfg.OutDir + ": " + err.Error() + "\n")
			break
		}
		// What the spec says now, against what the ledger says was built. This
		// runs every round because the user may edit the spec while it runs.
		delta := specReconcile(cfg, idx, covered)
		if rep := renderSpecDelta(delta); rep != "" && rep != lastDelta {
			say(rep)
			lastDelta = rep
		}
		var removals []string
		for _, id := range delta.Removed {
			ok, asked := removeOK[id]
			if !asked {
				ok = a.askSpecRemoval(ctx, sid, id)
				removeOK[id] = ok
				if !ok {
					delete(cfg.Items, id)
					if err := saveSpecConfig(sess.Cwd, cfg); err != nil {
						say("⚠ /spec: " + err.Error() + "\n")
					}
				}
			}
			if ok {
				removals = append(removals, id)
			}
		}

		mode := specModeItem
		setup := testCmd() == "" || testFiles == 0
		item, answer := specSetupID, ""
		switch {
		case setup:
			mode = specModeSetup
		case len(removals) > 0:
			// Deletions first: they keep the suite honest, and an item that is
			// gone must not be worked on by a later round.
			item, mode = removals[0], specModeRemove
		case len(delta.Changed) > 0:
			// Then items whose spec text moved: the code claims to implement
			// something the spec no longer says.
			item, mode = delta.Changed[0], specModeChange
		default:
			item, answer = nextSpecItem(idx, covered, cfg)
		}
		if item == "" {
			say(fmt.Sprintf("\n✅ **/spec finished**: every item in `%s/` is covered by a passing test, or blocked.\n\n", cfg.SpecDir))
			break
		}
		question := ""
		if answer != "" {
			if b := cfg.block(item); b != nil {
				question = b.Question
			}
			cfg.unblock(item)
			delete(cfg.Attempts, item)
		}

		// The pre-turn checks a typed prompt gets: settings reload, skills for a
		// stack the last round introduced, MCP reconcile.
		fixes = append(fixes, a.prepareChecks(ctx, sess, sid)...)

		// A change round is shown the spec edit itself; a removal round the text
		// that was deleted. Both come out of git, which has the old spec.
		note := ""
		switch mode {
		case specModeChange:
			note = specSpecDiff(ctx, sess.Cwd, cfg.Items[item].Commit, cfg.SpecDir+"/"+idx.docs[idx.items[item].Doc].rel)
		case specModeRemove:
			note = specRemovedText(ctx, sess.Cwd, cfg, item)
		}

		round++
		prompt, head := a.specRoundPrompt(sid, cfg, idx, item, mode, covered, testCmd(), reasons[item], question, answer, note)
		say(fmt.Sprintf("\n## /spec round %d · %s\n\n", round, head))
		turnErr := a.runPromptTurn(ctx, sess, prompt)
		if isCancelled(turnErr) {
			return stopped(turnErr)
		}
		res := specRoundResult{Mode: mode}
		var q *specQuestionError
		if errors.As(turnErr, &q) {
			res.Question = q.Question
		} else {
			if turnErr != nil {
				res.TurnErr = turnErr.Error()
			}
			if cmd := testCmd(); cmd == "" {
				res.TestTail = "no test command found: `" + cfg.OutDir + "/` has no justfile with a test recipe, no Cargo.toml, package.json or go.mod"
			} else {
				res.TestsPass, res.TestTail = a.runSpecTests(ctx, sid, outAbs, cfg.OutDir, cmd)
				if err := ctx.Err(); err != nil {
					return stopped(err)
				}
			}
			covered, testFiles, _ = specCoverage(outAbs, idx.order)
			if mode == specModeSetup {
				res.Covered = testFiles > 0
			} else {
				_, res.Covered = covered[item]
			}
		}

		// Commit before deciding: whether the round changed anything IS the
		// verdict for a change round (see specDecide), and a removal round has
		// deletions to record either way.
		sha := ""
		if res.TurnErr == "" {
			sha = a.specCommit(ctx, sid, sess.Cwd, cfg, idx, item, mode)
			res.Committed = sha != ""
		}
		done, block, reason := specDecide(cfg, item, res)
		switch {
		case done:
			delete(cfg.Attempts, item)
			delete(reasons, item)
			consecutive = 0
			switch mode {
			case specModeRemove:
				delete(cfg.Items, item)
				delete(removeOK, item)
				say(fmt.Sprintf("🗑 %s removed: the spec no longer has it.\n", item))
			case specModeSetup:
				say("✅ project setup is done.\n")
			default:
				// Record what the spec said, so a later edit to this section is
				// visible as a change instead of passing as still-implemented.
				cfg.Items[item] = specLedger{
					Hash:      specItemHash(idx, item),
					Title:     idx.items[item].Title,
					File:      idx.docs[idx.items[item].Doc].rel,
					CoveredBy: covered[item],
					Commit:    sha,
					At:        time.Now().UTC(),
				}
				say(fmt.Sprintf("✅ %s is covered.\n", item))
			}
		case block:
			cfg.Blocked = append(cfg.Blocked, specBlock{ID: item, Reason: firstLine(reason), Question: res.Question})
			delete(cfg.Attempts, item)
			delete(reasons, item)
			consecutive++
			msg := fmt.Sprintf("⛔ %s blocked: %s\n", item, firstLine(reason))
			if res.Question != "" {
				msg += "Question for you: " + res.Question + "\nAnswer it in `.codehalter/spec.toml` (the item's `answer`), then run /spec.\n"
			}
			say(msg)
		default:
			reasons[item] = reason
			say(fmt.Sprintf("↻ %s is not done yet, one more round: %s\n", item, firstLine(reason)))
		}
		if err := saveSpecConfig(sess.Cwd, cfg); err != nil {
			say("⚠ /spec: " + err.Error() + "\n")
		}
		if block && setup {
			say("The project setup could not be finished, and every item depends on it. Fix what the block says, then run /spec again.\n")
			break
		}
		if consecutive >= cfg.maxBlocked() {
			say(fmt.Sprintf("Stopping: %d items in a row ended blocked, which usually means one shared problem (the build, the test command, a missing toolchain). `/spec status` lists them.\n", consecutive))
			break
		}
	}

	a.drainSteer(ctx, sess)
	a.drainFixes(ctx, sid, dedupeFixes(fixes))
	return end, nil
}

// dedupeFixes drops repeated fix cards: the pre-turn checks run once per round,
// and a problem nobody has fixed yet is reported by every one of them.
func dedupeFixes(fixes []fixProblem) []fixProblem {
	seen := map[string]bool{}
	var out []fixProblem
	for _, f := range fixes {
		if !seen[f.desc] {
			seen[f.desc] = true
			out = append(out, f)
		}
	}
	return out
}

// specRoundPrompt renders the round's user message from res/SPEC.md (or
// SPEC-SETUP.md for the setup round) and returns it with a one-line heading
// for the chat.
func (a *agent) specRoundPrompt(sid string, cfg *specConfig, idx *specIndex, item string, mode specMode, covered map[string]string, testCmd, reason, question, answer, note string) (prompt, head string) {
	target := cfg.Target
	if target == "" {
		target = "No technology was given with /spec. Use what the spec implies; where it leaves the choice open, pick a mainstream option and state it in the project README."
	}
	context := ""
	if files := cfg.context(idx); len(files) > 0 {
		for i, f := range files {
			files[i] = "`" + cfg.SpecDir + "/" + f + "`"
		}
		context = "Standing rules for every item: " + strings.Join(files, ", ") + ". Read them if they are not already in this conversation."
	}
	previous := ""
	switch {
	case answer != "":
		previous = "## Your earlier question was answered\n\n"
		if question != "" {
			previous += "Question: " + question + "\n\n"
		}
		previous += "Answer: " + answer
	case reason != "":
		previous = "## The previous round on this item did not count\n\n" + reason + "\n\nFix that first."
	}

	if mode == specModeRemove {
		led := cfg.Items[item]
		body := a.loadPromptFile(sid, "SPEC-REMOVE.md")
		if body == "" {
			body = defaultSpecRemoveMD
		}
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
		r := strings.NewReplacer(
			"{{id}}", item, "{{title}}", title, "{{out_dir}}", cfg.OutDir,
			"{{test_cmd}}", testCmd, "{{covered_by}}", coveredBy, "{{old_text}}", old,
			"{{previous}}", previous,
		)
		return collapseBlankLines(r.Replace(body)), "remove " + item + " (gone from the spec)"
	}

	if mode == specModeSetup {
		body := a.loadPromptFile(sid, "SPEC-SETUP.md")
		if body == "" {
			body = defaultSpecSetupMD
		}
		r := strings.NewReplacer(
			"{{spec_dir}}", cfg.SpecDir, "{{out_dir}}", cfg.OutDir, "{{target}}", target,
			"{{context}}", context, "{{previous}}", previous,
			"{{items}}", fmt.Sprintf("%d requirement items", len(idx.order)),
		)
		return collapseBlankLines(r.Replace(body)), "project setup in `" + cfg.OutDir + "/`"
	}

	it := idx.items[item]
	sl := idx.slice(item, cfg.SpecDir)
	// The heading already starts with the id ("F2.3 Review every cut"); the
	// prompt names the id separately, so drop it from the title.
	title := strings.TrimSpace(strings.TrimPrefix(it.Title, item))
	if title == "" {
		title = item
	}
	var k [4]int
	var kd [4]int
	for _, id := range idx.order {
		k[idx.items[id].Kind]++
		if _, ok := covered[id]; ok {
			kd[idx.items[id].Kind]++
		}
	}
	progress := fmt.Sprintf("%d of %d items covered (flows %d/%d, sections %d/%d, parameters and tools %d/%d)",
		len(covered), len(idx.order), kd[specDefHeading], k[specDefHeading], kd[specDefSection], k[specDefSection], kd[specDefTableRow], k[specDefTableRow])
	screens := ""
	if len(sl.Images) > 0 {
		screens = "Screens this text shows (look at them with `screenshot path=...`): " + strings.Join(sl.Images, ", ") + "\n"
	}
	related := ""
	if len(sl.Related) > 0 {
		related = "Also relevant, read if you need it: " + strings.Join(sl.Related, ", ") + "\n"
	}
	body := a.loadPromptFile(sid, "SPEC.md")
	if body == "" {
		body = defaultSpecMD
	}
	if note != "" {
		previous = "## This item was implemented before, and the spec has changed since\n\n" +
			"The code and its test match the OLD text. Update both to the text below; where the diff removes something, remove it from the code too.\n\n```diff\n" +
			note + "\n```" + strings.TrimPrefix(previous, "## The previous round on this item did not count")
	}
	r := strings.NewReplacer(
		"{{id}}", item, "{{title}}", title, "{{file}}", cfg.SpecDir+"/"+idx.docs[it.Doc].rel,
		"{{progress}}", progress, "{{target}}", target, "{{spec_dir}}", cfg.SpecDir, "{{out_dir}}", cfg.OutDir,
		"{{test_cmd}}", testCmd, "{{token}}", specTestToken(item), "{{context}}", context,
		"{{slice}}", strings.TrimRight(sl.Text, "\n"), "{{screens}}", screens, "{{related}}", related,
		"{{previous}}", previous,
	)
	return collapseBlankLines(r.Replace(body)), fmt.Sprintf("%s %s · %s", item, title, progress)
}

// collapseBlankLines squeezes runs of blank lines to one, left behind where an
// optional placeholder rendered empty.
func collapseBlankLines(s string) string {
	for strings.Contains(s, "\n\n\n") {
		s = strings.ReplaceAll(s, "\n\n\n", "\n\n")
	}
	return s
}

// runSpecTests runs the project's test command in the output directory. It is
// codehalter's own check, not the model's, so it runs directly rather than in a
// visible terminal: the round already showed the model's test runs.
func (a *agent) runSpecTests(ctx context.Context, sid, outAbs, outRel, cmd string) (bool, string) {
	a.say(ctx, sid, fmt.Sprintf("🧪 checking: `%s` in `%s/`\n", cmd, outRel))
	tctx, cancel := context.WithTimeout(ctx, specTestTimeout)
	defer cancel()
	shell := "sh"
	if _, err := exec.LookPath("bash"); err == nil {
		shell = "bash"
	}
	// A login shell, so a toolchain installed into the user's profile PATH
	// (rustup puts cargo in ~/.cargo/bin) is found the same way a terminal
	// finds it.
	c := exec.CommandContext(tctx, shell, "-lc", cmd)
	c.Dir = outAbs
	start := time.Now()
	out, err := c.CombinedOutput()
	tail := string(out)
	if len(tail) > specTestTailBytes {
		tail = "…" + tailUTF8(tail, specTestTailBytes)
	}
	took := humanDuration(time.Since(start).Milliseconds())
	if err != nil {
		if tctx.Err() == context.DeadlineExceeded {
			tail += fmt.Sprintf("\n[codehalter stopped the test command after %s]", specTestTimeout)
		}
		a.say(ctx, sid, fmt.Sprintf("🧪 tests failed after %s\n", took))
		return false, tail
	}
	a.say(ctx, sid, fmt.Sprintf("🧪 tests passed in %s\n", took))
	return true, tail
}

// specCommit commits the output directory (and .devcontainer, where the
// toolchain lives) after an item passes, so every covered item is one commit
// and a cancelled or crashed loop loses nothing. Anything else in the tree is
// left for the user: an unrelated change of theirs must not ride along.
// specCommit stages the output directory and commits it, returning the new
// commit's sha (empty when there was nothing to commit, which is how a change
// round detects that the model did not touch the code).
func (a *agent) specCommit(ctx context.Context, sid, cwd string, cfg *specConfig, idx *specIndex, item string, mode specMode) string {
	git := func(args ...string) (string, error) {
		out, err := exec.CommandContext(ctx, "git", append([]string{"-C", cwd}, args...)...).CombinedOutput()
		return string(out), err
	}
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

// specSuggestTimeout bounds the one model call that reads the spec's entry page
// during setup. It is a page, not a conversation, and the user is waiting.
const specSuggestTimeout = 2 * time.Minute

// suggestSpecTarget reads the spec's own entry page for what to build and where
// to put it. The model goes first because a page says things a keyword scan
// cannot ("a desktop editor ... GTK 4 with libadwaita"); the keyword table is
// the fallback when there is no model, it fails, or it says unknown.
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

// specSetupDialog is the first /spec in a project: one card for where the spec
// is, then one for what to build from it and where. The second question's
// suggestion is read off the entry page of whatever the first answered, which
// is why they are two cards and not one.
//
// Autopilot takes the suggestion on both (askFormAuto answers with the first
// option), so an unattended run still starts, and says what it chose.
func (a *agent) specSetupDialog(ctx context.Context, sid string, sess *Session) (specDir, outDir, target string, ok bool) {
	say := func(s string) { a.say(ctx, sid, s) }
	cands := specDirCandidates(sess.Cwd)
	if len(cands) == 0 {
		say("No specification found: `/spec` looks for a directory holding at least two markdown files. Point it at one with `/spec <spec-dir> <out-dir> [technology]`.\n")
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
	var options []string
	if gOut != "" {
		options = append(options, gOut+" "+gTarget)
	}
	q := "What should be built from `" + specDir + "/`, and where? Answer as `<out-dir> <technology>`, for example `rust/ gtk4-rs with libadwaita`."
	if gOut != "" {
		q = "`" + specDir + "/" + entryRel + "` reads as **" + gTarget + "**. Build it into `" + gOut + "/`? Pick that, or type `<out-dir> <technology>`."
	}
	tcId2 := a.StartToolCall(ctx, sid, "What to build, and where?", "think", nil)
	answer2, err := a.askFormAuto(ctx, sid, tcId2, q, options, true)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId2, err.Error())
		return "", "", "", false
	}
	answer2 = strings.TrimSpace(answer2)
	if answer2 == "" {
		a.FailToolCall(ctx, sid, tcId2, "no target given")
		say("⚠ /spec: nothing to build into. Run `/spec " + specDir + " <out-dir> [technology]` when you know where it should go.\n")
		return "", "", "", false
	}
	outDir, target = answer2, ""
	if i := strings.IndexAny(answer2, " \t"); i >= 0 {
		outDir, target = answer2[:i], strings.TrimSpace(answer2[i+1:])
	}
	outDir = filepath.Clean(strings.Trim(outDir, "`\""))
	a.CompleteToolCall(ctx, sid, tcId2, []ToolCallContent{TextContent("Building into " + outDir + "/" + map[bool]string{true: "", false: " · target: " + target}[target == ""])})
	return specDir, outDir, target, true
}

// askSpecRemoval confirms deleting the code behind an item the spec dropped.
// The spec is the master, so removing is the first option, which is also the
// one autopilot takes; the round is one commit, so it is a revert away.
func (a *agent) askSpecRemoval(ctx context.Context, sid, id string) bool {
	ok, tcId, err := a.askYesNoWithCard(ctx, sid,
		id+" is gone from the spec. Remove its code and tests?", "think",
		"Remove it", "Keep the code")
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return false
	}
	answer := "Keeping the code; codehalter stops tracking the item"
	if ok {
		answer = "Removing it in the next round"
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(answer)})
	return ok
}

// specRemovedText recovers what the spec used to say about an item from the
// commit that implemented it, so the removal round knows what it is deleting.
func specRemovedText(ctx context.Context, cwd string, cfg *specConfig, id string) string {
	led := cfg.Items[id]
	if led.Commit == "" || led.File == "" {
		return ""
	}
	out, err := exec.CommandContext(ctx, "git", "-C", cwd, "show", led.Commit+":"+cfg.SpecDir+"/"+led.File).Output()
	if err != nil {
		return ""
	}
	return clipBytes(specSectionFromText(string(out), id), specSpecDiffBytes)
}

// specSpecDiff is what changed in an item's spec file since the commit that
// implemented it, so a change round sees the edit rather than the whole section
// again. Empty when the project is not in git or the commit is gone.
func specSpecDiff(ctx context.Context, cwd, commit, relPath string) string {
	if commit == "" {
		return ""
	}
	out, err := exec.CommandContext(ctx, "git", "-C", cwd, "diff", commit+"..HEAD", "--", relPath).CombinedOutput()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(clipBytes(string(out), specSpecDiffBytes))
}
