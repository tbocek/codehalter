package main

import (
	"context"
	"errors"
	"fmt"
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
	s.specMu.Lock()
	s.specFenceDir = dir
	s.specMu.Unlock()
}

// specFence returns the fenced spec dir, or "" when no loop is running.
func (s *Session) specFence() string {
	s.specMu.Lock()
	defer s.specMu.Unlock()
	return s.specFenceDir
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
type specRoundResult struct {
	Setup     bool
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
	if r.TurnErr == "" && r.Covered && r.TestsPass {
		return true, false, ""
	}
	var why []string
	if r.TurnErr != "" {
		why = append(why, "the round ended with an error: "+r.TurnErr)
	}
	if !r.Covered {
		if r.Setup {
			why = append(why, "no test source exists in the output directory yet")
		} else {
			why = append(why, fmt.Sprintf("no test in the output directory names %s (a test name containing `%s`, or `%s` in a comment or string inside a test)", item, specTestToken(item), item))
		}
	}
	if !r.TestsPass {
		why = append(why, "the test command did not pass. The end of its output:\n\n```\n"+strings.TrimSpace(r.TestTail)+"\n```")
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
	switch cmd {
	case "setup":
		specRel, outRel, err := specPaths(sess.Cwd, specArg, outArg)
		if err != nil {
			say("⚠ /spec: " + err.Error() + "\n")
			return end, nil
		}
		if cfg == nil || cfg.SpecDir != specRel {
			cfg = &specConfig{} // a different spec: its attempts and blocks don't carry over
		}
		cfg.SpecDir, cfg.OutDir, cfg.Target = specRel, outRel, target
		if err := os.MkdirAll(filepath.Join(sess.Cwd, outRel), 0o755); err != nil {
			say("⚠ /spec: creating " + outRel + ": " + err.Error() + "\n")
			return end, nil
		}
		if err := saveSpecConfig(sess.Cwd, cfg); err != nil {
			say("⚠ /spec: " + err.Error() + "\n")
			return end, nil
		}
	default:
		if cfg == nil {
			say("No spec loop set up in this project yet. Start one with `/spec <spec-dir> <out-dir> [technology prompt]`, for example `/spec spec/ rust/ use gtk4-rs libadwaita`.\n")
			return end, nil
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
		say(renderSpecStatus(cfg, idx, covered, testFiles, testCmd()) + "\n")
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
		setup := testCmd() == "" || testFiles == 0
		item, answer := specSetupID, ""
		if !setup {
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

		round++
		prompt, head := a.specRoundPrompt(sid, cfg, idx, item, setup, covered, testCmd(), reasons[item], question, answer)
		say(fmt.Sprintf("\n## /spec round %d · %s\n\n", round, head))
		sess.AddUser(prompt)
		sess.saveOrLog()

		turnErr := a.runTurn(ctx, sid)
		if isCancelled(turnErr) {
			return stopped(turnErr)
		}
		res := specRoundResult{Setup: setup}
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
			if setup {
				res.Covered = testFiles > 0
			} else {
				_, res.Covered = covered[item]
			}
		}

		done, block, reason := specDecide(cfg, item, res)
		switch {
		case done:
			delete(cfg.Attempts, item)
			delete(reasons, item)
			consecutive = 0
			say(fmt.Sprintf("✅ %s is covered.\n", item))
			a.specCommit(ctx, sid, sess.Cwd, cfg, idx, item)
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
func (a *agent) specRoundPrompt(sid string, cfg *specConfig, idx *specIndex, item string, setup bool, covered map[string]string, testCmd, reason, question, answer string) (prompt, head string) {
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

	if setup {
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
		tail = "…" + tail[len(tail)-specTestTailBytes:]
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
func (a *agent) specCommit(ctx context.Context, sid, cwd string, cfg *specConfig, idx *specIndex, item string) {
	git := func(args ...string) (string, error) {
		out, err := exec.CommandContext(ctx, "git", append([]string{"-C", cwd}, args...)...).CombinedOutput()
		return string(out), err
	}
	if _, err := git("rev-parse", "--is-inside-work-tree"); err != nil {
		return
	}
	paths := []string{cfg.OutDir}
	if fileExists(cwd, ".devcontainer") {
		paths = append(paths, ".devcontainer")
	}
	if out, err := git(append([]string{"add", "-A", "--"}, paths...)...); err != nil {
		a.say(ctx, sid, "⚠ /spec: git add failed: "+firstLine(out)+"\n")
		return
	}
	if _, err := git("diff", "--cached", "--quiet"); err == nil {
		return // nothing staged
	}
	msg := "spec: " + item
	if item == specSetupID {
		msg = "spec: set up " + cfg.OutDir + "/"
	} else if t := idx.items[item].Title; t != "" {
		msg = "spec: " + item + " " + strings.TrimPrefix(t, item+" ")
	}
	if out, err := git("commit", "-q", "-m", msg); err != nil {
		a.say(ctx, sid, "⚠ /spec: git commit failed, the work stays uncommitted: "+firstLine(out)+"\n")
		return
	}
	a.say(ctx, sid, "📌 committed: "+msg+"\n")
}
