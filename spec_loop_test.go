package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Done only when a test names the item and the suite passes; one retry, then blocked.
func TestSpecDecide(t *testing.T) {
	cfg := &specConfig{}
	if done, block, _, _ := specDecide(cfg, "F0.1", specRoundResult{Covered: true, TestsPass: true}, ""); !done || block {
		t.Errorf("covered and passing: done=%v block=%v, want done", done, block)
	}

	// Covered but the suite fails: a broken earlier item counts against the round.
	done, block, reason, _ := specDecide(cfg, "F0.2", specRoundResult{Covered: true, TestsPass: false, TestTail: "test f0_1 failed"}, "")
	if done || block || !strings.Contains(reason, "test f0_1 failed") {
		t.Errorf("first failure: done=%v block=%v reason=%q, want a retry carrying the test output", done, block, reason)
	}
	done, block, reason, _ = specDecide(cfg, "F0.2", specRoundResult{TestsPass: true}, "")
	if done || !block || !strings.Contains(reason, "f0_2") {
		t.Errorf("second failure: done=%v block=%v reason=%q, want blocked, naming the expected test token", done, block, reason)
	}

	if _, block, _, _ := specDecide(cfg, "F0.3", specRoundResult{Question: "keep or drop?"}, ""); !block {
		t.Error("a planner question did not block the item")
	}
	_, _, reason, _ = specDecide(&specConfig{}, specSetupID, specRoundResult{Mode: specModeSetup, TestsPass: true}, "")
	if !strings.Contains(reason, "no test source") {
		t.Errorf("setup without tests: reason %q", reason)
	}
}

// A second failure of another kind earns a third attempt; the same failure twice,
// or any third failure, blocks.
func TestSpecDecideThirdAttemptOnlyWhenTheFailureMoved(t *testing.T) {
	red := specRoundResult{Covered: true, TestsPass: false}
	unseen := specRoundResult{Covered: true, TestsPass: true, UIUnseen: []string{"src/ui.rs"}}

	cfg := &specConfig{}
	_, _, _, first := specDecide(cfg, "F0.1", red, "")
	if _, block, _, _ := specDecide(cfg, "F0.1", red, first); !block {
		t.Error("the same failure twice did not block")
	}

	cfg = &specConfig{}
	_, _, _, first = specDecide(cfg, "F0.2", red, "")
	_, block, _, second := specDecide(cfg, "F0.2", unseen, first)
	if block {
		t.Errorf("a failure that moved (%s -> %s) blocked at the second attempt", first, second)
	}
	if _, block, _, _ := specDecide(cfg, "F0.2", red, second); !block {
		t.Error("a third failure did not block")
	}
}

// The gates on what a round wrote hold an item back and say why; a refactor
// counts when its debt shrank.
func TestSpecDecideGates(t *testing.T) {
	green := specRoundResult{Covered: true, TestsPass: true, Committed: true}
	for name, r := range map[string]specRoundResult{
		"unreachable": {Covered: true, TestsPass: true, Unreachable: []string{"`run` (src/a.rs:3): only tests call it"}},
		"lint":        {Covered: true, TestsPass: true, Lint: []string{"src/a.rs:3: unused variable"}},
		"oversize":    {Covered: true, TestsPass: true, Oversize: []string{"`src/ui.rs` is 12000 lines, over the 1500-line budget"}},
		"standin":     {Covered: true, TestsPass: true, StandIns: []string{"`reply_for_test` (src/a.rs:3): the program asks a helper that exists for tests"}},
	} {
		done, _, reason, _ := specDecide(&specConfig{}, "F0.1", r, "")
		if done || !strings.Contains(reason, map[string]string{"unreachable": "only tests call it", "lint": "unused variable", "oversize": "1500-line budget", "standin": "is not done"}[name]) {
			t.Errorf("%s: done=%v reason=%q", name, done, reason)
		}
	}
	if done, _, _, _ := specDecide(&specConfig{}, "F0.1", green, ""); !done {
		t.Error("a clean round was held back")
	}
	cfg := &specConfig{}
	if done, _, reason, _ := specDecide(cfg, specRefactorID, specRoundResult{Mode: specModeRefactor, TestsPass: true, Committed: true, Debt: "a → b"}, ""); done || !strings.Contains(reason, "no measured progress") {
		t.Errorf("a refactor that shrank nothing: done=%v reason=%q", done, reason)
	}
	if done, _, _, _ := specDecide(cfg, specRefactorID, specRoundResult{Mode: specModeRefactor, TestsPass: true, Committed: true, Improved: true}, ""); !done {
		t.Error("a refactor that shrank the debt was held back")
	}
}

func TestSpecPaths(t *testing.T) {
	cwd := t.TempDir()
	if err := os.MkdirAll(filepath.Join(cwd, "spec"), 0o755); err != nil {
		t.Fatal(err)
	}
	spec, out, err := specPaths(cwd, "spec/", "rust/")
	if err != nil || spec != "spec" || out != "rust" {
		t.Errorf("= (%q %q %v), want spec rust", spec, out, err)
	}
	for _, tc := range []struct{ spec, out string }{
		{"missing", "rust"},      // spec dir must exist
		{"../elsewhere", "rust"}, // outside the project
		{"spec", "spec"},         // the same dir
		{"spec", "spec/rust"},    // output inside the fenced spec
		{".", "rust"},            // fencing the whole project
	} {
		if _, _, err := specPaths(cwd, tc.spec, tc.out); err == nil {
			t.Errorf("specPaths(%q, %q) accepted", tc.spec, tc.out)
		}
	}
}

// The file tools refuse the spec dir and nothing else.
func TestSpecFence(t *testing.T) {
	a, s := newTestAgent(t)
	spec := filepath.Join(s.Cwd, "spec")
	if msg := a.specFenceRefusal(s.ID, filepath.Join(spec, "a.md")); msg != "" {
		t.Errorf("no loop running, still refused: %q", msg)
	}
	s.setSpecFence(spec)
	if msg := a.specFenceRefusal(s.ID, filepath.Join(spec, "sub", "a.md")); msg == "" {
		t.Error("a write into the spec was allowed while the loop runs")
	}
	for _, p := range []string{filepath.Join(s.Cwd, "rust", "src", "lib.rs"), filepath.Join(s.Cwd, "spec-notes.md")} {
		if msg := a.specFenceRefusal(s.ID, p); msg != "" {
			t.Errorf("%s refused: %q", p, msg)
		}
	}
	s.setSpecFence("")
	if msg := a.specFenceRefusal(s.ID, filepath.Join(spec, "a.md")); msg != "" {
		t.Errorf("fence still up after the loop: %q", msg)
	}
}

// Both round prompts fill every placeholder and carry the item's facts.
func TestSpecRoundPrompt(t *testing.T) {
	a, s := newTestAgent(t)
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", Target: "use gtk4-rs libadwaita", Context: []string{"00-principles.md"},
		Items: map[string]specLedger{"§01-files#1-layout": {}}}

	prompt, head := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: "F0.1", Reason: "no test names f0_1"}, "just test", "cargo clippy --all-targets --quiet")
	for _, want := range []string{"**F0.1**", "`f0_1`", "just test", "use gtk4-rs libadwaita", "rust/", "S1 Switch.",
		"did not count", "no test names f0_1", "spec/00-principles.md", "1 of", "`cargo clippy --all-targets --quiet`, run from `rust/`",
		"over 1500 lines grows by at most 20"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("item prompt lacks %q", want)
		}
	}
	if strings.Contains(prompt, "{{") {
		t.Errorf("unfilled placeholder in the item prompt:\n%s", prompt)
	}
	if !strings.HasPrefix(head, "F0.1 Switch tab") {
		t.Errorf("heading = %q", head)
	}

	// A linter that did not run last round is this round's to install.
	install, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: "F0.1", LintMissing: "error: no such command: `clippy`"}, "just test", "cargo clippy --all-targets --quiet")
	if !strings.Contains(install, "It does not run in this container yet (error: no such command: `clippy`): install it first") {
		t.Errorf("the item prompt does not ask for the missing linter:\n%s", install)
	}

	refactor, head := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: specRefactorID, Mode: specModeRefactor,
		Debt: specDebt{overLines: 900, targets: "- `src/ui/window.rs`: 2400 lines\n"}}, "just test", "")
	if strings.Contains(refactor, "{{") || !strings.Contains(refactor, "`src/ui/window.rs`: 2400 lines") || !strings.Contains(head, "lines over the size budget 900") {
		t.Errorf("refactor prompt (%s):\n%s", head, refactor)
	}

	setup, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: specSetupID, Mode: specModeSetup}, "", "")
	if strings.Contains(setup, "{{") || !strings.Contains(setup, "use gtk4-rs libadwaita") || !strings.Contains(setup, "`rust/`") {
		t.Errorf("setup prompt:\n%s", setup)
	}

	qa := "## F0.2 · keep or drop?\n\n**Answer:** keep it"
	answered, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: "F0.2", Answered: qa}, "just test", "")
	if !strings.Contains(answered, qa) {
		t.Errorf("answered prompt lacks the question and answer:\n%s", answered)
	}
	// A changed item comes back with its answer; the diff fence must still close.
	changed, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: "F0.2", Mode: specModeChange, Note: "-old\n+new", Answered: qa}, "just test", "")
	if !strings.Contains(changed, "+new\n```\n\n## Answered in the spec's QUESTIONS.md") {
		t.Errorf("the answer runs into the diff fence:\n%s", changed)
	}
	// A retry keeps the answer beside why the last attempt failed.
	if retry, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: "F0.2", Answered: qa, Reason: "the suite failed"}, "just test", ""); !strings.Contains(retry, qa) || !strings.Contains(retry, "the suite failed") {
		t.Errorf("a retry lost the answer or the reason:\n%s", retry)
	}
	// cfg.Context is saved to spec.toml: rendering must not rewrite it.
	if cfg.Context[0] != "00-principles.md" {
		t.Errorf("rendering rewrote cfg.Context: %v", cfg.Context)
	}
}

// Each unanchored rule is named once; the anchored version of the same file is clean.
func TestSpecIgnoredProbes(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git")
	}
	cwd := t.TempDir()
	if out, err := exec.Command("git", "-C", cwd, "init", "-q").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v %s", err, out)
	}
	idx := &specIndex{docs: []specDoc{{rel: "05-cut.md"}, {rel: "07-narrate.md"}, {rel: "09-llm-and-tools.md"}}}
	gi := filepath.Join(cwd, ".gitignore")

	// llm/ hides the module named after the chapter's first word.
	os.WriteFile(gi, []byte("cut/\n*.json\ntest*\nout/\nllm/\n"), 0o644)
	got := specIgnoredProbes(t.Context(), cwd, "rust", idx)
	joined := strings.Join(got, "\n")
	for _, rule := range []string{"`cut/`", "`*.json`", "`test*`", "`llm/`"} {
		if strings.Count(joined, rule) != 1 {
			t.Errorf("rule %s reported %d times, want once:\n%s", rule, strings.Count(joined, rule), joined)
		}
	}
	if strings.Contains(joined, "`out/`") {
		t.Errorf("a rule that hides nothing under rust/ was reported:\n%s", joined)
	}

	os.WriteFile(gi, []byte("/cut/\n/*.json\n/test*\n/out/\n/llm/\n"), 0o644)
	if got := specIgnoredProbes(t.Context(), cwd, "rust", idx); len(got) != 0 {
		t.Errorf("anchored rules still reported: %v", got)
	}
}

// A change round needs a commit, since the old test still covers it; a removal is done when no
// test names the item.
func TestSpecDecideChangeAndRemove(t *testing.T) {
	cfg := &specConfig{}
	done, _, reason, _ := specDecide(cfg, "F0.1", specRoundResult{Mode: specModeChange, Covered: true, TestsPass: true}, "")
	if done || !strings.Contains(reason, "no code did") {
		t.Errorf("a change round that wrote nothing: done=%v reason=%q", done, reason)
	}
	if done, _, _, _ := specDecide(cfg, "F0.1", specRoundResult{Mode: specModeChange, Covered: true, TestsPass: true, Committed: true}, ""); !done {
		t.Error("a change round that committed should be done")
	}

	done, _, reason, _ = specDecide(cfg, "F0.2", specRoundResult{Mode: specModeRemove, TestsPass: true, Covered: true, Committed: true}, "")
	if done || !strings.Contains(reason, "still names") {
		t.Errorf("a removal with the test still in place: done=%v reason=%q", done, reason)
	}
	if done, _, _, _ := specDecide(cfg, "F0.2", specRoundResult{Mode: specModeRemove, TestsPass: true, Committed: true}, ""); !done {
		t.Error("a removal whose test is gone and whose suite passes should be done")
	}
	// The suite must still pass: deleting an item cannot take the build with it.
	if done, _, reason, _ := specDecide(cfg, "F0.3", specRoundResult{Mode: specModeRemove, TestTail: "3 failed"}, ""); done || !strings.Contains(reason, "did not pass") {
		t.Errorf("a removal that broke the suite: done=%v reason=%q", done, reason)
	}
}

func TestSpecDecideRedoNeedsACommit(t *testing.T) {
	cfg := &specConfig{}
	done, _, reason, _ := specDecide(cfg, "F0.1", specRoundResult{Redo: true, Covered: true, TestsPass: true}, "")
	if done || !strings.Contains(reason, "/spec redo") {
		t.Errorf("a redo round that wrote nothing: done=%v reason=%q", done, reason)
	}
	if done, _, _, _ := specDecide(cfg, "F0.1", specRoundResult{Redo: true, Covered: true, TestsPass: true, Committed: true}, ""); !done {
		t.Error("a redo round that committed should be done")
	}
}

// Target, spec files, test command, counts and blocked list are in, no placeholder left.
func TestSpecFinalPrompt(t *testing.T) {
	a, s := newTestAgent(t)
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", Target: "use gtk4-rs libadwaita",
		Items: map[string]specLedger{"F0.1": {}, "F0.2": {}}}
	stuck := idx.order[len(idx.order)-1]
	idx.questions = map[string][]specQuestion{"F0.3": {{ID: "F0.3", Question: "keep or drop?"}}}
	prompt := a.specFinalPrompt(s.ID, cfg, idx, "just test", map[string]string{stuck: "the build stayed red"})
	for _, want := range []string{"use gtk4-rs libadwaita", "`rust/README.md`", "just test", "2 items, 2 not done",
		stuck + ": the build stayed red", `F0.3: waits on the user's answer to "keep or drop?" in spec/QUESTIONS.md`, "spec/00-principles.md", "`--help`", "snapshot"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("final prompt lacks %q", want)
		}
	}
	if strings.Contains(prompt, "{{") {
		t.Errorf("unfilled placeholder:\n%s", prompt)
	}
	idx.questions = nil
	if !strings.Contains(a.specFinalPrompt(s.ID, &specConfig{SpecDir: "spec", OutDir: "rust"}, idx, "cargo test", nil), "none") {
		t.Error("no blocked items must render as none")
	}
}

func TestRunSpecTestsTailCarriesExitStatus(t *testing.T) {
	h := newTerminalHarness(t)
	pass, tail := h.agent.runSpecTests(context.Background(), h.sess.ID, t.TempDir(), "rust", "exit 3")
	if pass || !strings.Contains(tail, "printed nothing") || !strings.Contains(tail, "exit status 3") {
		t.Errorf("pass=%v tail=%q", pass, tail)
	}
	pass, tail = h.agent.runSpecTests(context.Background(), h.sess.ID, t.TempDir(), "rust", "echo boom; exit 4")
	if pass || !strings.Contains(tail, "boom") || !strings.Contains(tail, "exit status 4") {
		t.Errorf("pass=%v tail=%q", pass, tail)
	}
}

// Only a bare run in the output dir, exit 0 per the terminal, nothing written in
// it since, counts; the run may be a background job that exited this round.
func TestSpecRoundOwnGreenRun(t *testing.T) {
	a, sess := newTestAgent(t)
	out := filepath.Join(sess.Cwd, "rust")
	if err := os.MkdirAll(filepath.Join(out, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(out, "src", "lib.rs"), []byte("fn a() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	r := &specRun{a: a, sid: sess.ID, sess: sess, cfg: &specConfig{OutDir: "rust"}, outAbs: out}
	round := time.Now().Add(-time.Minute)
	ran := func(cmd, output string) ToolUse {
		return ToolUse{Name: "run_command", Input: `{"command":` + strconv.Quote(cmd) + `}`, Output: output, StartedAt: time.Now().Add(time.Second), DurationMs: 10}
	}
	edit := func(path string, after time.Duration) ToolUse {
		return ToolUse{Name: "edit_file", Input: `{"path":"` + path + `"}`, StartedAt: time.Now().Add(after)}
	}
	green := func(uses ...ToolUse) bool { return r.roundRanGreen(uses, "just test", round) }

	if !green(edit("rust/src/lib.rs", 0), ran("cd rust && just test > /tmp/t.log 2>&1", "exit 0\n\n")) {
		t.Error("a plain green run after the last edit did not count")
	}
	if !green(ran("cd "+out+" && just test", "exit 0\n\nok")) {
		t.Error("an absolute cd to the output dir did not count")
	}
	if !green(ran("cd rust && just test", "exit 0\n"), edit("AGENT.md", 2*time.Second)) {
		t.Error("an edit outside the output directory undid the green run")
	}
	for name, uses := range map[string][]ToolUse{
		"wrapped, exit is the echo's": {ran("cd rust && (just test > /tmp/t.log 2>&1; echo exit=$? >> /tmp/t.log)", "exit 0\n")},
		"red":                         {ran("cd rust && just test", "exit 101\n")},
		"edited after the run":        {ran("cd rust && just test", "exit 0\n"), edit("rust/src/lib.rs", 2*time.Second)},
		"wrong directory":             {ran("just test", "exit 0\n")},
		"handed over, no exit yet":    {ran("cd rust && just test", "still running after 2m")},
	} {
		if green(uses...) {
			t.Errorf("%s: counted as green", name)
		}
	}
	// A file written after the run, by whatever means.
	uses := []ToolUse{ran("cd rust && just test", "exit 0\n")}
	uses[0].StartedAt = time.Now().Add(-time.Minute)
	if green(uses...) {
		t.Error("a source file newer than the run did not force a re-run")
	}

	sess.recordJobRun(jobRun{cmd: "cd rust && just test > /tmp/gate.log 2>&1", code: 0, started: time.Now().Add(time.Second), ended: time.Now().Add(time.Second)})
	if !green() {
		t.Error("a background gate that exited 0 this round did not count")
	}
	sess.recordJobRun(jobRun{cmd: "cd rust && just test", code: 101, started: time.Now().Add(time.Second), ended: time.Now().Add(2 * time.Second)})
	if green(ran("cd rust && just test", "exit 0\n")) {
		t.Error("a red background run after the green one still counted as green")
	}
}

// A round that looked, or touched no UI file, is unaffected.
func TestSpecUIChangeNeedsALook(t *testing.T) {
	_, sess := newTestAgent(t)
	ui := filepath.Join(sess.Cwd, "rust", "src", "ui.rs")
	plain := filepath.Join(sess.Cwd, "rust", "src", "rules.rs")
	if err := os.MkdirAll(filepath.Dir(ui), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ui, []byte("use gtk::prelude::*;\n#[cfg(test)]\nmod tests {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Test code imports the toolkit too: an inline test module or a test file is not a screen.
	if err := os.WriteFile(plain, []byte("pub fn floor() -> f64 { 0.1 }\n#[cfg(test)]\nmod tests { use gtk::prelude::*; }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	uiTest := filepath.Join(sess.Cwd, "rust", "tests", "window.rs")
	if err := os.MkdirAll(filepath.Dir(uiTest), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(uiTest, []byte("use gtk::prelude::*;\n#[test]\nfn f1_1_click() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := uiEditedUnseen([]ToolUse{{Name: "write_file", Input: `{"path":"rust/tests/window.rs"}`}}, sess.Cwd); got != nil {
		t.Errorf("a test file edit demanded a look: %v", got)
	}
	editUI := ToolUse{Name: "edit_file", Input: `{"path":"rust/src/ui.rs"}`}
	editPlain := ToolUse{Name: "write_file", Input: `{"path":"rust/src/rules.rs"}`}
	// A README naming the toolkit is not a screen.
	readme := filepath.Join(sess.Cwd, "Readme.md")
	if err := os.WriteFile(readme, []byte("Built with gtk:: and libadwaita.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	editDoc := ToolUse{Name: "edit_file", Input: `{"path":"Readme.md"}`}
	if got := uiEditedUnseen([]ToolUse{editDoc}, sess.Cwd); got != nil {
		t.Errorf("a markdown edit demanded a look: %v", got)
	}
	look := ToolUse{Name: "screenshot", Input: `{"path":"rust/shots/03-window.png"}`}

	if got := uiEditedUnseen([]ToolUse{editPlain, editUI}, sess.Cwd); strings.Join(got, ",") != "rust/src/ui.rs" {
		t.Errorf("unseen = %v, want the UI file alone", got)
	}
	if got := uiEditedUnseen([]ToolUse{editUI, look}, sess.Cwd); got != nil {
		t.Errorf("a round that looked reports %v", got)
	}
	if got := uiEditedUnseen([]ToolUse{editPlain}, sess.Cwd); got != nil {
		t.Errorf("a round with no UI change reports %v", got)
	}
	done, block, reason, _ := specDecide(&specConfig{}, "F1.1", specRoundResult{Covered: true, TestsPass: true, UIUnseen: []string{"rust/src/ui.rs"}}, "")
	if done || block || !strings.Contains(reason, "never looked at it") {
		t.Errorf("decide = %v %v %q, want one more round with the reason", done, block, reason)
	}
}

// The audit prompt names the spec files and the way to answer.
func TestSpecAuditPrompt(t *testing.T) {
	a, s := newTestAgent(t)
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	prompt := a.specAuditPrompt(s.ID, &specConfig{SpecDir: "spec", OutDir: "rust", Target: "gtk4"}, idx, "just test")
	for _, want := range []string{"`redo`", "spec/00-principles.md", "just test", "gtk4", "snapshot", "report_only", "`F0.1`", "copied from the list below"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("audit prompt lacks %q", want)
		}
	}
	if strings.Contains(prompt, "{{") {
		t.Errorf("unfilled placeholder:\n%s", prompt)
	}
}

// A deleted section is removed without a card, a mass disappearance removes nothing, and a
// moved section keeps its record.
func TestSpecPickWorkRemovals(t *testing.T) {
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Every live item is built, so only the ledger's gone entries can make work.
	built := func(extra map[string]specLedger) map[string]specLedger {
		items := map[string]specLedger{}
		for _, id := range idx.order {
			items[id] = specLedger{Hash: specItemHash(idx, id), Title: idx.items[id].Title}
		}
		for id, led := range extra {
			items[id] = led
		}
		return items
	}
	pick := func(items map[string]specLedger) (specWork, *specRun, string) {
		t.Helper()
		h := newTerminalHarness(t)
		r := &specRun{a: h.agent, sid: h.sess.ID, sess: h.sess, idx: idx, reasons: map[string]string{},
			cfg: &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: "just test", Items: items}}
		named := map[string]string{} // built means a test is named after it
		for _, id := range idx.order {
			named[id] = "tests/a.rs"
		}
		w := r.pickWork(t.Context(), named, 1)
		if m := h.sentMethods(); len(m) != 0 {
			t.Errorf("pickWork asked the client %v, want no card", m)
		}
		log, err := os.ReadFile(sessionPath(h.sess.Cwd, h.sess.ID, "log"))
		if err != nil {
			t.Fatalf("no session log: %v", err)
		}
		return w, r, string(log)
	}

	const dropped = "§99-old#1-dropped"
	w, r, log := pick(built(map[string]specLedger{dropped: {Hash: "whatever", Title: "1. Dropped"}}))
	if w.Item != dropped || w.Mode != specModeRemove {
		t.Errorf("work = %+v, want a removal round for %s", w, dropped)
	}
	if _, ok := r.cfg.Items[dropped]; !ok || r.vanished {
		t.Errorf("the item must stay in the ledger until its removal round passes (vanished=%v)", r.vanished)
	}
	if !strings.Contains(log, "removed first: "+dropped) {
		t.Errorf("the change report does not announce the removal:\n%s", log)
	}

	// More gone entries than live ones: the spec looks missing, not edited.
	gone := map[string]specLedger{}
	for i := range len(idx.order) + 1 {
		gone[fmt.Sprintf("§99-old#%d-dropped", i)] = specLedger{Hash: "gone", Title: fmt.Sprintf("%d. Dropped", i)}
	}
	w, r, log = pick(built(gone))
	if w.Item != "" || !r.vanished {
		t.Errorf("work = %+v vanished=%v, want nothing scheduled", w, r.vanished)
	}
	for id := range gone {
		if _, ok := r.cfg.Items[id]; !ok {
			t.Errorf("%s left the ledger, want nothing forgotten", id)
		}
	}
	if !strings.Contains(log, "looks missing or unreadable") || !strings.Contains(log, "nothing is deleted") {
		t.Errorf("no warning line:\n%s", log)
	}
	// The remaining work still runs.
	items := built(gone)
	delete(items, idx.order[0])
	if w, _, _ := pick(items); w.Item != idx.order[0] || w.Mode != specModeItem {
		t.Errorf("work = %+v, want %s built while the removals are held", w, idx.order[0])
	}

	// A removal blocked in this run, or waiting on its question, is not picked every round.
	pickRemoval := func(blocked map[string]string) specWork {
		t.Helper()
		h := newTerminalHarness(t)
		r := &specRun{a: h.agent, sid: h.sess.ID, sess: h.sess, idx: idx, reasons: map[string]string{}, blocked: blocked,
			cfg: &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: "just test",
				Items: built(map[string]specLedger{dropped: {Hash: "whatever", Title: "1. Dropped"}})}}
		return r.pickWork(t.Context(), nil, 1)
	}
	if w := pickRemoval(map[string]string{dropped: "stuck"}); w.Item != "" {
		t.Errorf("work = %+v, want the removal blocked in this run skipped", w)
	}
	idx.questions = map[string][]specQuestion{dropped: {{ID: dropped, Question: "keep the helper?"}}}
	if w := pickRemoval(nil); w.Item != "" {
		t.Errorf("work = %+v, want the removal waiting on its question skipped", w)
	}
	idx.questions[dropped][0].Answer, idx.questions[dropped][0].Text = "keep it", "## x · keep the helper?\n\n**Answer:** keep it"
	if w := pickRemoval(nil); w.Item != dropped || w.Mode != specModeRemove || !strings.Contains(w.Answered, "keep it") {
		t.Errorf("work = %+v, want the answered removal picked with its answer", w)
	}
	idx.questions = nil

	const moved, now = "§98-moved#1-screen", "§03-shell#1-screen"
	items = built(map[string]specLedger{moved: {Hash: "moved", Title: idx.items[now].Title}})
	delete(items, now)
	w, r, _ = pick(items)
	if w.Item != "" {
		t.Errorf("work = %+v, want nothing: a moved section is not a removal", w)
	}
	if _, ok := r.cfg.Items[now]; !ok {
		t.Errorf("the record did not follow the moved section to %s", now)
	}
}

// A failed attempt is committed too; the next attempt, which only fixes the
// program, still counts the test the first one wrote.
func TestSpecSecondAttemptKeepsFirstAttemptsTest(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git")
	}
	a, sess := newTestAgent(t)
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: "false", LintCmd: "off", Items: map[string]specLedger{}}
	git := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("git", append([]string{"-C", sess.Cwd, "-c", "user.email=t@t", "-c", "user.name=t"}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v %s", args, err, out)
		}
	}
	write := func(rel, body string) {
		t.Helper()
		p := filepath.Join(sess.Cwd, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git("init", "-q")
	git("config", "user.email", "t@t")
	git("config", "user.name", "t")
	write("rust/src/lib.rs", "// v1\n")
	git("add", "-A")
	git("commit", "-q", "-m", "base")
	r := &specRun{a: a, sid: sess.ID, sess: sess, cfg: cfg, idx: idx, outAbs: filepath.Join(sess.Cwd, "rust"), reasons: map[string]string{}}

	write("rust/tests/switch.rs", "#[test]\nfn f0_1_switches() {}\n") // attempt 1: its test, and a red suite
	if done, _, err := r.finishRound(t.Context(), specWork{Item: "F0.1"}, nil, sess.toolMark(), time.Now()); err != nil || done {
		t.Fatalf("attempt 1: done=%v err=%v, want a failed attempt", done, err)
	}
	cfg.TestCmd = "true"
	write("rust/src/lib.rs", "// v2, the fix\n") // attempt 2 touches only the program
	done, _, err := r.finishRound(t.Context(), specWork{Item: "F0.1"}, nil, sess.toolMark(), time.Now())
	if err != nil || !done {
		t.Fatalf("attempt 2: done=%v err=%v reason=%q, want the first attempt's test to count", done, err, r.reasons["F0.1"])
	}
	if _, ok := cfg.Bases["F0.1"]; ok {
		t.Error("the base outlived the finished item")
	}
}

// A whole round: the executor adds a function nothing calls; the in-round check
// finds it before the round is judged, one more turn deletes it, and the item
// counts on its first attempt.
func TestSpecRoundCheckFixesBeforeJudging(t *testing.T) {
	write := func(id, path, content string) string {
		b, _ := json.Marshal(map[string]string{"path": path, "content": content})
		return sseToolCall(id, "write_file", string(b))
	}
	rig := newSpecLoopRig(t, map[string]string{
		"spec/01.md":        "# 01 Things\n\n### F0.1 Do it\n\nS1 do the thing.\n",
		"app/src/lib.rs":    "// the program\n",
		"app/tests/base.rs": "#[test]\nfn base_builds() {}\n",
	}, func(id string) bool { return id != "F0.1" },
		sseToolCall("p1", submitPlanToolName, `{"clear":true,"subtasks":[{"description":"build F0.1"}]}`),
		write("w1", "app/src/lib.rs", "pub fn helper() {}\n"),
		write("w2", "app/tests/f0_1.rs", "#[test]\nfn f0_1_does_the_thing() {}\n"),
		sseToolCall("r1", respondToolName, `{"message":"built"}`),
		sseToolCall("p2", submitPlanToolName, `{"clear":true,"subtasks":[{"description":"delete helper"}]}`),
		write("w3", "app/src/lib.rs", "// the program\n"),
		sseToolCall("r2", respondToolName, `{"message":"deleted"}`),
		sseToolCall("c1", submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[],"answer":"F0.1: DONE"}`),
	)
	said := rig.run(t)
	got := rig.ledger(t)
	if _, done := got.Items["F0.1"]; !done || got.Attempts["F0.1"] != 0 {
		t.Errorf("F0.1 done=%v attempts=%d, want done on its first attempt", done, got.Attempts["F0.1"])
	}
	if !strings.Contains(said, "Before this round counts: the program does not call these functions the round added: `helper`") {
		t.Errorf("the in-round check did not report helper:\n%s", said)
	}
	if rig.mock.callCount() != 8 {
		t.Errorf("model calls = %d, want 8 (the round, the fix turn, the completion check)", rig.mock.callCount())
	}
}

// specLoopRig is a git repo holding files, a ledger with every item done that
// done names, and a scripted main model; its summariser is a mock of its own,
// so each turn's background note does not take a scripted response.
type specLoopRig struct {
	h    *terminalHarness
	mock *mockLLM
}

func newSpecLoopRig(t *testing.T, files map[string]string, done func(id string) bool, responses ...string) *specLoopRig {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git")
	}
	h := newTerminalHarness(t)
	a, sess := h.agent, h.sess
	git := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("git", append([]string{"-C", sess.Cwd}, args...)...).CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v %s", args, err, out)
		}
	}
	files[".gitignore"] = ".codehalter/\n"
	for rel, body := range files {
		p := filepath.Join(sess.Cwd, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git("init", "-q")
	git("config", "user.email", "t@t")
	git("config", "user.name", "t")
	git("add", "-A")
	git("commit", "-q", "-m", "base")
	idx, err := scanSpec(filepath.Join(sess.Cwd, "spec"), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &specConfig{SpecDir: "spec", OutDir: "app", TestCmd: "true", LintCmd: "off", RefactorEvery: -1,
		Items: map[string]specLedger{}, Final: &specFinal{Items: len(idx.order)}}
	for _, id := range idx.order {
		if done(id) {
			cfg.Items[id] = specLedger{Hash: specItemHash(idx, id), Title: idx.items[id].Title}
		}
	}
	if err := saveSpecConfig(sess.Cwd, cfg); err != nil {
		t.Fatal(err)
	}
	mock := newMockLLM(t, responses...)
	t.Cleanup(mock.Close)
	// The pre-turn checks probe the LLM from the settings file: point it at the mock
	// and mark the probe done, so nothing reaches a real server.
	t.Setenv("HOME", t.TempDir())
	var notes []string
	for range 20 {
		notes = append(notes, sseText("turn note"))
	}
	summariser := newMockLLM(t, notes...)
	t.Cleanup(summariser.Close)
	settings := fmt.Sprintf("[[llm]]\nserver = %q\nmodel = \"m\"\n\n[[llm]]\nserver = %q\nmodel = \"s\"\npurpose = \"summary\"\n", mock.ts.URL, summariser.ts.URL)
	if err := os.WriteFile(filepath.Join(sess.Cwd, ".codehalter", "settings.toml"), []byte(settings), 0o644); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadSettings(sess.Cwd)
	if err != nil {
		t.Fatal(err)
	}
	a.connProbe = map[string]probeResult{mock.ts.URL + "\x00m": {Reachable: true, ModelKnown: true, ModelLoaded: true},
		summariser.ts.URL + "\x00s": {Reachable: true, ModelKnown: true, ModelLoaded: true}}
	a.cfgMu.Lock()
	a.setSettings(loaded) // builds the per-server slots the summary is routed by
	a.cfgMu.Unlock()
	a.mainSlotTokens.Store(100_000)
	sess.llmHash = hashSettingsFiles(sess.Cwd)
	a.tools.add()
	return &specLoopRig{h: h, mock: mock}
}

// run is one /spec; it returns everything the loop said.
func (g *specLoopRig) run(t *testing.T) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if _, err := g.h.agent.runSpec(ctx, g.h.sess.ID, g.h.sess, "", nil); err != nil {
		t.Fatal(err)
	}
	// Updates are recorded by the harness's reader, in order: once this marker is
	// in, so is everything the loop said before it.
	const marker = "\x00end of run"
	g.h.agent.say(ctx, g.h.sess.ID, marker)
	for deadline := time.Now().Add(5 * time.Second); ; {
		n := g.h.updatesOfKind(KindAgentMessage)
		if c, _ := n[len(n)-1]["content"].(map[string]any); c != nil && c["text"] == marker {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the loop's messages did not all arrive")
		}
		time.Sleep(5 * time.Millisecond)
	}
	var said strings.Builder
	for _, u := range g.h.updatesOfKind(KindAgentMessage) {
		if c, _ := u["content"].(map[string]any); c != nil {
			said.WriteString(fmt.Sprint(c["text"]))
		}
	}
	return said.String()
}

func (g *specLoopRig) ledger(t *testing.T) *specConfig {
	t.Helper()
	cfg, err := loadSpecConfig(g.h.sess.Cwd)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

// request is what the model was sent on call i, as JSON text.
func (g *specLoopRig) request(i int) string {
	b, _ := json.Marshal(g.mock.request(i))
	return string(b)
}

func TestSpecCompletionCheck(t *testing.T) {
	write := func(id, path, content string) string {
		b, _ := json.Marshal(map[string]string{"path": path, "content": content})
		return sseToolCall(id, "write_file", string(b))
	}
	rig := newSpecLoopRig(t, map[string]string{
		// F0.3 has no picture and still gets checked; the one it links to is F0.1's.
		"spec/01.md": "# 01 Screens\n\n### F0.1 Prepare\n\n![prepare](img/prepare.png)\n\nSources on the left, User Context on the right.\n\n" +
			"### F0.2 Cut\n\n![cut](img/cut.png) ![flow](img/flow.svg)\n\nThe toolbar on top.\n\n### F0.3 Help\n\nSee [Prepare](#f01-prepare).\n",
		"app/src/lib.rs":       "// the program\n",
		"app/tests/screens.rs": "#[test]\nfn f0_1_prepare() {}\n#[test]\nfn f0_2_cut() {}\n#[test]\nfn f0_3_help() {}\n",
	}, func(string) bool { return true },
		// Run 1: one round checks the file's three items; the planner looks and answers itself.
		sseToolCall("c1", submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[],"answer":"F0.1: MISSING sources list: the spec has it on the left, the program on the right\n- F0.2: DONE\n`+"`F0.3` \u2014 DONE"+`"}`),
		// The rebuild of F0.1.
		sseToolCall("p1", submitPlanToolName, `{"clear":true,"subtasks":[{"description":"swap the panels"}]}`),
		write("w1", "app/src/lib.rs", "// sources left, context right\n"),
		write("w2", "app/tests/screens.rs", "#[test]\nfn f0_1_prepare_sources_left() {}\n#[test]\nfn f0_2_cut() {}\n#[test]\nfn f0_3_help() {}\n"),
		sseToolCall("r1", respondToolName, `{"message":"swapped"}`),
		// Run 2 checks only the rebuilt item.
		sseToolCall("c2", submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[],"answer":"F0.1: DONE"}`),
	)
	said := rig.run(t)
	if rig.mock.callCount() != 5 {
		t.Fatalf("run 1 model calls = %d, want 5 (one check round, then the rebuild)\n%s", rig.mock.callCount(), said)
	}
	check := rig.request(0)
	for _, want := range []string{"`F0.1` Prepare (`spec/01.md:3`); the spec pictures it: `spec/img/prepare.png`", "`F0.2` Cut", "`spec/img/cut.png`", "`F0.3` Help (`spec/01.md:15`)"} {
		if !strings.Contains(check, want) {
			t.Errorf("the check round lacks %q:\n%s", want, check)
		}
	}
	if strings.Contains(check, "flow.svg`") || strings.Contains(check, "Help (`spec/01.md:15`); the spec pictures") {
		t.Errorf("an svg drawing or a linked section's picture was given as the item's own:\n%s", check)
	}
	if round := rig.request(1); !strings.Contains(round, "sources list: the spec has it on the left, the program on the right") {
		t.Errorf("the rebuild does not carry what is missing:\n%s", round)
	}
	got := rig.ledger(t)
	for _, id := range []string{"F0.2", "F0.3"} {
		if got.Items[id].Checked == "" {
			t.Errorf("%s was found done but is not marked checked", id)
		}
	}
	if !got.done("F0.1") || got.Items["F0.1"].Checked != "" || got.Redo["F0.1"] != "" {
		t.Errorf("F0.1 = %+v redo=%q, want rebuilt, done and not checked yet", got.Items["F0.1"], got.Redo["F0.1"])
	}

	said = rig.run(t)
	if rig.mock.callCount() != 6 {
		t.Fatalf("model calls after run 2 = %d, want 6 (one check of F0.1 alone)\n%s", rig.mock.callCount(), said)
	}
	// This run's check prompt, not the history before it.
	req := rig.request(5)
	again := req[strings.LastIndex(req, "# Completion check"):]
	if !strings.Contains(again, "`F0.1` Prepare") || strings.Contains(again, "`F0.2` Cut") {
		t.Errorf("run 2 should check only the rebuilt item:\n%s", again)
	}
	if rig.ledger(t).Items["F0.1"].Checked == "" {
		t.Error("F0.1 is not marked checked after run 2")
	}
}

// A question the spec cannot ground gets one corrective; the grounded one goes to
// the spec's QUESTIONS.md, the item waits there (no block in the ledger), and the
// answer written there reaches the next run's round and the item's fingerprint.
func TestSpecQuestionWaitsInTheSpecForItsAnswer(t *testing.T) {
	write := func(id, path, content string) string {
		b, _ := json.Marshal(map[string]string{"path": path, "content": content})
		return sseToolCall(id, "write_file", string(b))
	}
	good := `{"clear":false,"subtasks":[],"question":"In which order do the play buttons sit?",` +
		`"spec_quote":"The toolbar groups left to right: recording, cut.",` +
		`"options":[{"choice":"Side by side","example":"one row: [▶ recording] [▶✂ cut]"},{"choice":"Stacked","example":"▶ recording above ▶✂ cut"}]}`
	rig := newSpecLoopRig(t, map[string]string{
		"spec/01.md":        "# 01 Cut\n\n### F2.2 Play buttons\n\nThe toolbar groups left to right:\nrecording, cut.\n",
		"app/src/lib.rs":    "// the program\n",
		"app/tests/base.rs": "#[test]\nfn base_builds() {}\n",
	}, func(id string) bool { return id != "F2.2" },
		// Run 1: a question without the spec text it rests on, then a grounded one, then the final pass.
		sseToolCall("q1", submitPlanToolName, `{"clear":false,"subtasks":[],"question":"side by side or stacked?","choices":["side by side","stacked"]}`),
		sseToolCall("q2", submitPlanToolName, good),
		sseToolCall("f1", submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[],"answer":"README written"}`),
		// Run 2: the answered item, its completion check, then the final pass again.
		sseToolCall("p1", submitPlanToolName, `{"clear":true,"subtasks":[{"description":"side by side"}]}`),
		write("w1", "app/tests/f2_2.rs", "#[test]\nfn f2_2_play_buttons_side_by_side() {}\n"),
		sseToolCall("r1", respondToolName, `{"message":"built"}`),
		sseToolCall("c1", submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[],"answer":"F2.2: DONE"}`),
		sseToolCall("f2", submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[],"answer":"README updated"}`),
	)
	said := rig.run(t)
	if rig.mock.callCount() != 3 {
		t.Fatalf("run 1 model calls = %d, want 3\n%s", rig.mock.callCount(), said)
	}
	if retry := rig.request(1); !strings.Contains(retry, "Your question was not recorded: it has no `spec_quote`") {
		t.Errorf("the corrective did not say what the question lacked:\n%s", retry)
	}
	qpath := filepath.Join(rig.h.sess.Cwd, "spec", specQuestionsFile)
	qs, err := os.ReadFile(qpath)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"## F2.2 · In which order do the play buttons sit?", "`01.md:5`",
		"> The toolbar groups left to right: recording, cut.", "1. Side by side. Example: one row: [▶ recording] [▶✂ cut]",
		"2. Stacked. Example: ▶ recording above ▶✂ cut", "**Answer:** \n"} {
		if !strings.Contains(string(qs), want) {
			t.Errorf("QUESTIONS.md lacks %q:\n%s", want, qs)
		}
	}
	if !strings.Contains(said, "❓ F2.2 waits on your answer") || strings.Contains(said, "⛔") {
		t.Errorf("the question should wait, not block:\n%s", said)
	}
	if raw, _ := os.ReadFile(specConfigPath(rig.h.sess.Cwd)); strings.Contains(string(raw), "blocked]]") {
		t.Errorf("the ledger holds a block:\n%s", raw)
	}

	answered := strings.Replace(string(qs), "**Answer:** \n", "**Answer:** 1, side by side as the toolbar line says.\n", 1)
	if err := os.WriteFile(qpath, []byte(answered), 0o644); err != nil {
		t.Fatal(err)
	}
	said = rig.run(t)
	if rig.mock.callCount() != 8 {
		t.Fatalf("model calls after run 2 = %d, want 8\n%s", rig.mock.callCount(), said)
	}
	if round := rig.request(3); !strings.Contains(round, "Answered in the spec's QUESTIONS.md") || !strings.Contains(round, "1, side by side as the toolbar line says.") {
		t.Errorf("the round lacks the answer:\n%s", round)
	}
	idx, err := scanSpec(filepath.Join(rig.h.sess.Cwd, "spec"), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := rig.ledger(t); !got.done("F2.2") || got.Items["F2.2"].Hash != specItemHash(idx, "F2.2") {
		t.Errorf("F2.2 = %+v, want done with the answer in its fingerprint", got.Items["F2.2"])
	}
}

// An item that does not pass its attempts becomes a question codehalter words
// itself; the next run leaves it waiting and, with only questions left, stops.
func TestSpecStuckItemBecomesAQuestion(t *testing.T) {
	answer := func(id string) string {
		return sseToolCall(id, submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[],"answer":"looked, changed nothing"}`)
	}
	rig := newSpecLoopRig(t, map[string]string{
		"spec/01.md":        "# 01 Things\n\n### F0.1 Do it\n\nS1 do the thing.\n",
		"app/src/lib.rs":    "// the program\n",
		"app/tests/base.rs": "#[test]\nfn base_builds() {}\n",
	}, func(id string) bool { return id != "F0.1" },
		// Two attempts, each a round and its in-round check, that write no test; then the final pass.
		answer("a1"), answer("a2"), answer("a3"), answer("a4"), answer("f1"),
	)
	said := rig.run(t)
	if rig.mock.callCount() != 5 {
		t.Fatalf("run 1 model calls = %d, want 5\n%s", rig.mock.callCount(), said)
	}
	qs, err := os.ReadFile(filepath.Join(rig.h.sess.Cwd, "spec", specQuestionsFile))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"## F0.1 · This item did not pass its attempts. How should the rounds go on?", "`01.md:3`", "> F0.1 Do it",
		"What stopped the last attempt:\n\n```\nno test this round added or changed is named after F0.1", "1. Try again as the spec says.", "`spec/01.md`", "**Answer:** \n"} {
		if !strings.Contains(string(qs), want) {
			t.Errorf("QUESTIONS.md lacks %q:\n%s", want, qs)
		}
	}
	if !strings.Contains(said, "❓ F0.1 did not pass its attempts") {
		t.Errorf("the stuck item was not reported as a question:\n%s", said)
	}

	said = rig.run(t)
	if rig.mock.callCount() != 5 {
		t.Errorf("run 2 spent %d model call(s) on an item that waits on its question\n%s", rig.mock.callCount()-5, said)
	}
	if !strings.Contains(said, "1 question(s) wait on your answer in `spec/QUESTIONS.md`") {
		t.Errorf("run 2 did not stop on the waiting question:\n%s", said)
	}
}

// A server that refuses every request is no failure of the item: the loop pauses,
// counts no attempt and asks nothing. Three items once became questions about
// Halogen's 64-picture limit, and the loop stopped as if they were stuck.
func TestSpecPausesWhenTheModelCallFails(t *testing.T) {
	refused := `data: {"error":{"message":"the engine refused this request: IMG count outside 0..64"}}` + "\n\n"
	var resp []string
	for range 6 {
		resp = append(resp, refused)
	}
	rig := newSpecLoopRig(t, map[string]string{
		"spec/01.md":        "# 01 Things\n\n### F0.1 Do it\n\nS1 do the thing.\n",
		"app/src/lib.rs":    "// the program\n",
		"app/tests/base.rs": "#[test]\nfn base_builds() {}\n",
	}, func(id string) bool { return id != "F0.1" }, resp...)
	said := rig.run(t)
	if !strings.Contains(said, "/spec paused") || !strings.Contains(said, "IMG count outside 0..64") {
		t.Errorf("the loop did not pause on the refused call:\n%s", said)
	}
	if strings.Contains(said, "❓") || strings.Contains(said, "not done yet") {
		t.Errorf("a refused call was counted against the item:\n%s", said)
	}
	if _, err := os.Stat(filepath.Join(rig.h.sess.Cwd, "spec", specQuestionsFile)); err == nil {
		t.Error("a refused call became a question")
	}
	if got := rig.ledger(t); got.Attempts["F0.1"] != 0 {
		t.Errorf("attempts = %d, want none counted", got.Attempts["F0.1"])
	}
}

// A stop is acknowledged at once, before the step in flight ends, and the loop's
// last word is that it stopped, not that the editor aborted a request.
func TestSpecStopIsAcknowledged(t *testing.T) {
	rig := newSpecLoopRig(t, map[string]string{
		"spec/01.md":        "# 01 Things\n\n### F0.1 Do it\n\nS1 do the thing.\n",
		"app/src/lib.rs":    "// the program\n",
		"app/tests/base.rs": "#[test]\nfn base_builds() {}\n",
	}, func(id string) bool { return id != "F0.1" })
	ctx, cancel := context.WithCancel(t.Context())
	cancel() // the stop arrives before the first round
	if _, err := rig.h.agent.runSpec(ctx, rig.h.sess.ID, rig.h.sess, "", nil); err != nil {
		t.Fatal(err)
	}
	var said string
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		said = ""
		for _, u := range rig.h.updatesOfKind(KindAgentMessage) {
			if c, _ := u["content"].(map[string]any); c != nil {
				said += fmt.Sprint(c["text"])
			}
		}
		if strings.Contains(said, "Stopping /spec") && strings.Contains(said, "/spec stopped.") {
			break
		}
	}
	if !strings.Contains(said, "Stopping /spec") || !strings.Contains(said, "/spec stopped.") || strings.Contains(said, "aborted") {
		t.Errorf("the stop was not acknowledged plainly:\n%s", said)
	}
}
