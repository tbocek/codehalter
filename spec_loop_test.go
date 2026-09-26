package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestSpecDecide pins an item's fate after a round: done only when a test names
// it AND the suite passes, one retry otherwise, then blocked. A question blocks
// at once.
func TestSpecDecide(t *testing.T) {
	cfg := &specConfig{}
	if done, block, _ := specDecide(cfg, "F0.1", specRoundResult{Covered: true, TestsPass: true}); !done || block {
		t.Errorf("covered and passing: done=%v block=%v, want done", done, block)
	}

	// Covered but the suite fails: a broken earlier item counts against the round.
	done, block, reason := specDecide(cfg, "F0.2", specRoundResult{Covered: true, TestsPass: false, TestTail: "test f0_1 failed"})
	if done || block || !strings.Contains(reason, "test f0_1 failed") {
		t.Errorf("first failure: done=%v block=%v reason=%q, want a retry carrying the test output", done, block, reason)
	}
	done, block, reason = specDecide(cfg, "F0.2", specRoundResult{TestsPass: true})
	if done || !block || !strings.Contains(reason, "f0_2") {
		t.Errorf("second failure: done=%v block=%v reason=%q, want blocked, naming the expected test token", done, block, reason)
	}

	if _, block, _ := specDecide(cfg, "F0.3", specRoundResult{Question: "keep or drop?"}); !block {
		t.Error("a planner question did not block the item")
	}
	_, _, reason = specDecide(&specConfig{}, specSetupID, specRoundResult{Mode: specModeSetup, TestsPass: true})
	if !strings.Contains(reason, "no test source") {
		t.Errorf("setup without tests: reason %q", reason)
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

// TestSpecFence: while a loop runs, the file tools refuse the spec dir and
// nothing else.
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

// TestSpecRoundPrompt renders both round prompts from the embedded defaults and
// checks that every placeholder was filled and the item's facts are in it.
func TestSpecRoundPrompt(t *testing.T) {
	a, s := newTestAgent(t)
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", Target: "use gtk4-rs libadwaita"}

	prompt, head := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: "F0.1", Reason: "no test names f0_1"}, map[string]string{"§01-files#1-layout": "tests/x.rs"}, "just test")
	for _, want := range []string{"**F0.1**", "`f0_1`", "just test", "use gtk4-rs libadwaita", "rust/", "S1 Switch.",
		"did not count", "no test names f0_1", "spec/00-principles.md", "1 of"} {
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

	setup, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: specSetupID, Mode: specModeSetup}, nil, "")
	if strings.Contains(setup, "{{") || !strings.Contains(setup, "use gtk4-rs libadwaita") || !strings.Contains(setup, "`rust/`") {
		t.Errorf("setup prompt:\n%s", setup)
	}

	answered, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: "F0.2", Question: "keep or drop?", Answer: "keep it"}, nil, "just test")
	if !strings.Contains(answered, "keep or drop?") || !strings.Contains(answered, "keep it") {
		t.Errorf("answered prompt lacks the question and answer:\n%s", answered)
	}
}

func TestDedupeFixes(t *testing.T) {
	got := dedupeFixes([]fixProblem{{desc: "a"}, {desc: "b"}, {desc: "a"}})
	if len(got) != 2 || got[0].desc != "a" || got[1].desc != "b" {
		t.Errorf("= %+v", got)
	}
}

func TestSpecModuleNames(t *testing.T) {
	idx := &specIndex{docs: []specDoc{{rel: "README.md"}, {rel: "05-cut.md"}, {rel: "09-llm-and-tools.md"}, {rel: "inventory/cut.md"}}}
	got := specModuleNames(idx)
	want := []string{"cut", "llm_and_tools", "llm"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("= %v, want %v", got, want)
	}
}

// TestSpecIgnoredProbes reproduces the gitignore that motivated the check: the
// old code's unanchored output-folder rules, which also hide the rewrite's
// modules and tests of the same name. Each offending rule is named once; the
// anchored version of the same file is clean.
func TestSpecIgnoredProbes(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git")
	}
	cwd := t.TempDir()
	if out, err := exec.Command("git", "-C", cwd, "init", "-q").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v %s", err, out)
	}
	idx := &specIndex{docs: []specDoc{{rel: "05-cut.md"}, {rel: "07-narrate.md"}}}
	gi := filepath.Join(cwd, ".gitignore")

	os.WriteFile(gi, []byte("cut/\n*.json\ntest*\nout/\n"), 0o644)
	got := specIgnoredProbes(t.Context(), cwd, "rust", idx)
	joined := strings.Join(got, "\n")
	for _, rule := range []string{"`cut/`", "`*.json`", "`test*`"} {
		if strings.Count(joined, rule) != 1 {
			t.Errorf("rule %s reported %d times, want once:\n%s", rule, strings.Count(joined, rule), joined)
		}
	}
	if strings.Contains(joined, "`out/`") {
		t.Errorf("a rule that hides nothing under rust/ was reported:\n%s", joined)
	}

	os.WriteFile(gi, []byte("/cut/\n/*.json\n/test*\n/out/\n"), 0o644)
	if got := specIgnoredProbes(t.Context(), cwd, "rust", idx); len(got) != 0 {
		t.Errorf("anchored rules still reported: %v", got)
	}
}

// TestSpecDecideChangeAndRemove pins the two round kinds whose done rule is not
// "a test names it and the suite passes".
//
// A changed item is still covered by the test written for the OLD spec text, so
// coverage alone would declare it done before the model touched anything: the
// commit is the evidence. A removal is the inverse: it is done when no test
// names the item any more.
func TestSpecDecideChangeAndRemove(t *testing.T) {
	cfg := &specConfig{}
	done, _, reason := specDecide(cfg, "F0.1", specRoundResult{Mode: specModeChange, Covered: true, TestsPass: true})
	if done || !strings.Contains(reason, "no code did") {
		t.Errorf("a change round that wrote nothing: done=%v reason=%q", done, reason)
	}
	if done, _, _ := specDecide(cfg, "F0.1", specRoundResult{Mode: specModeChange, Covered: true, TestsPass: true, Committed: true}); !done {
		t.Error("a change round that committed should be done")
	}

	done, _, reason = specDecide(cfg, "F0.2", specRoundResult{Mode: specModeRemove, TestsPass: true, Covered: true, Committed: true})
	if done || !strings.Contains(reason, "still names") {
		t.Errorf("a removal with the test still in place: done=%v reason=%q", done, reason)
	}
	if done, _, _ := specDecide(cfg, "F0.2", specRoundResult{Mode: specModeRemove, TestsPass: true, Committed: true}); !done {
		t.Error("a removal whose test is gone and whose suite passes should be done")
	}
	// The suite must still pass: deleting an item cannot take the build with it.
	if done, _, reason := specDecide(cfg, "F0.3", specRoundResult{Mode: specModeRemove, TestTail: "3 failed"}); done || !strings.Contains(reason, "did not pass") {
		t.Errorf("a removal that broke the suite: done=%v reason=%q", done, reason)
	}
}

// TestSpecFinalPrompt: the round after the last item gets the whole picture:
// target, every spec file, the test command, the counts and the blocked list,
// with no placeholder left.
func TestSpecFinalPrompt(t *testing.T) {
	a, s := newTestAgent(t)
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", Target: "use gtk4-rs libadwaita",
		Items:   map[string]specLedger{"F0.1": {}, "F0.2": {}},
		Blocked: []specBlock{{ID: "§12-decisions#x", Reason: "the planner asked a question"}}}
	prompt := a.specFinalPrompt(s.ID, cfg, idx, "just test")
	for _, want := range []string{"use gtk4-rs libadwaita", "`rust/README.md`", "just test", "2 items, 1 blocked",
		"§12-decisions#x: the planner asked a question", "spec/00-principles.md", "`--help`", "snapshot"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("final prompt lacks %q", want)
		}
	}
	if strings.Contains(prompt, "{{") {
		t.Errorf("unfilled placeholder:\n%s", prompt)
	}
	if !strings.Contains(a.specFinalPrompt(s.ID, &specConfig{SpecDir: "spec", OutDir: "rust"}, idx, "cargo test"), "none") {
		t.Error("no blocked items must render as none")
	}
}

// TestRunSpecTestsTailCarriesExitStatus: a test command that dies before
// printing anything still tells why; an empty "end of its output" helped
// nobody.
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

// TestSpecRoundOwnGreenRun: the round's last plain run of the test command,
// exit 0 from the terminal, in the output directory, with nothing written
// since, is the check; anything less means codehalter runs the suite itself.
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
	ran := func(cmd, output string) ToolUse {
		return ToolUse{Name: "run_command", Input: `{"command":` + strconv.Quote(cmd) + `}`, Output: output, StartedAt: time.Now().Add(time.Second), DurationMs: 10}
	}
	edit := ToolUse{Name: "edit_file", Input: `{"path":"rust/src/lib.rs"}`}

	if !r.roundRanGreen([]ToolUse{edit, ran("cd rust && just test > /tmp/t.log 2>&1", "exit 0\n\n")}, "just test") {
		t.Error("a plain green run after the last edit did not count")
	}
	if !r.roundRanGreen([]ToolUse{ran("cd "+out+" && just test", "exit 0\n\nok")}, "just test") {
		t.Error("an absolute cd to the output dir did not count")
	}
	for name, uses := range map[string][]ToolUse{
		"wrapped, exit is the echo's": {ran("cd rust && (just test > /tmp/t.log 2>&1; echo exit=$? >> /tmp/t.log)", "exit 0\n")},
		"red":                         {ran("cd rust && just test", "exit 101\n")},
		"edited after the run":        {ran("cd rust && just test", "exit 0\n"), edit},
		"wrong directory":             {ran("just test", "exit 0\n")},
		"handed over, no exit yet":    {ran("cd rust && just test", "still running after 2m")},
	} {
		if r.roundRanGreen(uses, "just test") {
			t.Errorf("%s: counted as green", name)
		}
	}
	// A file written after the run, by whatever means.
	uses := []ToolUse{ran("cd rust && just test", "exit 0\n")}
	uses[0].StartedAt = time.Now().Add(-time.Minute)
	if r.roundRanGreen(uses, "just test") {
		t.Error("a source file newer than the run did not force a re-run")
	}
}

// TestSpecUIChangeNeedsALook: a round that edits a file importing a UI
// toolkit without one screenshot call is not done, with the reason; a round
// that looked, or one that touched no UI file, is unaffected.
func TestSpecUIChangeNeedsALook(t *testing.T) {
	_, sess := newTestAgent(t)
	ui := filepath.Join(sess.Cwd, "rust", "src", "ui.rs")
	plain := filepath.Join(sess.Cwd, "rust", "src", "rules.rs")
	if err := os.MkdirAll(filepath.Dir(ui), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ui, []byte("use gtk::prelude::*;\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(plain, []byte("pub fn floor() -> f64 { 0.1 }\n"), 0o644); err != nil {
		t.Fatal(err)
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
	done, block, reason := specDecide(&specConfig{}, "F1.1", specRoundResult{Covered: true, TestsPass: true, UIUnseen: []string{"rust/src/ui.rs"}})
	if done || block || !strings.Contains(reason, "never looked at it") {
		t.Errorf("decide = %v %v %q, want one more round with the reason", done, block, reason)
	}
}

// TestSpecAuditPrompt: a bare /spec redo asks the model which finished items
// fall short; the prompt names the spec files and the way to answer.
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
