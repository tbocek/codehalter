package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// The verdict on one round: done only when a test names the item, the suite
// passes and no gate holds it back; a change, redo or removal needs its own proof.
func TestSpecDecide(t *testing.T) {
	for _, c := range []struct {
		name      string
		item      string
		attempts  int // attempts already spent on the item
		r         specRoundResult
		done      bool
		block     bool
		reasonHas string
	}{
		{"covered and green", "F0.1", 0, specRoundResult{Covered: true, TestsPass: true}, true, false, ""},
		{"a red suite is a retry with its output", "F0.2", 0, specRoundResult{Covered: true, TestTail: "test f0_1 failed"}, false, false, "test f0_1 failed"},
		{"the second failure blocks, naming the test token", "F0.2", 1, specRoundResult{TestsPass: true}, false, true, "f0_2"},
		{"a planner question blocks", "F0.3", 0, specRoundResult{Question: "keep or drop?"}, false, true, ""},
		{"setup without a test source", specSetupID, 0, specRoundResult{Mode: specModeSetup, TestsPass: true}, false, false, "no test source"},
		{"gate: unreachable", "F0.1", 0, specRoundResult{Covered: true, TestsPass: true, Unreachable: []string{"`run` (src/a.rs:3): only tests call it"}}, false, false, "only tests call it"},
		{"gate: lint", "F0.1", 0, specRoundResult{Covered: true, TestsPass: true, Lint: []string{"src/a.rs:3: unused variable"}}, false, false, "unused variable"},
		{"gate: oversize", "F0.1", 0, specRoundResult{Covered: true, TestsPass: true, Oversize: []string{"`src/ui.rs` is 12000 lines, over the 1500-line budget"}}, false, false, "1500-line budget"},
		{"gate: stand-in", "F0.1", 0, specRoundResult{Covered: true, TestsPass: true, StandIns: []string{"`reply_for_test` (src/a.rs:3)"}}, false, false, "is not done"},
		{"refactor that shrank nothing", specRefactorID, 0, specRoundResult{Mode: specModeRefactor, TestsPass: true, Committed: true, Debt: "a -> b"}, false, false, "no measured progress"},
		{"refactor that shrank the debt", specRefactorID, 0, specRoundResult{Mode: specModeRefactor, TestsPass: true, Committed: true, Improved: true}, true, false, ""},
		{"change that wrote nothing", "F0.1", 0, specRoundResult{Mode: specModeChange, Covered: true, TestsPass: true}, false, false, "no code did"},
		{"change that committed", "F0.1", 0, specRoundResult{Mode: specModeChange, Covered: true, TestsPass: true, Committed: true}, true, false, ""},
		{"redo that wrote nothing", "F0.1", 0, specRoundResult{Redo: true, Covered: true, TestsPass: true}, false, false, "/spec redo"},
		{"redo that committed", "F0.1", 0, specRoundResult{Redo: true, Covered: true, TestsPass: true, Committed: true}, true, false, ""},
		{"removal with its test still there", "F0.2", 0, specRoundResult{Mode: specModeRemove, TestsPass: true, Covered: true, Committed: true}, false, false, "still names"},
		{"removal done", "F0.2", 0, specRoundResult{Mode: specModeRemove, TestsPass: true, Committed: true}, true, false, ""},
		{"removal that broke the suite", "F0.3", 0, specRoundResult{Mode: specModeRemove, TestTail: "3 failed"}, false, false, "did not pass"},
	} {
		cfg := &specConfig{Attempts: map[string]int{c.item: c.attempts}}
		done, block, reason, _ := specDecide(cfg, c.item, c.r, "")
		if done != c.done || block != c.block || !strings.Contains(reason, c.reasonHas) {
			t.Errorf("%s: done=%v block=%v reason=%q, want done=%v block=%v reason with %q", c.name, done, block, reason, c.done, c.block, c.reasonHas)
		}
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

	// llm/ hides the module named after the chapter's first word.
	writeTree(t, cwd, map[string]string{".gitignore": "cut/\n*.json\ntest*\nout/\nllm/\n"})
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

	writeTree(t, cwd, map[string]string{".gitignore": "/cut/\n/*.json\n/test*\n/out/\n/llm/\n"})
	if got := specIgnoredProbes(t.Context(), cwd, "rust", idx); len(got) != 0 {
		t.Errorf("anchored rules still reported: %v", got)
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
		stuck + ": the build stayed red", `F0.3: waits on the user's answer to "keep or drop?" in spec/QUESTIONS.md`, "spec/00-principles.md"} {
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
	for _, want := range []string{"spec/00-principles.md", "just test", "gtk4", "`F0.1`"} {
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
	write := func(rel, body string) {
		t.Helper()
		writeTree(t, sess.Cwd, map[string]string{rel: body})
	}
	write("rust/src/lib.rs", "// v1\n")
	gitInit(t, sess.Cwd)
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
	write := sseWriteFile
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
	h := newTerminalHarness(t)
	a, sess := h.agent, h.sess
	files[".gitignore"] = ".codehalter/\n"
	writeTree(t, sess.Cwd, files)
	gitInit(t, sess.Cwd)
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
	return g.said(t)
}

func (g *specLoopRig) ledger(t *testing.T) *specConfig {
	t.Helper()
	cfg, err := loadSpecConfig(g.h.sess.Cwd)
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

// said is everything the loop said so far, once the harness has it all.
func (g *specLoopRig) said(t *testing.T) string {
	t.Helper()
	const marker = "\x00said so far"
	g.h.agent.say(context.Background(), g.h.sess.ID, marker)
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(5 * time.Millisecond) {
		n := g.h.updatesOfKind(KindAgentMessage)
		if c, _ := n[len(n)-1]["content"].(map[string]any); c != nil && c["text"] == marker {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the loop's messages did not all arrive")
		}
	}
	var said strings.Builder
	for _, u := range g.h.updatesOfKind(KindAgentMessage) {
		if c, _ := u["content"].(map[string]any); c != nil {
			said.WriteString(fmt.Sprint(c["text"]))
		}
	}
	return said.String()
}

// request is what the model was sent on call i, as JSON text.
func (g *specLoopRig) request(i int) string {
	b, _ := json.Marshal(g.mock.request(i))
	return string(b)
}

func TestSpecCompletionCheck(t *testing.T) {
	write := sseWriteFile
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
	write := sseWriteFile
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
		t.Errorf("F2.2 = %+v, want done, its fingerprint the spec's as it now stands", got.Items["F2.2"])
	}
	if after, _ := os.ReadFile(qpath); strings.Contains(string(after), "## F2.2") {
		t.Errorf("F2.2 was built with its answer, but its question is still in the file:\n%s", after)
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

// Zed sends a message typed during a turn as a cancel first and the text only
// after the turn ended. So the cancel ends Zed's request at once and the loop
// finishes its round before it stops; `/spec abort` meanwhile stops it at once.
func TestSpecStopFinishesTheRoundAndAbortDoesNot(t *testing.T) {
	for _, abort := range []bool{false, true} {
		t.Run(map[bool]string{false: "stop", true: "abort"}[abort], func(t *testing.T) {
			entered, unblock := make(chan struct{}), make(chan struct{})
			var once sync.Once
			answer := func(id string) string {
				return sseToolCall(id, submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[],"answer":"looked"}`)
			}
			rig := newSpecLoopRig(t, map[string]string{
				"spec/01.md":        "# 01 Things\n\n### F0.1 Do it\n\nS1 do the thing.\n",
				"app/src/lib.rs":    "// the program\n",
				"app/tests/base.rs": "#[test]\nfn base_builds() {}\n",
			}, func(id string) bool { return id != "F0.1" },
				sseToolCall("p1", submitPlanToolName, `{"clear":true,"subtasks":[{"description":"do it"}]}`),
				sseToolCall("w1", "wait_tool", `{}`),
				sseToolCall("r1", respondToolName, `{"message":"did it"}`),
				answer("c1"), answer("c2"), answer("c3"))
			rig.h.agent.tools.add(Tool{Def: map[string]any{"type": "function", "function": map[string]any{"name": "wait_tool", "parameters": map[string]any{"type": "object"}}},
				Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
					once.Do(func() { close(entered) })
					select {
					case <-unblock:
					case <-ctx.Done():
					}
					return "waited", false
				}})
			released := make(chan struct{})
			turnCtx, cancel := context.WithCancel(t.Context())
			type out struct {
				detached bool
				err      error
			}
			ret := make(chan out, 1)
			go func() {
				_, err, d := rig.h.agent.runSpecTurn(turnCtx, rig.h.sess.ID, rig.h.sess, "", nil, func() { close(released) })
				ret <- out{d, err}
			}()
			select {
			case <-entered:
			case <-time.After(10 * time.Second):
				t.Fatal("the round never reached its step")
			}
			cancel() // Zed's cancel, sent the moment the message is entered
			select {
			case o := <-ret:
				if !o.detached || o.err != nil {
					t.Fatalf("Zed's request: detached=%v err=%v, want ended at once with the loop going on", o.detached, o.err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("Zed's request did not end on the cancel")
			}
			if abort {
				if !rig.h.sess.abortSpec() {
					t.Fatal("no running loop to abort")
				}
			} else {
				close(unblock) // the round's step ends by itself
			}
			select {
			case <-released:
			case <-time.After(20 * time.Second):
				t.Fatal("the loop never ended and gave the turn back")
			}
			said := rig.said(t)
			if !strings.Contains(said, "stops after the round in flight") || !strings.Contains(said, "/spec abort") {
				t.Errorf("the cancel was not explained:\n%s", said)
			}
			want := map[bool]string{false: "/spec stopped** as asked", true: "/spec aborted."}[abort]
			if !strings.Contains(said, want) {
				t.Errorf("want %q:\n%s", want, said)
			}
			if !abort && !strings.Contains(said, "🧪") {
				t.Errorf("the stopped loop skipped the round's own check:\n%s", said)
			}
		})
	}
}

// The check's answer names an item however the model writes it: an id with a colon
// of its own, a long section id shortened, an id followed by its title. Six items
// once stayed unread this way, each with a clear MISSING.
func TestSpecCompletionCheckReadsEveryNaming(t *testing.T) {
	spec := "# 01 Things\n\n### tool:set_policy Set the policy\n\nThe model sets the policy.\n\n" +
		"### F0.2 Cut\n\nThe toolbar on top.\n\n## 3. Project settings tab controls\n\nFreq, language, copy sources.\n"
	pre := t.TempDir()
	if err := os.WriteFile(filepath.Join(pre, "01.md"), []byte(spec), 0o644); err != nil {
		t.Fatal(err)
	}
	idx, err := scanSpec(pre, defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	section := ""
	for _, id := range idx.order {
		if strings.HasSuffix(id, "-tab-controls") {
			section = id
		}
	}
	if section == "" {
		t.Fatalf("no section item among %v", idx.order)
	}
	short := strings.TrimSuffix(section, "-tab-controls")
	answer := "tool:set_policy: DONE\n" + short + ": DONE\n- **F0.2 Cut** — DONE"
	b, _ := json.Marshal(map[string]any{"clear": true, "report_only": true, "subtasks": []any{}, "answer": answer})
	rig := newSpecLoopRig(t, map[string]string{
		"spec/01.md":          spec,
		"app/src/lib.rs":      "// the program\n",
		"app/tests/things.rs": "#[test]\nfn f0_2_cut() {}\n#[test]\nfn tool_set_policy() {}\n",
	}, func(string) bool { return true }, sseToolCall("c1", submitPlanToolName, string(b)))
	said := rig.run(t)
	got := rig.ledger(t)
	for _, id := range []string{"tool:set_policy", section, "F0.2"} {
		if it, ok := got.Items[id]; !ok || it.Checked == "" {
			t.Errorf("%s was answered DONE but is not done and checked (in ledger: %v)\n%s", id, ok, said)
		}
	}
	if strings.Contains(said, "no verdict") {
		t.Errorf("an answered item was taken for unanswered:\n%s", said)
	}
}

// A suite that fails and then passes unchanged is a flaky test, not the round's
// failure: the round counts, and the next item round is asked to fix the test.
func TestSpecFlakyTestIsNotTheRoundsFault(t *testing.T) {
	write := sseWriteFile
	rig := newSpecLoopRig(t, map[string]string{
		"spec/01.md":        "# 01 Things\n\n### F0.1 Do it\n\nS1 do the thing.\n\n### F0.2 Do more\n\nS1 do more.\n",
		"app/src/lib.rs":    "// the program\n",
		"app/tests/base.rs": "#[test]\nfn base_builds() {}\n",
	}, func(id string) bool { return id != "F0.1" && id != "F0.2" },
		sseToolCall("p1", submitPlanToolName, `{"clear":true,"subtasks":[{"description":"do it"}]}`),
		write("w1", "app/tests/f0_1.rs", "#[test]\nfn f0_1_does_it() {}\n"),
		sseToolCall("r1", respondToolName, `{"message":"done"}`),
		sseToolCall("p2", submitPlanToolName, `{"clear":true,"subtasks":[{"description":"do more"}]}`),
		write("w2", "app/tests/f0_2.rs", "#[test]\nfn f0_2_does_more() {}\n"),
		sseToolCall("r2", respondToolName, `{"message":"done"}`),
		sseToolCall("c1", submitPlanToolName, `{"clear":true,"report_only":true,"subtasks":[],"answer":"F0.1: DONE\nF0.2: DONE"}`),
	)
	// The suite fails its first run only, as a test that hangs on a timing would.
	once := filepath.Join(t.TempDir(), "ran")
	cfg := rig.ledger(t)
	cfg.TestCmd = "if [ -e " + once + " ]; then exit 0; fi; touch " + once + "; echo \"panicked at tests/cam.rs:254: and the status says so: playing\"; exit 101"
	if err := saveSpecConfig(rig.h.sess.Cwd, cfg); err != nil {
		t.Fatal(err)
	}
	said := rig.run(t)
	if !strings.Contains(said, "Flaky test") || !strings.Contains(said, "and the status says so") {
		t.Errorf("the flaky run was not reported:\n%s", said)
	}
	got := rig.ledger(t)
	if !got.done("F0.1") || !got.done("F0.2") {
		t.Errorf("F0.1 done=%v F0.2 done=%v, want both: the flaky test is no fault of theirs\n%s", got.done("F0.1"), got.done("F0.2"), said)
	}
	if round2 := rig.request(3); !strings.Contains(round2, "First: a flaky test") || !strings.Contains(round2, "and the status says so") {
		t.Errorf("the next round was not asked to fix the flaky test:\n%s", round2)
	}
	if got.Flaky != "" {
		t.Errorf("the flaky entry outlived the round that was asked to fix it: %q", got.Flaky)
	}
}

// Blocked items in a row mean one shared problem: the loop stops instead of burning every item.
func TestSpecStopsAfterBlockedInARow(t *testing.T) {
	// Turns that write nothing; the in-round check gives each round a second one.
	var turns []string
	for i, item := range []string{"F0.1", "F0.1", "F0.2", "F0.2"} {
		turns = append(turns,
			sseToolCall(fmt.Sprintf("p%d", i), submitPlanToolName, `{"clear":true,"subtasks":[{"description":"build `+item+`"}]}`),
			sseToolCall(fmt.Sprintf("r%d", i), respondToolName, `{"message":"could not"}`))
	}
	rig := newSpecLoopRig(t, map[string]string{
		"spec/01.md":        "# 01 Things\n\n### F0.1 One\n\nS1 one.\n\n### F0.2 Two\n\nS2 two.\n\n### F0.3 Three\n\nS3 three.\n",
		"app/src/lib.rs":    "// the program\n",
		"app/tests/base.rs": "#[test]\nfn base_builds() {}\n",
	}, func(string) bool { return false }, turns...)
	cfg := rig.ledger(t)
	cfg.MaxBlocked = 2
	cfg.Attempts = map[string]int{"F0.1": specMaxAttempts - 1, "F0.2": specMaxAttempts - 1}
	if err := saveSpecConfig(rig.h.sess.Cwd, cfg); err != nil {
		t.Fatal(err)
	}
	said := rig.run(t)
	if !strings.Contains(said, "Stopping: 2 items in a row did not pass") {
		t.Errorf("no stop after two blocked items:\n%s", said)
	}
	if strings.Contains(said, "F0.3") {
		t.Errorf("F0.3 was started after the stop:\n%s", said)
	}
}

// Only the output dir and .devcontainer are committed; the user's other edits stay theirs.
func TestSpecCommit(t *testing.T) {
	a, sess := newTestAgent(t)
	cwd := sess.Cwd
	writeTree(t, cwd, map[string]string{".gitignore": ".codehalter/\n", "app/main.go": "package main\n", ".devcontainer/Dockerfile": "FROM alpine\n", "notes.txt": "mine\n",
		"spec/01.md": "# 01\n\n### F0.1 Store notes\n\nKeep them.\n"})
	gitInit(t, cwd)
	idx, err := scanSpec(filepath.Join(cwd, "spec"), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &specConfig{SpecDir: "spec", OutDir: "app"}
	git := func(args ...string) string {
		t.Helper()
		out, err := specGit(t.Context(), cwd, args...)
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(out)
	}

	if sha := a.specCommit(t.Context(), sess.ID, cwd, cfg, idx, "F0.1", specModeItem); sha != "" {
		t.Errorf("nothing changed, got commit %s", sha)
	}
	for _, c := range []struct {
		item string
		mode specMode
		want string
	}{
		{"F0.1", specModeItem, "spec: F0.1 Store notes"},
		{"F0.1", specModeChange, "spec: update F0.1 Store notes"},
		{"§02#1-gone", specModeRemove, "spec: remove §02#1-gone"},
		{specRefactorID, specModeRefactor, "spec: refactor app/"},
		{specSetupID, specModeSetup, "spec: set up app/"},
	} {
		writeTree(t, cwd, map[string]string{"app/main.go": "package main\n// " + c.want + "\n", ".devcontainer/Dockerfile": "FROM alpine # " + c.want + "\n",
			"notes.txt": "mine, " + c.want + "\n"})
		sha := a.specCommit(t.Context(), sess.ID, cwd, cfg, idx, c.item, c.mode)
		if sha == "" || sha != git("rev-parse", "HEAD") {
			t.Fatalf("%s: commit = %q", c.want, sha)
		}
		if got := git("log", "-1", "--format=%s"); got != c.want {
			t.Errorf("message = %q, want %q", got, c.want)
		}
		if got := git("show", "--name-only", "--format=", "HEAD"); got != ".devcontainer/Dockerfile\napp/main.go" {
			t.Errorf("%s committed:\n%s", c.want, got)
		}
	}
	if got := git("status", "--porcelain"); got != "M notes.txt" {
		t.Errorf("status after the commits = %q, want only the user's notes.txt modified", got)
	}
}

// Every refactor_every finished items one round goes to the code's shape, but only when there is debt.
func TestSpecRefactorCadence(t *testing.T) {
	dir := t.TempDir()
	long := "package main\n\nfunc main() {}\n" + strings.Repeat("// line\n", 10)
	writeTree(t, dir, map[string]string{"spec/01.md": "# 01\n\n### F0.1 One\n\nS1.\n\n### F0.2 Two\n\nS2.\n\n### F0.3 Three\n\nS3.\n", "app/main.go": long})
	idx, err := scanSpec(filepath.Join(dir, "spec"), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	covered := map[string]string{"F0.1": "app/main_test.go", "F0.2": "app/main_test.go"}
	pick := func(refactorAt int, maxLines int) (specWork, *specConfig) {
		t.Helper()
		h := newTerminalHarness(t)
		cfg := &specConfig{SpecDir: "spec", OutDir: "app", TestCmd: "true", RefactorEvery: 2, RefactorAt: refactorAt, MaxFileLines: maxLines,
			Items: map[string]specLedger{"F0.1": {Hash: specItemHash(idx, "F0.1")}, "F0.2": {Hash: specItemHash(idx, "F0.2")}}}
		r := &specRun{a: h.agent, sid: h.sess.ID, sess: h.sess, cfg: cfg, idx: idx, outAbs: filepath.Join(dir, "app"), reasons: map[string]string{}}
		return r.pickWork(t.Context(), covered, 1), cfg
	}

	if w, _ := pick(0, 5); w.Mode != specModeRefactor || w.Debt.overLines == 0 {
		t.Errorf("two items since the last refactor and a long file: work = %+v, want a refactor round", w)
	}
	if w, _ := pick(1, 5); w.Item != "F0.3" {
		t.Errorf("one item since the last refactor: work = %+v, want F0.3", w)
	}
	w, cfg := pick(0, 2000)
	if w.Item != "F0.3" || cfg.RefactorAt != 2 {
		t.Errorf("no debt: work = %+v RefactorAt = %d, want F0.3 and the cadence restarted at 2", w, cfg.RefactorAt)
	}
}

// An item finished with its answer before the answer could be dropped loses the
// question at the next round, without counting as a spec change; an answer given
// after the build stays for the change round, an open question stays open.
func TestSpecDropsAnswersOfBuiltItems(t *testing.T) {
	h := newTerminalHarness(t)
	specAbs := filepath.Join(h.sess.Cwd, "spec")
	writeTree(t, specAbs, map[string]string{"01.md": "# 01\n\n### F0.1 Store\n\nS1.\n\n### F0.2 Font\n\nS2.\n\n### F0.3 Name\n\nS3.\n"})
	before, err := scanSpec(specAbs, defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"F0.1", "F0.2", "F0.3"} {
		if err := appendSpecQuestion(specAbs, id, specQuestion{ID: id, Question: "Which " + id + "?"}); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(specAbs, specQuestionsFile)
	data, _ := os.ReadFile(path)
	writeTree(t, specAbs, map[string]string{specQuestionsFile: strings.Replace(strings.Replace(string(data), "**Answer:** \n", "**Answer:** SQLite\n", 1), "**Answer:** \n", "**Answer:** serif\n", 1)})
	cfg := &specConfig{SpecDir: "spec", Items: map[string]specLedger{"F0.2": {Hash: specItemHash(before, "F0.2")}, "F0.3": {Hash: specItemHash(before, "F0.3")}}}
	r := &specRun{a: h.agent, sid: h.sess.ID, sess: h.sess, cfg: cfg, reasons: map[string]string{}}
	if err := r.scan(); err != nil {
		t.Fatal(err)
	}
	cfg.Items["F0.1"] = specLedger{Hash: specItemHash(r.idx, "F0.1")} // built with SQLite

	r.dropBuiltAnswers(t.Context())
	left, _ := os.ReadFile(path)
	if strings.Contains(string(left), "SQLite") || !strings.Contains(string(left), "serif") || !strings.Contains(string(left), "Which F0.3?") {
		t.Errorf("QUESTIONS.md after the sweep:\n%s", left)
	}
	if err := r.scan(); err != nil {
		t.Fatal(err)
	}
	if d := specReconcile(cfg, r.idx, nil); slices.Contains(d.Changed, "F0.1") || !slices.Contains(d.Changed, "F0.2") {
		t.Errorf("changed = %v, want F0.2 (answered after its build) and not F0.1", d.Changed)
	}
}
