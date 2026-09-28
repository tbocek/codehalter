package main

import (
	"context"
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

// The gates on what a round wrote hold an item back and say why; a refactor
// counts when its debt shrank.
func TestSpecDecideGates(t *testing.T) {
	green := specRoundResult{Covered: true, TestsPass: true, Committed: true}
	for name, r := range map[string]specRoundResult{
		"unreachable": {Covered: true, TestsPass: true, Unreachable: []string{"`run` (src/a.rs:3): only tests call it"}},
		"lint":        {Covered: true, TestsPass: true, Lint: []string{"src/a.rs:3: unused variable"}},
		"oversize":    {Covered: true, TestsPass: true, Oversize: []string{"`src/ui.rs` is 12000 lines, over the 1500-line budget"}},
	} {
		done, _, reason := specDecide(&specConfig{}, "F0.1", r)
		if done || !strings.Contains(reason, map[string]string{"unreachable": "only tests call it", "lint": "unused variable", "oversize": "1500-line budget"}[name]) {
			t.Errorf("%s: done=%v reason=%q", name, done, reason)
		}
	}
	if done, _, _ := specDecide(&specConfig{}, "F0.1", green); !done {
		t.Error("a clean round was held back")
	}
	cfg := &specConfig{}
	if done, _, reason := specDecide(cfg, specRefactorID, specRoundResult{Mode: specModeRefactor, TestsPass: true, Committed: true, Debt: "a → b"}); done || !strings.Contains(reason, "no measured progress") {
		t.Errorf("a refactor that shrank nothing: done=%v reason=%q", done, reason)
	}
	if done, _, _ := specDecide(cfg, specRefactorID, specRoundResult{Mode: specModeRefactor, TestsPass: true, Committed: true, Improved: true}); !done {
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

	refactor, head := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: specRefactorID, Mode: specModeRefactor,
		Debt: specDebt{overLines: 900, targets: "- `src/ui/window.rs`: 2400 lines\n"}}, "just test", "")
	if strings.Contains(refactor, "{{") || !strings.Contains(refactor, "`src/ui/window.rs`: 2400 lines") || !strings.Contains(head, "lines over the size budget 900") {
		t.Errorf("refactor prompt (%s):\n%s", head, refactor)
	}

	setup, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: specSetupID, Mode: specModeSetup}, "", "")
	if strings.Contains(setup, "{{") || !strings.Contains(setup, "use gtk4-rs libadwaita") || !strings.Contains(setup, "`rust/`") {
		t.Errorf("setup prompt:\n%s", setup)
	}

	answered, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: "F0.2", Question: "keep or drop?", Answer: "keep it"}, "just test", "")
	if !strings.Contains(answered, "keep or drop?") || !strings.Contains(answered, "keep it") {
		t.Errorf("answered prompt lacks the question and answer:\n%s", answered)
	}
	// A changed item that was blocked comes back with its answer; the diff fence must still close.
	changed, _ := a.specRoundPrompt(s.ID, cfg, idx, specWork{Item: "F0.2", Mode: specModeChange, Note: "-old\n+new", Answer: "keep it"}, "just test", "")
	if !strings.Contains(changed, "+new\n```\n\n## Your earlier question was answered") {
		t.Errorf("the answer runs into the diff fence:\n%s", changed)
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

func TestSpecDecideRedoNeedsACommit(t *testing.T) {
	cfg := &specConfig{}
	done, _, reason := specDecide(cfg, "F0.1", specRoundResult{Redo: true, Covered: true, TestsPass: true})
	if done || !strings.Contains(reason, "/spec redo") {
		t.Errorf("a redo round that wrote nothing: done=%v reason=%q", done, reason)
	}
	if done, _, _ := specDecide(cfg, "F0.1", specRoundResult{Redo: true, Covered: true, TestsPass: true, Committed: true}); !done {
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
	done, block, reason := specDecide(&specConfig{}, "F1.1", specRoundResult{Covered: true, TestsPass: true, UIUnseen: []string{"rust/src/ui.rs"}})
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

	// A blocked removal waits for its answer instead of being picked every round.
	blockedPick := func(answer string) specWork {
		t.Helper()
		h := newTerminalHarness(t)
		r := &specRun{a: h.agent, sid: h.sess.ID, sess: h.sess, idx: idx, reasons: map[string]string{},
			cfg: &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: "just test",
				Items:   built(map[string]specLedger{dropped: {Hash: "whatever", Title: "1. Dropped"}}),
				Blocked: []specBlock{{ID: dropped, Reason: "stuck", Answer: answer}}}}
		return r.pickWork(t.Context(), nil, 1)
	}
	if w := blockedPick(""); w.Item != "" {
		t.Errorf("work = %+v, want the unanswered blocked removal skipped", w)
	}
	if w := blockedPick("keep the helper"); w.Item != dropped || w.Mode != specModeRemove || w.Answer != "keep the helper" {
		t.Errorf("work = %+v, want the answered removal picked with its answer", w)
	}

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
	if done, _, err := r.finishRound(t.Context(), specWork{Item: "F0.1"}, nil, len(sess.Messages), time.Now()); err != nil || done {
		t.Fatalf("attempt 1: done=%v err=%v, want a failed attempt", done, err)
	}
	cfg.TestCmd = "true"
	write("rust/src/lib.rs", "// v2, the fix\n") // attempt 2 touches only the program
	done, _, err := r.finishRound(t.Context(), specWork{Item: "F0.1"}, nil, len(sess.Messages), time.Now())
	if err != nil || !done {
		t.Fatalf("attempt 2: done=%v err=%v reason=%q, want the first attempt's test to count", done, err, r.reasons["F0.1"])
	}
	if _, ok := cfg.Bases["F0.1"]; ok {
		t.Error("the base outlived the finished item")
	}
}
