package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// The hunks a round can be judged by are its program changes alone: a test file, an inline test
// module, a new file, a reworded comment or string are not, and undoing a hunk gives back the old text.
func TestSpecRoundHunks(t *testing.T) {
	orig := "fn a() -> i32 {\n    1\n}\n// old note\nfn b() -> &'static str {\n    \"old\"\n}\n\n#[cfg(test)]\nmod tests {\n    fn t() { let _ = 1; }\n}\n"
	cwd := gitRepo(t, map[string]string{"out/src/a.rs": orig, "out/tests/t.rs": "fn t() {}\n"})
	writeTree(t, cwd, map[string]string{
		"out/src/a.rs":   strings.NewReplacer("    1\n", "    2\n    + 3\n", "// old note", "// new note", "\"old\"", "\"new\"", "let _ = 1;", "let _ = 2;").Replace(orig),
		"out/src/b.rs":   "fn fresh() {}\n",
		"out/tests/t.rs": "fn t() { assert!(true) }\n",
	})
	ch := specRoundChanges(t.Context(), cwd, "out", "HEAD")
	hunks := specRoundHunks(t.Context(), cwd, "out", "HEAD", ch, 6)
	if len(hunks) != 1 || hunks[0].file != "src/a.rs" || hunks[0].start != 2 || !slices.Equal(hunks[0].old, []string{"    1"}) {
		t.Fatalf("hunks = %+v, want only a.rs's changed body", hunks)
	}
	now, err := os.ReadFile(filepath.Join(cwd, "out/src/a.rs"))
	if err != nil {
		t.Fatal(err)
	}
	undone := hunks[0].undo(string(now))
	if !strings.Contains(undone, "{\n    1\n}\n// new note") || strings.Contains(undone, "+ 3") {
		t.Errorf("undo gives back the old body and nothing else:\n%s", undone)
	}
	if got := specRoundHunks(t.Context(), cwd, "out", "HEAD", ch, 0); len(got) != 0 {
		t.Errorf("a limit of 0 puts nothing back, got %d", len(got))
	}
}

// A pure deletion is undone by putting the deleted lines back where they were.
func TestSpecHunkUndoesADeletion(t *testing.T) {
	h := specHunk{file: "a.go", start: 1, count: 0, old: []string{"two"}}
	if got := h.undo("one\nthree\n"); got != "one\ntwo\nthree\n" {
		t.Errorf("undo = %q", got)
	}
}

// The gate on a real round: the change the test checks is proven, the one it does not check is the
// finding, and every file is as the round left it afterwards. A filter the run does not show means
// the command cannot judge, and the gate stands down instead of reporting every change.
func TestUnprovenHunks(t *testing.T) {
	a, sess := newTestAgent(t)
	orig := "fn fixed() -> i32 {\n    0\n}\n\nfn other() -> i32 {\n    1\n}\n"
	writeTree(t, sess.Cwd, map[string]string{".gitignore": ".codehalter/\n", "out/src/a.rs": orig})
	gitInit(t, sess.Cwd)
	round := strings.NewReplacer("    0\n", "    42\n", "    1\n", "    7\n").Replace(orig)
	writeTree(t, sess.Cwd, map[string]string{"out/src/a.rs": round})
	// The item's "test": it checks the 42 and nothing about other().
	cfg := &specConfig{SpecDir: "spec", OutDir: "out", MutationCmd: "grep -q '    42' src/a.rs && echo ran {filter}"}
	r := &specRun{a: a, sid: sess.ID, sess: sess, cfg: cfg, outAbs: filepath.Join(sess.Cwd, "out"), reasons: map[string]string{}}
	ch := specRoundChanges(t.Context(), sess.Cwd, "out", "HEAD")

	got := r.unprovenHunks(t.Context(), "F1.2", "HEAD", "", ch)
	if len(got) != 1 || !strings.Contains(got[0], "src/a.rs:6") || !strings.Contains(got[0], "7") {
		t.Errorf("unproven = %v, want other()'s change alone", got)
	}
	if now, _ := os.ReadFile(filepath.Join(sess.Cwd, "out/src/a.rs")); string(now) != round {
		t.Errorf("the file is not as the round left it:\n%s", now)
	}
	if _, err := os.Stat(filepath.Join(sess.Cwd, sessionDir, specMutantFile)); !os.IsNotExist(err) {
		t.Errorf("the mutant backup is left behind: %v", err)
	}

	cfg.MutationCmd = "true"
	if got := r.unprovenHunks(t.Context(), "F1.2", "HEAD", "", ch); got != nil {
		t.Errorf("a run that shows no filter judges nothing, got %v", got)
	}
	cfg.MutationCmd = "off"
	if got := r.unprovenHunks(t.Context(), "F1.2", "HEAD", "", ch); got != nil {
		t.Errorf("off is off, got %v", got)
	}
}

// A codehalter stopped while a change was taken out puts the file back on the next run.
func TestRestoreSpecMutant(t *testing.T) {
	cwd := t.TempDir()
	path := filepath.Join(cwd, "a.rs")
	writeTree(t, cwd, map[string]string{"a.rs": "mutated\n", sessionDir + "/" + specMutantFile: `{"path":"` + path + `","content":"original\n"}`})
	if got := restoreSpecMutant(cwd); got != path {
		t.Fatalf("restored %q, want %q", got, path)
	}
	if now, _ := os.ReadFile(path); string(now) != "original\n" {
		t.Errorf("file = %q", now)
	}
	if restoreSpecMutant(cwd) != "" {
		t.Error("the backup is gone once used")
	}
}

// The filtered command follows the project's tooling, and a test command of the project's own
// leaves it to spec.toml.
func TestMutationCmd(t *testing.T) {
	for _, c := range []struct {
		files map[string]string
		cfg   specConfig
		want  string
	}{
		{map[string]string{"justfile": "test filter=\"\":\n    cargo test {{filter}}\n"}, specConfig{}, "just test {filter}"},
		{map[string]string{"justfile": "test:\n    cargo test\n", "Cargo.toml": ""}, specConfig{}, ""},
		{map[string]string{"Cargo.toml": ""}, specConfig{}, "cargo test {filter}"},
		{map[string]string{"go.mod": ""}, specConfig{}, "go test ./... -v -run '(?i){filter}'"},
		{map[string]string{"Cargo.toml": ""}, specConfig{TestCmd: "make check"}, ""},
		{map[string]string{"Cargo.toml": ""}, specConfig{MutationCmd: "off"}, ""},
		{map[string]string{}, specConfig{MutationCmd: "make t T={filter}"}, "make t T={filter}"},
	} {
		dir := t.TempDir()
		writeTree(t, dir, c.files)
		cfg := c.cfg
		r := &specRun{cfg: &cfg, outAbs: dir}
		if got := r.mutationCmd(); got != c.want {
			t.Errorf("%v %+v: %q, want %q", c.files, c.cfg, got, c.want)
		}
	}
}

// Wired into the round's final judgement: a green round whose test passes without one of its changes
// is not done, and the reason names the change.
func TestFinishRoundHoldsAnUnprovenChange(t *testing.T) {
	a, sess := newTestAgent(t)
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	writeTree(t, sess.Cwd, map[string]string{".gitignore": ".codehalter/\n", "rust/src/lib.rs": "pub fn v() -> i32 {\n    1\n}\n\npub fn w() -> i32 {\n    2\n}\n"})
	gitInit(t, sess.Cwd)
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: "true", LintCmd: "off", Items: map[string]specLedger{},
		MutationCmd: "grep -q '    41' src/lib.rs && echo {filter}"}
	r := &specRun{a: a, sid: sess.ID, sess: sess, cfg: cfg, idx: idx, outAbs: filepath.Join(sess.Cwd, "rust"), reasons: map[string]string{}}
	writeTree(t, sess.Cwd, map[string]string{
		"rust/src/lib.rs":      "pub fn v() -> i32 {\n    41\n}\n\npub fn w() -> i32 {\n    3\n}\n",
		"rust/tests/switch.rs": "#[test]\nfn f0_1_switches() { assert_eq!(v(), 41) }\n",
	})
	done, _, err := r.finishRound(t.Context(), specWork{Item: "F0.1"}, nil, sess.toolMark(), time.Now(), nil)
	if err != nil || done {
		t.Fatalf("done=%v err=%v: w()'s change is held by no test", done, err)
	}
	if why := r.reasons["F0.1"]; !strings.Contains(why, "src/lib.rs:6") || strings.Contains(why, "src/lib.rs:2") {
		t.Errorf("reason = %q, want w()'s change alone", why)
	}
}
