package main

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// codehalter's own test run gets a fresh $TMPDIR: what the run writes there is its own,
// what it writes into the project or at a fixed temp path is named and handed to the next
// item round, which is not this round's failure; the round that was asked clears it. A virtual
// display's lock file is the suite's display pool, not a leak.
func TestSpecTestRunMustStayInItsTempDir(t *testing.T) {
	sandbox := t.TempDir()
	t.Setenv("TMPDIR", sandbox) // the system temp dir, as the run sees it
	a, sess := newTestAgent(t)
	idx, err := scanSpec(writeSpecFixture(t), defaultSpecIDPatterns, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	writeTree(t, sess.Cwd, map[string]string{".gitignore": ".codehalter/\nrust/fixtures/out.mp4\n", "rust/src/lib.rs": "// v1\n", "rust/fixtures/demo.txt": "fixture\n"})
	gitInit(t, sess.Cwd)
	leaky := fmt.Sprintf(`echo scratch > "$TMPDIR/scratch.txt" && echo out > fixtures/out.mp4 && echo sock > %s/fixed.sock && echo 1 > %s/.X123-lock`, sandbox, sandbox)
	cfg := &specConfig{SpecDir: "spec", OutDir: "rust", TestCmd: leaky, LintCmd: "off", Items: map[string]specLedger{}}
	r := &specRun{a: a, sid: sess.ID, sess: sess, cfg: cfg, idx: idx, outAbs: filepath.Join(sess.Cwd, "rust"), reasons: map[string]string{}}

	writeTree(t, sess.Cwd, map[string]string{"rust/tests/switch.rs": "#[test]\nfn f0_1_switches() {}\n"})
	if done, _, err := r.finishRound(t.Context(), specWork{Item: "F0.1"}, nil, sess.toolMark(), time.Now(), nil); err != nil || !done {
		t.Fatalf("done=%v err=%v reason=%q: a leaky run is no failure of this round", done, err, r.reasons["F0.1"])
	}
	if !strings.Contains(cfg.Isolation, "rust/fixtures/out.mp4") || !strings.Contains(cfg.Isolation, filepath.Join(sandbox, "fixed.sock")) || strings.Contains(cfg.Isolation, "scratch.txt") || strings.Contains(cfg.Isolation, ".X123-lock") {
		t.Errorf("isolation = %q, want the project file and the fixed temp path, not the run's own TMPDIR nor a display's lock", cfg.Isolation)
	}

	// The next round is asked to isolate them; once it is through, the notice goes.
	cfg.TestCmd = `echo scratch > "$TMPDIR/scratch.txt"`
	writeTree(t, sess.Cwd, map[string]string{"rust/tests/switch.rs": "#[test]\nfn f0_1_switches() { let _ = 1; }\n"})
	cfg.reopen([]string{"F0.1"})
	if done, _, err := r.finishRound(t.Context(), specWork{Item: "F0.1", Redo: true, Isolation: cfg.Isolation}, nil, sess.toolMark(), time.Now(), nil); err != nil || !done {
		t.Fatalf("done=%v err=%v reason=%q", done, err, r.reasons["F0.1"])
	}
	if cfg.Isolation != "" {
		t.Errorf("isolation = %q after the round that was asked to fix it, and a clean run", cfg.Isolation)
	}
}
