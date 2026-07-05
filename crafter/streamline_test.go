package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// streamlineFixture returns a judge server that replies with the given rewrite,
// counting calls, plus a ground-skill file to streamline and the clean dir.
func streamlineFixture(t *testing.T, reply, source string) (judge ModelSpec, srcPath, cleanDir string, calls *atomic.Int32) {
	t.Helper()
	calls = &atomic.Int32{}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte(sse(fmt.Sprintf(`{"choices":[{"delta":{"content":%q}}]}`, reply))))
	}))
	t.Cleanup(ts.Close)
	root := t.TempDir()
	srcPath = filepath.Join(root, "SKILL-arch.md")
	if err := os.WriteFile(srcPath, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	return ModelSpec{Name: "main", Server: ts.URL, Model: "j"}, srcPath, filepath.Join(root, "clean-skills"), calls
}

const streamlineSrc = "# Arch skill\n## Git — writable, commit/push when asked\n- use `git status` first.\n"

// The clean file is generated once when absent, then owned by the user: reused
// as-is regardless of source edits, hand-edits respected, only a delete
// regenerates it.
func TestEnsureCleanSkillGeneratedOnceThenOwned(t *testing.T) {
	rewrite := "# Arch skill\n## Git\n- .git is writable; commit/push only when asked.\n- use `git status` first.\n"
	judge, src, cleanDir, calls := streamlineFixture(t, rewrite, streamlineSrc)

	p, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(p); string(got) != rewrite {
		t.Fatalf("clean file = %q, want the rewrite", got)
	}

	// Second call reuses the existing file — no judge call.
	if _, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("judge called %d times, want 1 (existing clean file must be reused)", calls.Load())
	}

	// Editing the SOURCE does NOT regenerate — the clean file is owned now.
	if err := os.WriteFile(src, []byte(streamlineSrc+"- new statement.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("judge called %d after a source edit, want 1 (clean file is owned, not regenerated)", calls.Load())
	}

	// A hand-edit of the clean file is respected verbatim.
	mine := "# my hand-edited clean skill\n- whatever I want.\n"
	if err := os.WriteFile(p, []byte(mine), 0o644); err != nil {
		t.Fatal(err)
	}
	p2, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(p2); string(got) != mine {
		t.Fatalf("hand-edit not respected: %q", got)
	}
	if calls.Load() != 1 {
		t.Fatalf("judge called %d, want 1 (hand-edit must not trigger a rewrite)", calls.Load())
	}

	// Deleting the clean file regenerates it (judge called again).
	if err := os.Remove(p); err != nil {
		t.Fatal(err)
	}
	if _, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("judge called %d after deleting the clean file, want 2", calls.Load())
	}
}

func TestEnsureCleanSkillUnwrapsFence(t *testing.T) {
	rewrite := "# Arch skill\n## Git\n- commit/push only when asked.\n- use `git status` first.\n"
	judge, src, cleanDir, _ := streamlineFixture(t, "```markdown\n"+rewrite+"```", streamlineSrc)
	p, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(p); string(got) != rewrite {
		t.Fatalf("fence not unwrapped: %q", got)
	}
}

func TestEnsureCleanSkillGuardsAgainstLoss(t *testing.T) {
	// The rewrite silently drops the `git status` statement — the guard must
	// reject it and write the verbatim original.
	lossy := "# Arch skill\n## Git\n- .git is writable; commit/push only when asked, and much more prose to defeat any size floor heuristic entirely.\n"
	judge, src, cleanDir, _ := streamlineFixture(t, lossy, streamlineSrc)
	p, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(p); string(got) != streamlineSrc {
		t.Fatalf("lossy rewrite accepted: %q", got)
	}
}

func TestEnsureCleanSkillGuardsAgainstShrink(t *testing.T) {
	// All spans survive but the rewrite collapses to a fraction of the size —
	// suspected statement loss, fall back to the original.
	src := streamlineSrc + strings.Repeat("- another plain statement without code that matters a lot.\n", 20)
	shrunk := "# Arch skill\n## Git\n- use `git status` first.\n"
	judge, srcPath, cleanDir, _ := streamlineFixture(t, shrunk, src)
	p, err := ensureCleanSkill(context.Background(), judge, "arch", srcPath, cleanDir)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(p); string(got) != src {
		t.Fatalf("shrunken rewrite accepted (%d bytes vs %d)", len(got), len(src))
	}
}

func TestMissingSpans(t *testing.T) {
	orig := "use `go vet` then {{cmd:date +%F}} and `go test`."
	if m := missingSpans(orig, "use `go vet` and {{cmd:date +%F}} plus `go test`."); len(m) != 0 {
		t.Fatalf("false positives: %v", m)
	}
	m := missingSpans(orig, "use `go vet` only.")
	if len(m) != 2 {
		t.Fatalf("missing = %v, want the cmd directive and `go test`", m)
	}
}
