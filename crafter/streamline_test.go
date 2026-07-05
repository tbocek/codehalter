package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// streamlineFixture returns a judge server that replies with the given
// rewrite, counting calls, plus a ground-skill file to streamline.
func streamlineFixture(t *testing.T, reply, source string) (judge ModelSpec, srcPath, cleanDir, cacheDir string, calls *atomic.Int32) {
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
	return ModelSpec{Name: "main", Server: ts.URL, Model: "j"},
		srcPath, filepath.Join(root, "clean-skills"), filepath.Join(root, ".cache", "streamline"), calls
}

const streamlineSrc = "# Arch skill\n## Git — writable, commit/push when asked\n- use `git status` first.\n"

func TestEnsureCleanSkillWritesAndCaches(t *testing.T) {
	rewrite := "# Arch skill\n## Git\n- .git is writable; commit/push only when asked.\n- use `git status` first.\n"
	judge, src, cleanDir, cacheDir, calls := streamlineFixture(t, rewrite, streamlineSrc)

	p1, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(p1)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != rewrite {
		t.Fatalf("clean file = %q, want the rewrite", got)
	}

	// Second call: cache hit, no judge call.
	if _, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir, cacheDir); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("judge called %d times, want 1 (second run must hit the cache)", calls.Load())
	}

	// Edited source: cache invalidates, judge called again.
	if err := os.WriteFile(src, []byte(streamlineSrc+"- new statement.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir, cacheDir); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("judge called %d times after source edit, want 2", calls.Load())
	}
}

func TestEnsureCleanSkillUnwrapsFence(t *testing.T) {
	rewrite := "# Arch skill\n## Git\n- commit/push only when asked.\n- use `git status` first.\n"
	judge, src, cleanDir, cacheDir, _ := streamlineFixture(t, "```markdown\n"+rewrite+"```", streamlineSrc)
	p, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(p)
	if string(got) != rewrite {
		t.Fatalf("fence not unwrapped: %q", got)
	}
}

func TestEnsureCleanSkillGuardsAgainstLoss(t *testing.T) {
	// The rewrite silently drops the `git status` statement — the guard must
	// reject it and fall back to the verbatim original.
	lossy := "# Arch skill\n## Git\n- .git is writable; commit/push only when asked, and much more prose to defeat any size floor heuristic entirely.\n"
	judge, src, cleanDir, cacheDir, _ := streamlineFixture(t, lossy, streamlineSrc)
	p, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(p)
	if string(got) != streamlineSrc {
		t.Fatalf("lossy rewrite accepted: %q", got)
	}
}

func TestEnsureCleanSkillGuardsAgainstShrink(t *testing.T) {
	// All spans survive but the rewrite collapses to a fraction of the size —
	// suspected statement loss, fall back to the original.
	src := streamlineSrc + strings.Repeat("- another plain statement without code that matters a lot.\n", 20)
	shrunk := "# Arch skill\n## Git\n- use `git status` first.\n"
	judge, srcPath, cleanDir, cacheDir, _ := streamlineFixture(t, shrunk, src)
	p, err := ensureCleanSkill(context.Background(), judge, "arch", srcPath, cleanDir, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(p)
	if string(got) != src {
		t.Fatalf("shrunken rewrite accepted (%d bytes vs %d)", len(got), len(src))
	}
}

func TestEnsureCleanSkillCachePersists(t *testing.T) {
	rewrite := "# Arch skill\n## Git\n- commit/push only when asked.\n- use `git status` first.\n"
	judge, src, cleanDir, cacheDir, _ := streamlineFixture(t, rewrite, streamlineSrc)
	p, err := ensureCleanSkill(context.Background(), judge, "arch", src, cleanDir, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	var c streamlineCache
	data, err := os.ReadFile(filepath.Join(cacheDir, "arch.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &c); err != nil || c.Hash == "" {
		t.Fatalf("cache sidecar invalid: %s (%v)", data, err)
	}
	if filepath.Base(p) != "SKILL-arch.md" {
		t.Fatalf("clean path = %s", p)
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
