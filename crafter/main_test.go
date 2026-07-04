package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDiscoverSkills(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"SKILL-go.md", "SKILL-base.md", "SKILL-ts.md", "README.md"} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// No filter: every SKILL-*.md, README ignored, sorted.
	all, err := discoverSkills(dir, Settings{})
	if err != nil {
		t.Fatal(err)
	}
	if got := skillNames(all); len(got) != 3 || got[0] != "base" || got[1] != "go" || got[2] != "ts" {
		t.Fatalf("discover all = %v", got)
	}
	// Filtered.
	some, err := discoverSkills(dir, Settings{Skills: []string{"go"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := skillNames(some); len(got) != 1 || got[0] != "go" {
		t.Fatalf("discover filtered = %v", got)
	}
}

func TestResultsRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "results.jsonl")
	r1 := ProbeResult{ClaimID: "go#00", Verdict: "keep", Keep: true}
	r2 := ProbeResult{ClaimID: "go#01", Verdict: "drop"}
	// Two lines for go#00: the later one wins on read.
	r1b := ProbeResult{ClaimID: "go#00", Verdict: "drop", Keep: false}
	for _, r := range []ProbeResult{r1, r2, r1b} {
		if err := appendResult(path, r); err != nil {
			t.Fatal(err)
		}
	}
	got := readResults(path)
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	if got["go#00"].Keep {
		t.Fatalf("last write for go#00 should win (drop)")
	}
	if got["go#01"].Verdict != "drop" {
		t.Fatalf("go#01 = %+v", got["go#01"])
	}
}

func TestReadResultsSkipsMalformed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "results.jsonl")
	// A good row, a half-written (malformed) row, a blank line, another good row.
	content := `{"claim_id":"go#00","verdict":"keep","keep":true}
{"claim_id":"go#01","verdict":"dr
` + "\n" + `{"claim_id":"go#02","verdict":"drop"}
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	got := readResults(path)
	// Malformed go#01 skipped (re-probed later); the two valid rows survive.
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2 (%v)", len(got), got)
	}
	if _, ok := got["go#00"]; !ok {
		t.Fatal("go#00 missing")
	}
	if _, ok := got["go#02"]; !ok {
		t.Fatal("go#02 missing")
	}
	if _, ok := got["go#01"]; ok {
		t.Fatal("malformed go#01 should not be present")
	}
}

func TestReadResultsMissingFile(t *testing.T) {
	got := readResults(filepath.Join(t.TempDir(), "nope.jsonl"))
	if len(got) != 0 {
		t.Fatalf("missing file should yield empty map, got %d", len(got))
	}
}

func TestFmtDuration(t *testing.T) {
	cases := []struct {
		sec  int
		want string
	}{
		{0, "00:00:00"},
		{5, "00:00:05"},
		{65, "00:01:05"},
		{3661, "01:01:01"},
	}
	for _, c := range cases {
		if got := fmtDuration(time.Duration(c.sec) * time.Second); got != c.want {
			t.Fatalf("fmtDuration(%ds) = %q, want %q", c.sec, got, c.want)
		}
	}
}

func TestTally(t *testing.T) {
	ms := ModelStats{Skills: []SkillStats{
		{Kept: 2, Dropped: 1, Errored: 0},
		{Kept: 1, Dropped: 3, Errored: 1},
	}}
	k, d, e := tally(ms)
	if k != 3 || d != 4 || e != 1 {
		t.Fatalf("tally = %d/%d/%d, want 3/4/1", k, d, e)
	}
}

func TestPct(t *testing.T) {
	cases := []struct {
		orig, pruned int
		want         string
	}{
		{100, 66, "-34%"},
		{100, 100, "+0%"},
		{0, 0, "0%"},
		{100, 120, "+20%"},
	}
	for _, c := range cases {
		if got := pct(c.orig, c.pruned); got != c.want {
			t.Fatalf("pct(%d,%d) = %q, want %q", c.orig, c.pruned, got, c.want)
		}
	}
}
