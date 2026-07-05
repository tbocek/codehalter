package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteReport(t *testing.T) {
	root := t.TempDir()
	modelsDir := filepath.Join(root, "models")
	gemma := filepath.Join(modelsDir, "gemma")
	if err := os.MkdirAll(gemma, 0o755); err != nil {
		t.Fatal(err)
	}
	// Seed a results.jsonl the report reads for example decisions.
	for _, r := range []ProbeResult{
		{Model: "gemma", Skill: "go", ClaimID: "go#00", Text: "always check errors", Verdict: "keep", Keep: true,
			Samples: []Sample{{Reason: "A ignored the error, B checked it"}}},
		{Model: "gemma", Skill: "go", ClaimID: "go#01", Text: "use gofmt tabs", Verdict: "drop", Keep: false,
			Samples: []Sample{{Reason: "both answers already used tabs"}}},
	} {
		if err := appendResult(filepath.Join(gemma, "results.jsonl"), r); err != nil {
			t.Fatal(err)
		}
	}

	stats := []ModelStats{{
		Model:       "gemma",
		OrigBytes:   1000,
		PrunedBytes: 700,
		Skills:      []SkillStats{{Skill: "go", Claims: 2, Kept: 1, Dropped: 1, OrigBytes: 1000, PrunedBytes: 700}},
	}}
	cfg := &Config{Judge: ModelSpec{Model: "big-judge"}, Settings: Settings{Samples: 3}}

	out := filepath.Join(root, "docs", "skill-crafter-report.html")
	if err := writeReport(out, stats, cfg, modelsDir); err != nil {
		t.Fatal(err)
	}
	html, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	s := string(html)
	for _, want := range []string{"gemma", "big-judge", "-30%", "KEEP", "DROP", "always check errors", "A ignored the error"} {
		if !strings.Contains(s, want) {
			t.Fatalf("report missing %q", want)
		}
	}
}
