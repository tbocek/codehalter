package main

import (
	"strings"
	"testing"
)

func TestPruneSkill(t *testing.T) {
	orig := "# Title\n- keep me\n- drop me\n- keep me too\n"
	pruned := pruneSkill(orig, map[int]bool{3: true})
	if strings.Contains(pruned, "drop me") {
		t.Fatalf("dropped line survived: %q", pruned)
	}
	if !strings.Contains(pruned, "keep me") || !strings.Contains(pruned, "keep me too") {
		t.Fatalf("kept lines missing: %q", pruned)
	}
}

func TestPruneSkillCollapsesBlankRuns(t *testing.T) {
	orig := "a\nb\nc\nd\n"
	// Dropping b and c (lines 2,3) leaves a\n\n\nd → collapse to a\n\nd? Here b,c
	// are content lines, not blanks, so removal just yields a\nd. Verify a real
	// blank-run collapse: drop only line 2, but line 3 is already blank.
	orig2 := "a\n\n\n\nb\n"
	got := pruneSkill(orig2, map[int]bool{})
	if strings.Contains(got, "\n\n\n") {
		t.Fatalf("blank run not collapsed: %q", got)
	}
	_ = orig
}

func TestDropLineSet(t *testing.T) {
	claims := []Claim{
		{ID: "go#00", StartLine: 2, EndLine: 2},
		{ID: "go#01", StartLine: 4, EndLine: 5},
		{ID: "go#02", StartLine: 7, EndLine: 7},
	}
	results := map[string]ProbeResult{
		"go#00": {ClaimID: "go#00", Keep: false},           // drop
		"go#01": {ClaimID: "go#01", Keep: true},            // keep
		"go#02": {ClaimID: "go#02", Keep: false, Err: "x"}, // errored → keep
	}
	drop := dropLineSet(claims, results)
	if !drop[2] {
		t.Fatalf("go#00 line 2 should be dropped")
	}
	if drop[4] || drop[5] {
		t.Fatalf("kept claim lines must not drop")
	}
	if drop[7] {
		t.Fatalf("errored claim must not drop")
	}
}

func TestSkillStats(t *testing.T) {
	claims := []Claim{{ID: "s#00"}, {ID: "s#01"}, {ID: "s#02"}}
	results := map[string]ProbeResult{
		"s#00": {Keep: true},
		"s#01": {Keep: false},
		"s#02": {Err: "boom"},
	}
	st := skillStats("go", "origbytes", "pruned", claims, results)
	if st.Kept != 1 || st.Dropped != 1 || st.Errored != 1 {
		t.Fatalf("counts = kept %d drop %d err %d", st.Kept, st.Dropped, st.Errored)
	}
	if st.OrigBytes != len("origbytes") || st.PrunedBytes != len("pruned") {
		t.Fatalf("bytes = %d/%d", st.OrigBytes, st.PrunedBytes)
	}
}
