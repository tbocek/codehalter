package main

import (
	"strings"
	"testing"
)

func TestPruneSkill(t *testing.T) {
	orig := "# Title\n- keep me\n- drop me\n- keep me too\n"
	pruned := pruneSkill(orig, []Claim{{Source: "- drop me", StartLine: 3, EndLine: 3}})
	if strings.Contains(pruned, "drop me") {
		t.Fatalf("dropped line survived: %q", pruned)
	}
	if !strings.Contains(pruned, "keep me") || !strings.Contains(pruned, "keep me too") {
		t.Fatalf("kept lines missing: %q", pruned)
	}
}

func TestPruneSkillCollapsesBlankRuns(t *testing.T) {
	got := pruneSkill("a\n\n\n\nb\n", nil)
	if strings.Contains(got, "\n\n\n") {
		t.Fatalf("blank run not collapsed: %q", got)
	}
}

func TestPruneSkillFragments(t *testing.T) {
	orig := "Base: arch. Pkg mgr pacman; yay for AUR. Rolling release.\n- probes: no sudo needed here\n"
	// Drop the middle sentence of line 1: the rest of the line survives, seam tidied.
	pruned := pruneSkill(orig, []Claim{{Source: "Pkg mgr pacman; yay for AUR.", StartLine: 1, EndLine: 1, Fragment: true}})
	if strings.Contains(pruned, "pacman") {
		t.Fatalf("dropped fragment survived: %q", pruned)
	}
	if !strings.Contains(pruned, "Base: arch. Rolling release.") {
		t.Fatalf("remaining sentences mangled: %q", pruned)
	}
	if !strings.Contains(pruned, "- probes: no sudo needed here") {
		t.Fatalf("untouched line changed: %q", pruned)
	}
}

func TestPruneSkillFragmentConsumesWholeLine(t *testing.T) {
	orig := "- only sentence on this line.\n- next line stays\n"
	// The fragment is the line's entire content: the orphaned "- " marker has
	// no words left, so the line goes away entirely.
	pruned := pruneSkill(orig, []Claim{{Source: "only sentence on this line.", StartLine: 1, EndLine: 1, Fragment: true}})
	if strings.Contains(pruned, "only sentence") || strings.Contains(strings.TrimSpace(pruned), "\n-\n") {
		t.Fatalf("consumed line not removed cleanly: %q", pruned)
	}
	if !strings.Contains(pruned, "- next line stays") {
		t.Fatalf("untouched line lost: %q", pruned)
	}
	if lead := strings.Split(pruned, "\n")[0]; strings.TrimSpace(lead) == "-" {
		t.Fatalf("orphan bullet survived: %q", pruned)
	}
}

func TestDroppedClaims(t *testing.T) {
	claims := []Claim{
		{ID: "go#00", StartLine: 2, EndLine: 2},
		{ID: "go#01", StartLine: 4, EndLine: 5},
		{ID: "go#02", StartLine: 7, EndLine: 7},
		{ID: "go#03", StartLine: 9, EndLine: 9, Fragment: true},
	}
	results := map[string]ProbeResult{
		"go#00": {ClaimID: "go#00", Keep: false},           // drop
		"go#01": {ClaimID: "go#01", Keep: true},            // keep
		"go#02": {ClaimID: "go#02", Keep: false, Err: "x"}, // errored → keep
		"go#03": {ClaimID: "go#03", Keep: false},           // drop (fragment)
	}
	got := droppedClaims(claims, results)
	if len(got) != 2 || got[0].ID != "go#00" || got[1].ID != "go#03" || !got[1].Fragment {
		t.Fatalf("droppedClaims = %+v", got)
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
