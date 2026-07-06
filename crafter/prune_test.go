package main

import (
	"strings"
	"testing"
)

func TestPruneSkill(t *testing.T) {
	orig := "# Title\n- keep me\n- drop me\n- keep me too\n"
	pruned := pruneSkill(orig, []Claim{{Source: "- drop me", StartLine: 3, EndLine: 3}}, nil)
	if strings.Contains(pruned, "drop me") {
		t.Fatalf("dropped line survived: %q", pruned)
	}
	if !strings.Contains(pruned, "keep me") || !strings.Contains(pruned, "keep me too") {
		t.Fatalf("kept lines missing: %q", pruned)
	}
}

func TestPruneSkillCollapsesBlankRuns(t *testing.T) {
	got := pruneSkill("a\n\n\n\nb\n", nil, nil)
	if strings.Contains(got, "\n\n\n") {
		t.Fatalf("blank run not collapsed: %q", got)
	}
}

func TestPruneSkillFragments(t *testing.T) {
	orig := "Base: arch. Pkg mgr pacman; yay for AUR. Rolling release.\n- probes: no sudo needed here\n"
	// Drop the middle sentence of line 1: the rest of the line survives, seam tidied.
	pruned := pruneSkill(orig, []Claim{{Source: "Pkg mgr pacman; yay for AUR.", StartLine: 1, EndLine: 1, Fragment: true}}, nil)
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
	pruned := pruneSkill(orig, []Claim{{Source: "only sentence on this line.", StartLine: 1, EndLine: 1, Fragment: true}}, nil)
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

func TestPruneSkillDropsOrphanHeaders(t *testing.T) {
	orig := `# Arch skill
Base line stays.

## Probe
- probe bullet one
- probe bullet two

## Search
- search bullet stays
`
	// Drop both Probe bullets (lines 5-6): the "## Probe" title must go too.
	pruned := pruneSkill(orig, []Claim{{Source: "- probe bullet one", StartLine: 5, EndLine: 5}, {Source: "- probe bullet two", StartLine: 6, EndLine: 6}}, nil)
	if strings.Contains(pruned, "## Probe") {
		t.Fatalf("orphan header survived: %q", pruned)
	}
	for _, want := range []string{"# Arch skill", "Base line stays.", "## Search", "- search bullet stays"} {
		if !strings.Contains(pruned, want) {
			t.Fatalf("kept content missing %q: %q", want, pruned)
		}
	}
}

func TestPruneSkillOrphanHeaderNesting(t *testing.T) {
	orig := "## Parent\n### Child\n- child bullet\n"
	// Nothing dropped: parent kept via populated subsection.
	if got := pruneSkill(orig, nil, nil); !strings.Contains(got, "## Parent") {
		t.Fatalf("parent with populated subsection must stay: %q", got)
	}
	// Child's only bullet dropped → child AND parent are orphans.
	got := pruneSkill(orig, []Claim{{Source: "- child bullet", StartLine: 3, EndLine: 3}}, nil)
	if strings.Contains(got, "## Parent") || strings.Contains(got, "### Child") {
		t.Fatalf("cascading orphan headers survived: %q", got)
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

func TestPruneSkillReplacesStrengthenedSpan(t *testing.T) {
	orig := "# Title\n- keep me\n- weak wording here\n- keep me too\n"
	pruned := pruneSkill(orig, nil, []replacement{{
		Claim: Claim{Source: "- weak wording here", StartLine: 3, EndLine: 3},
		Text:  "- STRONG wording: always do it, even when you think otherwise.",
	}})
	if strings.Contains(pruned, "weak wording") {
		t.Fatalf("original span survived: %q", pruned)
	}
	if !strings.Contains(pruned, "- STRONG wording: always do it, even when you think otherwise.") {
		t.Fatalf("replacement missing: %q", pruned)
	}
	if !strings.Contains(pruned, "- keep me\n") || !strings.Contains(pruned, "- keep me too") {
		t.Fatalf("neighbors damaged: %q", pruned)
	}
}

func TestPruneSkillReplacesFragment(t *testing.T) {
	orig := "Base: arch. Weak middle sentence. Rolling release.\n"
	pruned := pruneSkill(orig, nil, []replacement{{
		Claim: Claim{Source: "Weak middle sentence.", StartLine: 1, EndLine: 1, Fragment: true},
		Text:  "STRONG middle sentence, no exceptions.",
	}})
	if !strings.Contains(pruned, "Base: arch. STRONG middle sentence, no exceptions. Rolling release.") {
		t.Fatalf("fragment replacement wrong: %q", pruned)
	}
}

func TestPruneSkillDropAndReplaceTogether(t *testing.T) {
	orig := "- a\n- drop me\n- strengthen me\n- z\n"
	pruned := pruneSkill(orig,
		[]Claim{{Source: "- drop me", StartLine: 2, EndLine: 2}},
		[]replacement{{Claim: Claim{Source: "- strengthen me", StartLine: 3, EndLine: 3}, Text: "- MUCH STRONGER"}})
	want := "- a\n- MUCH STRONGER\n- z\n"
	if pruned != want {
		t.Fatalf("combined prune = %q, want %q", pruned, want)
	}
}

func TestStrengthenedClaims(t *testing.T) {
	claims := []Claim{{ID: "s#00"}, {ID: "s#01"}, {ID: "s#02"}, {ID: "s#03"}}
	results := map[string]ProbeResult{
		"s#00": {Keep: true, StrengthenedText: "stronger"},          // → replacement
		"s#01": {Keep: true},                                        // plain keep
		"s#02": {Keep: false, StrengthenedText: "tried but failed"}, // drop → no replacement
		"s#03": {Keep: true, StrengthenedText: "x", Err: "boom"},    // errored → no replacement
	}
	got := strengthenedClaims(claims, results)
	if len(got) != 1 || got[0].Claim.ID != "s#00" || got[0].Text != "stronger" {
		t.Fatalf("strengthenedClaims = %+v", got)
	}
}
