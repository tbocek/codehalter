package main

import (
	"regexp"
	"strings"
)

// SkillStats is the size accounting for one skill on one model: how many claims
// were kept (statement needed → stays in the pruned skill), dropped (redundant
// → removed), or errored (couldn't be tested → kept, to be safe), and the byte
// sizes before and after pruning.
type SkillStats struct {
	Skill       string `json:"skill"`
	Claims      int    `json:"claims"`
	Kept        int    `json:"kept"`
	Dropped     int    `json:"dropped"`
	Errored     int    `json:"errored"`
	OrigBytes   int    `json:"orig_bytes"`
	PrunedBytes int    `json:"pruned_bytes"`
}

// ModelStats aggregates a target model's per-skill results plus totals — the
// unit written to models/<name>/stats.json and rendered in the report.
type ModelStats struct {
	Model       string       `json:"model"`
	Skills      []SkillStats `json:"skills"`
	OrigBytes   int          `json:"orig_bytes"`
	PrunedBytes int          `json:"pruned_bytes"`
}

var blankRunRE = regexp.MustCompile(`\n{3,}`)

// pruneSkill removes the source spans of dropped claims from the original skill
// content, keeping everything else byte-for-byte. Kept and errored claims stay
// (errored = untested, so we never remove a statement we couldn't prove
// redundant). Runs of 3+ blank lines left by removals collapse to one.
//
// dropLines is the set of 1-based line numbers to delete, precomputed by the
// caller from each dropped claim's [StartLine,EndLine] span.
func pruneSkill(orig string, dropLines map[int]bool) string {
	lines := strings.Split(orig, "\n")
	kept := make([]string, 0, len(lines))
	for i, l := range lines {
		if dropLines[i+1] {
			continue
		}
		kept = append(kept, l)
	}
	out := strings.Join(kept, "\n")
	out = blankRunRE.ReplaceAllString(out, "\n\n")
	return out
}

// dropLineSet turns the dropped claims into the line-number set pruneSkill
// deletes. Only claims whose Keep is false AND that carry no error are dropped;
// an errored claim (untested) is left in place.
func dropLineSet(claims []Claim, results map[string]ProbeResult) map[int]bool {
	drop := map[int]bool{}
	for _, c := range claims {
		r, ok := results[c.ID]
		if !ok || r.Err != "" || r.Keep {
			continue
		}
		for ln := c.StartLine; ln <= c.EndLine; ln++ {
			drop[ln] = true
		}
	}
	return drop
}

// skillStats tallies keep/drop/error counts and byte sizes for one skill.
func skillStats(skill, orig, pruned string, claims []Claim, results map[string]ProbeResult) SkillStats {
	st := SkillStats{
		Skill:       skill,
		Claims:      len(claims),
		OrigBytes:   len(orig),
		PrunedBytes: len(pruned),
	}
	for _, c := range claims {
		r, ok := results[c.ID]
		switch {
		case !ok || r.Err != "":
			st.Errored++
		case r.Keep:
			st.Kept++
		default:
			st.Dropped++
		}
	}
	return st
}
