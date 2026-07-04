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

var (
	blankRunRE = regexp.MustCompile(`\n{3,}`)
	spaceRunRE = regexp.MustCompile(`[ \t]{2,}`)
)

// pruneSkill removes the source spans of dropped claims from the original skill
// content, keeping everything else byte-for-byte. Kept and errored claims stay
// (errored = untested, so we never remove a statement we couldn't prove
// redundant).
//
// Whole-line claims delete their [StartLine,EndLine] lines. Fragment claims
// remove just their Source substring from the line — the rest of the line's
// sentences survive. A line left with no words after fragment removal (e.g. an
// orphaned "- " bullet) is deleted; interior double spaces collapse; runs of
// 3+ blank lines collapse to one.
func pruneSkill(orig string, dropped []Claim) string {
	dropLine := map[int]bool{}
	fragsByLine := map[int][]string{}
	for _, c := range dropped {
		if c.Fragment {
			fragsByLine[c.StartLine] = append(fragsByLine[c.StartLine], c.Source)
			continue
		}
		for ln := c.StartLine; ln <= c.EndLine; ln++ {
			dropLine[ln] = true
		}
	}

	lines := strings.Split(orig, "\n")
	kept := make([]string, 0, len(lines))
	for i, l := range lines {
		n := i + 1
		if dropLine[n] {
			continue
		}
		if frags := fragsByLine[n]; len(frags) > 0 {
			for _, src := range frags {
				l = strings.Replace(l, src, "", 1)
			}
			if len(wordBag(l)) == 0 {
				continue // nothing but markers/punctuation left
			}
			// Tidy the seam: collapse interior space runs (keep indentation),
			// drop trailing whitespace.
			lead := l[:len(l)-len(strings.TrimLeft(l, " \t"))]
			l = lead + spaceRunRE.ReplaceAllString(strings.TrimRight(l[len(lead):], " \t"), " ")
		}
		kept = append(kept, l)
	}
	out := strings.Join(kept, "\n")
	out = blankRunRE.ReplaceAllString(out, "\n\n")
	return out
}

// droppedClaims filters to the claims pruneSkill removes: verdict "drop" with
// no error. An errored claim (untested) is left in place.
func droppedClaims(claims []Claim, results map[string]ProbeResult) []Claim {
	var out []Claim
	for _, c := range claims {
		r, ok := results[c.ID]
		if !ok || r.Err != "" || r.Keep {
			continue
		}
		out = append(out, c)
	}
	return out
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
