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
	Skill        string `json:"skill"`
	Claims       int    `json:"claims"`
	Kept         int    `json:"kept"`
	Strengthened int    `json:"strengthened,omitempty"` // kept, but only after a strengthened retry (subset of Kept)
	Dropped      int    `json:"dropped"`
	Errored      int    `json:"errored"`
	OrigBytes    int    `json:"orig_bytes"`
	PrunedBytes  int    `json:"pruned_bytes"`
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

// replacement is one claim whose source span is swapped for new text in the
// output skill — the strengthened wording a target model needed before it
// followed the statement.
type replacement struct {
	Claim Claim
	Text  string
}

// pruneSkill removes the source spans of dropped claims from the original skill
// content and splices replacement text over strengthened claims' spans, keeping
// everything else byte-for-byte. Kept and errored claims stay (errored =
// untested, so we never remove a statement we couldn't prove redundant).
//
// Whole-line claims delete their [StartLine,EndLine] lines (replacements emit
// their text at StartLine instead). Fragment claims remove (or swap) just their
// Source substring from the line — the rest of the line's sentences survive. A
// line left with no words after fragment removal (e.g. an orphaned "- " bullet)
// is deleted; interior double spaces collapse; runs of 3+ blank lines collapse
// to one.
func pruneSkill(orig string, dropped []Claim, replaced []replacement) string {
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
	replAt := map[int]string{} // whole-line replacement: emit text at StartLine …
	replSkip := map[int]bool{} // … and skip the original StartLine..EndLine
	fragRepl := map[int][]replacement{}
	for _, r := range replaced {
		if r.Claim.Fragment {
			fragRepl[r.Claim.StartLine] = append(fragRepl[r.Claim.StartLine], r)
			continue
		}
		replAt[r.Claim.StartLine] = r.Text
		for ln := r.Claim.StartLine; ln <= r.Claim.EndLine; ln++ {
			replSkip[ln] = true
		}
	}

	lines := strings.Split(orig, "\n")
	kept := make([]string, 0, len(lines))
	for i, l := range lines {
		n := i + 1
		if text, ok := replAt[n]; ok {
			kept = append(kept, strings.Split(text, "\n")...)
		}
		if replSkip[n] || dropLine[n] {
			continue
		}
		for _, r := range fragRepl[n] {
			l = strings.Replace(l, r.Claim.Source, r.Text, 1)
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
	// Drop orphaned section titles: a header whose section lost every content
	// line steers nothing (measured: both targets' pruned arch left "## Probe"
	// heading an empty section). A header survives iff some content line sits
	// between it and the next header of the same or shallower level — content
	// under a DEEPER header counts, so a parent with a populated subsection
	// stays.
	level := func(s string) int {
		t := strings.TrimSpace(s)
		n := 0
		for n < len(t) && t[n] == '#' {
			n++
		}
		if n == 0 || n >= len(t) || t[n] != ' ' {
			return 0 // not a header
		}
		return n
	}
	final := make([]string, 0, len(kept))
	for i, l := range kept {
		lv := level(l)
		if lv == 0 {
			final = append(final, l)
			continue
		}
		keepHeader := false
		for j := i + 1; j < len(kept); j++ {
			jlv := level(kept[j])
			if jlv > 0 && jlv <= lv {
				break // next sibling/parent section — ours had no content
			}
			if jlv == 0 && strings.TrimSpace(kept[j]) != "" {
				keepHeader = true
				break
			}
		}
		if keepHeader {
			final = append(final, l)
		}
	}

	out := strings.Join(final, "\n")
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

// strengthenedClaims filters to the claims kept only after a strengthened
// retry: their span is replaced with the strengthened wording in the output.
func strengthenedClaims(claims []Claim, results map[string]ProbeResult) []replacement {
	var out []replacement
	for _, c := range claims {
		r, ok := results[c.ID]
		if !ok || r.Err != "" || !r.Keep || r.StrengthenedText == "" {
			continue
		}
		out = append(out, replacement{Claim: c, Text: r.StrengthenedText})
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
			if r.StrengthenedText != "" {
				st.Strengthened++
			}
		default:
			st.Dropped++
		}
	}
	return st
}
