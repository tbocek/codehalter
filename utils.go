package main

import (
	"strings"
	"sync"
)

// parallel runs fn for each index [0, n) with up to `cap` concurrent
// goroutines. Callers pass an explicit upper bound matched to the work-list
// (e.g. launch_subagent's SubagentPinOrder length, probeAllLLMs's len(conns))
// so excess work queues instead of contending for slots.
func parallel(n, cap int, fn func(i int)) {
	if cap > n {
		cap = n
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, cap)
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			fn(i)
		}(i)
	}
	wg.Wait()
}

// failureSimilarityThreshold is the Jaccard ratio above which two failed-
// subtask reason bags are considered "the same problem." Tuned empirically
// for short LLM-generated strings: 0.6 catches "missing import" /
// "import is missing" (Jaccard 0.67) without collapsing genuinely different
// files. The orchestrator uses this to escalate the replan context — same
// failure recurring N times tells the planner the prior fix didn't work
// and a structurally different approach is needed.
const failureSimilarityThreshold = 0.6

// issueBag tokenises a list of issue strings into a single set of distinct
// lowercase alphanumeric words. Punctuation, casing and ordering are all
// discarded so two attempts reporting the same root cause in different
// phrasing collapse to comparable bags.
func issueBag(issues []string) map[string]bool {
	bag := make(map[string]bool)
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			bag[cur.String()] = true
			cur.Reset()
		}
	}
	for _, iss := range issues {
		for _, r := range strings.ToLower(iss) {
			switch {
			case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
				cur.WriteRune(r)
			default:
				flush()
			}
		}
		flush()
	}
	return bag
}

// jaccard returns |A ∩ B| / |A ∪ B| for two word sets. 1.0 = identical,
// 0.0 = disjoint. Two empty bags are treated as identical.
func jaccard(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	inter := 0
	for w := range a {
		if b[w] {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// trimJSON extracts a JSON object from an LLM response. Small models often
// wrap the JSON in prose ("Sure, here's the JSON: { … } Let me know!") or
// markdown fences; we just locate the first `{` and the matching `}` and
// keep that slice. Brace counting respects strings + escapes so braces inside
// string values don't confuse the scan. Returns the trimmed input unchanged
// if no balanced object is found — caller surfaces the parse error.
func trimJSON(s string) string {
	s = strings.TrimSpace(s)
	start := strings.IndexByte(s, '{')
	if start < 0 {
		return s
	}
	depth := 0
	inStr := false
	esc := false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return s[start : i+1]
			}
		}
	}
	return s
}
