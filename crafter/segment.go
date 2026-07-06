package main

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

//go:embed res/SEGMENT.md
var segmentPrompt string

//go:embed res/REPAIR.md
var repairPrompt string

// Claim is one atomic behavioral statement pulled from a SKILL file. Text is
// the self-contained rewrite used to author a test question; Source is the
// verbatim span it came from — pruning keeps or drops that exact span so the
// output skill stays byte-faithful to the input (no lossy regeneration).
//
// StartLine/EndLine are 1-based inclusive line numbers located in the original
// file; they are how the pruner removes a dropped claim without disturbing the
// rest of the file's formatting. Fragment marks a Source that is a sub-line
// span — skill files pack several independent sentences into one line, and the
// judge (correctly) claims them one by one; pruning then removes just that
// substring from the line instead of the whole line.
type Claim struct {
	ID        string `json:"id"`
	Skill     string `json:"skill"`
	Text      string `json:"text"`
	Source    string `json:"source"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Fragment  bool   `json:"fragment,omitempty"`
}

// rawClaim is one text+source pair as the judge returns it, before we locate
// the source span. Both segmentation and the repair pass return this shape.
type rawClaim struct {
	Text   string `json:"text"`
	Source string `json:"source"`
}

// segmentReply is the judge's raw output shape before we locate source spans.
type segmentReply struct {
	Claims []rawClaim `json:"claims"`
}

// segmentSkill splits one SKILL file into atomic claims. judgeA does the first
// pass; any claim whose source can't be located verbatim (paraphrased/ambiguous)
// is handed to judgeB in a REPAIR pass that re-quotes the exact span — a
// second-judge cross-check that recovers claims the first pass would have
// silently dropped. Only the flagged claims go to repair, so the ones that were
// already fine keep their (stable) IDs. The result is cached under
// cacheDir/<skill>.json; it depends only on the skill text + the two prompts, so
// a re-run skips both judge calls.
func segmentSkill(ctx context.Context, judgeA, judgeB ModelSpec, skill, skillPath, cacheDir string) ([]Claim, error) {
	content, err := os.ReadFile(skillPath)
	if err != nil {
		return nil, fmt.Errorf("read skill %s: %w", skillPath, err)
	}
	// Cache key covers the skill content AND both prompts: editing any of them
	// invalidates it. Content-only keying silently kept old claims after a
	// SEGMENT.md rule change.
	contentHash := hashOf([]byte(string(content) + segmentPrompt + repairPrompt))

	cachePath := filepath.Join(cacheDir, skill+".json")
	if claims, ok := readClaimCache(cachePath, contentHash); ok {
		// The {{cmd:}} fact filter (see buildClaims) also applies to claims
		// cached before the filter existed — filtering on read retires them
		// without invalidating the whole segmentation.
		return dropCmdFactClaims(skill, claims), nil
	}

	var reply segmentReply
	if err := chatJSON(ctx, judgeA, segmentPrompt, string(content), &reply); err != nil {
		return nil, fmt.Errorf("segment %s: %w", skill, err)
	}
	if len(reply.Claims) == 0 {
		return nil, fmt.Errorf("segment %s: judge returned no claims", skill)
	}

	lines := strings.Split(string(content), "\n")
	located, broken := buildClaims(skill, lines, reply.Claims)

	// Repair pass: re-quote the unlocatable sources on the second judge.
	if len(broken) > 0 {
		repaired, err := repairClaims(ctx, judgeB, skill, string(content), broken)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warn: %s: repair pass failed, %d claim(s) stay unlocatable: %v\n", skill, len(broken), err)
		} else {
			fixed, _ := buildClaims(skill, lines, repaired)
			recovered := map[string]bool{}
			for _, c := range fixed {
				recovered[c.Text] = true
			}
			located = append(located, fixed...)
			var still []rawClaim
			for _, b := range broken {
				if !recovered[strings.TrimSpace(b.Text)] {
					still = append(still, b)
				}
			}
			if len(fixed) > 0 {
				fmt.Fprintf(os.Stderr, "info: %s: repair recovered %d of %d unlocatable claim(s)\n", skill, len(fixed), len(broken))
			}
			broken = still
		}
	}
	for _, b := range broken {
		fmt.Fprintf(os.Stderr, "warn: %s source not locatable as a unique span (paraphrased, ambiguous, or crossing line boundaries), skipping: %q\n", skill, truncate(strings.TrimSpace(b.Source), 80))
	}

	// Dedup by ID: the repair pass could re-quote a span the first pass already
	// located, and a duplicate ID would collide in the downstream caches.
	claims := make([]Claim, 0, len(located))
	seen := map[string]bool{}
	for _, c := range located {
		if seen[c.ID] {
			continue
		}
		seen[c.ID] = true
		claims = append(claims, c)
	}
	if len(claims) == 0 {
		return nil, fmt.Errorf("segment %s: no claims had a locatable verbatim source", skill)
	}

	if err := writeClaimCache(cachePath, contentHash, claims); err != nil {
		return nil, fmt.Errorf("cache claims for %s: %w", skill, err)
	}
	return claims, nil
}

// buildClaims turns raw text+source pairs into located Claims, returning the
// ones whose source is a unique verbatim span plus the raw ones that aren't
// (empty text/source is dropped outright, not returned as broken). The claim ID
// is a content hash of the source span — stable across reordering, so a shifted
// line can't inherit another claim's verdict.
func buildClaims(skill string, lines []string, raw []rawClaim) (located []Claim, broken []rawClaim) {
	for _, rc := range raw {
		text := strings.TrimSpace(rc.Text)
		src := strings.TrimSpace(rc.Source)
		if text == "" || src == "" {
			continue
		}
		// A source carrying a {{cmd:...}} templating directive is a FACT line
		// (environment context, expanded to literal values when the real skill
		// is seeded), not a behavior — SEGMENT.md says to skip these, but the
		// judge ignores that often enough that it must be enforced here
		// (measured: "Base: Fedora ({{cmd:. /etc/os-release …}})" became a
		// claim demanding mandatory os-release verification, and the strengthen
		// loop then spliced that invented mandate into the output skill). Also
		// unprobeable by construction: arm B would show the target the RAW
		// directive, not the value the real agent sees. Skipped, not repaired —
		// the line stays in every output skill verbatim.
		if cmdRE.MatchString(src) {
			fmt.Fprintf(os.Stderr, "info: %s: claim over a {{cmd:}} fact line is not probed (kept verbatim): %q\n", skill, truncate(text, 80))
			continue
		}
		start, end, fragment := locateSpan(lines, src)
		if start == 0 {
			broken = append(broken, rc)
			continue
		}
		located = append(located, Claim{
			ID:        skill + "#" + hashOf([]byte(src))[:8],
			Skill:     skill,
			Text:      text,
			Source:    src,
			StartLine: start,
			EndLine:   end,
			Fragment:  fragment,
		})
	}
	return located, broken
}

// dropCmdFactClaims filters out claims whose source carries a {{cmd:}}
// templating directive — fact lines, never probed (see buildClaims). Applied to
// cache-loaded claims so pre-filter caches retire them too.
func dropCmdFactClaims(skill string, claims []Claim) []Claim {
	out := claims[:0:0]
	for _, c := range claims {
		if cmdRE.MatchString(c.Source) {
			fmt.Fprintf(os.Stderr, "info: %s: claim over a {{cmd:}} fact line is not probed (kept verbatim): %q\n", skill, truncate(c.Text, 80))
			continue
		}
		out = append(out, c)
	}
	return out
}

// repairClaims asks judgeB to re-quote the exact verbatim span for claims whose
// source the first pass couldn't locate, given the full skill text.
func repairClaims(ctx context.Context, judge ModelSpec, skill, content string, broken []rawClaim) ([]rawClaim, error) {
	brokenJSON, err := json.Marshal(broken)
	if err != nil {
		return nil, err
	}
	user := "SKILL (source text):\n" + content + "\n\nBROKEN CLAIMS (source not found verbatim in the SKILL above):\n" + string(brokenJSON)
	var reply segmentReply
	if err := chatJSON(ctx, judge, repairPrompt, user, &reply); err != nil {
		return nil, err
	}
	return reply.Claims, nil
}

// locateSpan finds the 1-based inclusive line range in fileLines that matches
// the verbatim source span, comparing on whitespace-trimmed lines so trailing
// spaces or the judge's minor reflowing don't defeat the match. Blank lines in
// the source are ignored on both sides so a span quoted with surrounding
// blanks still matches.
//
// When no whole-line match exists and the source is a single line, it falls
// back to a sub-line FRAGMENT match: skill files pack several sentences into
// one line and the judge claims them per sentence. The fragment must occur in
// exactly one file line — an ambiguous fragment can't be pruned safely.
// Returns (start, end, fragment); (0,0,false) when nothing matches.
func locateSpan(fileLines []string, source string) (int, int, bool) {
	srcLines := nonBlankTrimmed(strings.Split(source, "\n"))
	if len(srcLines) == 0 {
		return 0, 0, false
	}
	// Walk the file; at each position try to match the whole source sequence,
	// skipping blank file lines between source lines.
	for i := range fileLines {
		if strings.TrimSpace(fileLines[i]) != srcLines[0] {
			continue
		}
		fi, si := i, 0
		lastMatch := i
		for fi < len(fileLines) && si < len(srcLines) {
			ft := strings.TrimSpace(fileLines[fi])
			if ft == "" {
				fi++
				continue
			}
			if ft != srcLines[si] {
				break
			}
			lastMatch = fi
			si++
			fi++
		}
		if si == len(srcLines) {
			return i + 1, lastMatch + 1, false
		}
	}
	if len(srcLines) == 1 {
		// The fragment must occur exactly once in the whole file — counting
		// occurrences, not lines, so a fragment repeated within one line is
		// also rejected (Replace would remove an arbitrary one).
		found, at := 0, 0
		for i, l := range fileLines {
			if n := strings.Count(l, srcLines[0]); n > 0 {
				found += n
				at = i
			}
		}
		if found == 1 {
			return at + 1, at + 1, true
		}
	}
	return 0, 0, false
}

// nonBlankTrimmed trims each line and drops the empties.
func nonBlankTrimmed(lines []string) []string {
	out := make([]string, 0, len(lines))
	for _, l := range lines {
		if t := strings.TrimSpace(l); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// claimCache is the on-disk segmentation cache. Hash is the skill file's
// content hash at segmentation time; a read only hits when it still matches.
type claimCache struct {
	Hash   string  `json:"hash"`
	Claims []Claim `json:"claims"`
}

func readClaimCache(path, contentHash string) ([]Claim, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	var c claimCache
	if err := json.Unmarshal(data, &c); err != nil {
		fmt.Fprintf(os.Stderr, "warn: %s: unreadable segment cache, re-segmenting: %v\n", path, err)
		return nil, false
	}
	if c.Hash != contentHash || len(c.Claims) == 0 {
		return nil, false
	}
	return c.Claims, true
}

func writeClaimCache(path, contentHash string, claims []Claim) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(claimCache{Hash: contentHash, Claims: claims}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// hashOf returns the hex SHA-256 of b — used for the content-invalidated
// segment cache and (its first 8 chars) stable claim IDs.
func hashOf(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
