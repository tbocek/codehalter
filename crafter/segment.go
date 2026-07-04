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

// segmentReply is the judge's raw output shape before we locate source spans.
type segmentReply struct {
	Claims []struct {
		Text   string `json:"text"`
		Source string `json:"source"`
	} `json:"claims"`
}

// segmentSkill splits one SKILL file into atomic claims using the judge model,
// caching the result under cacheDir/<skill>.json. Segmentation depends only on
// the skill text and the judge (not on any target model), so caching lets a
// resumed or re-run job skip the judge call entirely.
func segmentSkill(ctx context.Context, judge ModelSpec, skill, skillPath, cacheDir string) ([]Claim, error) {
	content, err := os.ReadFile(skillPath)
	if err != nil {
		return nil, fmt.Errorf("read skill %s: %w", skillPath, err)
	}
	contentHash := hashOf(content)

	// Cache is keyed by the skill's content hash: editing the skill file
	// invalidates it, so we never reuse stale claims (whose line spans would
	// point at the wrong lines of the edited file).
	cachePath := filepath.Join(cacheDir, skill+".json")
	if claims, ok := readClaimCache(cachePath, contentHash); ok {
		return claims, nil
	}

	var reply segmentReply
	if err := chatJSON(ctx, judge, segmentPrompt, string(content), &reply); err != nil {
		return nil, fmt.Errorf("segment %s: %w", skill, err)
	}
	if len(reply.Claims) == 0 {
		return nil, fmt.Errorf("segment %s: judge returned no claims", skill)
	}

	lines := strings.Split(string(content), "\n")
	claims := make([]Claim, 0, len(reply.Claims))
	for i, rc := range reply.Claims {
		text := strings.TrimSpace(rc.Text)
		src := strings.TrimSpace(rc.Source)
		if text == "" || src == "" {
			continue
		}
		start, end, fragment := locateSpan(lines, src)
		if start == 0 {
			// The judge paraphrased the source instead of copying it (or the
			// fragment is ambiguous), so we can't prune it losslessly. Skip
			// with a warning rather than emit a claim we can't act on.
			fmt.Fprintf(os.Stderr, "warn: %s claim %d source not locatable as a unique span (paraphrased, ambiguous, or crossing line boundaries), skipping: %q\n", skill, i, truncate(src, 80))
			continue
		}
		claims = append(claims, Claim{
			// ID is a content hash of the source span, not a positional index,
			// so it's stable across reordering/insertion in the skill file —
			// results.jsonl and the authored-probe cache key on it, and a
			// shifted line must not silently inherit another claim's verdict.
			ID:        skill + "#" + hashOf([]byte(src))[:8],
			Skill:     skill,
			Text:      text,
			Source:    src,
			StartLine: start,
			EndLine:   end,
			Fragment:  fragment,
		})
	}
	if len(claims) == 0 {
		return nil, fmt.Errorf("segment %s: no claims had a locatable verbatim source", skill)
	}

	if err := writeClaimCache(cachePath, contentHash, claims); err != nil {
		return nil, fmt.Errorf("cache claims for %s: %w", skill, err)
	}
	return claims, nil
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
