package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log/slog"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// specTestReviewTimeout bounds the one review call; the main model may think first.
const specTestReviewTimeout = 10 * time.Minute

// Caps on what the review is shown: the round's test diff, and the names of the other test files.
const (
	specTestReviewDiffBytes  = 48 * 1024
	specTestReviewNamesBytes = 12 * 1024
)

type specTestFinding struct {
	Test    string `json:"test"`
	Problem string `json:"problem"`
	Fix     string `json:"fix"`
}

// reviewTests asks a fresh call, which has no stake in the round, whether the tests the round
// wrote are needed and only the item's: 538 test files had grown in one project, 354 of them
// named after no item, and one layout change broke 13 that each measured the same divider.
// It returns one line per finding; a review that cannot be had is said and returns nothing.
func (r *specRun) reviewTests(ctx context.Context, item, base string, ch specChanges) []string {
	var changed []string
	for _, f := range ch.files() {
		if isTestPath(f) {
			changed = append(changed, filepath.ToSlash(filepath.Join(r.cfg.OutDir, f)))
		}
	}
	if len(changed) == 0 {
		return nil
	}
	r.a.cfgMu.RLock()
	conn := r.a.settings.ConnAt(0, "execute")
	r.a.cfgMu.RUnlock()
	if conn == nil {
		return nil
	}
	// New files show in the diff through the same intent-to-add index the round's gates use.
	env, cleanup := intentToAddIndex(ctx, r.sess.Cwd, r.cfg.OutDir)
	defer cleanup()
	c := exec.CommandContext(ctx, "git", append([]string{"-C", r.sess.Cwd, "-c", "core.quotePath=false", "diff", base, "--find-renames", "--no-color", "--no-ext-diff", "--"}, changed...)...)
	c.Env = env
	out, err := c.CombinedOutput()
	if err != nil {
		slog.Debug("spec: test review diff", "item", item, "err", err, "out", truncate(string(out), 200))
		return nil
	}
	var others []string
	walkErr := filepath.WalkDir(r.outAbs, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if path != r.outAbs && skipWalkDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(r.sess.Cwd, path)
		if err == nil && isTestPath(rel) && !slices.Contains(changed, filepath.ToSlash(rel)) {
			others = append(others, filepath.ToSlash(rel))
		}
		return nil
	})
	if walkErr != nil {
		slog.Debug("spec: test review names", "err", walkErr)
	}
	slices.Sort(others)
	names := "(none)"
	if len(others) > 0 {
		names = clipBytes(strings.Join(others, "\n"), specTestReviewNamesBytes)
	}
	prompt := r.a.renderSpecPrompt(r.sid, "SPEC-TESTS.md", []string{
		"{{id}}", item, "{{section}}", r.idx.slice(item, r.cfg.SpecDir).Text,
		"{{diff}}", clipBytes(string(out), specTestReviewDiffBytes), "{{tests}}", names,
	})
	tctx, cancel := context.WithTimeout(ctx, specTestReviewTimeout)
	defer cancel()
	reply, _, _, err := r.a.llmStream(tctx, r.sid, conn, []llmMessage{{Role: "user", Content: prompt}}, nil, nil, nil, nil)
	var v struct {
		Findings []specTestFinding `json:"findings"`
	}
	if err == nil {
		start, end := strings.Index(reply, "{"), strings.LastIndex(reply, "}")
		switch {
		case start < 0 || end < start:
			err = fmt.Errorf("the reply holds no JSON object: %q", clipBytes(strings.TrimSpace(reply), 400))
		default:
			if jerr := json.Unmarshal([]byte(reply[start:end+1]), &v); jerr != nil {
				err = fmt.Errorf("the reply's JSON does not parse (%v): %q", jerr, clipBytes(reply[start:end+1], 400))
			}
		}
	}
	if err != nil {
		r.a.logSession(r.sid, "TESTS", "%s: no review: %v", item, err)
		r.say(ctx, fmt.Sprintf("⚠ test review of %s could not be had, so it holds nothing back: %v\n", item, err))
		return nil
	}
	var lines []string
	for _, f := range v.Findings {
		if strings.TrimSpace(f.Test+f.Problem) == "" {
			continue
		}
		line := fmt.Sprintf("`%s`: %s", f.Test, f.Problem)
		if f.Fix != "" {
			line += "; fix: " + f.Fix
		}
		lines = append(lines, line)
	}
	r.a.logSession(r.sid, "TESTS", "%s: %d finding(s) in %d changed test file(s)\n%s", item, len(lines), len(changed), strings.Join(lines, "\n"))
	if len(lines) == 0 {
		r.say(ctx, fmt.Sprintf("🧾 the tests this round wrote for %s are needed and its own\n", item))
	} else {
		r.say(ctx, fmt.Sprintf("🧾 test review of %s: %d finding(s)\n", item, len(lines)))
	}
	return lines
}
