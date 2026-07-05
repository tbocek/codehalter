package main

import (
	"context"
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

//go:embed res/STREAMLINE.md
var streamlinePrompt string

var (
	codeSpanRE = regexp.MustCompile("`[^`\n]+`")
	cmdRE      = regexp.MustCompile(`\{\{cmd:[^}]*\}\}`)
)

// ensureCleanSkill produces clean-skills/SKILL-<skill>.md: the ground skill
// rewritten by the main model so headers are pure category labels and every
// instruction lives in a body bullet. Recategorization is allowed; dropping
// statements is not — a mechanical guard (all code spans + {{cmd:}} directives
// must survive, size must not collapse) falls back to the verbatim original
// when the rewrite loses content, so the pipeline never runs on a lossy base.
//
// The clean file is generated ONCE, when it's absent, and then owned by you: if
// it already exists it is used as-is — no judge call, no hash check — so you can
// hand-edit clean-skills/SKILL-<skill>.md and the run respects your version.
// Delete the file to regenerate it from the ground skill. (Editing it re-triggers
// segmentation, which is keyed on the clean content.) Returns the clean path.
func ensureCleanSkill(ctx context.Context, judge ModelSpec, skill, srcPath, cleanDir string) (string, error) {
	cleanPath := filepath.Join(cleanDir, "SKILL-"+skill+".md")
	if _, err := os.Stat(cleanPath); err == nil {
		return cleanPath, nil // already present → yours to edit; never overwritten
	}

	content, err := os.ReadFile(srcPath)
	if err != nil {
		return "", fmt.Errorf("read skill %s: %w", srcPath, err)
	}
	out, err := chat(ctx, judge, streamlinePrompt, string(content))
	if err != nil {
		return "", fmt.Errorf("streamline %s: %w", skill, err)
	}
	// Models love fencing whole files despite instructions — unwrap one
	// full-file fence, tolerating a language tag on the opening line.
	out = strings.TrimSpace(out)
	if strings.HasPrefix(out, "```") {
		if i := strings.IndexByte(out, '\n'); i >= 0 {
			out = out[i+1:]
		}
		out = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(out), "```"))
	}
	out += "\n"

	// No-loss guard: recategorization is fine, dropped statements are not.
	// Code spans and {{cmd:}} directives are the least paraphrasable content —
	// any of them missing (or a collapsed size) means the rewrite lost
	// statements, and the verbatim original is written instead.
	if missing := missingSpans(string(content), out); len(missing) > 0 || len(out) < len(content)*4/5 {
		if len(missing) > 0 {
			fmt.Fprintf(os.Stderr, "warn: streamline %s dropped content (%d spans missing, e.g. %s) — wrote the original verbatim; edit or delete %s to change it\n",
				skill, len(missing), truncate(missing[0], 60), cleanPath)
		} else {
			fmt.Fprintf(os.Stderr, "warn: streamline %s shrank %d → %d bytes (suspected statement loss) — wrote the original verbatim; edit or delete %s to change it\n",
				skill, len(content), len(out), cleanPath)
		}
		out = string(content)
	}

	if err := os.MkdirAll(cleanDir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(cleanPath, []byte(out), 0o644); err != nil {
		return "", fmt.Errorf("write %s: %w", cleanPath, err)
	}
	return cleanPath, nil
}

// missingSpans returns the original's code spans and {{cmd:}} directives that
// no longer appear in the rewrite — each one is a lost statement fragment.
func missingSpans(orig, clean string) []string {
	var missing []string
	seen := map[string]bool{}
	for _, re := range []*regexp.Regexp{codeSpanRE, cmdRE} {
		for _, m := range re.FindAllString(orig, -1) {
			if seen[m] {
				continue
			}
			seen[m] = true
			if !strings.Contains(clean, m) {
				missing = append(missing, m)
			}
		}
	}
	return missing
}
