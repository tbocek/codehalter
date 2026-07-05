package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

//go:embed res/STREAMLINE.md
var streamlinePrompt string

// streamlineCache is the sidecar under .cache/streamline/<skill>.json. Hash
// covers the SOURCE skill content and STREAMLINE.md, so editing either
// regenerates the clean version; the clean file itself lives in clean-skills/.
type streamlineCache struct {
	Hash string `json:"hash"`
}

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
// Cached on hash(source + prompt): a re-run or the prefetch of an unchanged
// skill costs no judge call. Returns the clean file's path.
func ensureCleanSkill(ctx context.Context, judge ModelSpec, skill, srcPath, cleanDir, cacheDir string) (string, error) {
	content, err := os.ReadFile(srcPath)
	if err != nil {
		return "", fmt.Errorf("read skill %s: %w", srcPath, err)
	}
	hash := hashOf([]byte(string(content) + streamlinePrompt))
	cleanPath := filepath.Join(cleanDir, "SKILL-"+skill+".md")
	cachePath := filepath.Join(cacheDir, skill+".json")

	if data, err := os.ReadFile(cachePath); err == nil {
		var c streamlineCache
		if json.Unmarshal(data, &c) == nil && c.Hash == hash {
			if _, err := os.Stat(cleanPath); err == nil {
				return cleanPath, nil // cache hit: source, prompt and output all in place
			}
		}
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
	// statements, and the verbatim original is used instead.
	if missing := missingSpans(string(content), out); len(missing) > 0 || len(out) < len(content)*4/5 {
		if len(missing) > 0 {
			fmt.Fprintf(os.Stderr, "warn: streamline %s dropped content (%d spans missing, e.g. %s) — using the original verbatim; delete %s to retry\n",
				skill, len(missing), truncate(missing[0], 60), cachePath)
		} else {
			fmt.Fprintf(os.Stderr, "warn: streamline %s shrank %d → %d bytes (suspected statement loss) — using the original verbatim; delete %s to retry\n",
				skill, len(content), len(out), cachePath)
		}
		out = string(content)
	}

	if err := os.MkdirAll(cleanDir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(cleanPath, []byte(out), 0o644); err != nil {
		return "", fmt.Errorf("write %s: %w", cleanPath, err)
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}
	data, err := json.Marshal(streamlineCache{Hash: hash})
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(cachePath, data, 0o644); err != nil {
		return "", fmt.Errorf("cache streamline %s: %w", skill, err)
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
