package main

// The mutation gate: a test that passes with the fix taken back out proves nothing about the fix.
// In naivepost, putting each change back by hand and running the item's tests found five green
// tests that held nothing: a preview fix whose test never reached the state it fixed, a take band
// whose test passed with the old behaviour, a resize whose "fix" was the same line rewritten. So
// after a green final judgement each production hunk the round wrote is undone alone, the tests
// that name the item run again, and a hunk they still pass without is the round's finding.

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// specMutationHunks is how many changes one round puts back by default: each costs a filtered test
// run, and the largest changes carry the most behaviour.
const specMutationHunks = 6

// specMutantFile holds the file a mutant run changed, so a codehalter stopped mid-run puts it back.
const specMutantFile = "spec-mutant.json"

// specHunk is one change the round made to a program file: the new lines from start (1-based;
// for a pure deletion, the line the old lines followed) and the old lines they replaced.
type specHunk struct {
	file     string // relative to the output directory
	start    int
	count    int
	old, new []string
}

// where names the hunk as a finding names it: the file, the line and the first changed code line.
func (h specHunk) where() string {
	first := ""
	for _, l := range append(slices.Clone(h.new), h.old...) {
		if t := strings.TrimSpace(l); t != "" {
			first = t
			break
		}
	}
	if len(first) > 80 {
		first = first[:77] + "..."
	}
	return fmt.Sprintf("%s:%d (`%s`)", h.file, h.start, first)
}

// undo is content with this hunk taken back out: its new lines replaced by the old ones.
func (h specHunk) undo(content string) string {
	lines := strings.SplitAfter(content, "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	at := h.start - 1
	if h.count == 0 {
		at = h.start
	}
	at = min(max(at, 0), len(lines))
	end := min(at+h.count, len(lines))
	old := make([]string, len(h.old))
	for i, l := range h.old {
		old[i] = l + "\n"
	}
	return strings.Join(slices.Concat(lines[:at], old, lines[end:]), "")
}

var hunkFullRe = regexp.MustCompile(`^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@`)

// specRoundHunks lists the round's changes to program code that can each be undone alone, largest
// first, at most limit. Left out: test code (a test file, an inline test module), a new file (taking
// it out whole only breaks the build), moved lines, and a hunk whose code is the same once comments
// are dropped and strings blanked (a reworded log line or comment is no behaviour a test owes).
func specRoundHunks(ctx context.Context, cwd, outRel, base string, ch specChanges, limit int) []specHunk {
	env, cleanup := intentToAddIndex(ctx, cwd, outRel)
	defer cleanup()
	c := exec.CommandContext(ctx, "git", "-C", cwd, "-c", "core.quotePath=false", "diff", base, "--find-renames", "--no-color", "--no-ext-diff", "--no-prefix", "--relative", "-U0", "--", outRel)
	c.Env = env
	out, err := c.Output()
	if err != nil {
		slog.Debug("spec: mutation diff", "err", err)
		return nil
	}
	prefix := filepath.ToSlash(outRel) + "/"
	var all []specHunk
	cur, fresh := "", false
	var h *specHunk
	flush := func() {
		if h != nil && cur != "" && !fresh {
			all = append(all, *h)
		}
		h = nil
	}
	for _, l := range strings.Split(string(out), "\n") {
		switch {
		case strings.HasPrefix(l, "diff --git "):
			flush()
			cur, fresh = "", false
		case h == nil && strings.HasPrefix(l, "--- "):
			fresh = l[4:] == "/dev/null"
		case h == nil && strings.HasPrefix(l, "+++ "):
			cur = strings.TrimPrefix(strings.TrimRight(l[4:], "\t"), prefix)
			if l[4:] == "/dev/null" {
				cur = ""
			}
		case strings.HasPrefix(l, "@@"):
			flush()
			m := hunkFullRe.FindStringSubmatch(l)
			if m == nil {
				continue
			}
			start, _ := strconv.Atoi(m[3])
			count := 1
			if m[4] != "" {
				count, _ = strconv.Atoi(m[4])
			}
			h = &specHunk{file: cur, start: start, count: count}
		case h == nil:
		case strings.HasPrefix(l, "-"):
			h.old = append(h.old, l[1:])
		case strings.HasPrefix(l, "+"):
			h.new = append(h.new, l[1:])
		}
	}
	flush()

	var kept []specHunk
	for _, h := range all {
		data, err := os.ReadFile(filepath.Join(cwd, outRel, h.file))
		if err != nil {
			continue
		}
		prod := prodSource(h.file, string(data))
		if prod == "" || h.start+h.count-1 > strings.Count(prod, "\n") {
			continue
		}
		moved := h.count > 0
		for i := range h.count {
			moved = moved && ch.moved[h.file][h.start+i]
		}
		if moved {
			continue
		}
		code := func(lines []string) string {
			return strings.Join(strings.Fields(codeText(h.file, strings.Join(lines, "\n")+"\n", blankStrings)), " ")
		}
		if code(h.old) == code(h.new) {
			continue
		}
		kept = append(kept, h)
	}
	slices.SortStableFunc(kept, func(a, b specHunk) int { return (len(b.old) + len(b.new)) - (len(a.old) + len(a.new)) })
	return kept[:min(len(kept), limit)]
}

// mutationCmd is the test command narrowed to the tests that match {filter}: spec.toml's
// `mutation_cmd`, else one derived from the detected test command where its tool can filter (a
// justfile `test` recipe that takes an argument, cargo, go). A project that set its own test_cmd,
// or tooling that cannot be filtered, has none unless spec.toml gives one; "off" turns the gate off.
func (r *specRun) mutationCmd() string {
	switch r.cfg.MutationCmd {
	case "off":
		return ""
	case "":
	default:
		return r.cfg.MutationCmd
	}
	if r.cfg.TestCmd != "" {
		return ""
	}
	switch dir := r.outAbs; {
	case justRecipe(dir, "test"):
		if justRecipeTakesArg(dir, "test") {
			return "just test {filter}"
		}
	case fileExists(dir, "Cargo.toml"):
		return "cargo test {filter}"
	case fileExists(dir, "go.mod"):
		return "go test ./... -v -run '(?i){filter}'"
	}
	return ""
}

// justRecipeTakesArg: the recipe's line names a parameter before its colon (`test filter="":`).
func justRecipeTakesArg(dir, name string) bool {
	for _, file := range []string{"justfile", "Justfile", ".justfile"} {
		data, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil {
			continue
		}
		for _, ln := range strings.Split(string(data), "\n") {
			if rest, ok := strings.CutPrefix(ln, name+" "); ok && strings.Contains(rest, ":") {
				return strings.TrimSpace(strings.SplitN(rest, ":", 2)[0]) != ""
			}
		}
	}
	return false
}

// specMutantBackup is the file a mutant run has changed, as it was.
type specMutantBackup struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// restoreSpecMutant puts back a file a stopped mutant run left changed; the path it restored, or "".
func restoreSpecMutant(cwd string) string {
	file := filepath.Join(cwd, sessionDir, specMutantFile)
	data, err := os.ReadFile(file)
	if err != nil {
		return ""
	}
	var b specMutantBackup
	if err := json.Unmarshal(data, &b); err != nil || b.Path == "" {
		slog.Debug("spec: unreadable mutant backup", "err", err)
		return ""
	}
	if err := os.WriteFile(b.Path, []byte(b.Content), 0o644); err != nil {
		slog.Warn("spec: restoring the file a mutant run changed", "path", b.Path, "err", err)
		return ""
	}
	if err := os.Remove(file); err != nil {
		slog.Debug("spec: removing the mutant backup", "err", err)
	}
	return b.Path
}

// tryMutant writes h undone into its file, runs cmd, and puts the file back whatever happened. It
// answers whether the tests passed without the change.
func (r *specRun) tryMutant(ctx context.Context, h specHunk, n, of int, cmd string) (bool, error) {
	path := filepath.Join(r.outAbs, h.file)
	info, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	orig, err := os.ReadFile(path)
	if err != nil {
		return false, err
	}
	backup, err := json.Marshal(specMutantBackup{Path: path, Content: string(orig)})
	if err != nil {
		return false, err
	}
	mark := filepath.Join(r.sess.Cwd, sessionDir, specMutantFile)
	if err := os.WriteFile(mark, backup, 0o644); err != nil {
		return false, err
	}
	defer func() {
		if err := os.WriteFile(path, orig, info.Mode().Perm()); err != nil {
			slog.Warn("spec: putting a mutated file back", "path", path, "err", err)
			return
		}
		if err := os.Remove(mark); err != nil {
			slog.Debug("spec: removing the mutant backup", "err", err)
		}
	}()
	if err := os.WriteFile(path, []byte(h.undo(string(orig))), info.Mode().Perm()); err != nil {
		return false, err
	}
	ok, _, _ := r.a.runCheckCommand(ctx, r.sid, r.outAbs, fmt.Sprintf("Mutant %d/%d: %s put back as it was: %s", n, of, h.where(), cmd), cmd)
	return ok, ctx.Err()
}

// unprovenHunks is the mutation gate: the round's largest program changes, each undone alone, and
// those the item's tests still pass without. The filter is the item's test token, else the stem of
// the test file that names it; a filtered run that fails or does not show the filter on the code as
// it is cannot judge anything, and the gate says so and stands down.
func (r *specRun) unprovenHunks(ctx context.Context, item, base, coveredBy string, ch specChanges) []string {
	limit := cmp.Or(r.cfg.MutationHunks, specMutationHunks)
	tmpl := r.mutationCmd()
	if tmpl == "" || limit < 0 {
		return nil
	}
	hunks := specRoundHunks(ctx, r.sess.Cwd, r.cfg.OutDir, base, ch, limit)
	if len(hunks) == 0 {
		return nil
	}
	filters := []string{specTestToken(item)}
	if stem := strings.TrimSuffix(filepath.Base(coveredBy), filepath.Ext(coveredBy)); stem != "" && stem != filters[0] {
		filters = append(filters, stem)
	}
	cmd := ""
	for _, f := range filters {
		try := strings.ReplaceAll(tmpl, "{filter}", f)
		ok, tail, _ := r.a.runCheckCommand(ctx, r.sid, r.outAbs, "Mutation check, the item's tests as they are: "+try, try)
		if ctx.Err() != nil {
			return nil
		}
		if ok && strings.Contains(tail, f) {
			cmd = try
			break
		}
	}
	if cmd == "" {
		r.say(ctx, fmt.Sprintf("🧬 mutation check skipped: `%s` with %s does not run this item's tests green on the code as it is. `mutation_cmd` in .codehalter/spec.toml sets the command (`{filter}` is the item's test token), `mutation_cmd = \"off\"` none.\n", tmpl, strings.Join(filters, " or ")))
		return nil
	}
	r.say(ctx, fmt.Sprintf("🧬 putting back %d change(s) of this round one at a time: the item's tests must fail without each\n", len(hunks)))
	var survived []string
	for i, h := range hunks {
		passed, err := r.tryMutant(ctx, h, i+1, len(hunks), cmd)
		if errors.Is(err, context.Canceled) || ctx.Err() != nil {
			return nil
		}
		if err != nil {
			slog.Debug("spec: mutant not run", "hunk", h.where(), "err", err)
			continue
		}
		if passed {
			survived = append(survived, h.where())
		}
	}
	if len(survived) > 0 {
		r.say(ctx, fmt.Sprintf("🧬 %d change(s) can be taken out and the item's tests still pass: %s\n", len(survived), strings.Join(survived, "; ")))
	} else {
		r.say(ctx, "🧬 every change put back makes the item's tests fail\n")
	}
	return survived
}
