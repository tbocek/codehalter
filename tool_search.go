package main

import (
	"bytes"
	"context"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// The search tool's bounds.
const (
	searchDefaultContext = 2
	searchMaxContext     = 10
	searchDefaultLimit   = 50
	searchMaxLimit       = 200
	searchMaxFileBytes   = 2 << 20 // larger files are data, not code
	searchMaxLineRunes   = 300
)

// searchTool exists so the model never has a reason to be in the shell while it explores: sent to
// `grep -rn` through run_command, it read in the same call (`grep ...; sed -n ...`), 68 of one
// step's 152 calls. Qwen's own harness gives search a tool of its own for the same reason.
var searchTool = Tool{Def: map[string]any{
	"type": "function",
	"function": map[string]any{
		"name":        "search",
		"description": "Find text in the project's files: each match with its file, line number and the lines around it, in one call. Use this, never grep, rg or find through run_command. Output is grep's: `path:LINE:text` for a match, `path-LINE-text` around it. Read further with read_file.",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern":     map[string]any{"type": "string", "description": "The text to find, matched literally; with `regex` a regular expression (RE2)."},
				"regex":       map[string]any{"type": "boolean", "description": "Treat `pattern` as a regular expression."},
				"path":        map[string]any{"type": "string", "description": "A file or folder to search in, relative to the project. Default: the whole project."},
				"glob":        map[string]any{"type": "string", "description": "Only files matching it: `*.rs` by name, `tests/**/*.rs` by path."},
				"context":     map[string]any{"type": "integer", "description": fmt.Sprintf("Lines shown before and after each match, 0 to %d (default %d).", searchMaxContext, searchDefaultContext)},
				"ignore_case": map[string]any{"type": "boolean"},
				"limit":       map[string]any{"type": "integer", "description": fmt.Sprintf("At most this many matches (default %d, at most %d).", searchDefaultLimit, searchMaxLimit)},
			},
			"required": []string{"pattern"},
		},
	},
}, Execute: searchExecute}

func searchExecute(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	args := parseArgs(rawArgs)
	pattern := args.str("pattern")
	if pattern == "" {
		return "error: `pattern` is required: the text to find.", true
	}
	sess := a.getSession(sid)
	if sess == nil {
		return "error: no session", true
	}
	expr := pattern
	if !args.flag("regex") {
		expr = regexp.QuoteMeta(pattern)
	}
	if args.flag("ignore_case") {
		expr = "(?i)" + expr
	}
	re, err := regexp.Compile(expr)
	if err != nil {
		return fmt.Sprintf("error: %q is not a regular expression this search reads (RE2: no lookaround, no backreferences): %v. Without `regex` the pattern is matched literally.", pattern, err), true
	}
	scope := sess.Cwd
	if p := args.str("path"); p != "" {
		if scope, err = a.resolvePath(sid, p); err != nil {
			return "error: " + err.Error(), true
		}
		if _, err := os.Stat(scope); err != nil {
			return fmt.Sprintf("error: %s does not exist; leave `path` out to search the whole project.", p), true
		}
	}
	glob, err := searchGlob(args.str("glob"))
	if err != nil {
		return "error: " + err.Error(), true
	}
	around := searchDefaultContext
	if n, ok := args.num("context"); ok {
		around = min(max(n, 0), searchMaxContext)
	}
	limit := searchDefaultLimit
	if n, ok := args.num("limit"); ok {
		limit = min(max(n, 1), searchMaxLimit)
	}

	// A glob matches the file's name, its path in the project, or its path under `path`.
	under := ""
	if r, err := filepath.Rel(sess.Cwd, scope); err == nil && r != "." {
		under = filepath.ToSlash(r) + "/"
	}
	var out strings.Builder
	matches, files := 0, 0
	for _, rel := range searchFiles(ctx, sess.Cwd, scope) {
		if glob != nil && !glob.MatchString(rel) && !glob.MatchString(filepath.Base(rel)) && !glob.MatchString(strings.TrimPrefix(rel, under)) {
			continue
		}
		data, err := os.ReadFile(filepath.Join(sess.Cwd, rel))
		if err != nil || len(data) > searchMaxFileBytes || bytes.IndexByte(data[:min(len(data), 8192)], 0) >= 0 {
			continue // unreadable, data or binary
		}
		// The newline that ends the last line starts no line of its own.
		lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
		var hits []int
		for i, line := range lines {
			if re.MatchString(line) {
				hits = append(hits, i)
			}
		}
		if len(hits) == 0 {
			continue
		}
		files++
		shownTo := -1 // the last line already printed for this file
		for _, h := range hits {
			matches++
			if matches > limit {
				continue // counted for the total, not shown
			}
			from, to := max(h-around, 0), min(h+around, len(lines)-1)
			if shownTo >= 0 && from > shownTo+1 || shownTo < 0 && out.Len() > 0 {
				out.WriteString("--\n")
			}
			for i := max(from, shownTo+1); i <= to; i++ {
				sep := "-"
				if re.MatchString(lines[i]) {
					sep = ":"
				}
				fmt.Fprintf(&out, "%s%s%d%s%s\n", rel, sep, i+1, sep, searchClip(lines[i]))
			}
			shownTo = max(shownTo, to)
		}
	}
	switch {
	case matches == 0:
		return fmt.Sprintf("No match for %q in %s.", pattern, searchWhere(sess.Cwd, scope, args.str("glob"))), false
	case matches > limit:
		fmt.Fprintf(&out, "\nShowing the first %d of %d matches, in %d files: narrow `pattern`, `path` or `glob`, or raise `limit` (at most %d).", limit, matches, files, searchMaxLimit)
	default:
		fmt.Fprintf(&out, "\n%d matches in %d files.", matches, files)
	}
	return out.String(), false
}

// searchFiles lists the files under scope, relative to root: what git tracks plus new files it
// does not ignore, else every file outside the build and dot directories.
func searchFiles(ctx context.Context, root, scope string) []string {
	rel, err := filepath.Rel(root, scope)
	if err != nil {
		return nil
	}
	c := exec.CommandContext(ctx, "git", "-C", root, "-c", "core.quotePath=false", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", filepath.ToSlash(rel))
	if out, err := c.Output(); err == nil {
		var files []string
		seen := map[string]bool{}
		for _, f := range strings.Split(string(out), "\x00") {
			if f != "" && !seen[f] {
				seen[f] = true
				files = append(files, f)
			}
		}
		return files
	}
	var files []string
	walkErr := filepath.WalkDir(scope, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if path != scope && skipWalkDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if r, err := filepath.Rel(root, path); err == nil {
			files = append(files, filepath.ToSlash(r))
		}
		return nil
	})
	if walkErr != nil {
		slog.Debug("search: walking the project", "scope", scope, "err", walkErr)
	}
	return files
}

// searchGlob turns `*.rs` or `tests/**/*.rs` into a regular expression over a file's path; nil
// for no glob.
func searchGlob(glob string) (*regexp.Regexp, error) {
	if glob == "" {
		return nil, nil
	}
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(glob); i++ {
		switch c := glob[i]; {
		case c == '*' && i+1 < len(glob) && glob[i+1] == '*':
			b.WriteString(".*")
			i++
			if i+1 < len(glob) && glob[i+1] == '/' {
				i++ // `**/` also matches no folder at all
				b.WriteString("/?")
			}
		case c == '*':
			b.WriteString("[^/]*")
		case c == '?':
			b.WriteString("[^/]")
		default:
			b.WriteString(regexp.QuoteMeta(string(c)))
		}
	}
	b.WriteString("$")
	re, err := regexp.Compile(b.String())
	if err != nil {
		return nil, fmt.Errorf("`glob` %q is not a pattern this search reads: %v", glob, err)
	}
	return re, nil
}

func searchClip(line string) string {
	r := []rune(strings.TrimRight(line, "\r"))
	if len(r) <= searchMaxLineRunes {
		return string(r)
	}
	return string(r[:searchMaxLineRunes]) + "…"
}

func searchWhere(root, scope, glob string) string {
	where := "the project"
	if rel, err := filepath.Rel(root, scope); err == nil && rel != "." {
		where = filepath.ToSlash(rel)
	}
	if glob != "" {
		where += " (" + glob + ")"
	}
	return where
}
