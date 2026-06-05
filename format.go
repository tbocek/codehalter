package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// formatterCmd returns the argv that formats a file of path's type by reading
// the source on STDIN and writing the formatted source to STDOUT — the filter
// shape every supported formatter is invoked in. We format the content string,
// never the on-disk file, because over ACP the file may be a pending Zed edit
// not yet flushed to disk, so a subprocess touching the path would race it.
// nil means "no known formatter for this extension". cwd lets us prefer a
// project-local prettier over a global one.
func formatterCmd(path, cwd string) []string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".go":
		return lookCmd("gofmt")
	case ".rs":
		return lookCmd("rustfmt", "--emit", "stdout", "--quiet")
	case ".py":
		return lookCmd("ruff", "format", "-")
	case ".zig":
		return lookCmd("zig", "fmt", "--stdin")
	case ".sh", ".bash":
		return lookCmd("shfmt")
	case ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".java", ".proto":
		return lookCmd("clang-format", "--assume-filename="+path)
	case ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".jsonc",
		".css", ".scss", ".less", ".html", ".vue", ".md", ".mdx",
		".yaml", ".yml", ".graphql":
		if bin := prettierBin(cwd); bin != "" {
			return []string{bin, "--stdin-filepath", path}
		}
	}
	return nil
}

// lookCmd returns {bin, args...} when bin is on PATH, else nil.
func lookCmd(bin string, args ...string) []string {
	if _, err := exec.LookPath(bin); err != nil {
		return nil
	}
	return append([]string{bin}, args...)
}

// prettierBin prefers the project-local prettier (the version the repo pins)
// over a global one; "" when neither exists. npx is deliberately NOT used — its
// cold start would tax every edit.
func prettierBin(cwd string) string {
	if cwd != "" {
		local := filepath.Join(cwd, "node_modules", ".bin", "prettier")
		if st, err := os.Stat(local); err == nil && !st.IsDir() {
			return local
		}
	}
	if p, err := exec.LookPath("prettier"); err == nil {
		return p
	}
	return ""
}

// runFormatter pipes src through the formatter argv and returns its stdout. ok
// is false when the formatter isn't available, errors (e.g. a syntax error in
// mid-edit source), or times out — callers then keep the unformatted source.
func runFormatter(argv []string, src, dir string) (string, bool) {
	if len(argv) == 0 {
		return "", false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = dir // resolve .prettierrc / .clang-format / .rustfmt.toml near the file
	cmd.Stdin = strings.NewReader(src)
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return "", false
	}
	return out.String(), true
}

// formatGuarded is defensive auto-formatting: it formats newContent ONLY when
// the pre-edit source was ALREADY canonical (a dry run leaves it unchanged). If
// the file wasn't formatted to begin with, the edit is returned untouched —
// reformatting an unformatted file would reflow unrelated lines and bury the
// real change in a noisy diff (and surprise the model's next edit). A new/empty
// file is clean by definition, so its content is formatted. Returns the content
// to write.
func (a *agent) formatGuarded(sid, path, oldContent, newContent string) string {
	cwd := ""
	if sess := a.getSession(sid); sess != nil {
		cwd = sess.Cwd
	}
	argv := formatterCmd(path, cwd)
	if argv == nil {
		return newContent
	}
	dir := filepath.Dir(path)
	// Dry run on the pre-edit source: only proceed if it was already canonical.
	if formattedOld, ok := runFormatter(argv, oldContent, dir); !ok || formattedOld != oldContent {
		return newContent
	}
	formattedNew, ok := runFormatter(argv, newContent, dir)
	if !ok {
		return newContent // mid-edit source doesn't parse — don't impose
	}
	return formattedNew
}
