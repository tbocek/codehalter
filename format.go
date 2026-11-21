package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// Every formatter runs as a stdin/stdout filter, never on the path: over ACP the
// file may be a pending Zed edit not yet flushed to disk.
func formatterCmd(path, cwd string) []string {
	switch strings.ToLower(filepath.Ext(path)) {
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

func lookCmd(bin string, args ...string) []string {
	if _, err := exec.LookPath(bin); err != nil {
		return nil
	}
	return append([]string{bin}, args...)
}

// npx is deliberately not used: its cold start would tax every edit.
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

// formatResult is one formatter run. parsed says the formatter accepted the source:
// a formatter that exits non-zero on stdin reports either style it cannot fix or a
// syntax error, and detail carries its first complaint for either case.
type formatResult struct {
	text   string
	parsed bool
	detail string
}

func runFormatter(argv []string, src, dir string) formatResult {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = dir // resolve .prettierrc / .clang-format / .rustfmt.toml near the file
	cmd.Stdin = strings.NewReader(src)
	var out, errOut bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errOut
	if err := cmd.Run(); err != nil {
		return formatResult{detail: firstLine(strings.TrimSpace(errOut.String()))}
	}
	return formatResult{text: out.String(), parsed: true}
}

// formatGuarded formats only when the pre-edit source was already canonical:
// reformatting an unformatted file would bury the real change in a noisy diff.
// A new content the formatter cannot read is left alone here: parseRefusal refuses it.
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
	oldRes := runFormatter(argv, oldContent, dir)
	if !oldRes.parsed || oldRes.text != oldContent {
		return newContent
	}
	formattedNew := runFormatter(argv, newContent, dir)
	if formattedNew.parsed {
		return formattedNew.text
	}
	return newContent
}

// parseRefusal is the guard beside formatGuarded: content that no longer parses must
// not reach the disk. It refuses only when the file parsed BEFORE this write, so a
// source that was already broken is never blamed for staying as it is. The language's
// own parser answers first; a formatter that is also a parser is the fallback, and a
// style-only tool like prettier never refuses anything. An empty return means write it.
func (a *agent) parseRefusal(sid, path, oldContent, newContent string) string {
	// A file that is not on disk yet cannot have been broken by this write: a brand new
	// unparsable file is written like any other, and the next edit to it is judged.
	// Over ACP the editor may hold an unsaved buffer, so its view counts as existence.
	if _, err := os.Stat(path); err != nil && !a.bufferedFile(path) {
		return ""
	}
	if verdict := probeParses(path, newContent); verdict.have {
		if verdict.ok {
			return ""
		}
		if before := probeParses(path, oldContent); !before.have || !before.ok {
			return ""
		}
		return refuseUnparsable(path, verdict.detail)
	}
	cwd := ""
	if sess := a.getSession(sid); sess != nil {
		cwd = sess.Cwd
	}
	argv := formatterCmd(path, cwd)
	if argv == nil {
		return ""
	}
	dir := filepath.Dir(path)
	if oldRes := runFormatter(argv, oldContent, dir); !oldRes.parsed || oldRes.text != oldContent {
		return ""
	}
	newRes := runFormatter(argv, newContent, dir)
	if newRes.parsed {
		return ""
	}
	return refuseUnparsable(path, newRes.detail)
}

// bufferedFile reports whether the editor may hold this path as an unsaved buffer, in
// which case os.Stat says nothing about whether the file exists. Without ACP fs.write
// the write goes straight to disk, so a missing file really is a new one.
func (a *agent) bufferedFile(path string) bool { return a.clientCan("write") }

func refuseUnparsable(path, detail string) string {
	if detail != "" {
		return fmt.Sprintf("refused: %s does not parse after this edit, so NOTHING was written. %s Re-read that region and redo the edit; the file on disk is unchanged.", path, detail)
	}
	return fmt.Sprintf("refused: %s does not parse after this edit, so NOTHING was written. The file on disk is unchanged: re-read that region and redo the edit.", path)
}
