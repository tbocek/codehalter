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

// Every formatter runs as a stdin/stdout filter, never on the path: over ACP the
// file may be a pending Zed edit not yet flushed to disk.
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

// formatGuarded formats only when the pre-edit source was already canonical:
// reformatting an unformatted file would bury the real change in a noisy diff.
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
	if formattedOld, ok := runFormatter(argv, oldContent, dir); !ok || formattedOld != oldContent {
		return newContent
	}
	formattedNew, ok := runFormatter(argv, newContent, dir)
	if !ok {
		return newContent
	}
	return formattedNew
}
