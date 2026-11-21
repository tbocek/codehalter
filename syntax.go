package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// A syntax probe is a parse-only check: unlike a formatter it answers "can this be
// read at all", and unlike a compiler it needs no project context (no classpath, no
// include paths, no tsconfig), because parsing is context-free. That is what lets one
// table cover every language without knowing how the project builds.
//
// Deliberately absent: clang/javac/tsc, whose CLI also reports SEMANTIC errors on a
// lone file (a missing #include, an unresolved symbol), which would refuse valid edits.
// Prettier is absent too: with no parser plugin it accepts anything.
func syntaxProbe(path string) []string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go":
		return lookCmd("gofmt") // gofmt parses: a non-zero exit is a syntax complaint
	case ".rs":
		return lookCmd("rustfmt", "--emit", "stdout", "--quiet")
	case ".sh", ".bash":
		if argv := lookCmd("bash", "-n"); argv != nil {
			return argv
		}
		return lookCmd("sh", "-n")
	case ".py":
		return lookCmd("python3", "-c", "import ast,sys; ast.parse(sys.stdin.read())")
	case ".json":
		return lookCmd("python3", "-c", "import json,sys; json.load(sys.stdin)")
	case ".toml":
		if !tomllibAvailable() {
			return nil
		}
		return lookCmd("python3", "-c", "import tomllib,sys; tomllib.loads(sys.stdin.read())")
	case ".zig":
		return lookCmd("zig", "ast-check")
	case ".rb":
		return lookCmd("ruby", "-c")
	case ".mjs", ".cjs":
		// node --check takes a file, never stdin: probeParses writes the candidate out.
		// .js is deliberately not probed: node picks the module kind from the nearest
		// package.json, and a temp copy cannot carry that, so valid ESM in a .js would be
		// refused as "Cannot use import statement outside a module". .mjs/.cjs are
		// unambiguous by their own extension.
		return lookCmd("node", "--check")
	case ".php":
		return lookCmd("php", "-l")
	}
	return nil
}

// Probes that take a file rather than stdin. The extension decides where the content
// goes, since these tools pick their parser from the file name.
func probeNeedsFile(argv []string) bool {
	if len(argv) == 0 {
		return false
	}
	switch filepath.Base(argv[0]) {
	case "node", "php":
		return true
	}
	return false
}

// tomllib landed in the stdlib in 3.11. An older python3 fails on valid TOML, so the
// probe is dropped after one failed run against an empty document.
var tomllibCheck = sync.OnceValue(func() bool {
	argv := lookCmd("python3", "-c", "import tomllib,sys; tomllib.loads(sys.stdin.read())")
	if argv == nil {
		return false
	}
	res := runFilter(argv, "", "")
	return res.parsed
})

func tomllibAvailable() bool { return tomllibCheck() }

type syntaxVerdict struct {
	ok     bool
	detail string
	have   bool
}

// firstErrorLine drops a Python traceback's preamble and quotes the exception line,
// which is the part that says what is wrong. Other tools are already down to one line.
func firstErrorLine(tool, detail string) string {
	if tool != "python3" || !strings.Contains(detail, "Traceback") {
		return detail
	}
	lines := strings.Split(detail, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if ln := strings.TrimSpace(lines[i]); strings.Contains(ln, "Error:") {
			return ln
		}
	}
	return detail
}

// firstErrorLine drops a Python traceback's preamble and quotes the exception line,
// which is the part that says what is wrong. Other tools are already down to one line.
// probeParses answers whether content can be read by the language's own parser.
// have=false means nothing here can tell (no probe for the extension, or its tool is
// not installed): the write must go through, never blocked for a missing tool.
func probeParses(path, content string) syntaxVerdict {
	argv := syntaxProbe(path)
	if argv == nil {
		return syntaxVerdict{}
	}
	if probeNeedsFile(argv) {
		tmp, err := os.CreateTemp("", "codehalter-probe-*"+strings.ToLower(filepath.Ext(path)))
		if err != nil {
			return syntaxVerdict{have: true}
		}
		defer os.Remove(tmp.Name())
		if _, err := tmp.WriteString(content); err != nil {
			tmp.Close()
			return syntaxVerdict{have: true}
		}
		if err := tmp.Close(); err != nil {
			return syntaxVerdict{have: true}
		}
		res := runFilter(append(argv, tmp.Name()), "", filepath.Dir(path))
		return syntaxVerdict{ok: res.parsed, detail: res.detail, have: true}
	}
	res := runFilter(argv, content, filepath.Dir(path))
	return syntaxVerdict{ok: res.parsed, detail: res.detail, have: true}
}

// runFilter runs argv with src on stdin, keeping stdout and stderr's telling line.
func runFilter(argv []string, src, dir string) formatResult {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Stdin = strings.NewReader(src)
	var out, errOut bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errOut
	if err := cmd.Run(); err != nil {
		detail := firstErrorLine(filepath.Base(argv[0]), strings.TrimSpace(errOut.String()))
		return formatResult{detail: detail}
	}
	return formatResult{text: out.String(), parsed: true}
}
