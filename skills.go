package main

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// detectStacks returns the set of language/stack identifiers active in cwd,
// in a stable order. Used by ensureDefaults (to seed only relevant SKILL files)
// and loadSkills (to assemble the systemPrompt prefix).
func detectStacks(cwd string) []string {
	var stacks []string

	if exists(filepath.Join(cwd, "go.mod")) {
		stacks = append(stacks, "go")
	}

	hasPkg := exists(filepath.Join(cwd, "package.json"))
	hasTS := exists(filepath.Join(cwd, "tsconfig.json")) || hasFileWithExt(cwd, ".ts", ".tsx")
	if hasTS {
		stacks = append(stacks, "ts")
	}
	if hasPkg && !hasTS {
		stacks = append(stacks, "js")
	}

	if exists(filepath.Join(cwd, "pom.xml")) ||
		exists(filepath.Join(cwd, "build.gradle")) ||
		exists(filepath.Join(cwd, "build.gradle.kts")) {
		stacks = append(stacks, "java")
	}

	if hasFileWithExt(cwd, ".sh", ".bash") {
		stacks = append(stacks, "bash")
	}

	return stacks
}

// loadSkills concatenates every SKILL-*.md present in .codehalter/. Detection
// (detectStacks) decides which to seed initially, but loading honors whatever
// the user actually has on disk — drop a SKILL-rust.md in there manually and
// it gets picked up; delete one and it stops loading. Called once per session
// (from systemPrompt) so the concatenated text lives in the first user
// message and stays cache-stable thereafter.
func loadSkills(cwd string) string {
	dir := filepath.Join(cwd, ".codehalter")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		n := e.Name()
		if strings.HasPrefix(n, "SKILL-") && strings.HasSuffix(n, ".md") {
			names = append(names, n)
		}
	}
	if len(names) == 0 {
		return ""
	}
	sort.Strings(names) // deterministic order → stable cache prefix
	var b strings.Builder
	for _, n := range names {
		data, err := os.ReadFile(filepath.Join(dir, n))
		if err != nil {
			continue
		}
		b.Write(data)
		if !strings.HasSuffix(string(data), "\n") {
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}
	return b.String()
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// hasFileWithExt returns true if cwd contains any file (non-recursive) whose
// extension matches one of exts. Cheap directory scan; we only care about the
// root because deeper detection is the user's job to override.
func hasFileWithExt(cwd string, exts ...string) bool {
	entries, err := os.ReadDir(cwd)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := filepath.Ext(e.Name())
		for _, want := range exts {
			if ext == want {
				return true
			}
		}
	}
	return false
}
