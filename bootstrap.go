package main

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// .gitignore
// ---------------------------------------------------------------------------

// ensureGitignore makes sure .gitignore has an entry for .codehalter/ — either
// a real ignore line or a comment marking it as intentionally tracked. Asks
// the user once per repo; later sessions short-circuit because the written
// entry is detected on next startup.
//
// Only runs inside a git-managed directory: requires either a .git directory
// in cwd or an existing .gitignore.
func (a *agent) ensureGitignore(ctx context.Context, cwd string, sid SessionId) {
	gitignorePath := filepath.Join(cwd, ".gitignore")
	hasGit := dirExists(filepath.Join(cwd, ".git"))
	hasGitignore := fileExists(gitignorePath)
	if !hasGit && !hasGitignore {
		return
	}

	data, _ := os.ReadFile(gitignorePath)
	content := string(data)

	for _, line := range strings.Split(content, "\n") {
		if strings.Contains(strings.ToLower(line), "codehalter") {
			return
		}
	}

	tcId := a.StartToolCall(ctx, sid, "Add .codehalter/ to .gitignore?", "think", nil)
	ok, err := a.askYesNoAuto(ctx, sid, tcId, "Ignore", "Track", true)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}

	var entry, note string
	if ok {
		entry = ".codehalter/"
		note = "Added .codehalter/ to .gitignore"
	} else {
		entry = "# .codehalter/ is intentionally tracked"
		note = "Marked .codehalter/ as tracked in .gitignore"
	}

	if len(content) > 0 && !strings.HasSuffix(content, "\n") {
		content += "\n"
	}
	content += entry + "\n"

	if err := os.WriteFile(gitignorePath, []byte(content), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(note)})
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(note+"\n")))
}

// ---------------------------------------------------------------------------
// .devcontainer
// ---------------------------------------------------------------------------

// ensureDevcontainer offers to seed .devcontainer/Dockerfile and
// .devcontainer/devcontainer.json when codehalter is running outside a
// container and the project has no devcontainer config yet.
//
// Asks at most once per project: a "yes" creates the files; a "no" writes
// an empty .codehalter/devcontainer.skip marker so we stop nagging. Delete
// either path (or just .codehalter/devcontainer.skip) to be asked again.
func (a *agent) ensureDevcontainer(ctx context.Context, cwd string, sid SessionId) {
	if containerKind() != "" {
		return
	}
	dir := filepath.Join(cwd, ".devcontainer")
	if dirExists(dir) {
		return
	}
	skipMarker := filepath.Join(cwd, sessionDir, "devcontainer.skip")
	if fileExists(skipMarker) {
		return
	}

	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(
		"To sandbox file edits and task runs, I can write a Debian-based "+
			".devcontainer/Dockerfile and .devcontainer/devcontainer.json template "+
			"you can then edit. Reopen the project in the container to use it. "+
			"Skip is remembered in .codehalter/devcontainer.skip.\n\n")))

	tcId := a.StartToolCall(ctx, sid, "Write .devcontainer/Dockerfile and devcontainer.json?", "think", nil)
	ok, err := a.askYesNoAuto(ctx, sid, tcId, "Create", "Skip", true)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if !ok {
		_ = os.WriteFile(skipMarker, nil, 0o644)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Skipped")})
		return
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if err := os.WriteFile(filepath.Join(dir, "Dockerfile"), []byte(defaultDevcontainerDockerfile), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if err := os.WriteFile(filepath.Join(dir, "devcontainer.json"), []byte(defaultDevcontainerJSON), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}

	note := "Wrote .devcontainer/Dockerfile and .devcontainer/devcontainer.json. Reopen the project in the container to use them."
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(note)})
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(note+"\n")))
}

// ---------------------------------------------------------------------------
// Stack detection & SKILL-*.md loading
// ---------------------------------------------------------------------------

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

	if dirExists(filepath.Join(cwd, ".devcontainer")) {
		stacks = append(stacks, "devcontainer")
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

// ---------------------------------------------------------------------------
// Path-existence helpers (shared by bootstrap routines above)
// ---------------------------------------------------------------------------

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

func dirExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.IsDir()
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
