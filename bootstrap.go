package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
)

// ---------------------------------------------------------------------------
// .gitignore
// ---------------------------------------------------------------------------

// ensureGitignore makes sure .gitignore mentions .codehalter/ (as an ignore
// line or a tracked-on-purpose marker). Asks once per repo — later sessions
// short-circuit on the existing entry. Skipped outside git-managed dirs
// (requires .git/ or an existing .gitignore).
func (a *agent) ensureGitignore(ctx context.Context, cwd string, sid string) {
	gitignorePath := filepath.Join(cwd, ".gitignore")
	gitInfo, gitErr := os.Stat(filepath.Join(cwd, ".git"))
	hasGit := gitErr == nil && gitInfo.IsDir()
	ignoreInfo, ignoreErr := os.Stat(gitignorePath)
	hasGitignore := ignoreErr == nil && !ignoreInfo.IsDir()
	if !hasGit && !hasGitignore {
		return
	}

	var content string
	if hasGitignore {
		data, err := os.ReadFile(gitignorePath)
		if err != nil {
			a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "Failed to read .gitignore: " + err.Error() + "\n"}})
			return
		}
		content = string(data)
		for _, line := range strings.Split(content, "\n") {
			if strings.Contains(strings.ToLower(line), "codehalter") {
				return
			}
		}
	}

	title := "Add .codehalter/ to .gitignore?"
	labels := []string{"Ignore", "Track"}
	if !hasGitignore {
		title = "No .gitignore found — create one for .codehalter/?"
		labels = []string{"Add .gitignore, ignore .codehalter", "Add .gitignore, track .codehalter"}
	}
	tcId := a.StartToolCall(ctx, sid, title, "think", nil)
	choice, err := a.askChoiceAuto(ctx, sid, tcId, labels)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}

	var entry, note string
	switch choice {
	case labels[0]:
		entry, note = ".codehalter/", "Added .codehalter/ to .gitignore"
	case labels[1]:
		entry, note = "# .codehalter/ is intentionally tracked", "Marked .codehalter/ as tracked in .gitignore"
	default:
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Cancelled")})
		return
	}

	if hasGitignore {
		sep := ""
		if !strings.HasSuffix(content, "\n") {
			sep = "\n"
		}
		f, err := os.OpenFile(gitignorePath, os.O_APPEND|os.O_WRONLY, 0o644)
		if err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return
		}
		_, writeErr := f.WriteString(sep + entry + "\n")
		closeErr := f.Close()
		if writeErr != nil {
			a.FailToolCall(ctx, sid, tcId, writeErr.Error())
			return
		}
		if closeErr != nil {
			a.FailToolCall(ctx, sid, tcId, closeErr.Error())
			return
		}
	} else {
		if err := os.WriteFile(gitignorePath, []byte(entry+"\n"), 0o644); err != nil {
			a.FailToolCall(ctx, sid, tcId, err.Error())
			return
		}
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(note)})
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: note + "\n"}})
}

// ---------------------------------------------------------------------------
// .devcontainer
// ---------------------------------------------------------------------------

// ensureDevcontainer offers to seed .devcontainer/Dockerfile +
// devcontainer.json when codehalter runs outside a container and none exists.
// Re-asks every session while unsandboxed — accepting creates the dir (future
// sessions then short-circuit); skipping only dismisses for this session.
func (a *agent) ensureDevcontainer(ctx context.Context, cwd string, sid string) {
	if containerKind() != "" {
		return
	}
	dir := filepath.Join(cwd, ".devcontainer")
	if info, err := os.Stat(dir); err == nil && info.IsDir() {
		return
	}

	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "To sandbox file edits and task runs, I can write a " +
		".devcontainer/Dockerfile and .devcontainer/devcontainer.json template " +
		"you can then edit. Reopen the project in the container to use it.\n\n"}})

	tcId := a.StartToolCall(ctx, sid, "Write .devcontainer/Dockerfile and devcontainer.json?", "think", nil)
	choice, err := a.askChoiceAuto(ctx, sid, tcId, []string{"Alpine", "Arch", "Debian", "Fedora", "Ubuntu"})
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}

	var dockerfile string
	switch choice {
	case "Alpine":
		dockerfile = defaultDevcontainerDockerfileAlpine
	case "Arch":
		dockerfile = defaultDevcontainerDockerfileArch
	case "Debian":
		dockerfile = defaultDevcontainerDockerfileDebian
	case "Fedora":
		dockerfile = defaultDevcontainerDockerfileFedora
	case "Ubuntu":
		dockerfile = defaultDevcontainerDockerfileUbuntu
	default:
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Skipped")})
		return
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if err := os.WriteFile(filepath.Join(dir, "Dockerfile"), []byte(dockerfile), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if err := os.WriteFile(filepath.Join(dir, "devcontainer.json"), []byte(defaultDevcontainerJSON), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}

	note := "Wrote .devcontainer/Dockerfile (" + choice + ") and .devcontainer/devcontainer.json. Reopen the project in the container to use them."
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(note)})
	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: note + "\n"}})
}

// ---------------------------------------------------------------------------
// Stack detection
// ---------------------------------------------------------------------------

// detectStacks returns the language/stack identifiers active in cwd, in a
// fixed order (load-bearing: tests assert it, and the seed loop in
// ensureDefaults walks it). Used to seed only the relevant SKILL files.
func detectStacks(cwd string) []string {
	var stacks []string

	if _, err := os.Stat(filepath.Join(cwd, "go.mod")); err == nil {
		stacks = append(stacks, "go")
	}

	_, pkgErr := os.Stat(filepath.Join(cwd, "package.json"))
	_, tsconfigErr := os.Stat(filepath.Join(cwd, "tsconfig.json"))
	hasTS := tsconfigErr == nil || hasFileWithExt(cwd, ".ts", ".tsx")
	if hasTS {
		stacks = append(stacks, "ts")
	}
	if pkgErr == nil && !hasTS {
		stacks = append(stacks, "js")
	}

	for _, n := range []string{"pom.xml", "build.gradle", "build.gradle.kts"} {
		if _, err := os.Stat(filepath.Join(cwd, n)); err == nil {
			stacks = append(stacks, "java")
			break
		}
	}

	if _, err := os.Stat(filepath.Join(cwd, "Cargo.toml")); err == nil {
		stacks = append(stacks, "rust")
	}

	for _, n := range []string{"build.zig", "build.zig.zon"} {
		if _, err := os.Stat(filepath.Join(cwd, n)); err == nil {
			stacks = append(stacks, "zig")
			break
		}
	}

	if hasFileWithExt(cwd, ".sh", ".bash") {
		stacks = append(stacks, "bash")
	}

	if info, err := os.Stat(filepath.Join(cwd, ".devcontainer")); err == nil && info.IsDir() {
		stacks = append(stacks, "devcontainer")
	}

	return stacks
}

// hasFileWithExt reports whether cwd contains a file (non-recursive) with one
// of the given extensions. Root-only by design — deeper detection is the
// user's job to override.
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
