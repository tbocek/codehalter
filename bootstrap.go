package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
)

// ---------------------------------------------------------------------------
// .devcontainer
// ---------------------------------------------------------------------------

// ensureDevcontainer gates the whole session on running inside a container.
// Returns true only when we're already inside one. Otherwise it scaffolds
// .devcontainer/ (prompting for the base OS if no template exists yet), sets
// a.abortReason so Prompt refuses every turn, and returns false — codehalter
// does not run unsandboxed.
func (a *agent) ensureDevcontainer(ctx context.Context, cwd string, sid string) bool {
	if containerKind() != "" {
		return true
	}

	dir := filepath.Join(cwd, ".devcontainer")
	dirInfo, statErr := os.Stat(dir)
	hasDevcontainer := statErr == nil && dirInfo.IsDir()

	const reopen = "Reopen the project in the container to continue."
	const restart = "Start a new Agent Thread (the + button at the top) to re-open the devcontainer setup menu."

	if hasDevcontainer {
		a.sendUpdateAndAbort(ctx, sid, "codehalter is running outside the .devcontainer. "+reopen)
		return false
	}

	a.sendUpdate(ctx, sid, messageChunk{Kind: KindAgentMessage, Content: ContentBlock{Type: "text", Text: "codehalter must run inside a container. I can scaffold " +
		".devcontainer/Dockerfile and .devcontainer/devcontainer.json for you to edit, then you can reopen the project in the container.\n\n"}})

	choice, tcId, err := a.askChoiceWithCard(ctx, sid, "Write .devcontainer/Dockerfile and devcontainer.json?", "think", []string{"Alpine", "Arch", "Debian", "Fedora", "Ubuntu"})
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		a.sendUpdateAndAbort(ctx, sid, "codehalter requires a sandbox. "+restart)
		return false
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
		a.sendUpdateAndAbort(ctx, sid, "Devcontainer setup cancelled. "+restart)
		return false
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		a.sendUpdateAndAbort(ctx, sid, "codehalter requires a sandbox. "+restart)
		return false
	}
	if err := os.WriteFile(filepath.Join(dir, "Dockerfile"), []byte(dockerfile), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		a.sendUpdateAndAbort(ctx, sid, "codehalter requires a sandbox. "+restart)
		return false
	}
	if err := os.WriteFile(filepath.Join(dir, "devcontainer.json"), []byte(defaultDevcontainerJSON), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		a.sendUpdateAndAbort(ctx, sid, "codehalter requires a sandbox. "+restart)
		return false
	}

	// Seed BOOTSTRAP-<os>-<stack>.md alongside the devcontainer for each
	// detected stack that has a template. Skip stacks with no template, and
	// never overwrite — these are one-shot install prompts scoped to a
	// freshly-scaffolded devcontainer.
	osName := strings.ToLower(choice)
	var seeded []string
	for _, stack := range detectStacks(cwd) {
		body, ok := defaultBootstraps[osName+"-"+stack]
		if !ok {
			continue
		}
		name := "BOOTSTRAP-" + osName + "-" + stack + ".md"
		path := filepath.Join(cwd, ".codehalter", name)
		if _, err := os.Stat(path); err == nil {
			continue
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			continue
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			continue
		}
		seeded = append(seeded, name)
	}

	note := "Wrote .devcontainer/Dockerfile (" + choice + ") and .devcontainer/devcontainer.json. " + reopen
	if len(seeded) > 0 {
		note += " Also seeded " + strings.Join(seeded, ", ") + " for per-stack dev-tool install."
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(note)})
	a.sendUpdateAndAbort(ctx, sid, note)
	return false
}

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
	labels := []string{"Ignore, add '.codehalter' to .gitignore", "Track, add '#.codehalter' to .gitignore"}
	if !hasGitignore {
		title = "No .gitignore found — create one for .codehalter/?"
		labels = []string{"Add .gitignore, ignore .codehalter", "Add .gitignore, track .codehalter"}
	}
	choice, tcId, err := a.askChoiceWithCard(ctx, sid, title, "think", labels)
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
