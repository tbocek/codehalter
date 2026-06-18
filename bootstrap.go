package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
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

	const reopen = "Reopen the project in the container to continue. In Zed, press Ctrl-Alt-Shift-O and choose \"Connect Dev Container\"."
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

	// Per-stack dev-tool installs are handled by the prepare phase on the
	// next session (inside the container) — it asks the user, installs live,
	// persists in this Dockerfile, and wires MCP. Nothing to seed here.
	note := "Wrote .devcontainer/Dockerfile (" + choice + ") and .devcontainer/devcontainer.json. " + reopen
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(note)})
	a.sendUpdateAndAbort(ctx, sid, note)
	return false
}

// ---------------------------------------------------------------------------
// .gitignore
// ---------------------------------------------------------------------------

// gitignoreSettingsEntry is the one path codehalter always keeps out of git: the
// project-local settings.toml can hold an api_key. Added whenever settings.toml
// is scaffolded, independent of the whole-dir ignore/track choice in
// ensureGitignore — so a team that TRACKS .codehalter/ (to share PLAN.md/skills)
// still never commits the secrets file.
const gitignoreSettingsEntry = sessionDir + "/settings.toml"

// ensureSettingsGitignored appends gitignoreSettingsEntry to cwd/.gitignore when
// absent (creating the file if the project is a git repo), returning whether the
// entry is now present. Best-effort and non-interactive: it runs at scaffold
// time so the secrets file is excluded before the user fills in real values. No
// point in a non-git project, so it no-ops there.
func ensureSettingsGitignored(cwd string) bool {
	gitInfo, gerr := os.Stat(filepath.Join(cwd, ".git"))
	hasGit := gerr == nil && gitInfo.IsDir()
	gitignorePath := filepath.Join(cwd, ".gitignore")
	data, rerr := os.ReadFile(gitignorePath)
	if !hasGit && rerr != nil {
		return false // not a git repo and no existing .gitignore — nothing to do
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == gitignoreSettingsEntry {
			return true // already ignored
		}
	}
	sep := ""
	if len(data) > 0 && !strings.HasSuffix(string(data), "\n") {
		sep = "\n"
	}
	return os.WriteFile(gitignorePath, []byte(string(data)+sep+gitignoreSettingsEntry+"\n"), 0o644) == nil
}

// ensureGitignore makes sure .gitignore mentions .codehalter/ (as an ignore
// line or a tracked-on-purpose marker). Asks once per repo — later sessions
// short-circuit on the existing entry. Skipped outside git-managed dirs
// (requires .git/ or an existing .gitignore).
func (a *agent) ensureGitignore(ctx context.Context, cwd string, sid string) {
	gitignorePath := filepath.Join(cwd, ".gitignore")
	// .git is a DIR in a normal repo but a FILE (a "gitdir:" pointer) in a linked
	// worktree or a submodule — both are git-managed, so any successful stat counts.
	_, gitErr := os.Stat(filepath.Join(cwd, ".git"))
	hasGit := gitErr == nil
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
// fixed order (load-bearing: tests assert it, and ensureSkills walks it).
// Used to seed only the relevant SKILL files.
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

	_, cmakeErr := os.Stat(filepath.Join(cwd, "CMakeLists.txt"))
	if cmakeErr == nil || hasFileWithExt(cwd, ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx") {
		stacks = append(stacks, "c")
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
		if slices.Contains(exts, filepath.Ext(e.Name())) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Environment detection
// ---------------------------------------------------------------------------

// containerKind reports the sandbox kind codehalter is running inside ("" when
// on the bare host). discoverSandbox gates run_command on this, and
// ensureDevcontainer aborts the session when it's empty.
func containerKind() string {
	if os.Getenv("REMOTE_CONTAINERS") == "true" || os.Getenv("DEVCONTAINER") == "true" {
		return "devcontainer"
	}
	if _, err := os.Stat("/.dockerenv"); err == nil {
		return "docker"
	}
	if _, err := os.Stat("/run/.containerenv"); err == nil {
		return "podman"
	}
	if v := os.Getenv("container"); v != "" {
		return v
	}
	return ""
}

// osInfo holds the result of parsing /etc/os-release. ID is the
// supported-distro slug we use to pick a SKILL-*.md (one of "alpine",
// "arch", "debian", "fedora", "ubuntu"; "" when missing/unsupported).
// Fields is every key=value pair from the file (un-lowercased values,
// quotes stripped) — used to substitute {{VERSION_ID}}, {{PRETTY_NAME}},
// etc. into the per-OS skill body so the LLM doesn't have to probe.
type osInfo struct {
	ID     string
	Fields map[string]string
}

// readOSInfo parses /etc/os-release. ID_LIKE is consulted as a fallback
// so Linux Mint maps to ubuntu, Manjaro to arch, etc. Cheap file read —
// safe to call from prepare on every turn.
func readOSInfo() osInfo {
	info := osInfo{Fields: map[string]string{}}
	data, err := os.ReadFile("/etc/os-release")
	if err != nil {
		return info
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		eq := strings.IndexByte(line, '=')
		if eq <= 0 {
			continue
		}
		k := line[:eq]
		v := strings.Trim(line[eq+1:], `"'`)
		info.Fields[k] = v
	}
	supported := map[string]bool{"alpine": true, "arch": true, "debian": true, "fedora": true, "ubuntu": true}
	if id := strings.ToLower(info.Fields["ID"]); supported[id] {
		info.ID = id
		return info
	}
	for _, alt := range strings.Fields(strings.ToLower(info.Fields["ID_LIKE"])) {
		if supported[alt] {
			info.ID = alt
			return info
		}
	}
	return info
}
