package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// ensureDevcontainer returns true only inside a container. Otherwise it scaffolds
// .devcontainer/ and aborts the session: codehalter does not run unsandboxed.
func (a *agent) ensureDevcontainer(ctx context.Context, cwd string, sid string) bool {
	if containerKind() != "" {
		return true
	}

	dir := filepath.Join(cwd, ".devcontainer")
	dirInfo, statErr := os.Stat(dir)
	hasDevcontainer := statErr == nil && dirInfo.IsDir()

	// Standalone only lands here when the launcher declined, which for an existing
	// .devcontainer means no docker/podman with the compose plugin on PATH.
	reopen := "Reopen the project in the container to continue. In Zed, press Ctrl-Shift-P and type \"open dev container\"."
	restart := "Start a new Agent Thread (the + button at the top) to re-open the devcontainer setup menu."
	if a.standalone {
		reopen = "codehalter --cli starts that container itself once docker or podman with the compose plugin is on PATH. " +
			"Install one of those and run it again, or start the container yourself: devcontainer exec --workspace-folder . codehalter --cli"
		restart = "Run codehalter --cli again to re-open the devcontainer setup menu."
	}

	if hasDevcontainer {
		a.sendUpdateAndAbort(ctx, sid, "codehalter is running outside the .devcontainer. "+reopen)
		return false
	}

	a.say(ctx, sid, "codehalter must run inside a container. I can scaffold "+
		".devcontainer/Dockerfile and .devcontainer/devcontainer.json for you to edit, then you can reopen the project in the container.\n\n")

	distros := []string{"Alpine", "Arch", "Debian", "Fedora", "Ubuntu"}
	choice, tcId, err := a.askCard(ctx, sid, "Write .devcontainer/Dockerfile and devcontainer.json?", "think", choiceOptions(distros))
	fail := func(err error) bool {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		a.sendUpdateAndAbort(ctx, sid, "codehalter requires a sandbox. "+restart)
		return false
	}
	if err != nil {
		return fail(err)
	}
	if !slices.Contains(distros, choice) {
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Skipped")})
		a.sendUpdateAndAbort(ctx, sid, "Devcontainer setup cancelled. "+restart)
		return false
	}
	dockerfile, err := devcontainerDockerfiles.ReadFile("res/Dockerfile.devcontainer." + strings.ToLower(choice))
	if err != nil {
		return fail(err)
	}

	// A bind mount whose host source is missing fails the container start, so only
	// offer mounts whose source exists.
	gitWritable, gitconfig := false, false
	// Only a .git directory: a .git file (worktree/submodule link) can't back the bind mount.
	if info, err := os.Stat(filepath.Join(cwd, ".git")); err == nil && info.IsDir() {
		yes, gtc, gerr := a.askYesNoWithCard(ctx, sid, "Mount your repo's .git (writable) and ~/.gitconfig (when present) into the container, so git uses your real history and identity for commit/push?", "think", "Yes, mount", "No")
		if gerr != nil {
			a.FailToolCall(ctx, sid, gtc, gerr.Error())
		} else {
			gitWritable = yes
			gitconfig = yes && loadGlobalConfig().HasGitconfigInHome
			done := "Not mounting .git or .gitconfig."
			if yes {
				done = "Will mount .git writable"
				if gitconfig {
					done += " and ~/.gitconfig."
				} else {
					done += " (no ~/.gitconfig recorded on the host, so it isn't mounted)."
				}
			}
			a.CompleteToolCall(ctx, sid, gtc, []ToolCallContent{TextContent(done)})
		}
	}
	sshAgent := false
	if hostSSHAgentAvailable() {
		yes, stc, serr := a.askYesNoWithCard(ctx, sid, "Forward your host SSH agent into the container? Lets git push / ssh use your host SSH keys.", "think", "Yes, forward SSH agent", "No")
		if serr != nil {
			a.FailToolCall(ctx, sid, stc, serr.Error())
		} else {
			sshAgent = yes
			done := "Not forwarding the SSH agent."
			if yes {
				done = "Will forward the host SSH agent."
			}
			a.CompleteToolCall(ctx, sid, stc, []ToolCallContent{TextContent(done)})
		}
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fail(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "Dockerfile"), dockerfile, 0o644); err != nil {
		return fail(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "devcontainer.json"), []byte(buildDevcontainerJSON(gitWritable, gitconfig, sshAgent)), 0o644); err != nil {
		return fail(err)
	}

	// Per-stack dev tools are installed later by the prepare phase inside the container.
	note := "Wrote .devcontainer/Dockerfile (" + choice + ") and .devcontainer/devcontainer.json, the mounts you chose apply once you (re)start the container. " + reopen
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(note)})
	a.sendUpdateAndAbort(ctx, sid, note)
	return false
}

// Optional bind mounts, kept out of res/devcontainer.json because a missing bind
// source fails the container start. Extras are spliced in after configMountAnchor.
const (
	configMountAnchor = `"source=${localEnv:HOME}/.config/codehalter,target=/home/dev/.config/codehalter,type=bind,readonly"`
	gitMount          = `"source=${localWorkspaceFolder}/.git,target=${containerWorkspaceFolder}/.git,type=bind"`
	gitconfigMount    = `"source=${localEnv:HOME}/.gitconfig,target=/home/dev/.gitconfig,type=bind,readonly"`
	sshMount          = `"source=${localEnv:SSH_AUTH_SOCK},target=/ssh-agent,type=bind"`
)

func buildDevcontainerJSON(gitWritable, gitconfig, sshAgent bool) string {
	out := defaultDevcontainerJSON
	var extras []string
	if gitWritable {
		extras = append(extras, gitMount)
		if gitconfig {
			extras = append(extras, gitconfigMount)
		}
	}
	if sshAgent {
		extras = append(extras, sshMount)
	}
	if len(extras) > 0 {
		ins := configMountAnchor
		for _, m := range extras {
			ins += ",\n    " + m
		}
		out = strings.Replace(out, configMountAnchor, ins, 1)
	}
	if sshAgent {
		out = strings.Replace(out, `"DEVCONTAINER": "true"`, `"DEVCONTAINER": "true", "SSH_AUTH_SOCK": "/ssh-agent"`, 1)
	}
	return out
}

// ensureTerminals refuses the session when the client lacks ACP terminal support.
// Every command runs as an ACP terminal; there is deliberately no in-process fallback.
func (a *agent) ensureTerminals(ctx context.Context, sid string) bool {
	if a.clientCan("terminal") {
		return true
	}
	a.sendUpdateAndAbort(ctx, sid, "This editor did not advertise ACP terminal support "+
		"(clientCapabilities.terminal in its initialize request), so codehalter has no way to run commands: "+
		"builds, tests, and every run_command are unavailable. Use a client with ACP terminal support (Zed does), "+
		"then start a new Agent Thread.")
	return false
}

func hostSSHAgentAvailable() bool {
	sock := os.Getenv("SSH_AUTH_SOCK")
	if sock == "" {
		return false
	}
	_, err := os.Stat(sock)
	return err == nil
}

// Always ignored, even when .codehalter/ is tracked: the project settings.toml can hold an api_key.
const gitignoreSettingsEntry = sessionDir + "/settings.toml"

// .git is a directory in a clone, a "gitdir:" file in a linked worktree or submodule.
func gitManaged(cwd string) bool { return fileExists(cwd, ".git") }

// appendGitignore adds entry as its own line after content, the file's current text.
func appendGitignore(cwd, content, entry string) error {
	if content != "" && !strings.HasSuffix(content, "\n") {
		entry = "\n" + entry
	}
	return appendFile(filepath.Join(cwd, ".gitignore"), entry+"\n")
}

func ensureSettingsGitignored(cwd string) bool {
	data, rerr := os.ReadFile(filepath.Join(cwd, ".gitignore"))
	if !gitManaged(cwd) && rerr != nil {
		return false
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == gitignoreSettingsEntry {
			return true
		}
	}
	return appendGitignore(cwd, string(data), gitignoreSettingsEntry) == nil
}

func (a *agent) ensureGitignore(ctx context.Context, cwd string, sid string) {
	gitignorePath := filepath.Join(cwd, ".gitignore")
	ignoreInfo, ignoreErr := os.Stat(gitignorePath)
	hasGitignore := ignoreErr == nil && !ignoreInfo.IsDir()
	if !gitManaged(cwd) && !hasGitignore {
		return
	}

	var content string
	if hasGitignore {
		data, err := os.ReadFile(gitignorePath)
		if err != nil {
			a.say(ctx, sid, "Failed to read .gitignore: "+err.Error()+"\n")
			return
		}
		content = string(data)
		for _, line := range strings.Split(content, "\n") {
			// The settings-only entry is not the whole-dir decision and must not suppress the prompt.
			if strings.TrimSpace(line) == gitignoreSettingsEntry {
				continue
			}
			if strings.Contains(strings.ToLower(line), "codehalter") {
				return
			}
		}
	}

	title := "Add .codehalter/ to .gitignore?"
	labels := []string{"Ignore, add '.codehalter' to .gitignore", "Track, add '#.codehalter' to .gitignore"}
	if !hasGitignore {
		title = "No .gitignore found: create one for .codehalter/?"
		labels = []string{"Add .gitignore, ignore .codehalter", "Add .gitignore, track .codehalter"}
	}
	choice, tcId, err := a.askCard(ctx, sid, title, "think", choiceOptions(labels))
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

	if err := appendGitignore(cwd, content, entry); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(note)})
	a.say(ctx, sid, note+"\n")
}

// detectStacks returns stacks in a fixed order that skillSet relies on.
func detectStacks(cwd string) []string {
	var stacks []string

	if fileExists(cwd, "go.mod") {
		stacks = append(stacks, "go")
	}

	hasTS := fileExists(cwd, "tsconfig.json") || hasFileWithExt(cwd, ".ts", ".tsx")
	if hasTS {
		stacks = append(stacks, "ts")
	}
	if fileExists(cwd, "package.json") && !hasTS {
		stacks = append(stacks, "js")
	}

	if hasFileWithExt(cwd, ".css", ".scss", ".sass", ".less", ".html", ".htm") {
		stacks = append(stacks, "css")
	}

	if fileExists(cwd, "pom.xml") || fileExists(cwd, "build.gradle") || fileExists(cwd, "build.gradle.kts") {
		stacks = append(stacks, "java")
	}

	if fileExists(cwd, "Cargo.toml") {
		stacks = append(stacks, "rust")
	}

	if fileExists(cwd, "build.zig") || fileExists(cwd, "build.zig.zon") {
		stacks = append(stacks, "zig")
	}

	if fileExists(cwd, "CMakeLists.txt") || hasFileWithExt(cwd, ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hxx") {
		stacks = append(stacks, "c")
	}

	return stacks
}

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

// ID is set only for distros with a shipped SKILL-<id>.md, else "".
type osInfo struct {
	ID     string
	Fields map[string]string
}

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
	ids := append([]string{strings.ToLower(info.Fields["ID"])}, strings.Fields(strings.ToLower(info.Fields["ID_LIKE"]))...)
	for _, id := range ids {
		if id != "" && shipped("SKILL-"+id+".md") {
			info.ID = id
			return info
		}
	}
	return info
}
