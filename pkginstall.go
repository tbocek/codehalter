package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// Syntax probes need a tool per language, and a container image rarely has them all.
// Everything here runs without a model: the project's languages say what is needed,
// the distro says how to install it, and the only judgement call is the user's yes.

// distroFamilies orders the package-name families to try for this container, from
// /etc/os-release ID and then ID_LIKE (linuxmint → debian, centos → fedora, …).
func distroFamilies(info osInfo) []string {
	known := map[string]bool{"arch": true, "debian": true, "fedora": true, "alpine": true, "suse": true}
	var out []string
	for _, id := range append([]string{strings.ToLower(info.Fields["ID"])}, strings.Fields(strings.ToLower(info.Fields["ID_LIKE"]))...) {
		if known[id] && !slices.Contains(out, id) {
			out = append(out, id)
		}
	}
	return out
}

// syntaxToolPackage maps a probe binary to the package that provides it per family.
// A name untested on a family is not retried blindly: the install fails, the re-check
// after it still finds no probe, and that is reported as-is.
//
// python3 and node are absent on purpose: python3 ships in every base image here, and
// Node is installed as the project's toolchain, never just to get `node --check`.
type syntaxToolPackage struct {
	bin  string
	pkgs map[string]string
}

var syntaxToolPackages = []syntaxToolPackage{
	{"shfmt", map[string]string{"arch": "shfmt", "debian": "shfmt", "fedora": "shfmt", "alpine": "shfmt"}},
	{"clang-format", map[string]string{"arch": "clang", "debian": "clang-format", "fedora": "clang", "alpine": "clang"}},
	{"rustfmt", map[string]string{"arch": "rust", "debian": "rustfmt", "fedora": "rustfmt", "alpine": "rust"}},
	{"zig", map[string]string{"arch": "zig", "debian": "zig", "fedora": "zig", "alpine": "zig"}},
	{"ruff", map[string]string{"arch": "ruff", "debian": "python3-ruff", "fedora": "python3-ruff", "alpine": "py3-ruff"}},
	// gofmt comes with the Go toolchain, so a missing gofmt means Go itself is missing.
	{"go", map[string]string{"arch": "go", "debian": "golang-go", "fedora": "golang", "alpine": "go"}},
}

func syntaxToolPackageFor(bin string) (syntaxToolPackage, bool) {
	for _, t := range syntaxToolPackages {
		if t.bin == bin {
			return t, true
		}
	}
	return syntaxToolPackage{}, false
}

// packageName picks the first family this container reports that has an entry.
func (t syntaxToolPackage) packageName(families []string) (string, bool) {
	for _, f := range families {
		if pkg, ok := t.pkgs[f]; ok {
			return pkg, true
		}
	}
	return "", false
}

// pkgInstallCmd is the non-interactive install line for the first family we know.
func pkgInstallCmd(families []string, pkg string) ([]string, bool) {
	for _, f := range families {
		switch f {
		case "arch":
			return []string{"sudo", "pacman", "-S", "--noconfirm", "--needed", pkg}, true
		case "debian":
			return []string{"sudo", "apt-get", "install", "-y", "--no-install-recommends", pkg}, true
		case "fedora":
			return []string{"sudo", "dnf", "-y", "install", pkg}, true
		case "alpine":
			return []string{"sudo", "apk", "add", pkg}, true
		case "suse":
			return []string{"sudo", "zypper", "-n", "install", pkg}, true
		}
	}
	return nil, false
}

// dockerfileInstallLine mirrors how a scaffolded image lists its packages, so an added
// line reads like the ones around it rather than like a bolted-on one-liner.
func dockerfileInstallLine(families []string, pkg, bin string) (string, bool) {
	comment := "# codehalter: syntax check for " + bin + "\n"
	switch firstFamily(families) {
	case "arch":
		return comment + "RUN pacman -Syu --noconfirm --needed \\\n        " + pkg + " \\\n    && rm -rf /var/cache/pacman/pkg/*\n", true
	case "debian":
		return comment + "RUN apt-get update && apt-get install -y --no-install-recommends \\\n        " + pkg + " \\\n    && rm -rf /var/lib/apt/lists/*\n", true
	case "fedora":
		return comment + "RUN dnf -y install \\\n        " + pkg + " \\\n    && dnf clean all\n", true
	case "alpine":
		return comment + "RUN apk add --no-cache \\\n        " + pkg + "\n", true
	case "suse":
		return comment + "RUN zypper -n install \\\n        " + pkg + " \\\n    && zypper clean\n", true
	}
	return "", false
}

func firstFamily(families []string) string {
	if len(families) == 0 {
		return ""
	}
	return families[0]
}

// ensureDockerfileLine persists one package in the image, textually and idempotently:
// nothing else in the file is rewritten. An already-listed package changes nothing, a
// matching existing RUN list gains one line, and anything else appends its own layer.
func ensureDockerfileLine(dockerfilePath string, families []string, pkg, bin string) (string, error) {
	data, err := os.ReadFile(dockerfilePath)
	if err != nil {
		return "", err
	}
	text := string(data)
	if packageListed(text, pkg) {
		return "already listed in " + dockerfilePath, nil
	}
	line, ok := dockerfileInstallLine(families, pkg, bin)
	if !ok {
		return "", fmt.Errorf("%s does not name a package manager codehalter knows (%s)", dockerfilePath, strings.Join(families, "/"))
	}
	if body, pm := installListToExtend(text, families); body != "" {
		updated := insertIntoInstallList(text, body, pm, pkg)
		if updated == "" {
			return "", fmt.Errorf("could not find where to add %s to %s", pkg, dockerfilePath)
		}
		if err := os.WriteFile(dockerfilePath, []byte(updated), 0o644); err != nil {
			return "", err
		}
		return "added " + pkg + " to the existing install list in " + dockerfilePath, nil
	}
	if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	if err := os.WriteFile(dockerfilePath, []byte(text+"\n"+line), 0o644); err != nil {
		return "", err
	}
	return "added a new install layer for " + pkg + " to " + dockerfilePath, nil
}

// packageListed matches a whole word, so "go" is not satisfied by "golang-go".
func packageListed(text, pkg string) bool {
	for _, line := range strings.Split(text, "\n") {
		fields := strings.Fields(strings.TrimSuffix(strings.TrimSpace(line), `\`))
		for _, f := range fields {
			if f == pkg {
				return true
			}
		}
	}
	return false
}

// installListToExtend finds the package list of the container's own package manager:
// the RUN line that calls it with an install flag, whose continuation lines hold packages.
func installListToExtend(text string, families []string) (block, pm string) {
	want := map[string]string{"arch": "pacman", "debian": "apt-get", "fedora": "dnf", "alpine": "apk", "suse": "zypper"}
	for _, f := range families {
		pm, ok := want[f]
		if !ok {
			continue
		}
		lines := strings.Split(text, "\n")
		for i, ln := range lines {
			t := strings.TrimSpace(ln)
			if !strings.HasPrefix(t, "RUN "+pm+" ") {
				continue
			}
			j := i
			for ; j < len(lines); j++ {
				if !strings.HasSuffix(strings.TrimRight(lines[j], " "), `\`) {
					break
				}
			}
			return strings.Join(lines[i:j+1], "\n"), pm
		}
	}
	return "", ""
}

// insertIntoInstallList puts the package before the block's last line, which is the
// `&& cleanup` continuation, keeping one package per line with its trailing backslash.
func insertIntoInstallList(text, block, pm, pkg string) string {
	lines := strings.Split(block, "\n")
	insertAt := -1
	for i := len(lines) - 1; i > 0; i-- {
		t := strings.TrimSpace(lines[i])
		if strings.HasPrefix(t, "&&") || strings.HasPrefix(t, "&&&") {
			insertAt = i
			break
		}
	}
	if insertAt <= 0 {
		return ""
	}
	out := append([]string{}, lines[:insertAt]...)
	out = append(out, "        "+pkg+" \\")
	out = append(out, lines[insertAt:]...)
	return strings.Replace(text, block, strings.Join(out, "\n"), 1)
}

// syntaxGap is one language whose parser is missing here.
type syntaxGap struct {
	bin   string // the probe binary that is missing
	langs []string
	pkg   string
	cmd   []string
}

// extensionProbes pairs an extension with the probe that must answer for it and the
// package that brings it. Only extensions the tree actually holds are asked about.
var extensionProbes = []struct {
	exts []string
	lang string
	bin  string
}{
	{[]string{".go"}, "Go", "gofmt"},
	{[]string{".rs"}, "Rust", "rustfmt"},
	{[]string{".sh", ".bash"}, "shell script", "bash"},
	{[]string{".py"}, "Python", "python3"},
	{[]string{".zig"}, "Zig", "zig"},
	{[]string{".rb"}, "Ruby", "ruby"},
	{[]string{".php"}, "PHP", "php"},
	{[]string{".mjs", ".cjs"}, "JavaScript module", "node"},
}

// toolForMissingProbe names the package to install for a probe binary that is absent.
// bash and python3 are in every base image, so their absence is not something a
// package fixes; node arrives with the project's toolchain, not from here.
func toolForMissingProbe(bin string) (string, bool) {
	switch bin {
	case "gofmt":
		return "go", true
	case "rustfmt":
		return "rustfmt", true
	case "shfmt":
		return "shfmt", true
	case "zig":
		return "zig", true
	case "ruff":
		return "ruff", true
	case "clang-format":
		return "clang-format", true
	}
	return "", false
}

// detectSyntaxGaps lists the languages this tree holds whose parser is missing.
// Nothing here blocks a write: a gap only means the parse gate cannot cover that file.
func detectSyntaxGaps(stacks []string, cwd string) []syntaxGap {
	families := distroFamilies(readOSInfo())
	var gaps []syntaxGap
	seen := map[string]bool{}
	for _, ep := range extensionProbes {
		if !hasFileWithExt(cwd, ep.exts...) {
			continue
		}
		if syntaxProbe(filepath.Join(cwd, "sample"+ep.exts[0])) != nil {
			continue
		}
		if seen[ep.bin] {
			continue
		}
		pkgBin, ok := toolForMissingProbe(ep.bin)
		if !ok {
			continue
		}
		t, ok := syntaxToolPackageFor(pkgBin)
		if !ok {
			continue
		}
		pkg, ok := t.packageName(families)
		if !ok {
			continue
		}
		cmd, ok := pkgInstallCmd(families, pkg)
		if !ok {
			continue
		}
		seen[ep.bin] = true
		gaps = append(gaps, syntaxGap{bin: ep.bin, langs: []string{ep.lang}, pkg: pkg, cmd: cmd})
	}
	return gaps
}

// syntaxToolProblem is the one card for every language whose parser is missing. Its
// accepted branch installs in Go: the choice needs no judgement, only a yes.
func syntaxToolProblem(a *agent, cwd string, gaps []syntaxGap) fixProblem {
	if !a.clientCan("elicitation") && !a.clientCan("terminal") {
		// Nobody can answer: say what is missing and how to add it, install nothing.
		var lines []string
		for _, g := range gaps {
			lines = append(lines, "  "+strings.Join(g.cmd, " "))
		}
		return fixProblem{desc: "🟡 Syntax check unavailable (" + strings.Join(binNames(gaps), ", ") + "); nobody to ask, so nothing was installed:\n" + strings.Join(lines, "\n"), notice: true}
	}
	return fixProblem{
		desc:    "\U0001f7e1 Syntax check: install " + strings.Join(binNames(gaps), ", ") + " (codehalter refuses an edit that no longer parses)",
		install: func(ctx context.Context, sid string) { a.installSyntaxTools(ctx, sid, cwd, gaps) },
	}
}

func binNames(gaps []syntaxGap) []string {
	var names []string
	for _, g := range gaps {
		names = append(names, g.bin)
	}
	return names
}

// installSyntaxTools installs each gap with the distro's own package manager, verifies the
// probe answers afterwards, and persists what worked in .devcontainer/Dockerfile. Nothing is
// installed unless the user said yes, and nothing is claimed as fixed without the re-check.
func (a *agent) installSyntaxTools(ctx context.Context, sid, cwd string, gaps []syntaxGap) {
	stopBeat := a.heartbeat(ctx, sid)
	defer stopBeat()
	families := distroFamilies(readOSInfo())
	dockerfile := filepath.Join(cwd, ".devcontainer", "Dockerfile")
	hasImage := fileExists(filepath.Join(cwd, ".devcontainer"), "Dockerfile")
	var done, failed, notes []string
	for _, g := range gaps {
		if ctx.Err() != nil {
			return
		}
		tcId := a.StartToolCall(ctx, sid, "Installing "+g.pkg+" ("+strings.Join(g.langs, "/")+" syntax check)", "execute", nil)
		out, err := runPkgInstall(ctx, g.cmd)
		if err != nil {
			failed = append(failed, firstLine(err.Error()))
			a.FailToolCall(ctx, sid, tcId, firstLine(err.Error()))
			continue
		}
		if syntaxProbe(filepath.Join(cwd, "probe"+probeExtFor(g.bin))) == nil {
			failed = append(failed, g.pkg+" installed but "+g.bin+" still does not answer")
			a.FailToolCall(ctx, sid, tcId, g.pkg+" installed, but `"+g.bin+"` is still not on PATH")
			continue
		}
		done = append(done, g.bin)
		note := g.pkg + " is installed here but there is no .devcontainer/Dockerfile to record it in"
		if hasImage {
			added, derr := ensureDockerfileLine(dockerfile, families, g.pkg, g.bin)
			if derr != nil {
				note = firstLine(derr.Error())
			} else {
				note = added
			}
		}
		notes = append(notes, note)
		a.CompleteToolCallTitled(ctx, sid, tcId, "Installed "+g.pkg+": "+g.bin+" answers",
			[]ToolCallContent{TextContent(tailOf(out, 500) + "\n" + note)})
	}
	var b strings.Builder
	if len(done) > 0 {
		fmt.Fprintf(&b, "\u2705 Syntax check now covers %s.\n", strings.Join(done, ", "))
	}
	for _, n := range notes {
		fmt.Fprintf(&b, "%s\n", n)
	}
	for _, f := range failed {
		fmt.Fprintf(&b, "\U0001f7e1 %s\n", f)
	}
	if b.Len() > 0 {
		a.say(ctx, sid, "\n"+b.String())
	}
}

// probeExtFor names a file whose probe would use the given tool, for the after-install check.
func probeExtFor(bin string) string {
	switch bin {
	case "gofmt":
		return ".go"
	case "rustfmt":
		return ".rs"
	case "bash", "shfmt":
		return ".sh"
	case "python3":
		return ".py"
	case "zig":
		return ".zig"
	case "ruby":
		return ".rb"
	case "php":
		return ".php"
	case "node":
		return ".js"
	}
	return ".txt"
}

func tailOf(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return "…" + s[len(s)-n:]
}

// runPkgInstall runs one package-manager command, returning its tail for the tool card.
func runPkgInstall(ctx context.Context, argv []string) (string, error) {
	runCtx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(runCtx, argv[0], argv[1:]...)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	err := cmd.Run()
	tail := strings.TrimSpace(out.String())
	if len(tail) > 2000 {
		tail = tail[len(tail)-2000:]
	}
	if err != nil {
		return tail, fmt.Errorf("%s: %w", strings.Join(argv, " "), err)
	}
	return tail, nil
}
