package main

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// templatePlaceholder marks where args go; its presence makes args required.
const templatePlaceholder = "{{}}"

type availableCommandsUpdate struct {
	Kind     string             `json:"sessionUpdate"`
	Commands []availableCommand `json:"availableCommands"`
}

// availableCommand must be an object: ACP clients deserialise a bare string array to no commands.
type availableCommand struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Input       *commandInput `json:"input,omitempty"`
}

type commandInput struct {
	Hint string `json:"hint"`
}

func templateNames(cwd string) []string {
	set := map[string]bool{}
	add := func(entries []fs.DirEntry) {
		for _, e := range entries {
			if n := e.Name(); strings.HasPrefix(n, "TEMPLATE-") && strings.HasSuffix(n, ".md") {
				set[strings.TrimSuffix(strings.TrimPrefix(n, "TEMPLATE-"), ".md")] = true
			}
		}
	}
	if embedded, err := resMD.ReadDir("res"); err == nil {
		add(embedded)
	}
	if disk, err := os.ReadDir(filepath.Join(cwd, ".codehalter")); err == nil {
		add(disk)
	}
	names := make([]string, 0, len(set))
	for n := range set {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

func loadTemplate(cwd, name string) (body string, ok bool) {
	return builtin(cwd, "TEMPLATE-"+name+".md")
}

// renderMacro returns a stopMsg instead of a prompt when {{}} has no args; the caller runs no turn.
func renderMacro(name, body, args string) (rendered, stopMsg string) {
	args = strings.TrimSpace(args)
	if strings.Contains(body, templatePlaceholder) {
		if args == "" {
			return "", fmt.Sprintf("⚠ /%s expects a prompt: type `/%s <your text>`.", name, name)
		}
		return strings.ReplaceAll(body, templatePlaceholder, args), ""
	}
	if args != "" {
		return body + "\n\n" + args, ""
	}
	return body, ""
}

func handleClean(cwd string) string {
	dir := filepath.Join(cwd, ".codehalter")
	matched := []string{}
	for _, pattern := range []string{"session_*.log", "session_*.toml"} {
		entries, _ := filepath.Glob(filepath.Join(dir, pattern))
		matched = append(matched, entries...)
	}
	if len(matched) == 0 {
		return "✓ No session log files found in .codehalter/"
	}
	var errs []string
	for _, f := range matched {
		if err := os.Remove(f); err != nil {
			errs = append(errs, err.Error())
		}
	}
	if len(errs) > 0 {
		return fmt.Sprintf("⚠ Cleaned %d file(s), %d error(s): %s", len(matched)-len(errs), len(errs), strings.Join(errs, "; "))
	}
	return fmt.Sprintf("✓ Cleaned %d session file(s) from .codehalter/", len(matched))
}

// handleSettings re-probes because ensureLLM only does so on a settings change. The sources
// are said first since a llama.cpp router blocks the probe while it loads the model.
func (a *agent) handleSettings(ctx context.Context, sid, cwd string) string {
	a.say(ctx, sid, renderSettingsSources(cwd)+"Probing every configured `[[llm]]`")
	stopBeat := a.heartbeat(ctx, sid)
	a.probeAllLLMs(ctx)
	stopBeat()
	return "\n\n" + a.renderLLMStatus()
}

func splitMacro(userText string) (name, args string) {
	if !strings.HasPrefix(userText, "/") {
		return "", ""
	}
	rest := strings.TrimPrefix(userText, "/")
	name = rest
	if i := strings.IndexAny(rest, " \t\r\n"); i >= 0 {
		name, args = rest[:i], rest[i+1:]
	}
	return name, args
}

// expandMacro: handled=false means run userText as-is; a stopMsg means show it and run no turn.
func (a *agent) expandMacro(ctx context.Context, sid, cwd, userText string) (rendered, stopMsg string, handled bool) {
	name, args := splitMacro(userText)
	// No cwd is a prompt for an unknown session, which the turn then refuses.
	if name == "" || cwd == "" {
		return "", "", false
	}
	switch name {
	case "clean":
		return "", handleClean(cwd), true
	case "settings":
		return "", a.handleSettings(ctx, sid, cwd), true
	}
	body, ok := loadTemplate(cwd, name)
	if !ok {
		return "", "", false
	}
	r, msg := renderMacro(name, body, args)
	return r, msg, true
}

func templateSummary(name, body string) string {
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(line), "#"))
		if line == "" || line == templatePlaceholder {
			continue
		}
		return clipUTF8(line, 120)
	}
	return "Run the " + name + " prompt template"
}

func (a *agent) sendAvailableCommands(ctx context.Context, sid string) {
	cwd := ""
	if sess := a.getSession(sid); sess != nil {
		cwd = sess.Cwd
	}
	names := templateNames(cwd)
	cmds := make([]availableCommand, 0, len(names)+1)
	cmds = append(cmds,
		availableCommand{Name: "clean", Description: "Delete session log files from .codehalter/"},
		availableCommand{Name: "settings", Description: "Show which settings.toml is in use and re-probe every configured model"},
		availableCommand{
			Name:        "spec",
			Description: "Implement a spec item by item until every requirement has a passing test. /spec starts or resumes (the first run asks where the spec is, where to build, with what); `status` reports the ledger; `stop` ends the loop after the round in flight; `abort` stops it at once; `redo` finds the finished items that do not deliver and rebuilds them",
			Input:       &commandInput{Hint: "status | stop | abort | redo"},
		},
	)
	for _, n := range names {
		body, _ := loadTemplate(cwd, n)
		cmd := availableCommand{Name: n, Description: templateSummary(n, body)}
		if strings.Contains(body, templatePlaceholder) {
			cmd.Input = &commandInput{Hint: "<your text>"}
		}
		cmds = append(cmds, cmd)
	}
	a.sendUpdate(ctx, sid, availableCommandsUpdate{Kind: "available_commands_update", Commands: cmds})
}
