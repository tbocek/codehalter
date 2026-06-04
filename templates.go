package main

import (
	"context"
	"embed"
	"fmt"
	"sort"
	"strings"
)

// Template macros are slash commands: `/<name> <args>` expands the embedded
// docs/TEMPLATE-<name>.md into the turn's user message and runs it as a normal
// prompt. Command name = filename minus the TEMPLATE- prefix and .md suffix.

//go:embed docs/TEMPLATE-*.md
var templateFS embed.FS

// templatePlaceholder marks where a macro injects the user's args; its presence
// also makes the arg required — invoking with no args is rejected, not run.
const templatePlaceholder = "{{}}"

type availableCommandsUpdate struct {
	Kind     string   `json:"sessionUpdate"` // "available_commands_update"
	Commands []string `json:"availableCommands"`
}

// templateNames returns the macro command names (sorted) from the embedded
// docs/TEMPLATE-<name>.md filenames.
func templateNames() []string {
	entries, _ := templateFS.ReadDir("docs")
	var names []string
	for _, e := range entries {
		n := e.Name()
		if strings.HasPrefix(n, "TEMPLATE-") && strings.HasSuffix(n, ".md") {
			names = append(names, strings.TrimSuffix(strings.TrimPrefix(n, "TEMPLATE-"), ".md"))
		}
	}
	sort.Strings(names)
	return names
}

// renderMacro applies the body→prompt rules: {{}} replaced with args; {{}} with
// no args → stopMsg (caller shows it to the user, runs no turn); no {{}} → args
// appended after the body.
func renderMacro(name, body, args string) (rendered, stopMsg string) {
	args = strings.TrimSpace(args)
	if strings.Contains(body, templatePlaceholder) {
		if args == "" {
			return "", fmt.Sprintf("⚠ /%s expects a prompt — type `/%s <your text>`.", name, name)
		}
		return strings.ReplaceAll(body, templatePlaceholder, args), ""
	}
	if args != "" {
		return body + "\n\n" + args, ""
	}
	return body, ""
}

// expandMacro turns a `/<name> <args>` message into the prompt to run when
// <name> is a known macro. handled=false → not a macro, run userText as-is.
// handled=true + stopMsg → macro needs an arg it lacks; show stopMsg, run no
// turn. Otherwise `rendered` replaces the user message.
func expandMacro(userText string) (rendered, stopMsg string, handled bool) {
	if !strings.HasPrefix(userText, "/") {
		return "", "", false
	}
	rest := strings.TrimPrefix(userText, "/")
	name, args := rest, ""
	if i := strings.IndexAny(rest, " \t\r\n"); i >= 0 {
		name, args = rest[:i], rest[i+1:]
	}
	data, err := templateFS.ReadFile("docs/TEMPLATE-" + name + ".md")
	if err != nil {
		return "", "", false
	}
	r, msg := renderMacro(name, string(data), args)
	return r, msg, true
}

// sendAvailableCommands advertises the macro commands to the editor's slash
// menu. Re-sent every turn (from prepare) so the menu stays live.
func (a *agent) sendAvailableCommands(ctx context.Context, sid string) {
	a.sendUpdate(ctx, sid, availableCommandsUpdate{Kind: "available_commands_update", Commands: templateNames()})
}
