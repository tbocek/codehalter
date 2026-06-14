package main

import (
	"context"
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Template macros are slash commands: `/<name> <args>` expands a
// TEMPLATE-<name>.md body into the turn's user message and runs it as a normal
// prompt. Command name = filename minus the TEMPLATE- prefix and .md suffix.
// Defaults ship embedded and are seeded into .codehalter/ on first run
// (seedTemplates); from then on the on-disk copy wins, so users can edit a
// shipped template or drop in their own TEMPLATE-<name>.md.

//go:embed res/TEMPLATE-*.md
var templateFS embed.FS

// templatePlaceholder marks where a macro injects the user's args; its presence
// also makes the arg required — invoking with no args is rejected, not run.
const templatePlaceholder = "{{}}"

// templateOptPlaceholder is the OPTIONAL counterpart: a macro carrying {{?}} runs
// even with no args (the placeholder then renders empty), so the arg is a
// take-it-or-leave-it value, not a gate. /improve uses it for the submit API key
// so the analysis runs on a bare `/improve`.
const templateOptPlaceholder = "{{?}}"

type availableCommandsUpdate struct {
	Kind     string             `json:"sessionUpdate"` // "available_commands_update"
	Commands []availableCommand `json:"availableCommands"`
}

// availableCommand is one entry in the ACP slash-command menu. The protocol
// requires an object with name + description; a bare string array deserialises
// to zero valid commands client-side (Zed shows "Available commands: none" and
// rejects the slash), which is exactly what an earlier []string version did.
type availableCommand struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

// isTemplateFile reports whether a filename is a TEMPLATE-<name>.md and returns
// the bare <name>.
func isTemplateFile(n string) (name string, ok bool) {
	if strings.HasPrefix(n, "TEMPLATE-") && strings.HasSuffix(n, ".md") {
		return strings.TrimSuffix(strings.TrimPrefix(n, "TEMPLATE-"), ".md"), true
	}
	return "", false
}

// seedTemplates copies each embedded TEMPLATE-*.md into .codehalter/ when absent
// (seed-once, like the phase prompts), so the user has editable copies.
func seedTemplates(cwd string) error {
	entries, _ := templateFS.ReadDir("res")
	for _, e := range entries {
		n := e.Name()
		if _, ok := isTemplateFile(n); !ok {
			continue
		}
		path := filepath.Join(cwd, ".codehalter", n)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			data, _ := templateFS.ReadFile("res/" + n)
			if err := os.WriteFile(path, data, 0o644); err != nil {
				return fmt.Errorf("seeding %s: %w", path, err)
			}
		}
	}
	return nil
}

// templateNames returns the macro command names (sorted, deduped) from both the
// user's .codehalter/TEMPLATE-*.md and the embedded defaults, so a user-dropped
// template shows up in the slash menu alongside the shipped ones.
func templateNames(cwd string) []string {
	set := map[string]bool{}
	add := func(entries []os.DirEntry) {
		for _, e := range entries {
			if name, ok := isTemplateFile(e.Name()); ok {
				set[name] = true
			}
		}
	}
	embedded, _ := templateFS.ReadDir("res")
	add(embedded)
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

// loadTemplate returns a template body, preferring the user's editable
// .codehalter copy over the embedded default. ok=false when neither exists.
func loadTemplate(cwd, name string) (body string, ok bool) {
	if data, err := os.ReadFile(filepath.Join(cwd, ".codehalter", "TEMPLATE-"+name+".md")); err == nil {
		return string(data), true
	}
	if data, err := templateFS.ReadFile("res/TEMPLATE-" + name + ".md"); err == nil {
		return string(data), true
	}
	return "", false
}

// renderMacro applies the body→prompt rules: {{}} replaced with args; {{}} with
// no args → stopMsg (caller shows it to the user, runs no turn); no {{}} → args
// appended after the body.
func renderMacro(name, body, args string) (rendered, stopMsg string) {
	args = strings.TrimSpace(args)
	// Optional placeholder: substitute the args (empty when none were given) and
	// run regardless — unlike the required placeholder below, it never rejects.
	if strings.Contains(body, templateOptPlaceholder) {
		return strings.ReplaceAll(body, templateOptPlaceholder, args), ""
	}
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

// handleClean is a code-level slash command: /clean deletes all session log
// files (session_*.log, session_*.toml) from .codehalter/ and returns a
// confirmation. It runs before template expansion in expandMacro.
func handleClean(cwd string) (message string, handled bool) {
	if cwd == "" {
		return "", false
	}
	dir := filepath.Join(cwd, ".codehalter")
	matched := []string{}
	for _, pattern := range []string{"session_*.log", "session_*.toml"} {
		entries, _ := filepath.Glob(filepath.Join(dir, pattern))
		matched = append(matched, entries...)
	}
	if len(matched) == 0 {
		return "✓ No session log files found in .codehalter/", true
	}
	var errs []string
	for _, f := range matched {
		if err := os.Remove(f); err != nil {
			errs = append(errs, err.Error())
		}
	}
	if len(errs) > 0 {
		return fmt.Sprintf("⚠ Cleaned %d file(s), %d error(s): %s", len(matched)-len(errs), len(errs), strings.Join(errs, "; ")), true
	}
	return fmt.Sprintf("✓ Cleaned %d session file(s) from .codehalter/", len(matched)), true
}

// expandMacro turns a `/<name> <args>` message into the prompt to run when
// <name> is a known macro. handled=false → not a macro, run userText as-is.
// handled=true + stopMsg → macro needs an arg it lacks; show stopMsg, run no
// turn. Otherwise `rendered` replaces the user message.
// splitMacro parses a slash command "/name args" into its name and args. name
// is "" when userText is not a slash command. Shared by expandMacro and the
// Prompt handler, which arms the /improve ask cap from the parsed name.
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

func expandMacro(cwd, userText string) (rendered, stopMsg string, handled bool) {
	name, args := splitMacro(userText)
	if name == "" {
		return "", "", false
	}
	// Code-level slash commands run before template expansion.
	if name == "clean" {
		msg, handled := handleClean(cwd)
		if handled {
			return "", msg, true
		}
	}
	body, ok := loadTemplate(cwd, name)
	if !ok {
		return "", "", false
	}
	r, msg := renderMacro(name, body, args)
	return r, msg, true
}

// sendAvailableCommands advertises the macro commands to the editor's slash
// menu. Re-sent every turn (from prepare) so the menu stays live.
func (a *agent) sendAvailableCommands(ctx context.Context, sid string) {
	cwd := ""
	if sess := a.getSession(sid); sess != nil {
		cwd = sess.Cwd
	}
	names := templateNames(cwd)
	cmds := make([]availableCommand, 0, len(names)+1)
	// Code-level slash commands first.
	cmds = append(cmds, availableCommand{Name: "clean", Description: "Delete session log files from .codehalter/"})
	for _, n := range names {
		cmds = append(cmds, availableCommand{Name: n, Description: "Run the " + n + " prompt template"})
	}
	a.sendUpdate(ctx, sid, availableCommandsUpdate{Kind: "available_commands_update", Commands: cmds})
}
