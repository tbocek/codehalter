package main

import (
	"context"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"
)

// skillSet is sorted so the cached prompt prefix is byte-stable. It keys on what is present
// in the tree, never on PATH: the model needs SKILL-justfile.md before it installs `just`.
func skillSet(cwd string, stacks []string) []string {
	names := []string{"SKILL-base.md"}
	if slices.Contains(stacks, "css") {
		names = append(names, "SKILL-layout.md")
	}
	for _, n := range []string{"justfile", "Justfile", ".justfile"} {
		if _, err := os.Stat(filepath.Join(cwd, n)); err == nil {
			names = append(names, "SKILL-justfile.md")
			break
		}
	}
	if n := "SKILL-" + readOSInfo().ID + ".md"; shipped(n) {
		names = append(names, n)
	}
	// A shipped name on disk is an override, loaded only where that skill applies.
	if entries, err := os.ReadDir(filepath.Join(cwd, sessionDir)); err == nil {
		for _, e := range entries {
			n := e.Name()
			if !e.IsDir() && strings.HasPrefix(n, "SKILL-") && strings.HasSuffix(n, ".md") && !shipped(n) && !slices.Contains(names, n) {
				names = append(names, n)
			}
		}
	}
	sort.Strings(names)
	return names
}

// cmdPlaceholder is expanded once at session init and stored, so the cached prefix never
// shifts mid-session. Commands are project-controlled, the same trust level as mcp.toml.
var cmdPlaceholder = regexp.MustCompile(`\{\{cmd:(.+?)\}\}`)

// expandCmdPlaceholders leaves a failing command's placeholder verbatim rather than "".
func expandCmdPlaceholders(body string) string {
	return cmdPlaceholder.ReplaceAllStringFunc(body, func(m string) string {
		cmd := m[len("{{cmd:") : len(m)-2]
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, "sh", "-c", cmd).Output()
		if err != nil {
			slog.Warn("skill {{cmd:}} failed — leaving the placeholder in place", "cmd", cmd, "err", err)
			return m
		}
		return strings.TrimSpace(string(out))
	})
}

func overriddenBuiltins(cwd string) []string {
	entries, err := os.ReadDir(filepath.Join(cwd, sessionDir))
	if err != nil {
		return nil
	}
	var over []string
	for _, e := range entries {
		if !e.IsDir() && shipped(e.Name()) {
			over = append(over, e.Name())
		}
	}
	sort.Strings(over)
	return over
}

// skillBody returns "" for an unknown name, which is how a deleted user skill stops loading.
func skillBody(cwd, name string) string {
	body, ok := builtin(cwd, name)
	if !ok {
		return ""
	}
	return expandCmdPlaceholders(body)
}

// loadSkills runs only at session start and compaction; a skill that applies later is sent
// by checkEnv as a user message, so the cached prefix stays stable.
func loadSkills(cwd string, names []string) string {
	var b strings.Builder
	for _, n := range names {
		content := skillBody(cwd, n)
		if content != "" {
			b.WriteString(content)
			if !strings.HasSuffix(content, "\n") {
				b.WriteString("\n")
			}
			b.WriteString("\n")
		}
	}
	return b.String()
}
