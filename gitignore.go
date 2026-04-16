package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
)

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
	ok, err := a.conn.AskYesNo(ctx, sid, tcId, "Ignore", "Track")
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

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}

func dirExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.IsDir()
}
