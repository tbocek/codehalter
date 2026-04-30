package main

import (
	"context"
	"os"
	"path/filepath"
)

// ensureDevcontainer offers to seed .devcontainer/Dockerfile and
// .devcontainer/devcontainer.json when codehalter is running outside a
// container and the project has no devcontainer config yet.
//
// Asks at most once per project: a "yes" creates the files; a "no" writes
// an empty .codehalter/devcontainer.skip marker so we stop nagging. Delete
// either path (or just .codehalter/devcontainer.skip) to be asked again.
func (a *agent) ensureDevcontainer(ctx context.Context, cwd string, sid SessionId) {
	if containerKind() != "" {
		return
	}
	dir := filepath.Join(cwd, ".devcontainer")
	if dirExists(dir) {
		return
	}
	skipMarker := filepath.Join(cwd, sessionDir, "devcontainer.skip")
	if fileExists(skipMarker) {
		return
	}

	tcId := a.StartToolCall(ctx, sid, "Create .devcontainer/ template?", "think", nil)
	ok, err := a.askYesNoAuto(ctx, sid, tcId, "Create", "Skip", true)
	if err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if !ok {
		_ = os.WriteFile(skipMarker, nil, 0o644)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("Skipped")})
		return
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if err := os.WriteFile(filepath.Join(dir, "Dockerfile"), []byte(defaultDevcontainerDockerfile), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}
	if err := os.WriteFile(filepath.Join(dir, "devcontainer.json"), []byte(defaultDevcontainerJSON), 0o644); err != nil {
		a.FailToolCall(ctx, sid, tcId, err.Error())
		return
	}

	note := "Wrote .devcontainer/Dockerfile and .devcontainer/devcontainer.json. Reopen the project in the container to use them."
	a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent(note)})
	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(note+"\n")))
}
