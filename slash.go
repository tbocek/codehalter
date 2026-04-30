package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	improveLogTailBytes = 64 * 1024
)

// handleSlashCommand dispatches `/improve` and `/clean`. Returns true when
// the prompt was a recognised slash command; in that case the response has
// already been emitted and the caller should end the turn without going to
// the LLM.
func (a *agent) handleSlashCommand(ctx context.Context, sid SessionId, userText string) bool {
	text := strings.TrimSpace(userText)
	switch {
	case text == "/improve" || strings.HasPrefix(text, "/improve "):
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(a.buildImproveSnapshot(sid))))
		return true
	case text == "/clean" || strings.HasPrefix(text, "/clean "):
		a.runClean(ctx, sid)
		return true
	}
	return false
}

// runClean archives the session's full state to a "session_archive_*" file,
// then resets the live session in place — Messages and History are wiped,
// the same SessionId continues. Zed cannot be told to refresh its panel
// (ACP has no agent → client "reload" notification), so this leaves the
// prior turns visible in Zed's UI; the next prompt starts the agent with a
// clean slate but the user has to close+reopen the session for a visually
// empty panel. The title is left intact and will be regenerated when the
// next user message arrives (isFirstMessage returns true on empty history).
func (a *agent) runClean(ctx context.Context, sid SessionId) {
	sess := a.getSession(sid)
	if sess == nil {
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ /clean: no session.\n")))
		return
	}
	sess.mu.Lock()
	if len(sess.Messages) == 0 && len(sess.History) == 0 {
		sess.mu.Unlock()
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Session already empty.\n")))
		return
	}
	archiveID, err := sess.rotate(nil, nil)
	if err != nil {
		sess.mu.Unlock()
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("⚠ /clean failed: "+err.Error()+"\n")))
		return
	}
	_ = sess.saveLocked()
	sess.mu.Unlock()

	a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(fmt.Sprintf(
		"🧹 Session reset — archived as %s.\n"+
			"Zed cannot be told to refresh its chat panel from here (ACP has no agent → client reload). "+
			"The agent has no memory of prior turns; close and reopen this session for a fully empty view.\n\n",
		archiveID))))
}

// buildImproveSnapshot returns a markdown report covering version,
// environment, settings (API keys redacted), discovered runners, the active
// session's recent messages, and the tail of its debug log. Designed to be
// copy-pasted into another tool when asking "how can codehalter be improved?".
func (a *agent) buildImproveSnapshot(sid SessionId) string {
	var b strings.Builder
	b.WriteString("# /improve — codehalter session snapshot\n\n")
	fmt.Fprintf(&b, "codehalter v0.1.0 / go %s / %s/%s\n\n",
		runtime.Version(), runtime.GOOS, runtime.GOARCH)

	b.WriteString("## Environment\n")
	if kind := containerKind(); kind != "" {
		fmt.Fprintf(&b, "- container: %s\n", kind)
	} else {
		b.WriteString("- container: none (running on host)\n")
	}
	if path, err := findFirefox(); err == nil {
		fmt.Fprintf(&b, "- firefox: %s\n", path)
	} else {
		b.WriteString("- firefox: not found\n")
	}
	fmt.Fprintf(&b, "- images supported: %v\n\n", a.imagesSupported)

	b.WriteString("## Settings\n")
	if a.settings.path == "" {
		b.WriteString("- (no settings.toml loaded)\n\n")
	} else {
		fmt.Fprintf(&b, "- path: %s\n", a.settings.path)
		for i, c := range a.settings.LLMConnections {
			tier := "subagent"
			if i == 0 {
				tier = "main"
			}
			var roles []string
			if c.ExtraBodyThinking != nil {
				roles = append(roles, "thinking")
			}
			if c.ExtraBodyExecute != nil {
				roles = append(roles, "execute")
			}
			if c.ExtraBodySummary != nil {
				roles = append(roles, "summary")
			}
			apiKey := ""
			if c.APIKey != "" {
				apiKey = " api_key=<set>"
			}
			fmt.Fprintf(&b, "- [%d] tier=%s url=%s model=%s roles=[%s]%s\n",
				i, tier, c.URL, c.Model, strings.Join(roles, ","), apiKey)
		}
		b.WriteString("\n")
	}

	b.WriteString("## Project tooling\n")
	a.mu.Lock()
	caps := a.capabilities
	empty := a.emptyProject
	a.mu.Unlock()
	switch {
	case empty:
		b.WriteString("- empty project\n\n")
	case len(caps.runners) == 0:
		b.WriteString("- no task runner detected\n\n")
	default:
		fmt.Fprintf(&b, "- runners: %s\n", strings.Join(caps.runners, ", "))
		fmt.Fprintf(&b, "- build:  %s\n", joinOrNone(caps.build))
		fmt.Fprintf(&b, "- test:   %s\n", joinOrNone(caps.test))
		fmt.Fprintf(&b, "- lint:   %s\n", joinOrNone(caps.lint))
		fmt.Fprintf(&b, "- format: %s\n\n", joinOrNone(caps.format))
	}

	sess := a.getSession(sid)
	b.WriteString("## Session\n")
	if sess == nil {
		b.WriteString("- (no session)\n\n")
		return b.String()
	}
	fmt.Fprintf(&b, "- id: %s\n", sess.ID)
	fmt.Fprintf(&b, "- depth: %d\n", sess.Depth)
	if sess.Title != "" {
		fmt.Fprintf(&b, "- title: %s\n", sess.Title)
	}
	fmt.Fprintf(&b, "- messages: %d, history levels: %d\n\n",
		len(sess.Messages), len(sess.History))

	for i, h := range sess.History {
		fmt.Fprintf(&b, "## History summary level %d (#%d)\n%s\n\n", h.Level, i, h.Content)
	}

	for i, m := range sess.Messages {
		fmt.Fprintf(&b, "## Message [%d] %s\n%s\n", i, m.Role, m.Content)
		for _, t := range m.ToolUses {
			fmt.Fprintf(&b, "\n**tool_use** `%s` input=`%s`\noutput:\n```\n%s\n```\n", t.Name, t.Input, t.Output)
		}
		b.WriteString("\n")
	}

	logPath := filepath.Join(sess.Cwd, sessionDir, fmt.Sprintf("session_%s.log", sid))
	if data, err := readTail(logPath, improveLogTailBytes); err == nil && len(data) > 0 {
		b.WriteString("## Session log (tail)\n```\n")
		b.Write(data)
		if !strings.HasSuffix(string(data), "\n") {
			b.WriteString("\n")
		}
		b.WriteString("```\n")
	}

	return b.String()
}

func joinOrNone(s []string) string {
	if len(s) == 0 {
		return "(none)"
	}
	return strings.Join(s, ", ")
}

// readTail returns up to maxBytes from the end of path. Used to grab the
// recent debug log without loading huge files into memory.
func readTail(path string, maxBytes int) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := info.Size()
	offset := int64(0)
	if size > int64(maxBytes) {
		offset = size - int64(maxBytes)
	}
	if _, err := f.Seek(offset, 0); err != nil {
		return nil, err
	}
	buf := make([]byte, size-offset)
	n, err := f.Read(buf)
	if err != nil {
		return nil, err
	}
	return buf[:n], nil
}
