package main

import (
	"context"
)

// executeConn returns the LLM connection for the execution phase.
// Falls back to "thinking" if no "execute" role is configured in settings.
func (a *agent) executeConn() *LLMConnection {
	if c := a.settings.LLM("execute"); c != nil {
		return c
	}
	return a.settings.LLM("thinking")
}

// execute runs the execution phase. It prepends EXECUTE.md (if present) to the
// last user message, excludes web tools (information retrieval belongs to the
// planning phase), and runs the agentic tool loop with write-enabled tools.
// extraExclude lists additional tool names to exclude (e.g. launch_subagent
// when a subagent has reached its max nesting depth).
func (a *agent) execute(ctx context.Context, sid SessionId, messages []llmMessage, extraExclude ...string) (toolLoopResult, error) {
	if executeMD := a.loadPromptFile(sid, "EXECUTE.md"); executeMD != "" && len(messages) > 0 {
		last := len(messages) - 1
		if messages[last].Role == "user" {
			if content, ok := messages[last].Content.(string); ok {
				messages[last].Content = executeMD + "\n\n---\n\n" + content
			}
		}
	}
	exclude := map[string]bool{"web_search": true, "web_read": true}
	for _, name := range extraExclude {
		exclude[name] = true
	}
	return a.runToolLoopFiltered(ctx, sid, a.executeConn(), messages, toolFilter{
		readOnly: false,
		exclude:  exclude,
	})
}
