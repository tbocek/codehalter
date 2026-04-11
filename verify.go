package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type verifyResult struct {
	Success  bool     `json:"success"`
	Issues   []string `json:"issues"`
	FixSteps []string `json:"fix_steps,omitempty"`
}

const maxVerifyAttempts = 2

// verify runs a self-check after execution. If the LLM finds issues, it gets
// another chance to fix them. Returns the final result and the verify outcome.
func (a *agent) verify(ctx context.Context, sid SessionId, conn *LLMConnection, messages []llmMessage, res toolLoopResult, userText string, planSteps []string) (toolLoopResult, *verifyResult, error) {
	verifyPrompt := a.loadPromptFile(sid, "VERIFY.md")
	if verifyPrompt == "" {
		return res, &verifyResult{Success: true}, nil
	}

	for attempt := 0; attempt < maxVerifyAttempts; attempt++ {
		// Build verification context.
		var prompt strings.Builder
		prompt.WriteString(verifyPrompt)
		prompt.WriteString("\n\nUser request: ")
		prompt.WriteString(userText)
		if len(planSteps) > 0 {
			prompt.WriteString("\n\nApproved plan:\n")
			for i, step := range planSteps {
				fmt.Fprintf(&prompt, "%d. %s\n", i+1, step)
			}
		}
		prompt.WriteString("\n\nYour response was:\n")
		prompt.WriteString(res.Text)

		// Run verify with read-only tools + run_task.
		verifyMessages := append(messages, llmMessage{Role: "user", Content: prompt.String()})
		verifyRes, err := a.runToolLoopFiltered(ctx, sid, conn, verifyMessages, toolFilter{
			readOnly: true,
			include:  map[string]bool{"run_task": true},
		})
		if err != nil {
			return res, nil, nil
		}

		trimmed := trimJSON(verifyRes.Text)

		var result verifyResult
		if err := json.Unmarshal([]byte(trimmed), &result); err != nil {
			return res, nil, nil
		}

		if result.Success {
			res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
			return res, &result, nil
		}

		// If there are fix_steps, return to the caller for a full retry cycle.
		if len(result.FixSteps) > 0 {
			res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
			return res, &result, nil
		}

		// No fix_steps — try an inline fix within this attempt.
		var issueText strings.Builder
		issueText.WriteString("⚠ Self-check found issues:\n")
		for _, issue := range result.Issues {
			fmt.Fprintf(&issueText, "- %s\n", issue)
		}
		issueText.WriteString("\nFixing...\n")
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(issueText.String())))

		fixPrompt := "Your previous response had these issues:\n"
		for _, issue := range result.Issues {
			fixPrompt += "- " + issue + "\n"
		}
		fixPrompt += "\nPlease fix these issues now."

		messages = append(messages,
			llmMessage{Role: "assistant", Content: res.Text},
			llmMessage{Role: "user", Content: fixPrompt},
		)

		fixRes, err := a.runToolLoop(ctx, sid, conn, messages, false)
		if err != nil {
			return res, &result, nil
		}
		res.Text = res.Text + "\n" + fixRes.Text
		res.ToolUses = append(res.ToolUses, verifyRes.ToolUses...)
		res.ToolUses = append(res.ToolUses, fixRes.ToolUses...)
	}

	return res, &verifyResult{Success: true}, nil
}
