package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

const planPrompt = `Analyze the user's request. Reply with ONLY a JSON object, no other text:
{
  "clear": true/false,       // is the request clear enough to act on?
  "choices": [],             // if not clear: up to 2 interpretations as short strings
  "question": "",            // if not clear: what to ask the user
  "complexity": "simple",    // "simple" or "complex"
  "steps": ["step1", ...]   // brief plan of action
}

Rules:
- "simple": single file read/edit, short answer, quick lookup
- "complex": multi-file changes, architecture decisions, debugging, refactoring
- If the request is ambiguous, set clear=false and provide up to 2 choices
- Keep steps brief (1 line each)`

type planResult struct {
	Clear      bool     `json:"clear"`
	Choices    []string `json:"choices"`
	Question   string   `json:"question"`
	Complexity string   `json:"complexity"`
	Steps      []string `json:"steps"`
}

// planAndRoute analyzes the user's request, asks for clarification if needed,
// and returns which LLM connection to use for execution.
func (a *agent) planAndRoute(ctx context.Context, sid SessionId, userText string) (*LLMConnection, error) {
	fast := a.settings.LLM("fast")
	if fast == nil {
		return nil, fmt.Errorf("no 'fast' connection in .codehalter/settings.toml")
	}

	// In discussion mode, always use fast — no planning needed.
	a.mu.Lock()
	mode := a.mode
	a.mu.Unlock()
	if mode == "discussion" {
		return fast, nil
	}

	// Ask fast to plan.
	messages := []llmMessage{{Role: "user", Content: planPrompt + "\n\nUser request: " + userText}}
	response, _, err := a.llmRequest(ctx, "", fast, messages)
	if err != nil {
		return fast, nil
	}

	// Parse the JSON response.
	response = strings.TrimSpace(response)
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimPrefix(response, "```")
	response = strings.TrimSuffix(response, "```")
	response = strings.TrimSpace(response)

	var plan planResult
	if err := json.Unmarshal([]byte(response), &plan); err != nil {
		return fast, nil
	}

	sess := a.getSession(sid)

	// If the request is unclear, ask the user.
	if !plan.Clear && len(plan.Choices) > 0 {
		question := plan.Question
		if question == "" {
			question = "I'm not sure what you mean. Which of these?"
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(question)))

		tcId := a.StartToolCall(ctx, sid, "Clarification needed", "think", nil)
		choice, err := a.conn.AskChoice(ctx, sid, tcId, plan.Choices)
		a.CompleteToolCall(ctx, sid, tcId, []ToolCallContent{TextContent("User chose: " + choice)})

		if err != nil || choice == "abort" {
			if sess != nil {
				sess.AddAssistant(question + "\n(choices: " + strings.Join(plan.Choices, ", ") + ")\nUser aborted.")
				_ = sess.Save()
			}
			return nil, fmt.Errorf("user aborted")
		}

		if sess != nil {
			sess.AddAssistant(question + "\n(choices: " + strings.Join(plan.Choices, ", ") + ")\nUser chose: " + choice)
			_ = sess.Save()
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock("Understood: "+choice+"\n")))
	}

	// Show the plan.
	if len(plan.Steps) > 0 {
		var planText strings.Builder
		planText.WriteString("Plan:\n")
		for i, step := range plan.Steps {
			fmt.Fprintf(&planText, "%d. %s\n", i+1, step)
		}
		a.sendUpdate(ctx, sid, AgentMessageChunk(TextBlock(planText.String())))
	}

	// Route based on complexity.
	if plan.Complexity == "complex" {
		if thinking := a.settings.LLM("thinking"); thinking != nil {
			return thinking, nil
		}
	}

	return fast, nil
}
