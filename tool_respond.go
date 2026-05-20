package main

import (
	"context"
	"encoding/json"
)

// respondToolName is the synthetic terminal tool that captures the model's
// final user-facing message. Inspired by forge's respond_tool
// (https://github.com/antoinezambelli/forge): exposing a `respond(message)`
// tool keeps small local models inside the tool-calling grammar they're best
// at, so the "should I emit prose or another tool call?" decision — which
// 8B-class models reliably get wrong — never has to be made. The execute and
// subagent phases include it; plan/verify/document exclude it (they emit
// structured JSON or are one-shot text).
const respondToolName = "respond"

func init() {
	RegisterTool(Tool{Terminal: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": respondToolName,
			"description": "Emit your final user-facing message and end the turn. " +
				"Call this exactly once when the task is complete; everything you " +
				"would have written as a free-text reply goes in `message`. After " +
				"this call returns, no further tools run.",
			"parameters": map[string]any{
				"type":     "object",
				"required": []string{"message"},
				"properties": map[string]any{
					"message": map[string]any{
						"type":        "string",
						"description": "The full user-facing message — what you would have written as the assistant's final reply.",
					},
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid SessionId, rawArgs string) (string, bool) {
		var args struct {
			Message string `json:"message"`
		}
		_ = json.Unmarshal([]byte(rawArgs), &args)
		return args.Message, false
	}})
}
