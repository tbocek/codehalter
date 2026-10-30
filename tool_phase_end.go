package main

import (
	"context"
	"encoding/json"
	"log/slog"
)

// Terminal tools end a phase without side effects: the tool loop hands up what
// Execute returns as the phase's result (submit_plan's arguments ARE the plan).

const submitPlanToolName = "submit_plan"

var submitPlanTool = Tool{Def: map[string]any{
	"type": "function",
	"function": map[string]any{
		"name":        submitPlanToolName,
		"description": "Submit the finished plan and end planning. Call it exactly once, when you have gathered everything and decided how to proceed. A pure lookup you can already answer: the COMPLETE answer in `answer`, no subtasks (your reasoning is never shown, and text beside a tool call is often dropped). No planning tools run after it.",
		"parameters": map[string]any{
			"type":     "object",
			"required": []string{},
			"properties": map[string]any{
				"choices": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Up to 2 short interpretations, with `question`.",
				},
				"question": map[string]any{
					"type":        "string",
					"description": "One sentence asking which interpretation, only when you ask.",
				},
				"spec_quote": map[string]any{
					"type":        "string",
					"description": "Only in a /spec round, when asking: the spec text that comes closest to deciding the question, copied exactly from a spec file. codehalter checks that it is there.",
				},
				"options": map[string]any{
					"type":        "array",
					"description": "Only in a /spec round, when asking, instead of choices: 2 or 3 ways to decide the question, your pick first.",
					"items": map[string]any{
						"type":     "object",
						"required": []string{"choice", "example"},
						"properties": map[string]any{
							"choice":  map[string]any{"type": "string", "description": "The option in a few words."},
							"example": map[string]any{"type": "string", "description": "What the user would see or the program would do with it: a label, a layout, a value, a line of a file."},
						},
					},
				},
				"subtasks": map[string]any{
					"type":        "array",
					"description": "The units of work for the executor. None with `answer` or `question`.",
					"items": map[string]any{
						"type":     "object",
						"required": []string{"description"},
						"properties": map[string]any{
							"description": map[string]any{
								"type":        "string",
								"description": "Self-contained instruction naming exact files, commands, packages.",
							},
							"verify": map[string]any{
								"type":        "array",
								"items":       map[string]any{"type": "string"},
								"description": "Concrete checks the executor runs before declaring done. Empty only for pure-lookup subtasks that edit nothing.",
							},
						},
					},
				},
				"redo": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Only in the /spec redo audit, whose prompt asks for it: the ids of the items that fall short. Refused anywhere else.",
				},
				"answer": map[string]any{
					"type":        "string",
					"description": "The complete answer for the user, with no subtasks. This is what the user reads: everything you found, in full, not a promise to write it. Empty otherwise.",
				},
			},
		},
	},
}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	return rawArgs, false
}}

// respond keeps small models inside the tool-calling grammar, so they never
// choose between prose and another tool call (after forge's respond_tool).
const respondToolName = "respond"

var respondTool = Tool{Def: map[string]any{
	"type": "function",
	"function": map[string]any{
		"name":        respondToolName,
		"description": "Your final user-facing message; ends the turn. Call it exactly once, when the task is done: everything you would write as a reply goes in `message`. No tools run after it.",
		"parameters": map[string]any{
			"type":     "object",
			"required": []string{"message"},
			"properties": map[string]any{
				"message": map[string]any{
					"type":        "string",
					"description": "The full user-facing message, what you would write as your final reply.",
				},
				"failed": map[string]any{
					"type":        "boolean",
					"description": "True only when THIS task is not done: its change could not be made to work, or it is impossible. Codehalter then plans again. A test that was already failing, or is not about this task, is no reason.",
				},
			},
		},
	},
}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
	var args struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		// Fall back to the raw payload so the user still sees the final text.
		slog.Debug("respond: arguments not valid JSON, using raw text", "err", err)
		return rawArgs, false
	}
	return args.Message, false
}}
