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
		"description": "Submit the finished plan and end planning. Call it exactly once, when you have gathered everything and decided how to proceed. A pure lookup you can already answer: the COMPLETE answer in `answer`, report_only=true, no subtasks (your reasoning is never shown, and text beside a tool call is often dropped). No planning tools run after it.",
		"parameters": map[string]any{
			"type":     "object",
			"required": []string{"clear", "subtasks", "report_only"},
			"properties": map[string]any{
				"clear": map[string]any{
					"type":        "boolean",
					"description": "True when the request is actionable. False when it needs clarification: then fill choices and question, no subtasks.",
				},
				"choices": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Up to 2 short interpretations, only when clear=false.",
				},
				"question": map[string]any{
					"type":        "string",
					"description": "One sentence asking which interpretation, only when clear=false.",
				},
				"spec_quote": map[string]any{
					"type":        "string",
					"description": "Only in a /spec round, with clear=false: the spec text that comes closest to deciding the question, copied exactly from a spec file. codehalter checks that it is there.",
				},
				"options": map[string]any{
					"type":        "array",
					"description": "Only in a /spec round, with clear=false, instead of choices: 2 or 3 ways to decide the question, your pick first.",
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
					"description": "One or more units of work for the executor. Empty only when clear=false (clarification) or report_only with the answer in the `answer` argument.",
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
				"report_only": map[string]any{
					"type":        "boolean",
					"description": "True when the request is informational and you have the answer: no edits, no commands. Skips the execute-confirmation gate.",
				},
				"redo": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Only in the /spec redo audit, whose prompt asks for it: the ids of the items that fall short. Refused anywhere else.",
				},
				"answer": map[string]any{
					"type":        "string",
					"description": "The complete answer for the user, when report_only=true and subtasks is empty. This is what the user reads: everything you found, in full, not a promise to write it. Empty otherwise.",
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
					"description": "True when the task is not done: a check fails and you cannot fix it, or the task is wrong or impossible. Say why in `message`; codehalter plans again.",
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
