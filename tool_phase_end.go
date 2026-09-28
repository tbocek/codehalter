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
		"name": submitPlanToolName,
		"description": "Submit your finished plan and end the planning phase. Call this exactly " +
			"once when you have gathered everything and decided how to proceed. Put the structured " +
			"plan in the arguments. If the request is a pure lookup you can already answer, put the " +
			"COMPLETE answer in the `answer` argument and call this with report_only=true and an " +
			"empty subtasks list: your reasoning is never shown, and message text beside a tool call " +
			"is often dropped. After this call returns, no further planning tools run.",
		"parameters": map[string]any{
			"type":     "object",
			"required": []string{"clear", "subtasks", "report_only"},
			"properties": map[string]any{
				"clear": map[string]any{
					"type":        "boolean",
					"description": "True when the request is actionable. False when it needs clarification — then fill choices + question and leave subtasks empty.",
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
					"description": "True when the whole request is informational and you already have the answer — no edits, no commands. Skips the execute-confirmation gate.",
				},
				"redo": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "string"},
					"description": "Only in a project built with /spec: the ids of the spec items this request concerns when it is more than one plan's worth of work (a page to build, a program to make work). codehalter reopens them and rebuilds them one per round. No subtasks with this.",
				},
				"spec": map[string]any{
					"type":        "array",
					"description": "Only in a project WITHOUT a spec, when the request is more than one plan's worth of work: the request written down as a specification, one or more markdown files, one headed section per requirement with the statements a test can check. codehalter writes them, shows the user, and runs /spec on them. No subtasks with this.",
					"items": map[string]any{
						"type":     "object",
						"required": []string{"path", "content"},
						"properties": map[string]any{
							"path":    map[string]any{"type": "string", "description": "File name inside the spec directory, for example `01-files.md`."},
							"content": map[string]any{"type": "string", "description": "The markdown."},
						},
					},
				},
				"spec_dir": map[string]any{"type": "string", "description": "With `spec`: the directory to write it to, relative to the project root. Default `spec`."},
				"out_dir":  map[string]any{"type": "string", "description": "With `spec`: where the program is built, relative to the project root (an existing manifest's directory, or a new one)."},
				"target":   map[string]any{"type": "string", "description": "With `spec`: the technology to build it with, in a few words."},
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
