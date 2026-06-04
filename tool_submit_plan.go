package main

import "context"

// submitPlanToolName is the terminal tool for the PLANNING phase — the planner's
// counterpart to `respond` in execute. The planner calls it exactly once with
// the structured plan (clarity, subtasks, report_only) as the tool arguments;
// the loop unmarshals those arguments straight into planResult, so the plan
// never has to be parsed out of free-text prose. Any user-facing answer the
// planner has in hand (a pure-lookup result) it writes as normal message text,
// which arrives separately as the loop's accumulated content — the two channels
// (structured plan vs. prose answer) stay physically separate and can't be
// mashed into one string the way the old "emit JSON as text" convention did.
const submitPlanToolName = "submit_plan"

func init() {
	RegisterTool(Tool{Terminal: true, Def: map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": submitPlanToolName,
			"description": "Submit your finished plan and end the planning phase. Call this exactly " +
				"once when you have gathered everything and decided how to proceed. Put the structured " +
				"plan in the arguments. If the request is a pure lookup you can already answer, write " +
				"that answer as your normal message text FIRST, then call this with report_only=true and " +
				"an empty subtasks list. After this call returns, no further planning tools run.",
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
					"subtasks": map[string]any{
						"type":        "array",
						"description": "One or more units of work for the executor. Empty only when clear=false (clarification) or report_only with the answer already given in your message text.",
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
				},
			},
		},
	}, Execute: func(ctx context.Context, a *agent, sid string, rawArgs string) (string, bool) {
		// The arguments ARE the plan. Echo them back as the terminal output so
		// runToolLoop hands them up as res.Text, where runPlanPhase unmarshals
		// them into planResult. No work to do here — submit_plan has no side
		// effects, it's purely the structured exit.
		return rawArgs, false
	}})
}
