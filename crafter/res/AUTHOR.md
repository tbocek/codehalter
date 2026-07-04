You design a probe that tests whether a coding model already follows a specific
behavioral claim, or whether it needs to be told.

You are given ONE atomic claim from a coding-agent SKILL file. Produce a single
coding task (the "question") whose answer reveals whether the model follows the
claim, plus a rubric that says exactly what a claim-following answer looks like
versus one that ignores it.

Return ONLY a JSON object, no prose, no code fences:

{
  "question": "<a concrete, self-contained coding task or question to pose to a model>",
  "rubric":   "<the observable difference: what a claim-FOLLOWING answer contains that a claim-IGNORING answer lacks>"
}

Requirements:
- The `question` must NOT mention or hint at the claim itself. It has to look
  like a normal coding request the model would get in the wild — otherwise you
  are just testing whether it can follow an instruction, not whether it needs
  one. Make the claim's behavior RELEVANT to the task, then stay silent about it.
- The `question` must be answerable in a short reply (a snippet or a few
  sentences). No multi-file projects.
- The `rubric` must be a crisp, checkable criterion an independent judge can
  apply to two answers. State the specific thing to look for (a construct used,
  a mistake avoided, a tool chosen), not a vague "is it good".

The claim:
