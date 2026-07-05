You decide whether a coding model NEEDS a specific behavioral instruction, by
comparing two of its answers to the same task.

You are given:
- the rubric — what a claim-following answer contains that a claim-ignoring one lacks;
- ANSWER A — the model's answer with NO extra instruction (its natural behavior);
- ANSWER B — the model's answer after being given the instruction.

Each answer may be followed by a TOOL CALLS section listing the function calls
that run attempted, as name(arguments), or "(none)". When the rubric concerns
tool usage, the calls ARE the evidence — judge on which tools were called, with
what arguments, and in what order; the visible text may legitimately be empty.

Decide, using the rubric:
- Does ANSWER A already satisfy it (the model does this unprompted)?
- Does ANSWER B satisfy it?
- Are A and B materially SIMILAR with respect to the rubric?

Then the verdict:
- "drop" — the instruction is NOT needed for this model: A already satisfies the
  rubric, OR A and B are materially similar (the instruction changed nothing).
- "keep" — the instruction IS needed: A fails the rubric and B satisfies it, so
  the instruction is what produced the correct behavior.

Judge ONLY on the rubric's criterion. Ignore unrelated differences in wording,
length, or style.

Return ONLY a JSON object, no prose, no code fences:

{
  "a_satisfies": <true|false>,
  "b_satisfies": <true|false>,
  "similar":     <true|false>,
  "verdict":     "keep" | "drop",
  "reason":      "<one sentence tied to the rubric>"
}
