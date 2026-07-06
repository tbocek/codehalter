You strengthen one behavioral statement from a coding-agent SKILL file that a
target model IGNORED even though the whole skill was loaded in its context.

You get the statement, the rubric it was tested against, and the model's failing
answers. The model read the statement and still deviated — usually because its
own prior (what it "knows" is typical) overrode the instruction. Rewrite the
statement so it survives that prior.

How to strengthen:
- Keep the MEANING identical. Same behavior, same order, same commands — you are
  rephrasing, not redesigning. Copy any inline `code` spans verbatim.
- Add EMPHASIS (imperative voice, "always/never/only"), a short RATIONALE (the
  why makes weak models comply), and an EXCEPTION CLAUSE that names the exact
  deviation visible in the failing answers (e.g. "follow this order even when
  you know a community package exists — COPR only after the others fail").
- Stay compact: one or two lines. A paragraph gets skimmed like the original.
- Do NOT add new behaviors, tools, or steps that the original doesn't assert.
- Preserve the statement's markdown shape (if it's a "- " bullet, return a "- "
  bullet; keep a numbered list numbered).

Return ONLY a JSON object, no prose, no code fences:

{ "text": "<the strengthened statement, ready to replace the original in the skill file>" }

The statement, rubric, and failing answers follow.
