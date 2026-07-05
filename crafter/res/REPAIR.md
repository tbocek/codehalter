You fix broken source citations produced by a first segmentation pass.

A SKILL file was split into atomic claims. Each claim has a `text` (a
self-contained rewrite of one statement) and a `source` (the verbatim span in
the skill the statement came from). For the claims below, the `source` could NOT
be found verbatim in the skill — it was paraphrased, reworded, truncated, or is
ambiguous — so each will be DROPPED (never tested) unless you fix its `source`.

You are given the FULL skill text and the broken claims. For each claim, return
the EXACT verbatim span from the skill text that this statement came from —
copied character-for-character (same words, punctuation, casing, symbols), so it
can be found by a plain string search.

Rules:
- Copy the `source` from the skill text VERBATIM. Do NOT paraphrase, summarize,
  translate, reflow, or "clean up". If you cannot copy it exactly, you have not
  found it.
- Keep each claim's `text` UNCHANGED — you are only fixing `source`.
- The span must be UNIQUE in the skill (appear in exactly one place). If the
  minimal span is ambiguous, extend it with a few neighboring words until it is
  unique.
- Prefer the SHORTEST verbatim span that still uniquely covers the statement.
- Keep the claims in the SAME ORDER you were given, one output claim per input.
- If a claim's statement genuinely is NOT present in the skill text (the first
  pass hallucinated it), set its `source` to "" — do not invent one.

Return ONLY a JSON object, no prose, no code fences:

{
  "claims": [
    { "text": "<the claim text, unchanged>", "source": "<exact verbatim span from the skill, or \"\">" }
  ]
}

The skill text and the broken claims follow.
