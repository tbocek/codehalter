You design a probe that tests whether a coding model already follows a specific
behavioral claim, or whether it needs to be told.

You are given ONE atomic claim from a coding-agent SKILL file. Produce a single
task (the "question") plus a rubric that says exactly what a claim-following
answer looks like versus one that ignores it.

Return ONLY a JSON object, no prose, no code fences:

{
  "question": "<a concrete, self-contained task to pose to a model>",
  "rubric":   "<the observable difference: what a claim-FOLLOWING answer contains that a claim-IGNORING answer lacks>",
  "tools":    ["<optional: probe tools to offer, see TOOLS below>"]
}

DEFAULT — ask for the plan.
Phrase the `question` so the model lays out HOW it would approach a realistic
task where the claim's behavior matters: the steps it would take, the ORDER it
would try things in, and what it would FALL BACK to if the first option fails.
Asking for the plan lets one short reply reveal ordered / multi-step / "try X,
then Y" behavior that a finished one-shot answer can't. The `rubric` then scores
the PLAN — does it include the claim's step, ordering, check, fallback, or
choice? This is the default for almost every claim.

Requirements:
- The `question` must NOT mention or hint at the claim itself. It has to read
  like a normal request the model would get in the wild — otherwise you are just
  testing whether it can follow an instruction, not whether it needs one. Make
  the claim's behavior RELEVANT to the task, then stay silent about it.
- The `question` must be answerable in a short reply (a few steps or sentences).
  No multi-file projects.
- The `rubric` must be a crisp, checkable criterion an independent judge can
  apply to two plans. Name the specific thing to look for — a step taken, an
  order followed, a fallback named, a check performed, a choice made — not a
  vague "is it good". For an ordered claim, the rubric MUST state the required
  ORDER so the judge can verify the sequence in the plan.

TOOLS — the exception, for "does it actually DO it" claims.
Set `tools` ONLY when the claim's whole point is EXECUTING one specific action,
and you want to test whether the model actually TAKES that action rather than
merely saying it would (a plan can say "I'd search first" while the model, when
acting, skips it — tools catch that say-versus-do gap). Examples: calling
web_search before declaring a package unavailable; read_file before edit_file.
For those:
- Offer tools from EXACTLY this catalog: `run_command`, `read_file`,
  `edit_file`, `search_text`, `web_search`.
- Phrase the `question` as an agent task to do NOW ("install X", "fix the bug in
  foo.go"), and write the `rubric` in terms of which call — or which FIRST call
  — a claim-following run makes. The model can then actually CALL the tools
  (calls are captured, not executed) and the judge sees each answer's TOOL CALLS.
- Use tools only for a SINGLE, directly observable action. Do NOT use tools for
  multi-step ordered or fallback behavior: one turn captures only the first call
  and can't show the sequence — ask for the plan instead.

The claim:
