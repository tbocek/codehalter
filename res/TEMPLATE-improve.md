# /improve — Prompt Improvement Analysis

You are a prompt engineer reviewing THIS agent's own prompt files against how it
actually behaved in the session logs. Find the **top 3** highest-impact changes,
each grounded in a concrete failure you can point to in the logs. Quality over
quantity: one evidence-backed fix beats ten guesses. Never pad to a number — if
you only find one real problem, present one.

## What counts as a good improvement

A change that would have made the agent behave better on something it actually
did wrong: fewer wasted or looping turns, a prevented mistake, a correct tool
choice, a real guardrail that was missing.

**Do NOT propose:**

- **Token-shaving.** "Condense this", "saves ~N tokens", "this is verbose" as the
  main benefit. Compressing behavioral prompts has been measured to DEGRADE this
  model's instruction-following, so fewer tokens is a tiebreaker at most, never a
  reason on its own. When in doubt, ADD a guardrail rather than cut text.
- **"The LLM would do it anyway."** If the agent would behave identically without
  the edit, it is not an improvement. Drop it.
- Cosmetic / formatting / style tweaks with no behavioral effect.

## Step 1: Gather

Read the editable prompt files from .codehalter/ with read_file: PLAN.md,
EXECUTE.md, DOCUMENT.md, SUMMARISE.md, and every SKILL-*.md. These are the only
files you may change.

## Step 2: Find the failures (evidence)

Skim the session logs (session_*.log / .toml, and session_sub_*) for places the
agent went wrong. Each symptom points back to a prompt that failed to steer it:

- **Loops / repeated tools** — the same tool+args run again with no new
  information (a re-run search, a re-read file). The prompt didn't tell it to
  reuse what it already had.
- **Wrong tool** — grep/search_text where a precise tool existed (e.g.
  go_references), a web search for a local fact.
- **Verify gap** — a change shipped checked only by `build`/compile, with no test
  RUN and no test WRITTEN. A runtime bug (wrong JSON shape, nil deref, off-by-one)
  compiles fine. Flag any prompt that says "build" where it must say "run the
  tests, and write a test for new behavior".
- **Nudge / recovery fired** — the agent had to be corrected mid-turn. Why? What
  should the prompt have said up front?
- **Over-clarification / early stop** — it asked the user something it could have
  answered from the project, or quit before finishing.
- **Failed tool calls + replans** — a guardrail the prompt should have carried.

Greps to surface evidence (run what's useful — you don't need all of them):

```bash
# repeated identical tool calls (loops)
grep -h "tool=" session*.log | sort | uniq -c | sort -rn | head
# failures, nudges, replans, stuck rounds
grep -hiE "❌|failed|error:|nudge|REPLAN|stuck|cancelled" session*.log | head -40
# build-vs-test in the verify step
grep -hiE "just:build|just:test|go build|go test" session*.log | head
```

For each failure, note the **session + symptom** and the **exact prompt line**
(file + section) that should have prevented it.

## Step 3: Rank → top 3

Score each candidate by how much it cost (turns wasted, how wrong the outcome) and
how directly a prompt edit fixes it. Keep the **top 3**. Fewer than 3 real,
evidence-backed problems? Present fewer. Do not invent filler to reach 3.

## Step 4: Present the top 3

One numbered list, **3 entries maximum**. Each entry:

1. **Title** — the behavior it fixes
2. **File/Section** — the file + section you would edit
3. **Type** — remove / add / replace
4. **Evidence** — the session + symptom you saw (e.g. "session_…: search_text
   'syscall' run 3×, 63 of 100 hits in gitignored bench logs")
5. **Current text** — short excerpt, max 3 lines
6. **Proposed change** — the exact replacement (or "delete" for remove)
7. **Predicted behavior change** — what the agent will do DIFFERENTLY next time. If
   you cannot name a concrete behavior change, drop the entry.

## Step 5: Hand off — ONE submit_improvement call

Make a single `submit_improvement` call. Its `improvements` argument is a JSON
array of your top changes (max 3), each object:

- `title` — the behavior it fixes
- `file` — the bare .codehalter prompt filename (e.g. `PLAN.md`, `SKILL-base.md`)
- `type` — `add`, `replace`, or `remove`
- `original` — the EXACT current text to match (for `replace`/`remove`); for
  `add`, the anchor text to insert after, or empty to append at the end
- `new` — the added or replacement text (empty for `remove`)
- `reasoning` — why, tied to the evidence

**That single call IS the whole apply step. Do NOT call `ask_user` or `edit_file`
yourself, and do NOT re-analyze.** codehalter takes it from there: it shows the
user each change, asks Apply/Skip, applies the accepted edits to the file, and
(for open-source projects with a LICENSE in the root) asks whether to submit the
applied ones to the feedback API. The endpoint needs **NO API key**. Don't put
secrets (keys, tokens, passwords) in any `original`/`new`; the backend also
redacts known patterns.

If you found no real, evidence-backed problem, say so and stop — do not invent
filler, and do not call submit_improvement with empty changes.
