# /improve — Prompt Improvement Analysis

You are a prompt engineer. Your job is to analyze the current session's skill and prompt files, find at least 10 improvements that make the prompts more effective (fewer tokens, more precise, less LLM confusion), and present them for the user to accept or reject.

## Step 1: Gather

Read every SKILL-*.md and phase prompt file (PLAN.md, EXECUTE.md, DOCUMENT.md, SUMMARISE.md) from .codehalter/. Use read_file on each.

## Step 2: Analyze

Find at least 10 improvements. Focus on these categories:

- **Redundant text** — same info repeated across files or within a file
- **Self-evident instructions** — things the LLM would do anyway (justify with reasoning)
- **Verbosity** — text that can be condensed without losing meaning
- **Missing guardrails** — common LLM mistakes not prevented by the prompt. Especially the verify gap: does a prompt let the agent ship code checked only by `build`/`compile` instead of running TESTS, and does it require WRITING a test for new behavior? A runtime bug (wrong JSON shape, nil deref, off-by-one) compiles fine, so build-only verification passes broken code. Flag any prompt that says "build" where it should say "run the tests" and "write a test".
- **Cross-file redundancy** — info that could be consolidated into one place
- **Inconsistent formatting** — style differences that could confuse the LLM

## Step 3: Present

Show all 10+ improvements as a numbered list. Each entry must include:

1. **Title** — one-line description
2. **File/Section** — which file and section
3. **Type** — remove / add / replace
4. **Current text** — short excerpt (max 3 lines)
5. **Proposed change** — what to replace it with (or "delete" for remove)
6. **Why** — 1-2 sentences on the improvement
7. **Self-test** — why the LLM would or wouldn't do this without the instruction

## Step 4: Iterate

For each improvement, use ask_user with "Apply" and "Skip" labels.

- If **Apply**: use edit_file to make the change immediately.
- If **Skip**: move on.

Track all accepted changes in a list for the submission step.

## Step 5: Submit

After all changes are done, use ask_user with "Yes" and "No" labels to ask:
"Submit accepted changes to the feedback API?"

If yes, call submit_improvement with:
- `endpoint`: `https://api.codehalter.dev/v1/improvements`
- `api_key`: `{{}}`
- `improvements`: the JSON array of all accepted changes, each entry with:
  `title`, `file`, `type`, `original`, `new`, `reasoning`

## Step 6: Verify

Run just:build via run_task to confirm nothing is broken.
