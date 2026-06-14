# /improve — Prompt Improvement Analysis

You are a prompt engineer. Your job is to analyze the current session logs (in .codehalter), skill and prompt files, find at least 10 improvements that make the prompts more effective (fewer tokens, more precise, less LLM confusion), and present them for the user to accept or reject.

## Step 1: Gather

Read every SKILL-*.md and phase prompt file (PLAN.md, EXECUTE.md, DOCUMENT.md, SUMMARISE.md) from .codehalter/. Use read_file on each. Analyze the logs session_date*.log|toml and session_sub*.log|toml.

## Step 1.5: Quick Diagnostics

Run these commands to get a quantitative picture of the session(s) before diving in:

```bash
# Most common tools used
grep -h "tool=" session*.log | sed 's/.*tool=\([^ ]*\).*/\1/' | sort | uniq -c | sort -rn | head -20

# Tool failures and errors
grep -hi "❌\|failed\|error:\|Error\|FAIL" session*.log | head -30

# Repeated tool calls (same tool in a row suggests unclear prompt)
grep -h "tool=" session*.log | sed 's/.*tool=\([^ ]*\).*/\1/' | uniq -c | sort -rn | head -20

# How many times did the agent re-read the same file?
grep -h "read_file" session*.log | wc -l

# Sessions that required replanning
grep -h "REPLAN" session*.log | wc -l

# Count total tool calls per session
for f in session_*.log; do echo -n "$f: "; grep -c "tool=" "$f"; done | sort -t: -k2 -rn

# Cache analysis: look for cache hits/misses
grep -hi "cache" session*.log | head -50

# Nudge analysis: find where nudges appear
grep -hi "nudge" session*.log | head -50

# Look at what was in the system prompt
grep -h "^system_prompt\|^summary\|^messages" session*.toml | head -5

# Find commands that could be improved (slow or failed)
grep -h "slow\|timeout\|hang" session*.log | head -30
```

**What these reveal:**

| Metric | What to flag in prompts |
|--------|------------------------|
| High read_file count | Prompt doesn't constrain reading; agent wastes turns |
| Repeated tool names | Prompt instruction was unclear, agent retried |
| TOOL_CALL_FAILED + replan cycles | Missing guardrails — the prompt didn't prevent the failure |
| Very long LLM responses in .log | Prompt doesn't constrain output length |
| Tools used that shouldn't be (web_search for local things) | Prompt didn't establish "check local first" clearly |
| build-only verify (no `just:test`) | Prompt says "build" where it should say "test" |
| **Cache misses** | **Skill change or compaction — check if skills were updated during analysis** |
| **Nudges** | **Why was a nudge needed? Could the prompt language be clearer?** |

**Cache Consistency Analysis:**
- **Track**: Every cache hit/miss across tool calls. Cache should only miss on skill file changes or compaction events.
- **Look for**: Repeated cache misses on the same tool call — this suggests the prompt isn't giving the agent enough context to be self-sufficient.
- **Flag**: If a cache miss isn't due to a skill change or compaction, the prompt might be forcing unnecessary re-computation.

**Nudge Analysis:**
- **Track**: Where nudges appear in the conversation flow.
- **Analyze why**: Was the original prompt too vague? Did the agent need additional context?
- **Commands to improve**: When nudging was required, what specific command or clarification was given? Could the original prompt have been written to include that information?

## Step 2: Analyze

Find at least 10 improvements. Focus on these categories:

- **Less agent/human turns** — prompts that would lead to fewer turns
- **Commands with less tokens** — commands that would have produced the same result but with fewer tokens
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

**Prerequisite**: the project must have an open-source license (MIT, BSD, Apache, GPL, etc.) in its root. If no such license is present, the submission will be rejected — inform the user and do not call submit_improvement.

**Note**: do not include sensitive data (API keys, tokens, passwords, secrets) in the improvement text. The backend will redact known patterns, but it's better to avoid them entirely.

## Step 6: Verify

Run just:build via run_task to confirm nothing is broken.
