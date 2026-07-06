# Skill Crafter

Different models need different skills. codehalter steers coding agents with
**SKILL files** — short lists of behavioral statements ("wrap errors with `%w`",
"read a file before editing it", "on Arch, reach for `pacman` before the AUR").
But a statement only earns its place if the model would get it *wrong without
it*: some behaviors are baked into a model's weights, others aren't, and a
finetune moves the line. Skill Crafter **measures which statements each target
model actually needs** and prunes the rest, producing a per-model SKILL file.

## The pipeline

Each skill flows through streamline → segment → (per claim) author → generate →
score → prune:

```
ground-skills/SKILL-x.md
  │ STREAMLINE  judge   → clean-skills/SKILL-x.md   headers become categories, instructions become bullets
  │ SEGMENT     judge A → atomic claims, each mapped to its verbatim source span
  │             judge B → repair pass: re-quotes any source that didn't locate
  ▼  per claim:
  │ AUTHOR      judge   → a disguised "how would you approach this?" question + a rubric
  │ GENERATE    target  → the model's answer with NO skill (arm A) and with the WHOLE clean skill (arm B)
  │ SCORE       judge   → keep the statement if arm B follows it and arm A doesn't
  ▼
models/<model>/SKILL-x.md   the clean skill with the un-needed lines pruned out
```

### The A/B probe

For each claim the judge authors a disguised question. By **default** it asks the
model to lay out its **plan / approach** — so ordered and multi-step behavior
("try X, then fall back to Y") surfaces in a single reply. The target answers it,
`samples` times, in two arms:

- **Arm A** — no system prompt: the model's natural behavior.
- **Arm B** — the **whole clean skill** loaded: the realistic context the agent
  actually receives (not the single line out of context).

Each answer is judged against the rubric. A statement is **kept** when arm B
follows it and arm A doesn't, and **dropped** when the model already behaves that
way with no skill at all. This is keep-biased on purpose (see `keep_threshold`):
a wrong drop silently regresses a behavior; a wrong keep costs a little prefill.

A drop where the model failed the rubric even *with* the skill loaded (every
sample `B_sat=false`) is a different animal: the statement isn't redundant, it's
**ignored** — usually the model's own prior overriding the instruction. Those get
one **strengthen-and-retry**: the judge rewrites the statement (same meaning,
added emphasis, rationale, and an exception clause naming the observed
deviation), the target regenerates arm B with the rewrite spliced into the skill
(arm A is reused — no skill in it), and the judge re-scores. If the rewrite
works, the claim is kept and the **strengthened wording replaces the original**
in that model's output skill; if it still fails, the drop is flagged
`INEFFECTIVE` — a real finding: this model won't follow that policy regardless
of phrasing. Both attempts are recorded in `results.jsonl`.

For claims whose whole point is *executing* an action (call `web_search` before
declaring a package missing; read before edit), the author instead offers real
**tools** and scores the actual tool calls — the one case where "does it *do* it"
beats "does it *say* it".

### The scheduler

Prep, author, generate, and score are scheduled tasks over four unbounded queues.
A free judge slot always does the task that refills the emptiest **upstream**
queue — priority **AUTHOR > PREP > SCORE** — so the target is fed first, claims
are made only when it runs out of them, and scoring drains once everything
upstream is healthy. The target runs generation in parallel; every generated
answer is persisted, so a run is resumable at any point. Several `[[judge]]`
endpoints form an interchangeable pool — a second judge is roughly linear extra
throughput on the bottleneck.

## Configure — `crafter.toml`

```toml
[settings]
samples = 3          # A/B repetitions per claim
keep_threshold = 1   # min "keep" votes (of samples) to keep a statement;
                     # 1 = keep unless ALL samples agree to drop. 2 = strict majority of 3.
# skills = ["go", "base"]   # restrict to these SKILL-<name>.md; empty = all

# One or more judges — the strong reference model that segments, authors, and
# scores. Any judge does any judge task, so they should serve the same (or an
# equivalent) model; a second [[judge]] is pure throughput.
[[judge]]
server = "http://localhost:9001"     # host root only; crafter appends /v1/chat/completions
model  = "Qwen3.5 (122B-A10B; ...)"  # must match the id at /v1/models exactly
temperature = 0.6
# api_key   = "..."
# top_p     = 0.95
# params    = { top_k = 20, min_p = 0.0, presence_penalty = 1.5 }
# parallel  = 2      # omit to auto-detect the server's slot count (llama.cpp -np)

# One [[model]] per target being profiled. `name` is its folder under models/.
[[model]]
name    = "gemma-4-31b"
server  = "https://your-endpoint"
model   = "Gemma-4 (31B; ...)"
api_key = "..."
```

A preflight pings every endpoint (and auto-detects slot counts) before any real
work, so a wrong key or model id fails fast.

## Run

```
# copy the skills you want to probe into ground-skills/SKILL-*.md, then:
go run .                              # or: go build -o crafter . && ./crafter
```

Flags: `-ground` (input dir, default `ground-skills`), `-models` (output,
`models`), `-clean` (`clean-skills`), `-cache` (`.cache`), `-config`
(`crafter.toml`), `-docs` (generated HTML report), `-llmlog` (wire-level trace),
`-skip-preflight`.

## Files & caching

| path | what | owner |
|---|---|---|
| `ground-skills/SKILL-*.md` | the skills you want to probe | you (input) |
| `clean-skills/SKILL-*.md` | the streamlined skill that's actually probed | **you** — generated once when absent, then used as-is; hand-edit freely, delete to regenerate |
| `models/<model>/SKILL-*.md` | the pruned per-model skill | output |
| `models/<model>/results.jsonl` | every verdict + both answers + the judge's reasoning | resume ledger |
| `models/<model>/samples.jsonl` | generated A/B pairs | resume ledger |
| `models/<model>/stats.json` | byte sizes + keep/drop/error counts | output |
| `.cache/segments/*.json` | segmented claims, per skill | cache |
| `.cache/authored/*.json` | authored questions, shared across models | cache |

Everything is resumable — a re-run skips claims already decided. To force a fresh
probe of a target, delete its `results.jsonl` + `samples.jsonl`. Editing a clean
skill re-segments it; editing a prompt in `res/` invalidates that stage's cache
and re-runs only it. `clean-skills/` is yours to edit — it is never overwritten
while it exists.

## Prompts (`res/`, embedded at build)

- `STREAMLINE.md` — rewrite a skill so headers are categories and instructions
  are body bullets, dropping nothing. A mechanical no-loss guard (every code span
  and `{{cmd:}}` directive must survive) falls back to the verbatim original.
- `SEGMENT.md` — split into atomic claims with **verbatim** source spans.
- `REPAIR.md` — the second-judge pass that re-quotes sources the first pass
  couldn't locate.
- `AUTHOR.md` — the plan-eliciting question + rubric (tools for execution claims).
- `JUDGE.md` — score arm A vs arm B against the rubric.
- `STRENGTHEN.md` — rewrite an ignored statement (same meaning, more force) for
  the one-shot retry.

## Contribute

This works best with breadth — many models, many finetunes, many probed
statements. Run the crafter against a model we don't have and open a PR with its
skill set + stats; propose statements via `/improve` during real sessions;
challenge a verdict (every keep/drop ships with both answers and the judge's
reasoning); or sharpen the method — question authoring, rubrics, judge bias. It's
all open.
