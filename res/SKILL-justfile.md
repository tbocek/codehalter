# Justfile skill
Project has `justfile` (or `Justfile`/`.justfile`). Use declared recipes via `run_task`, NOT raw `just`.

## Probe available recipes
- `just --list` → every public recipe + docstring. Use first.
- `just --show <recipe>` → print a recipe body without running it.
- `just --evaluate` → dump every variable's resolved value.
run_task already enumerated recipes this turn → reuse, don't re-parse.

## Editing
- **Recipes are NOT incremental.** `just` has no dependency tracking and no output-freshness check: it ALWAYS runs the body, every time. Skipping "because the output is up to date" does not exist — don't claim it, and don't rely on it.
- Consistent indentation (spaces OR tabs, not both — `just` refuses mixed).
- `{{var}}` is evaluated by just BEFORE the shell sees the line; `$var` is shell expansion after. Confusing the two → recipes that "work in the shell" but fail under `just`.
- Recipe lines each run in their own shell (`sh -cu` by default; override with a shebang or `set shell := ["bash", "-cu"]`).

## When proposing a new recipe
1. Name it compatibly with the build/test/lint/format classifier (the codehalter task router groups by these keywords).
2. Add a leading docstring comment — that's what `just --list` shows.
3. Declare prerequisites as recipe dependencies, NOT inline `just <other>` calls (the latter spawns a new `just` process).
4. Depends on a tool not in the base image → surface that; `apt install` inside a recipe hides install latency.
