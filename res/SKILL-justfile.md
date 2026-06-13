# Justfile skill
Project has `justfile` (or `Justfile`/`.justfile`). Use declared recipes via `run_task`, NOT raw `just`. `just` = Make-alternative: cleaner syntax, no implicit dependency tracking, recipes run in `sh` by default (or shebang of choice).

## Probe available recipes
- `just --list` → every public recipe + docstring. Use first.
- `just --list --unsorted` → preserves file order.
- `just --show <recipe>` → print recipe body without running.
- `just --evaluate` → dump every variable's resolved value.
- `just --summary` → terse one-line recipe names.
run_task already enumerated recipes this turn → reuse, don't re-parse.

## Recipe basics
```just
default: build      # first recipe runs when `just` has no args
build:
    cargo build --release
# Comment above a recipe = its docstring.
test args="":
    cargo test {{args}}
# Recipes can depend on other recipes.
lint: build
    cargo clippy -- -D warnings
```
Hard rules when editing:
- **Consistent indentation** (spaces OR tabs, not both — `just` refuses mixed). Most projects use 4 spaces.
- **`{{var}}` for variable interpolation** (Jinja-like, NOT shell `$var`). `$var` = shell expansion inside recipe body.
- **`@cmd`** suppresses echo of command before running.
- **Recipes NOT incremental** — `just` always runs body. No built-in "output newer than input" check.
- **First recipe = default** unless you declare `default` explicitly.

## Parameters / defaults
```just
deploy env="staging":
    ./deploy.sh {{env}}
# `just deploy` → "staging"; `just deploy prod` → "prod".
# Variadic with `+` (one or more) or `*` (zero or more):
test +files:
    cargo test {{files}}
```

## Shell choice
Recipes default `sh -cu`. Override per-recipe with shebang, or globally `set shell := ["bash", "-cu"]` at top.

## When proposing new recipe
1. Pick name compatible with build/test/lint/format classifier (codehalter task router groups by these keywords).
2. Add leading docstring comment.
3. Declare prerequisites as recipe dependencies, NOT inline `just <other>` calls (latter spawns new `just` process).
4. Depends on tools not in base image → surface that; `apt install` inside a recipe hides install latency.

## Gotchas
- Mixed tabs/spaces → parse error. Pick one.
- `{{var}}` vs `$var`: `just` evaluates `{{var}}` BEFORE shell sees line; `$var` = shell expansion AFTER. Confusion here → recipes that "work in shell" but fail under `just` or reverse.
- Comments in recipe body must start with `#` on own line.
- `just --dry-run <recipe>` shows what would run without executing.
- Each recipe line runs in own shell. Use `\` for continuation, or wrap whole thing in shebang recipe.