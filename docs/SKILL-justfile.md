# Justfile skill

This project has a `justfile` (or `Justfile`/`.justfile`). Use declared
recipes via `run_task`, not raw `just`. `just` is a Make-alternative:
cleaner syntax, no implicit dependency tracking, recipes run in `sh` by
default (or a shebang of your choice).

## Probe the available recipes

- `just --list` — every public recipe with its docstring. Use first.
- `just --list --unsorted` — preserves file order.
- `just --show <recipe>` — print recipe body without running it.
- `just --evaluate` — dump every variable's resolved value.
- `just --summary` — terse one-line recipe names.

If `run_task` already enumerated recipes this turn, reuse — don't re-parse.

## Recipe basics

```just
default: build      # first recipe runs when `just` has no args

build:
    cargo build --release

# A comment above a recipe becomes its docstring.
test args="":
    cargo test {{args}}

# Recipes can depend on other recipes.
lint: build
    cargo clippy -- -D warnings
```

Hard rules when editing:

- **Consistent indentation** (spaces OR tabs, not both — `just` refuses
  to parse mixed). Most projects use 4 spaces.
- **`{{var}}` for variable interpolation** (Jinja-like, NOT shell `$var`).
  `$var` is shell expansion inside the recipe body.
- **`@cmd`** suppresses echo of the command before running it.
- **Recipes are NOT incremental** — `just` always runs the body. No
  built-in "output newer than input" check.
- **First recipe is the default** unless you declare `default` explicitly.

## Parameters and defaults

```just
deploy env="staging":
    ./deploy.sh {{env}}
# `just deploy` → "staging"; `just deploy prod` → "prod".

# Variadic with `+` (one or more) or `*` (zero or more):
test +files:
    cargo test {{files}}
```

## Shell choice

Recipes default to `sh -cu`. Override per-recipe with a shebang, or
globally with `set shell := ["bash", "-cu"]` at the top.

## When proposing a new recipe

1. Pick a name compatible with build/test/lint/format classifier
   (codehalter's task router groups by these keywords).
2. Add a leading docstring comment.
3. Declare prerequisites as recipe dependencies, not inline `just <other>`
   calls (the latter spawns a new `just` process).
4. If it depends on tools not in the base image, surface that — adding
   `apt install` inside a recipe hides install latency.

## Common gotchas

- Mixed tabs/spaces → parse error. Pick one.
- `{{var}}` vs `$var`: `just` evaluates `{{var}}` BEFORE the shell sees
  the line; `$var` is shell expansion AFTER. Confusion here causes
  recipes that "work in the shell" but fail under `just` or vice versa.
- Comments in a recipe body must start with `#` on their own line.
- `just --dry-run <recipe>` shows what would run without executing.
- Each recipe line runs in its own shell. Use `\` for continuation, or
  wrap the whole thing in a shebang recipe.
