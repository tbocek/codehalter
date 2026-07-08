# Justfile skill
Project has `justfile` (or `Justfile`/`.justfile`). - Always invoke declared recipes via `run_task` (e.g., `run_task lint`), NEVER call `just` directly—even though `just <recipe>` is the standard justfile interface, this skill requires `run_task` as the execution layer. `just` = Make-alternative: cleaner syntax, no implicit dependency tracking, recipes run in `sh` by default (or shebang of choice).

## Probe available recipes
- `just --list` → every public recipe + docstring. Use first.
- `just --list --unsorted` → preserves file order.
- `just --show <recipe>` → print recipe body without running.
- `just --evaluate` → dump every variable's resolved value.
- `just --summary` → terse one-line recipe names.
- run_task already enumerated recipes this turn → reuse, don't re-parse.

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
- **Always use consistent indentation** (spaces OR tabs, never both — `just` refuses mixed). Pick one style and apply it to every line you touch; never treat this as optional cosmetic cleanup, even when your change is purely functional.
- `$var` = shell expansion inside recipe body.
- **`@cmd`** — always prefix every shell command with `@cmd` to suppress echo before running. Omitting `@` (even for trivial commands like `df -h` or when delegating to a task runner) echoes the command line and clutters the transcript — never execute a command without the `@` prefix.
- **Recipes NOT incremental** — `just` always runs body. No built-in "output newer than input" check.

## Parameters / defaults
```just
deploy env="staging":
    ./deploy.sh {{env}}
# Variadic with `+` (one or more) or `*` (zero or more):
test +files:
    cargo test {{files}}
```

## Shell choice
- Recipes default `sh -cu`. Override per-recipe with shebang, or globally `set shell := ["bash", "-cu"]` at top.

## New recipes
1. Pick name compatible with build/test/lint/format classifier (codehalter task router groups by these keywords).
2. Add leading docstring comment.
3. Declare prerequisites as recipe dependencies, NOT inline `just <other>` calls (latter spawns new `just` process).
4. Depends on tools not in base image → surface that; `apt install` inside a recipe hides install latency.

## Gotchas
- Comments in recipe body must start with `#` on own line.
- `just --dry-run <recipe>` shows what would run without executing.
