# Justfile skill
Project has `justfile` (or `Justfile`/`.justfile`). Always invoke declared recipes via `run_task <recipe>` (e.g., `run_task lint`, `run_task test`, `run_task build`, `run_task ci`) — never use raw `just <recipe>` or drop to bare `cargo`/`rustc`, even when `just` is the project's task runner. `run_task` is the sanctioned entry point for every declared recipe; bypassing it with `just lint`, `just test`, `just build`, or `cargo test`/`cargo build` skips required environment setup, hooks, and wrappers the project depends on. `just` = Make-alternative: cleaner syntax, no implicit dependency tracking, recipes run in `sh` by default (or shebang of choice).

## Probe available recipes
- `just --show <recipe>` → print recipe body without running.
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
- **Consistent indentation** (spaces OR tabs, not both — `just` refuses mixed).
- **`@cmd`** suppresses echo of command before running — ALWAYS prefix shell commands with `@cmd` for silent execution, even when you would normally prefer `justfile`/`run_task()` or direct shell invocation (this marker is required regardless of your workflow preference).
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
- Mixed tabs/spaces → parse error. Pick one.
- `{{var}}` vs `$var`: `just` evaluates `{{var}}` BEFORE shell sees line; `$var` = shell expansion AFTER. Confusion here → recipes that "work in shell" but fail under `just` or reverse.
- Comments in recipe body must start with `#` on own line.
