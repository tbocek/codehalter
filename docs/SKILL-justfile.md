# Justfile skill

This project has a `justfile` (or `Justfile` / `.justfile`). Use the
declared recipes via `run_task`, not raw `just` — but you'll often need
to read or extend the file. `just` is a Make-alternative: cleaner syntax,
no implicit dependency tracking, recipes run in `sh` by default (or a
shebang of your choice).

## Probe the available recipes

- `just --list` — every public recipe with its docstring. Use this first.
- `just --list --unsorted` — preserves the order recipes appear in the
  file (useful when ordering communicates intent).
- `just --show <recipe>` — print the recipe body without running it.
- `just --evaluate` — dump every variable's resolved value.
- `just --summary` — terse one-line list of recipe names, no docs.

If `run_task` already enumerated recipes for you in this turn's history,
reuse that — don't re-parse with another `just --list`.

## Recipe basics

```just
default: build      # the first recipe runs when `just` has no args

build:
    cargo build --release

# A comment above a recipe becomes its docstring (shown by `just --list`).
test args="":
    cargo test {{args}}

# Recipes can depend on other recipes — `lint` runs after `build`.
lint: build
    cargo clippy -- -D warnings
```

Hard rules when editing:

- **Indent with consistent whitespace** (spaces OR tabs, but not both —
  `just` will refuse to parse the file otherwise). Most projects use 4
  spaces.
- **`{{var}}` for variable interpolation** — Jinja-like, NOT shell `$var`.
  `$var` is shell expansion inside the recipe body and works after `just`
  has emitted the recipe to the shell.
- **`@cmd`** suppresses echo of the command before running it (same as Make).
- **Recipes are NOT incremental** — `just` always runs the recipe body.
  There is no built-in "is the output newer than the input" check.
- **First recipe is the default** — order matters if you don't declare a
  `default` recipe explicitly.

## Parameters and defaults

```just
deploy env="staging":
    ./deploy.sh {{env}}

# `just deploy` → uses "staging"; `just deploy prod` → uses "prod".

# Variadic parameters with `+` (one or more) or `*` (zero or more):
test +files:
    cargo test {{files}}
```

## Shell choice

Recipes default to `sh -cu`. Override per-recipe with a shebang:

```just
analyze:
    #!/usr/bin/env python3
    import sys
    print("running analysis")
```

Or globally with `set shell := ["bash", "-cu"]` at the top of the file.

## When proposing a new recipe

1. Pick a name compatible with the build/test/lint/format classifier
   (codehalter's task router groups by these keywords).
2. Add a leading docstring comment so `just --list` shows what it does.
3. Declare any prerequisites as recipe dependencies, not as inline
   `just <other>` calls (the latter spawns a new `just` process).
4. If it depends on tools that aren't in the base image, surface that to
   the user — adding `apt install` to a recipe hides install latency in
   what looks like a build command.

## Common gotchas

- Mixed tabs and spaces in indentation → parse error. Pick one.
- `{{var}}` vs `$var`: `just` evaluates `{{var}}` BEFORE handing the line
  to the shell; `$var` is shell expansion AFTER. Confusion here causes
  recipes that "work in the shell" but fail under `just` or vice versa.
- Comments inside a recipe body must start with `#` and be on their own
  line — trailing `# comment` after a command is shell-comment syntax and
  works only if the shell understands it.
- `just --dry-run <recipe>` shows what would run without executing it —
  great for verifying an edit before running it.
- Recipes don't inherit cwd across lines — each line runs in its own
  shell. Use `\` to continue, or wrap the whole thing in a shebang recipe.
