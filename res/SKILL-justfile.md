# Justfile skill
Project has `justfile` (or `Justfile`/`.justfile`): recipes run as `just <recipe>` through `run_command`.

## Probe
- `just --list` → every public recipe with its docstring. Use before guessing a name.
- `just --show <recipe>` → print a recipe body without running it.

## What just does NOT do
- **Recipes are not incremental.** No dependency tracking, no output-freshness check: the body ALWAYS runs. "Skipping, the output is up to date" does not exist here. Don't claim it, don't rely on it.
- Mixed indentation is refused: spaces OR tabs inside a recipe, never both.
- `{{var}}` is just's own substitution, applied BEFORE the shell sees the line; `$var` is shell expansion after. Confusing the two gives recipes that work pasted into a shell and fail under `just`.
- Each recipe line runs in its own shell (`sh -cu`; override with a shebang or `set shell := ["bash", "-cu"]`), so a `cd` on one line is gone by the next.

## Writing a recipe
- Lead with a `#` comment: that is what `just --list` shows.
- Declare prerequisites as recipe dependencies, not as inline `just <other>` calls: the latter spawns a second `just`.
