# Justfile skill
Recipes run as `just <recipe>` through `run_command`. `just --list` shows them all; `just --show <recipe>` prints one without running it.

- A recipe body ALWAYS runs: `just` tracks no dependencies and no output freshness.
- A recipe's indentation is spaces or tabs, never both.
- `{{var}}` is substituted by `just` before the shell runs; `$var` is expanded by the shell after.
- Each line runs in its own shell, so a `cd` is gone by the next line (unless the recipe has a shebang).
- Start a recipe with a `#` comment: `just --list` shows it.
- Declare prerequisites as recipe dependencies, not inline `just <other>` calls.
