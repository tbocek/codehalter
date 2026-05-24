# Makefile skill

This project has a `Makefile` (or `makefile`/`GNUmakefile`). Use declared
targets via `run_task`, not raw `make`.

## Probe the available targets

- `make -pRrq | awk '/^[a-zA-Z0-9_-]+:/ {print $1}' | sort -u` — every
  defined target including pattern rules. Noisy but exhaustive.
- `grep '^[a-zA-Z][^:]*:' Makefile` — quick-and-dirty target list.
- `awk '/^\.PHONY:/ {for (i=2;i<=NF;i++) print $i}' Makefile` — declared
  phony targets, which usually correspond to user-facing tasks.

If `run_task` already enumerated tasks this turn, reuse — don't re-parse.

## Hard rules when editing

- **Tabs, not spaces**, before recipe commands. Space-indented recipes
  produce "missing separator" errors that look unrelated.
- **`:=` for immediate eval, `=` for deferred (lazy) eval**. `:=` is
  almost always what you want.
- **`@cmd`** suppresses echo of the command. Use `@` for `echo` /
  informational lines; leave it off for real work.
- **`$$`** to escape a literal dollar sign in a recipe.
- **`.PHONY: <name>`** for any target that doesn't produce a file with
  that name. Without it, Make may skip the target when a same-named
  file exists.

## Automatic variables (recipe context)

- `$@` — the target being built.
- `$<` — the first prerequisite.
- `$^` — all prerequisites, space-separated, deduped.
- `$?` — prerequisites newer than the target.
- `$*` — the stem (`%`-matched portion) in a pattern rule.

## When proposing a new target

1. Add to `.PHONY` if it doesn't produce a file.
2. Tab-indent the recipe lines.
3. If it depends on tools not in the base image, surface that — hidden
   `apt install` in a target hides install latency.
4. Keep names compatible with build/test/lint/format classifier
   (codehalter's task router groups by these keywords).

## Common gotchas

- Variables in recipes: `$VAR` is shell, `$(VAR)` is Make. Mixing is fine
  but easy to get wrong.
- Each recipe line runs in its own subshell unless joined with `\`. So
  `cd foo` then `make` on the next line returns to the original cwd.
- `make -n <target>` shows what *would* run without executing —
  invaluable for sanity-checking an edit.
- `make --warn-undefined-variables` exposes typos in `$(VAR_NAME)`.
