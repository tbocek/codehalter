# Makefile skill
Project has `Makefile` (or `makefile`/`GNUmakefile`). Use declared targets via `run_task`, NOT raw `make`.

## Probe available targets
- `make -pRrq | awk '/^[a-zA-Z0-9_-]+:/ {print $1}' | sort -u` → every defined target incl pattern rules. Noisy but exhaustive.
- `grep '^[a-zA-Z][^:]*:' Makefile` → quick-and-dirty target list.
- `awk '/^\.PHONY:/ {for (i=2;i<=NF;i++) print $i}' Makefile` → declared phony targets, usually user-facing tasks.
run_task already enumerated tasks this turn → reuse, don't re-parse.

## Hard rules when editing
- **Tabs, NOT spaces**, before recipe commands. Space-indented recipes → "missing separator" errors that look unrelated.
- **`:=` immediate eval, `=` deferred (lazy) eval**. `:=` almost always what you want.
- **`@cmd`** suppresses echo. Use `@` for echo/informational lines; leave off for real work.
- **`$$`** escapes literal dollar sign in recipe.
- **`.PHONY: <name>`** for any target not producing a file with that name. Without it, Make may skip target when same-named file exists.

## Automatic variables (recipe context)
- `$@` → target being built.
- `$<` → first prerequisite.
- `$^` → all prerequisites, space-separated, deduped.
- `$?` → prerequisites newer than target.
- `$*` → stem (`%`-matched portion) in pattern rule.

## When proposing new target
1. Add to `.PHONY` if not producing a file.
2. Tab-indent recipe lines.
3. Depends on tools not in base image → surface that; hidden `apt install` in a target hides install latency.
4. Keep names compatible with build/test/lint/format classifier (codehalter task router groups by these keywords).

## Gotchas
- Recipe vars: `$VAR` = shell, `$(VAR)` = Make. Mixing fine but easy to get wrong.
- Each recipe line runs in own subshell unless joined with `\`. So `cd foo` then `make` next line returns to original cwd.
- `make -n <target>` shows what *would* run without executing — invaluable for sanity-checking an edit.
- `make --warn-undefined-variables` exposes typos in `$(VAR_NAME)`.