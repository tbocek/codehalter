# Makefile skill

## Overview
- Project has `Makefile` (or `makefile`/`GNUmakefile`).
- Use declared targets via `run_task`, NOT raw `make`.

## Target probing
- `make -pRrq | awk '/^[a-zA-Z0-9_-]+:/ {print $1}' | sort -u` → every defined target incl pattern rules. Noisy but exhaustive.
- `awk '/^\.PHONY:/ {for (i=2;i<=NF;i++) print $i}' Makefile` → declared phony targets, usually user-facing tasks.
- run_task already enumerated tasks this turn → reuse, don't re-parse.

## Hard rules
- **`:=` immediate eval, `=` deferred (lazy) eval**. `:=` almost always what you want.
- **`@cmd`** suppresses echo. Use `@` for echo/informational lines; leave off for real work.

## Automatic variables
- Available in recipe context.
- `$^` → all prerequisites, space-separated, deduped.
- `$?` → prerequisites newer than target.
- `$*` → stem (`%`-matched portion) in pattern rule.

## Gotchas
- `make --warn-undefined-variables` exposes typos in `$(VAR_NAME)`.
