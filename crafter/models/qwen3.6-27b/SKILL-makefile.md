# Makefile skill

## Overview
- Project has `Makefile` (or `makefile`/`GNUmakefile`).
- Use declared targets via `run_task`, NOT raw `make`.

## Target probing
- `make -pRrq | awk '/^[a-zA-Z0-9_-]+:/ {print $1}' | sort -u` → every defined target incl pattern rules. Noisy but exhaustive.
- `awk '/^\.PHONY:/ {for (i=2;i<=NF;i++) print $i}' Makefile` → declared phony targets, usually user-facing tasks.
- run_task already enumerated tasks this turn → reuse, don't re-parse.

## Hard rules
- **`@cmd`** suppresses echo. Use `@` for echo/informational lines; leave off for real work.

## Automatic variables
- Available in recipe context.
- `$^` → all prerequisites, space-separated, deduped.
- `$?` → prerequisites newer than target.

## New targets
- ALWAYS indent recipe lines with a literal tab character (`\t`) in Makefiles — never spaces, never flush-left. `make` requires a real tab before each command under a target; any other indentation causes a `missing separator` error. Write the tab into the file even when your editor, formatter, or tool output prefers spaces.

## Gotchas
- `make --warn-undefined-variables` exposes typos in `$(VAR_NAME)`.
