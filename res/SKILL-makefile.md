# Makefile skill
Project has `Makefile` (or `makefile`/`GNUmakefile`). Use declared targets via `run_task`, NOT raw `make`.

## Probe available targets
- `make -pRrq | awk '/^[a-zA-Z0-9_-]+:/ {print $1}' | sort -u` → every defined target incl. pattern rules. Noisy but exhaustive.
- `grep '^[a-zA-Z][^:]*:' Makefile` → quick-and-dirty target list.
run_task already enumerated tasks this turn → reuse, don't re-parse.

## Editing
- **`.PHONY: <name>`** for any target not producing a file with that name. Without it, Make skips the target when a same-named file exists. Easiest rule to forget.
- Tab-indent recipe lines (spaces → "missing separator"). `:=` immediate eval, `=` deferred — `:=` is almost always what you want.
- New target → keep the name compatible with the build/test/lint/format classifier (the codehalter task router groups by these keywords).
- Depends on a tool not in the base image → surface that; a hidden `apt install` inside a target hides install latency.

## Gotcha
Each recipe line runs in its own subshell unless joined with `\`. So `cd foo` on one line, `make` on the next, runs in the original cwd.
