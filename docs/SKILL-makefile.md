# Makefile skill

This project has a `Makefile` (or `makefile` / `GNUmakefile`). Use the
declared targets via `run_task`, not raw `make` — but you'll often need
to read or extend the file. These conventions matter.

## Probe the available targets

- `make -pRrq | awk '/^[a-zA-Z0-9_-]+:/ {print $1}' | sort -u` — every
  defined target, including pattern rules. Noisy but exhaustive.
- `grep '^[a-zA-Z][^:]*:' Makefile` — quick-and-dirty target list (misses
  `.PHONY` declarations and rules with prerequisites on the same line).
- `awk '/^\.PHONY:/ {for (i=2;i<=NF;i++) print $i}' Makefile` — the
  user-declared phony targets, which usually correspond to the
  "user-facing" tasks (`test`, `build`, `lint`).

If `run_task` already enumerated tasks for you in this turn's history,
reuse that — don't re-parse the Makefile.

## Hard rules when editing

- **Tabs, not spaces**, before recipe commands. A space-indented recipe
  produces "missing separator" errors that look unrelated. When you
  add a recipe line, the indentation MUST be one literal tab character.
- **`:=` for immediate eval, `=` for deferred (lazy) eval**. `:=` is
  almost always what you want; `=` re-runs the right-hand side every
  time the variable is expanded, which can be slow and surprising.
- **`@cmd` suppresses the echo of the command itself** before running it.
  Use `@` for `echo` and informational lines; leave it off for the real
  work so users see what's running.
- **`$$` to escape a literal dollar sign** in a recipe (shell expansion
  happens after Make's expansion).
- **`.PHONY: <name>`** for any target that doesn't produce a file with
  that exact name. Without it, Make may skip the target when a same-named
  file already exists (stale or unrelated).

## Automatic variables (recipe context)

- `$@` — the target being built.
- `$<` — the first prerequisite.
- `$^` — all prerequisites, space-separated, deduplicated.
- `$?` — prerequisites newer than the target (for incremental rules).
- `$*` — the stem (the `%`-matched portion) in a pattern rule.

## When proposing a new target

1. Add it to `.PHONY` if it doesn't produce a file.
2. Put the recipe lines tab-indented underneath.
3. If it depends on tools that aren't in the base image, surface that to
   the user — adding `apt install` to a target hides install latency in
   what looks like a build command.
4. Keep target names compatible with the build/test/lint/format
   classifier (codehalter's task router groups by these keywords).

## Common gotchas

- Variables in recipes use shell expansion (`$VAR` is shell, `$(VAR)` is
  Make). Mixing them is fine but easy to get wrong.
- Each recipe line runs in its own subshell unless you join with `\`. So
  `cd foo` then `make` on the next line returns to the original cwd.
- `make -n <target>` shows what *would* run without executing it —
  invaluable for sanity-checking a Makefile edit before running it.
- `make --warn-undefined-variables` exposes typos in `$(VAR_NAME)`
  references.
