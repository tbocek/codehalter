# Bash skill

## Script header
- Start every script with `#!/usr/bin/env bash` and `set -euo pipefail`.
- `IFS=$'\n\t'` if the script does any list iteration over `$@` or split data.

## Quoting
- Quote ALL variable expansions: `"$var"`, `"$@"`, `"${arr[@]}"`. Unquoted expansions split on whitespace and glob.
- Never write `$*` for argument forwarding — use `"$@"`.

## Tests and conditionals
- `[[ ... ]]` for tests, not `[ ... ]`. Pattern matching, regex, and safer parsing.
- Exit-status checks: `if cmd; then ...` not `if [ $? = 0 ]`.

## Substitution
- `$(cmd)`, never backticks.
- `${var:-default}` for fallback; `${var:?error}` for required.

## Idioms
- Prefer `mapfile`/`readarray` over while-read loops for collecting lines.
- Use `local` inside functions to avoid leaking variables.
- Use `trap` to clean up temp files / processes on EXIT.
- Long-running pipelines: check `${PIPESTATUS[@]}` if you need to know which stage failed.

## Style
- Run `shellcheck` mentally — silence its top warnings (SC2086 quoting, SC2046 word splitting).
- Avoid Bashisms (`[[`, arrays, `${var//pat/repl}`) in scripts that need to run under `/bin/sh`.

## Debugging "failed on line N"

When a script (or `just` recipe / `make` target) bails with `failed on line N with exit code 1`, that's the *wrapper's* error — the real cause is hidden upstream in the output. Don't stop at the wrapper message.

1. Open the script and look up line N. That is the failing command — but the actual diagnostic (e.g. `package site is not in std`) was printed by a sub-process a few lines earlier in the output.
2. Re-run just that line via `run_command` in isolation. Strip any surrounding cd/setup noise. This gets you a clean error without the wrapper's "exit 1" footer drowning it out.
3. If the failing line invokes another bash script (`bash foo.sh ARG`, `./foo.sh ARG`), re-run it with `bash -x foo.sh ARG`. The `-x` trace prints each command with a `+` prefix before executing — the last `+` line before the error tells you which inner line actually failed.
4. The fix targets the *inner* command's error, not the wrapper's line number. Reporting "verify failed on line 23" is not a diagnosis; running line 23 directly is.

Example: a task fails with `recipe <name> failed on line N with exit code 1`. Open the recipe, find line N — say it's `bash <subscript>.sh <args>`. Re-run that as `bash -x <subscript>.sh <args>` via `run_command` and the trace will show each command before it ran, surfacing the exact step inside `<subscript>.sh` that broke alongside its real error message.
