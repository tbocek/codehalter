# Bash skill

## Script header
- `#!/usr/bin/env bash` + `set -euo pipefail`.
- `IFS=$'\n\t'` if the script iterates over `$@` or split data.

## Quoting
- Quote ALL variable expansions: `"$var"`, `"$@"`, `"${arr[@]}"`. Unquoted
  expansions split on whitespace and glob.
- Never `$*` for argument forwarding — use `"$@"`.

## Tests and conditionals
- `[[ ... ]]`, not `[ ... ]`. Pattern matching, regex, safer parsing.
- `if cmd; then ...` not `if [ $? = 0 ]`.

## Substitution
- `$(cmd)`, never backticks.
- `${var:-default}` for fallback; `${var:?error}` for required.

## Idioms
- `mapfile`/`readarray` over while-read loops for collecting lines.
- `local` inside functions to avoid leaking variables.
- `trap` for EXIT cleanup (temp files, processes).
- Check `${PIPESTATUS[@]}` for long pipelines if you need to know which
  stage failed.

## Style
- Run `shellcheck` mentally — silence SC2086 (quoting), SC2046 (word splitting).
- Avoid Bashisms (`[[`, arrays, `${var//pat/repl}`) in scripts that need
  to run under `/bin/sh`.

## Debugging "failed on line N"

When a script (or `just` recipe / `make` target) bails with `failed on
line N with exit code 1`, that's the *wrapper's* error — the real cause
is hidden upstream in the output. Don't stop at the wrapper message.

1. Open the script, look up line N — that's the failing command, but the
   diagnostic (e.g. `package site is not in std`) was printed by a
   sub-process a few lines earlier in the output.
2. Re-run just that line via `run_command` in isolation. Gets the clean
   error without the wrapper's "exit 1" footer drowning it.
3. If the failing line invokes another bash script (`bash foo.sh ARG`),
   re-run with `bash -x foo.sh ARG`. The `-x` trace prefixes each command
   with `+` — the last `+` line before the error tells you which inner
   line broke.
4. The fix targets the *inner* command's error, not the wrapper's line
   number. "Failed on line 23" is not a diagnosis; running line 23
   directly is.
