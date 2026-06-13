# Bash skill
## Script header
- `#!/usr/bin/env bash` + `set -euo pipefail`.
- `IFS=$'\n\t'` if script iterates `$@` or split data.

## Quoting
- Quote ALL var expansions: `"$var"`, `"$@"`, `"${arr[@]}"`. Unquoted → splits on whitespace + globs.
- Never `$*` for arg forwarding → use `"$@"`.

## Tests / conditionals
- `[[ ... ]]` not `[ ... ]` → pattern match, regex, safer parse.
- `if cmd; then ...` not `if [ $? = 0 ]`.

## Substitution
- `$(cmd)` never backticks.
- `${var:-default}` = fallback; `${var:?error}` = required.

## Idioms
- mapfile/readarray over while-read loops for collecting lines.
- `local` in functions → no var leak.
- `trap` for EXIT cleanup (temp files, processes).
- Check `${PIPESTATUS[@]}` on long pipelines if need which stage failed.

## Style
- Mental shellcheck → silence SC2086 (quoting), SC2046 (word split).
- No Bashisms (`[[`, arrays, `${var//pat/repl}`) in scripts that run under /bin/sh.

## Debug "failed on line N"
Script (or just recipe / make target) bails `failed on line N exit code 1` → that = *wrapper's* error, real cause hidden upstream in output. Don't stop at wrapper msg.
1. Open script, find line N → that's failing cmd, BUT diagnostic (e.g. `package site is not in std`) was printed by sub-process a few lines earlier in output.
2. Re-run just that line via run_command isolated → clean error without wrapper's "exit 1" footer drowning it.
3. Failing line invokes another bash script (`bash foo.sh ARG`) → re-run `bash -x foo.sh ARG`. `-x` trace prefixes each cmd with `+` → last `+` line before error = which inner line broke.
4. Fix targets *inner* cmd's error, NOT wrapper's line number. "Failed on line 23" = not a diagnosis; running line 23 direct = is.