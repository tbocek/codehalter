# Bash skill

## Header
- `#!/usr/bin/env bash` + `set -euo pipefail`.
- Bash-only: `pipefail`, `[[`, arrays break under `sh`/dash → don't use `#!/bin/sh` with them.
- `shopt -s inherit_errexit` (bash 4.4+) → command substitution respects `set -e`.
- `IFS=$'\n\t'` only if iterating split data. Opinionated (changes word-splitting globally); not a blind default.

## `set -e` blind spots
Cases where `set -e` silently won't fire — know before trusting it:
- Command substitution: `x=$(false; echo hi)` → no exit (unless `inherit_errexit`).
- `local x=$(cmd)` → `local` masks failure. Split: `local x; x=$(cmd)`.
- Function used as conditional (`if f`, `f &&`, `f ||`, `while f`) → `set -e` disabled *inside* f.
- Arithmetic: `((count--))` returns 1 when result is 0 → exits. Use `((count--)) || true`.
- Escape hatch: `|| true` on commands allowed to fail non-zero.

## Quoting
- Quote ALL expansions: `"$var"`, `"$@"`, `"${arr[@]}"`. Unquoted → word-split + glob.
- Arg forwarding: `"$@"` never `$*`.
- `--` before user/glob input → stops a leading-`-` filename being reparsed as a flag.

## Tests / conditionals
- `[[ ... ]]` not `[ ... ]`.
- `if cmd; then` not `if [ $? = 0 ]`.

## Substitution
- `$(cmd)` never backticks.
- `${var:-default}` fallback; `${var:?msg}` required; `${var}` brace-form for clarity.

## Idioms
- `local` in functions → no leak.
- `trap ... EXIT` → cleanup (temp files, procs).
- `trap 'echo "line $LINENO failed" >&2' ERR` → error reporting.
- `mapfile`/`readarray` over `while read` for collecting lines.
- `${PIPESTATUS[@]}` → which pipe stage failed.
- Lowercase var names; UPPERCASE only for exported.

## Lint
- Run real `shellcheck` (not "mentally") + `bash -n` (syntax). Fix SC2086 (quoting), SC2046 (word-split).
- No bashisms in `/bin/sh` scripts.

## "failed on line N" debugging
Wrapper (script / make target / harness) prints `failed on line N exit 1` → that's the *wrapper's* line. Real cause is upstream in the output.
1. Open script, line N = failing cmd. But the real diagnostic (e.g. `package X not in std`) was printed earlier by a sub-process.
2. Re-run that line isolated → clean error without the wrapper's `exit 1` footer drowning it.
3. Line invokes another script (`bash foo.sh ARG`) → re-run `bash -x foo.sh ARG`. Last `+`-prefixed trace line before the error = the inner cmd that broke.
4. Fix the *inner* cmd. "Failed on line 23" is not a diagnosis; running line 23 directly is.