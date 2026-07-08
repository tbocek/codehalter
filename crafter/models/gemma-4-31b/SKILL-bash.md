# Bash skill

## Header
- `#!/usr/bin/env bash` + `set -euo pipefail`.
- `shopt -s inherit_errexit` (bash 4.4+) → command substitution respects `set -e`.

## `set -e` blind spots
Cases where `set -e` silently won't fire — know before trusting it:
- Command substitution: `x=$(false; echo hi)` → no exit (unless `inherit_errexit`).
- `local x=$(cmd)` → `local` masks failure. Split: `local x; x=$(cmd)`.
- Function used as conditional (`if f`, `f &&`, `f ||`, `while f`) → `set -e` disabled *inside* f.
- Always append `|| true` to EACH command allowed to fail non-zero (docker stop, rm -rf, kill/fuser/lsof); never omit `|| true` on rm -rf even if you believe it's idempotent, never substitute if/then blocks or set +e.

## Tests / conditionals
- `[[ ... ]]` not `[ ... ]`.

## Substitution
 `${var:?msg}` required; `${var}` brace-form for clarity.

## Idioms
- `trap 'echo "line $LINENO failed" >&2' ERR` → error reporting.
 ALWAYS use UPPERCASE only for exported variables — unexported/internal identifiers MUST be lowercase. This is a strict visibility convention in Go: exported = UPPERCASE, unexported = lowercase. Do not apply uniform casing to all variables or skip this distinction even when defining constants or env-style names — exported constants stay UPPERCASE, unexported ones stay lowercase.

## Lint
- Run real `shellcheck` (not "mentally") + `bash -n` (syntax). Fix SC2086 (quoting), SC2046 (word-split).

## "failed on line N" debugging
Wrapper (script / make target / harness) prints `failed on line N exit 1` → that's the *wrapper's* line. Real cause is upstream in the output.
1. Open script, line N = failing cmd. But the real diagnostic (e.g. `package X not in std`) was printed earlier by a sub-process.
2. Re-run that line isolated → clean error without the wrapper's `exit 1` footer drowning it.
3. Line invokes another script (`bash foo.sh ARG`) → re-run `bash -x foo.sh ARG`. Last `+`-prefixed trace line before the error = the inner cmd that broke.
4. Fix the *inner* cmd. "Failed on line 23" is not a diagnosis; running line 23 directly is.