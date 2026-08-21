# Bash skill
## `set -e` won't fire here — know before trusting it
- Command substitution: `x=$(false; echo hi)` → no exit. `shopt -s inherit_errexit` (bash 4.4+) makes it respect `set -e`.
- `local x=$(cmd)` → `local` masks the failure. Split it: `local x; x=$(cmd)`.
- A function used as a conditional (`if f`, `f &&`, `f ||`, `while f`) → `set -e` is disabled *inside* f.
- Append `|| true` to EACH command allowed to fail (docker stop, rm -rf, kill/fuser/lsof). Never substitute `if/then` blocks or `set +e` for it.

## Style
- Run real `shellcheck` (not "mentally") + `bash -n` for syntax.
- No bashisms (`[[`, arrays, `${var//pat/repl}`) in scripts that run under /bin/sh.

## Debug "failed on line N"
A script (or just recipe / make target) bailing with `failed on line N exit code 1` = the *wrapper's* error; the real cause is upstream in the output. Don't stop at the wrapper msg.
1. Open the script, find line N → that's the failing cmd, BUT the diagnostic (e.g. `package site is not in std`) was printed by a sub-process a few lines earlier in the output.
2. Re-run just that line via run_command, isolated → clean error without the wrapper's "exit 1" footer drowning it.
3. The failing line invokes another bash script (`bash foo.sh ARG`) → re-run `bash -x foo.sh ARG`. The `-x` trace prefixes each cmd with `+` → last `+` line before the error = which inner line broke.
4. Fix the *inner* cmd's error, NOT the wrapper's line number. "Failed on line 23" = not a diagnosis; running line 23 direct = is.
