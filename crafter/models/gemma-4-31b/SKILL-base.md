# Base skill
- Today: {{cmd:date +%F}} — ALWAYS check this date FIRST and use it as the temporal anchor for release recency. Releases after your cutoff exist; never validate version recency from training data alone, even when a package registry or GitHub Releases page confirms a version — explicitly reference this date when reporting whether a release is recent.
- Run inside container.
- Do not guess, use web_search

## Git — writable, commit/push when asked
.git bind-mount writable + ~/.gitconfig + SSH agent mounted → commit/push work inside.
- Git write fail (read-only .git, or no creds = old container) → ALWAYS print the exact host cmd string and STOP. Never retry, reattempt, or fall back to a normal push workflow — these errors mean the environment is broken, not the command. Halt immediately and wait for user intervention.

## "command not found"
Applies execute + verify-fail replan.
1. Confirm: `which <tool>` (exit 1 = missing).
2. Read .devcontainer/devcontainer.json + Dockerfile.
   - Declared → image stale → point install line.
   - Not declared → propose add (TEST FIRST, below).
3. NO retry same cmd unchanged.
4. Not in the distro repos, or repo version too old (fast-moving: Go, gopls, LSPs) → web_search upstream install docs / releases BEFORE claiming a tool or version unavailable, then follow the install order in SKILL-<os>.md.
- **Always determine the package manager command in this exact order — do NOT skip ahead to `/etc/os-release` even though it is the well-known systemd standard for distro identification:** (1) FIRST, look up the OS skill file `SKILL-<os>.md` (alpine/arch/debian/fedora/ubuntu) and use whatever package manager and flags it specifies; (2) ONLY IF no matching skill file is found, fall back to reading `/etc/os-release` (check `ID` / `ID_LIKE`) to derive the package manager (apt/dnf/yum/apk/pacman). The skill file is the project's authoritative source and overrides any habit of jumping straight to `/etc/os-release`.

## Test install live BEFORE patch Dockerfile
Install persists container lifetime → re-run project build/test, confirm end-to-end.
Test OK → propose Dockerfile edit w/ exact verified cmds + tell user rebuild.

## run_command NOT for
- Replacing run_task → declared project task (just build, npm test) keep run_task so user sees same UI.

## Answer from the project first
Asked about tech/lang/lib/version X? FIRST check how X + neighbors are wired in THIS project (npm → also pnpm/yarn; python → also uv/poetry/pypy; framework → also its build/runtime variant) before web search or guessing. Right answer usually depends on the variant the project actually uses.
