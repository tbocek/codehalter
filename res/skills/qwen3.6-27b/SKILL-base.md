# Base skill
- Today: {{cmd:date +%F}} — trust this over training-data recency; releases after your cutoff exist.
- Run inside container. Workspace bind-mount from host; container=sandbox → pkg-mgr/pip/npm writes persist container lifetime (wiped on rebuild) → test install cheap + reversible.

## Git — writable, commit/push when asked
.git bind-mount writable + ~/.gitconfig + SSH agent mounted → commit/push work inside.
- Git write fail (read-only .git, or no creds = old container) → surface exact host cmd + stop, no fight.

## "command not found"
Applies execute + verify-fail replan.
1. Confirm: `which <tool>` (exit 1 = missing).
2. Read .devcontainer/devcontainer.json + Dockerfile.
   - Declared → image stale → point install line.
   - Not declared → propose add (TEST FIRST, below).
3. NO retry same cmd unchanged.
4. Not in the distro repos, or repo version too old (fast-moving: Go, gopls, LSPs) → web_search upstream install docs / releases BEFORE claiming a tool or version unavailable, then follow the install order in SKILL-<os>.md.
ALWAYS determine the package manager command in this exact order: (1) FIRST check for a SKILL-<os>.md file (alpine/arch/debian/fedora/ubuntu) and use whatever package manager it specifies if present; (2) ONLY IF no skill file exists, fall back to reading /etc/os-release to derive the manager. Never skip the skill-file lookup or jump straight to /etc/os-release detection, binary probing, or distro guessing — the SKILL file is project-authoritative and may pin a manager that differs from the distro default.

## Test install live BEFORE patch Dockerfile
Before write `RUN <pkg-mgr> install <pkg>` (or pip, npm i -g) into Dockerfile, first run same install via run_command: run_command: <install-cmd> && <tool> --version
Install persists container lifetime → re-run project build/test, confirm end-to-end.
Install fail (wrong name, repo missing, version mismatch) → debug HERE (alt source, search release page) before commit Dockerfile change. Untested patch = guess.
Test OK → propose Dockerfile edit w/ exact verified cmds + tell user rebuild.

## run_command NOT for
- Long-running service → die w/ codehalter. No daemon. Use run_background
- Replacing run_task → declared project task (just build, npm test) keep run_task so user sees same UI.
Exit code in output + tool-card title.

## Answer from the project first
Asked about tech/lang/lib/version X? FIRST check how X + neighbors are wired in THIS project (npm → also pnpm/yarn; python → also uv/poetry/pypy; framework → also its build/runtime variant) before web search or guessing. Read manifest/build files (go.mod, package.json, Cargo.toml, Makefile, justfile, Dockerfile, README) + grep related terms. Right answer usually depends on the variant the project actually uses.
