# Container skill
Today: {{cmd:date +%F}} — trust this over training-data recency; releases after your cutoff exist.
Run inside container. Workspace bind-mount from host; container=sandbox → pkg-mgr/pip/npm writes persist container lifetime (wiped on rebuild) → test install cheap + reversible.

reuse from history

## Git — writable, commit/push when asked
.git bind-mount writable + ~/.gitconfig + SSH agent mounted → commit/push work inside. BUT git action ONLY when user explicit ask (see EXECUTE.md) — never commit/push self.
- **OK on request**: commit, push, + safe reads status/log/diff/show/blame/fetch.
- **Avoid unless explicit**: history rewrite shared branch — reset --hard, push --force, filter-branch, gc --prune.
- Git write fail (read-only .git, or no creds = old container) → surface exact host cmd + stop, no fight.

## "command not found"
Applies execute + verify-fail replan. In plan: emit install + Dockerfile-edit steps, let execute run.
1. Confirm: `which <tool>` (exit 1 = missing).
2. Read .devcontainer/devcontainer.json + Dockerfile.
   - Declared → image stale → point install line.
   - Not declared → propose add (TEST FIRST, below).
3. NO retry same cmd unchanged.
4. Not in the distro repos, or repo version too old (fast-moving: Go, gopls, LSPs) → web_search upstream install docs / releases BEFORE claiming a tool or version unavailable, then follow the install order below.
Pkg-mgr cmd depends base image → check SKILL-<os>.md (alpine/arch/debian/fedora/ubuntu) or fall back /etc/os-release.

## Install order (any missing tool, incl. lang-ecosystem: gopls, ruff, prettier…)
1. Distro pkg mgr.
2. Upstream website — web_search the official install docs, then either:
   - install script: `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images);
   - prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin;
   - vendor repo, when the docs offer one → add it with the custom-repo recipe in SKILL-<os>.md, then install via the pkg mgr (repo installs keep getting updates — prefer over a one-off binary when both exist).
3. Language installer — go install / pipx / npm i -g / cargo install, when upstream documents it. In container these hit root-vs-dev env mismatch (GOBIN/PATH) → binary can land off PATH: verify with `which <tool>` after install and add the installer's bin dir to PATH if missing.
4. Community repos: AUR, COPR, PPA, backports.

## Test install live BEFORE patch Dockerfile
**Execute phase only. Plan stays read-only.**
Before write `RUN <pkg-mgr> install <pkg>` (or pip, npm i -g) into Dockerfile, first run same install via run_command: run_command: <install-cmd> && <tool> --version
Install persists container lifetime → re-run project build/test, confirm end-to-end.
Install fail (wrong name, repo missing, version mismatch) → debug HERE (alt source, search release page) before commit Dockerfile change. Untested patch = guess.
Test OK → propose Dockerfile edit w/ exact verified cmds + tell user rebuild.

## run_command for probes too
- which <tool>, <tool> --version → exists?
- Type-check/lint (cargo check, tsc --noEmit, go vet ./...).
- ls -la <path>, cat <config> → inspect state.
Probe workspace writes (e.g. cargo check fills target/) real. OK for build artifacts; no destructive shell cmd vs source → use edit_file/write_file.

## run_command NOT for
- Long-running service → die w/ codehalter. No daemon.
- Replacing run_task → declared project task (just build, npm test) keep run_task so user sees same UI.
- Editing project file → use edit_file/write_file so change hits diff/approval UI. Raw `sed -i` or `>` skip that.
Exit code in output + tool-card title. which <tool> exit 1 = binary missing, not tool error.

## Answer from the project first
Asked about tech/lang/lib/version X? FIRST check how X + neighbors are wired in THIS project (npm → also pnpm/yarn; python → also uv/poetry/pypy; framework → also its build/runtime variant) before web search or guessing. Read manifest/build files (go.mod, package.json, Cargo.toml, Makefile, justfile, Dockerfile, README) + grep related terms. Right answer usually depends on the variant the project actually uses.
