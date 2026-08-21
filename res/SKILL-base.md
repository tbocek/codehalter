# Container skill
Today: {{cmd:date +%F}} — trust over training recency; releases after cutoff exist.
Run inside container. Workspace bind-mounted from host. Container=sandbox → pkg-mgr/pip/npm writes persist container lifetime, wiped on rebuild → test install cheap + reversible.

reuse from history

## Git — writable, commit/push on ask
.git writable + ~/.gitconfig + SSH agent mounted → commit/push work inside. Git action ONLY on explicit user ask (EXECUTE.md). Never self-commit/push.
- **OK on request**: commit, push, reads status/log/diff/show/blame/fetch.
- **Avoid unless explicit**: shared-branch history rewrite — reset --hard, push --force, filter-branch, gc --prune.
- Write fails (read-only .git, no creds = old container) → print exact host cmd, stop. No fight.

## "command not found"
Execute + verify-fail replan. In plan: emit install + Dockerfile-edit steps, let execute run.
1. `which <tool>`, exit 1 = missing.
2. Read .devcontainer/devcontainer.json + Dockerfile. Declared → image stale, point at install line. Not declared → propose add (TEST FIRST, below).
3. NO retry of same cmd unchanged.
4. Absent from distro repos, or repo version too old (fast movers: Go, gopls, LSPs) → web_search upstream docs/releases BEFORE claiming unavailable, then install order below.
Pkg-mgr cmds depend on base image → SKILL-<os>.md (alpine/arch/debian/fedora/ubuntu), else /etc/os-release.

## Install order (any missing tool, incl. gopls, ruff, prettier…)
1) distro pkg mgr 2) upstream site 3) language installer 4) community repos (AUR, COPR, PPA, backports). Next option only when previous has no package, or too old.
- "Not in repos" = claim, not fact. Prove with pkg-mgr exact-name query (SKILL-<os>.md). Never grep search output: listings print `name-version` → `grep "^name "` matches 0 lines = false absence. "Too old" → quote the version the repo offers.
- Upstream site, one of:
  - install script `curl -fsSL https://…/install.sh | sh` — use the shell the docs name, sh vs bash matters on minimal images;
  - prebuilt binary/tarball → ~/.local/bin (on PATH) or /usr/local/bin;
  - vendor repo when docs offer one → add via custom-repo recipe in SKILL-<os>.md, install via pkg mgr. Repo installs keep updating → prefer over one-off binary.
- Language installer (go install / pipx / npm i -g / cargo install) only when upstream documents it. Root-vs-dev env mismatch (GOBIN/PATH) lands binary off PATH → `which <tool>` after install, add installer bin dir to PATH if missing.

## Test install live BEFORE patching Dockerfile
**Execute only. Plan stays read-only.**
Before writing `RUN <pkg-mgr> install <pkg>` (or pip, npm i -g) into Dockerfile → run_command `<install-cmd> && <tool> --version` first.
Install persists container lifetime → re-run project build/test, confirm end-to-end.
Install fails (wrong name, missing repo, version mismatch) → debug HERE (alt source, release page) before touching Dockerfile. Untested patch = guess.
Test OK → propose Dockerfile edit with exact verified cmds + tell user to rebuild.
`FROM` tag freezes the whole package set → stale tag = why the packaged toolchain is behind (alpine:3.23 ships older Go than alpine:3.24). Touching a Dockerfile for ANY reason → first check the tag is still current stable (web_search distro releases, or Docker Hub `latest`). Moved → propose bump in same edit, name what it refreshes. Stable tags ONLY, never edge/rawhide/sid/devel/testing.

## run_command for probes too
- `which <tool>`, `<tool> --version` → exists?
- Type-check/lint: cargo check, tsc --noEmit, go vet ./...
- `ls -la <path>`, `cat <config>` → inspect state.
Probe writes into workspace (cargo check fills target/) are real. OK for build artifacts. Never destructive shell cmds against source → edit_file/write_file.

## run_command NOT for
- Long-running service → dies with codehalter. No daemon.
- Replacing run_task → declared project task (just build, npm test) stays run_task, same UI for user.
- Editing project file → edit_file/write_file so change hits diff/approval UI. Raw `sed -i` or `>` skip it.
Exit code in output + tool-card title. `which <tool>` exit 1 = binary missing, not tool error.

## Answer from the project first
Asked about tech/lang/lib/version X? FIRST check how X + neighbors are wired in THIS project (npm → also pnpm/yarn; python → also uv/poetry/pypy; framework → also its build/runtime variant) before web search or guessing. Read manifests (go.mod, package.json, Cargo.toml, Makefile, justfile, Dockerfile, README) + grep related terms. Right answer usually depends on the variant this project uses.
