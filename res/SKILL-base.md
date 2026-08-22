# Container skill
Today: {{cmd:date +%F}} — trust over training recency; releases after cutoff exist.
Run inside container. Workspace bind-mounted from host. Container=sandbox → pkg-mgr/pip/npm writes persist container lifetime, wiped on rebuild → test install cheap + reversible.

## Unfamiliar project
Repo you don't already know (first turn here, or the request names files/commands you can't place) → orient BEFORE acting: `list_files` at root, and if the listing shows a README (`README.md`/`.rst`/`.txt`/`README`) `read_file` it. Layout, build/test commands and local conventions come from there, not from the ecosystem's defaults.
- One pass, not a survey: root listing + README. Deeper dirs only where the task points.
- Already listed root this turn → reuse that listing, don't re-list.

## Tool choice
- Declared project task (just build, npm test, make) → `run_task`, NOT `run_command`: same UI for the user.
- Editing a project file → `edit_file`/`write_file` so the change hits the diff/approval UI. Raw `sed -i` or `>` skip it.
- No long-running service: it dies with codehalter. No daemon.
- Probe writes into the workspace (cargo check fills target/) are fine — build artifacts, not source.

## "command not found"
Pkg-mgr commands depend on the base image → SKILL-<os>.md (alpine/arch/debian/fedora/ubuntu), else /etc/os-release.
- NO retry of the same cmd unchanged.
- Absent from distro repos, or repo version too old (fast movers: Go, gopls, LSPs) → web_search upstream docs/releases BEFORE claiming unavailable.

## Install order (any missing tool, incl. gopls, ruff, prettier…)
1) distro pkg mgr 2) upstream site 3) language installer 4) community repos (AUR, COPR, PPA, backports). Next option only when the previous has no package, or too old.
- "Not in repos" = claim, not fact. Prove with the pkg-mgr exact-name query (SKILL-<os>.md). Never grep search output: listings print `name-version`, so `grep "^name "` matches 0 lines = false absence. "Too old" → quote the version the repo offers.
- Vendor repo when upstream docs offer one → custom-repo recipe in SKILL-<os>.md. Repo installs keep updating → prefer over a one-off binary.
- Language installer (go install / pipx / npm i -g / cargo install) lands the binary off PATH when root and dev env differ → `which <tool>` after install.

## `FROM` tag freezes the whole package set
Stale tag = why the packaged toolchain is behind (alpine:3.23 ships Go 1.25.10, alpine:3.24 ships 1.26.3). So a too-old packaged tool is FIRST a tag question, not a reason to hand-install a tarball.
Touching a Dockerfile for ANY reason → check the tag is still current stable (web_search distro releases, or Docker Hub `latest`). Moved → propose the bump in the same edit, name what it refreshes. Stable tags ONLY, never edge/rawhide/sid/devel/testing.
Never write a `RUN <pkg-mgr> install <pkg>` line you have not run live in the container first (`<install-cmd> && <tool> --version`) — an untested patch is a guess.

## Persist only what the plan keeps
Installed to answer one question (a linter run once, a CLI to inspect a file, a candidate lib you then rejected) → leave it in the container and let the rebuild wipe it. NO `.devcontainer/Dockerfile` line, no devcontainer.json feature: a throwaway probe in the image makes every later rebuild slower and lies about what the project needs.
Persist ONLY when the tool stays part of the project: build/test/lint chain, runtime dep, something the next turn or the next person needs. Then it IS a Dockerfile edit, tested live first (above).
Borderline → not a keeper unless dropping it breaks a documented command. Say what you installed and that you left it unpersisted.
