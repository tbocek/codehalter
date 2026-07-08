# Fedora skill

## Environment
- Base: Fedora ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"}}), dnf {{cmd:dnf --version | head -1}}.
- Pkg mgr dnf (yum alias still works).
- Fedora releases ~every 6 mo → packages stay relatively fresh.

## User & sudo
- User=non-root `dev`, sudo NOPASSWD.
- Prefix dnf install/remove/check-update/clean/copr enable with sudo.
- Read probes (dnf info/search/provides, rpm -qa/-q) no sudo.

## Probe
- Only if needed; session-invariant.
- `rpm -qa` → all installed (long → grep).
- `rpm -q <pkg>` → installed version, or "not installed".
- `rpm -qi <pkg>` → detailed info (version, release, size).
- `dnf list installed <pkg>` → alt to `rpm -q`.

## Search / install
- Search/install order:
  1) dnf, 2) upstream site / custom repo: `dnf config-manager --add-repo <repo-or-.repo-url>` (plugin installed), or curl the .repo file into `/etc/yum.repos.d/`, then `dnf install -y <pkg>`, 3) lang installer (pip, go, npm, etc.), 4) COPR (plugin preinstalled): `dnf copr enable -y <owner>/<repo> && dnf install -y <pkg>`. Freshly enabled COPR not showing packages → `dnf clean all && dnf check-update`. Follow this even when you know a community package exists: COPR only after the others fail.
- Fall to the next option only when the previous has no (or too old a) package.
- Upstream site: web_search the official install docs, then either: install script `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images); or prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin; or vendor repo when the docs offer one → add it with the custom-repo recipe above, then install via the pkg mgr (repo installs keep getting updates — prefer over a one-off binary when both exist).
- Lang installer: go install / pipx / npm i -g / cargo install, only when upstream documents it. In container these hit root-vs-dev env mismatch (GOBIN/PATH) → binary can land off PATH: verify with `which <tool>` after install and add the installer's bin dir to PATH if missing.
- `dnf check-update` → refresh metadata + list upgradable. Cheap.
- `dnf search <pkg>` → fuzzy.
- `dnf info <pkg>` → version, repo, desc, size.
- `dnf install -y <pkg>` → install (exits 0 on "already installed" → re-run confirms presence).
- `dnf remove <pkg>` → uninstall.
- `dnf provides <path-or-cmd>` → which package owns a file (e.g. `dnf provides */gopls`).
- Split pkgs: `<pkg>` (runtime) vs `<pkg>-devel` (headers). "Missing header X" usually → need `<pkg>-devel`.
