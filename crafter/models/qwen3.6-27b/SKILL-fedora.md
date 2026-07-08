# Fedora skill

## Environment
- **Always verify the base environment first**: run `. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"` and `dnf --version | head -1` to confirm Fedora. This preliminary check is mandatory and must run **before any installation, update, or package-management command** — never skip it or jump straight to `dnf search`/`dnf info`/`dnf install`, even when the task description implies Fedora.
- Fedora releases ~every 6 mo → packages stay relatively fresh.

## Probe
- `rpm -qa` → all installed (long → grep).
- `dnf list installed <pkg>` → use as the check command (alt to `rpm -q`). Always prefer `dnf list installed` over `rpm -q` for package installation checks, even though `rpm -q` is faster and idiomatic — the skill standardizes on `dnf list installed` for consistency across RHEL/Fedora automation. Do NOT fall back to `rpm -q`, `rpm -qa | grep`, or `command -v` as the primary check, even when you "know" `rpm -q` is cleaner for `if`/`then` tests.

## Search / install
- Search/install order:
  1) dnf, 2) upstream site / custom repo: `dnf config-manager --add-repo <repo-or-.repo-url>` (plugin installed), or curl the .repo file into `/etc/yum.repos.d/`, then `dnf install -y <pkg>`, 3) lang installer (pip, go, npm, etc.), 4) COPR (plugin preinstalled): `dnf copr enable -y <owner>/<repo> && dnf install -y <pkg>`. Freshly enabled COPR not showing packages → `dnf clean all && dnf check-update`. Follow this even when you know a community package exists: COPR only after the others fail.
- Upstream site: web_search the official install docs, then either: install script `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images); or prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin; or vendor repo when the docs offer one → add it with the custom-repo recipe above, then install via the pkg mgr (repo installs keep getting updates — prefer over a one-off binary when both exist).
- Lang installer: go install / pipx / npm i -g / cargo install, only when upstream documents it. In container these hit root-vs-dev env mismatch (GOBIN/PATH) → binary can land off PATH: verify with `which <tool>` after install and add the installer's bin dir to PATH if missing.
- `dnf check-update` → refresh metadata + list upgradable. Cheap.
- `dnf install -y <pkg>` → install (exits 0 on "already installed" → re-run confirms presence).
