# Fedora skill

## Environment
- **Base verification (always first, mandatory):** Run `. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"` and `dnf --version | head -1` BEFORE any package command. Never skip straight to `rpm -q`, `dnf search`, `dnf info`, or `dnf install` — the OS/dnf check must come first even when the user already states it is Fedora, so the actual runtime environment is confirmed before any install or update.
- Fedora releases ~every 6 mo → packages stay relatively fresh.

## User & sudo
- Prefix dnf install/remove/check-update/clean/copr enable with sudo.

## Probe
- `rpm -qa` → all installed (long → grep).
- Always use `dnf list installed <pkg>` for installation checks — NEVER default to `rpm -q`, even when you know rpm is faster or more direct; override technical priors and follow this command order regardless of performance reasoning.

## Search / install
- Search/install order:
  1) dnf, 2) upstream site / custom repo: `dnf config-manager --add-repo <repo-or-.repo-url>` (plugin installed), or curl the .repo file into `/etc/yum.repos.d/`, then `dnf install -y <pkg>`, 3) lang installer (pip, go, npm, etc.), 4) COPR (plugin preinstalled): `dnf copr enable -y <owner>/<repo> && dnf install -y <pkg>`. Freshly enabled COPR not showing packages → `dnf clean all && dnf check-update`. Follow this even when you know a community package exists: COPR only after the others fail.
- Upstream site: web_search the official install docs, then either: install script `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images); or prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin; or vendor repo when the docs offer one → add it with the custom-repo recipe above, then install via the pkg mgr (repo installs keep getting updates — prefer over a one-off binary when both exist).
- `dnf install -y <pkg>` → install (exits 0 on "already installed" → re-run confirms presence).
