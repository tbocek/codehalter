# Fedora skill
Base: Fedora ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"}}), dnf {{cmd:dnf --version | head -1}}. Pkg mgr dnf (yum alias still works). Fedora releases ~every 6 mo → packages stay relatively fresh.
User=non-root `dev`, sudo NOPASSWD. Prefix dnf install/remove/check-update/clean/copr enable with sudo. Read probes (dnf info/search/provides, rpm -qa/-q) no sudo.

## Probe (only if needed; session-invariant)
- rpm -qa → all installed (long → grep).
- rpm -q <pkg> → installed version, or "not installed".
- rpm -qi <pkg> → detailed info (version, release, size).
- dnf list installed <pkg> → alt to rpm -q.

## Search / install
Order: 1) dnf 2) upstream site / custom repo (below) 3) lang installer 4) COPR — details in container skill.
- dnf check-update → refresh metadata + list upgradable. Cheap.
- dnf search <pkg> → fuzzy.
- dnf info <pkg> → version, repo, desc, size.
- dnf install -y <pkg> → install (exits 0 on "already installed" → re-run confirms presence).
- dnf remove <pkg> → uninstall.
- dnf provides <path-or-cmd> → which package owns a file (e.g. `dnf provides */gopls`).
- Split pkgs: <pkg> (runtime) vs <pkg>-devel (headers). "Missing header X" usually → need <pkg>-devel.
- Custom repo: dnf config-manager --add-repo <repo-or-.repo-url> (plugin installed), or curl the .repo file into /etc/yum.repos.d/, then dnf install -y <pkg>.
- 4th choice COPR (plugin preinstalled): dnf copr enable -y <owner>/<repo> && dnf install -y <pkg>. Freshly enabled COPR not showing packages → dnf clean all && dnf check-update.
