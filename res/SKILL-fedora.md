# Fedora skill
Base: Fedora ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}). Pkg mgr dnf (yum alias still works). Fedora releases ~every 6 mo → packages stay relatively fresh.
User=non-root `dev`, sudo NOPASSWD. Prefix dnf install/remove/check-update/clean/copr enable with sudo. Read probes (dnf info/search/provides, rpm -qa/-q) no sudo.

## Probe (only if needed; session-invariant)
- /etc/os-release in header → no re-run.
- dnf --version
- rpm -qa → all installed (long → grep).
- rpm -q <pkg> → installed version, or "not installed".
- rpm -qi <pkg> → detailed info (version, release, size).
- dnf list installed <pkg> → alt to rpm -q.

## Search / install
- dnf check-update → refresh metadata + list upgradable. Cheap.
- dnf search <pkg> → fuzzy.
- dnf info <pkg> → version, repo, desc, size.
- dnf install -y <pkg> → install.
- dnf remove <pkg> → uninstall.
- dnf provides <path-or-cmd> → which package owns a file (e.g. `dnf provides */gopls`).

## Version staleness
dnf info shows older than expected → web_search upstream. Workarounds:
- **COPR**: Fedora user repos. `dnf copr enable <owner>/<repo> && dnf install -y <pkg>`. Many language toolchains have COPRs.
- **RPM Fusion** for media/non-free codecs.
- **Upstream dnf repo**: many projects publish a .repo file → `curl ... > /etc/yum.repos.d/<name>.repo && dnf install -y <pkg>`.
Don't claim "version X unavailable" without checking official repos + COPR.

## Gotchas
- dnf caches metadata aggressively; freshly enabled COPR doesn't show packages → `dnf clean all && dnf check-update`.
- Split pkgs: <pkg> (runtime) vs <pkg>-devel (headers). "Missing header X" usually → need <pkg>-devel.
- dnf install exits 0 on "already installed" → re-run confirms presence.