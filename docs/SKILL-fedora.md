# Fedora skill

Container base is Fedora (`/etc/os-release` ID=fedora). Package manager
is `dnf` (the `yum` alias still works for legacy commands). Fedora
releases every ~6 months, so packages stay relatively fresh.

## Probe first

- `cat /etc/os-release` — version (VERSION_ID like "40", "41").
- `dnf --version` — dnf + python-dnf versions.
- `rpm -qa` — every installed package (long; pipe to grep).
- `rpm -q <pkg>` — installed version, or "package not installed".
- `rpm -qi <pkg>` — detailed info (version, release, install date, size).
- `dnf list installed <pkg>` — alternative to `rpm -q`.

Session-invariant — reuse from this turn's history rather than re-running.

## Search and install

- `dnf check-update` — refresh metadata + list upgradable. Cheap.
- `dnf search <pkg>` — fuzzy search across names and summaries.
- `dnf info <pkg>` — version, repo source, description, size.
- `dnf install -y <pkg>` — install (the `-y` skips the y/n prompt).
- `dnf remove <pkg>` — uninstall.
- `dnf provides <path-or-cmd>` — find which package owns a file (e.g.
  `dnf provides */gopls`).

## Version staleness

Fedora is reasonably current but not rolling. If `dnf info` shows an
older version than expected, check the upstream release page with
`web_search` first. Workarounds:

- **COPR**: Fedora's user repo system. `dnf copr enable <owner>/<repo> &&
  dnf install -y <pkg>`. Many language toolchains have COPRs.
- **RPM Fusion** for media / non-free codecs (less relevant for dev work
  but worth knowing): `dnf install -y https://mirrors.rpmfusion.org/...`.
- **Upstream dnf repo**: many projects publish a `.repo` file —
  `curl ... > /etc/yum.repos.d/<name>.repo && dnf install -y <pkg>`.

Don't claim "version X is not available" without checking the official
repos and COPR.

## Common gotchas

- `dnf` uses metadata caching aggressively; if a freshly enabled COPR
  doesn't show packages, `dnf clean all && dnf check-update`.
- Some packages split between `<pkg>` (runtime) and `<pkg>-devel` (headers).
  Compile failures with "missing header X" usually mean you need
  `<pkg>-devel`.
- `dnf install` exits 0 on "already installed" — re-running is a cheap
  idempotent way to confirm presence.
