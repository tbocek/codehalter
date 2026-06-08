# Fedora skill

Base is Fedora ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}). Pkg manager
is `dnf` (`yum` alias still works). Fedora releases every ~6 months —
packages stay relatively fresh.

Run as non-root `dev` (sudo NOPASSWD). Prefix `dnf install`/`remove`/
`check-update`/`clean`/`copr enable` with `sudo`. Read-only probes
(`dnf info`/`search`/`provides`, `rpm -qa`/`-q`) don't need it.

## Probe (only if needed)

`/etc/os-release` is in the header — do NOT re-run. Session-invariant:

- `dnf --version`
- `rpm -qa` — every installed package (long; pipe to grep).
- `rpm -q <pkg>` — installed version, or "not installed".
- `rpm -qi <pkg>` — detailed info (version, release, size).
- `dnf list installed <pkg>` — alternative to `rpm -q`.

## Search and install

- `dnf check-update` — refresh metadata + list upgradable. Cheap.
- `dnf search <pkg>` — fuzzy search.
- `dnf info <pkg>` — version, repo, description, size.
- `dnf install -y <pkg>` — install.
- `dnf remove <pkg>` — uninstall.
- `dnf provides <path-or-cmd>` — find which package owns a file
  (e.g. `dnf provides */gopls`).

## Version staleness

If `dnf info` shows older than expected, check upstream with `web_search`.
Workarounds:

- **COPR**: Fedora's user repos. `dnf copr enable <owner>/<repo> &&
  dnf install -y <pkg>`. Many language toolchains have COPRs.
- **RPM Fusion** for media/non-free codecs.
- **Upstream dnf repo**: many projects publish a `.repo` file —
  `curl ... > /etc/yum.repos.d/<name>.repo && dnf install -y <pkg>`.

Don't claim "version X unavailable" without checking official repos + COPR.

## Common gotchas

- `dnf` caches metadata aggressively; if a freshly enabled COPR doesn't
  show packages, `dnf clean all && dnf check-update`.
- Split packages: `<pkg>` (runtime) vs `<pkg>-devel` (headers). "Missing
  header X" usually means you need `<pkg>-devel`.
- `dnf install` exits 0 on "already installed" — re-running confirms presence.
