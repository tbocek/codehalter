# Ubuntu skill

Container base is Ubuntu ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}},
codename={{UBUNTU_CODENAME}}). Tooling is Debian-derived: `apt` /
`apt-get` / `dpkg` work identically. The main differences from Debian are
the suite codenames (jammy, noble, oracular, …) and PPAs as an additional
source of fresher packages.

You are running as the non-root user `dev` (sudo is configured NOPASSWD).
Prefix all package-management commands (`apt update`, `apt install`,
`apt remove`, `apt-get ...`, `add-apt-repository`, `dpkg -i`) and any
other root-only operation with `sudo`. Read-only probes (`apt-cache
search`, `apt-cache policy`, `dpkg -l`, `dpkg -s`) do NOT need sudo.

## Probe (only if needed)

`/etc/os-release` is already reflected in the header above — do NOT re-run
`cat /etc/os-release`. The remaining probes ARE worth running on demand:

- `apt --version`.
- `dpkg -l` — every installed package (long; pipe to grep).
- `dpkg -l <pkg>` — installed version, or "no packages found".
- `dpkg -s <pkg>` — detailed status.

Session-invariant — reuse from this turn's history rather than re-running.

## Search and install

- `apt update` — refresh the index. Required after `add-apt-repository` or
  edits to `/etc/apt/sources.list*`.
- `apt-cache search <pkg>` — fuzzy search.
- `apt-cache policy <pkg>` — installed + candidate version + source.
  Always check this before claiming a version isn't available.
- `apt install -y <pkg>` — install.
- `apt remove <pkg>` / `apt purge <pkg>` — uninstall.

## Version staleness — PPAs are the Ubuntu shortcut

Ubuntu LTS releases freeze versions for years. Three escalating workarounds:

- **PPA**: `add-apt-repository -y ppa:<owner>/<name> && apt update && apt
  install -y <pkg>`. Many language toolchains (Go, Python, Node) have
  well-maintained PPAs.
- **Upstream apt repo**: NodeSource for Node, deadsnakes for Python, the
  official Go / Rust / Docker repos all publish a `.list` file + GPG key.
- **Snap**: `snap install <pkg>` for desktop / sandboxed tools. Snap
  often doesn't work inside a container without extra setup; prefer apt
  or upstream repos here.

Don't claim "version X is not available" without checking `apt-cache
policy` and the relevant PPA / upstream repo.

## Common gotchas

- `add-apt-repository` requires `software-properties-common` — install
  that first if it's missing.
- `DEBIAN_FRONTEND=noninteractive apt-get install -y` if a config prompt
  would hang the install.
- `apt` adds progress lines unsuitable for scripts — use `apt-get` there.
- Snap doesn't run in most containers; don't try to install via snap as
  a fallback unless you've verified snapd is running here.
