# Debian skill

Container base is Debian ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}},
codename={{VERSION_CODENAME}}). Package manager is `apt` / `apt-get`; the
C library is glibc. Stable repos prioritise stability over freshness, so
versions can be months or years behind upstream. The codename above
determines which suites apt sees.

## Probe (only if needed)

`/etc/os-release` is already reflected in the header above — do NOT re-run
`cat /etc/os-release`. The remaining probes ARE worth running on demand:

- `apt --version` — apt version.
- `dpkg -l` — every package currently installed (long; pipe to grep).
- `dpkg -l <pkg>` — installed version + arch for one package, or "no
  packages found" if absent.
- `dpkg -s <pkg>` — detailed status (depends, conflicts, size).

These are session-invariant for the life of the container — once you see
the answer in a tool result this turn, don't re-run them.

## Search and install

- `apt update` — refresh the index. Required after any change to
  `/etc/apt/sources.list*`; otherwise occasional. Cheap.
- `apt-cache search <pkg>` — fuzzy search across names and descriptions.
- `apt-cache policy <pkg>` — installed version + candidate version + which
  source suite ships it. Use this BEFORE installing to see what you'd get.
- `apt-cache show <pkg>` — full record.
- `apt install -y <pkg>` — install (the `-y` skips the y/n prompt).
- `apt remove <pkg>`, `apt purge <pkg>` — uninstall (purge removes config too).

## Version staleness

`apt-cache policy <pkg>` is the source of truth — if the Candidate looks
old for a fast-moving tool (Go, Node, gopls, language servers), check the
upstream release page with `web_search` first. Common workarounds:

- **Backports**: enable in `/etc/apt/sources.list.d/` then
  `apt install -t bookworm-backports <pkg>`.
- **Upstream apt repo**: many projects publish a `.list` file + GPG key
  (Node via NodeSource, Go via various, etc.) — fetch and install per their
  docs.
- **Direct download**: tarball / .deb from the upstream release page.

Don't claim "version X is not available" without checking the candidate
and considering backports.

## Common gotchas

- `apt` (the friendly CLI) and `apt-get` (the scriptable CLI) overlap but
  `apt` adds a "Reading package lists..." progress line that breaks naive
  parsing. Use `apt-get` in scripts.
- `DEBIAN_FRONTEND=noninteractive apt-get install -y` if a package config
  prompt would otherwise hang the install.
- `apt install <pkg>=<version>` to pin a specific version when the
  candidate isn't what you want.
