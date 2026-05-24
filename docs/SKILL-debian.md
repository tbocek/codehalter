# Debian skill

Base is Debian ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}},
codename={{VERSION_CODENAME}}). Pkg manager is `apt`/`apt-get`; libc is
glibc. Stable repos prioritise stability over freshness — versions can be
months/years behind upstream.

Run as non-root `dev` (sudo NOPASSWD). Prefix `apt update`/`install`/
`remove`/`apt-get`/`dpkg -i` with `sudo`. Read-only probes (`apt-cache
search`/`policy`, `dpkg -l`/`-s`) don't need it.

## Probe (only if needed)

`/etc/os-release` is reflected in the header — do NOT re-run. Session-
invariant:

- `apt --version`
- `dpkg -l` — every installed package (long; pipe to grep).
- `dpkg -l <pkg>` — installed version + arch, or "no packages found".
- `dpkg -s <pkg>` — detailed status (depends, conflicts, size).

## Search and install

- `apt update` — refresh index. Required after editing sources; cheap.
- `apt-cache search <pkg>` — fuzzy search.
- `apt-cache policy <pkg>` — installed + candidate + source suite. Check
  this BEFORE installing to see what you'd get.
- `apt-cache show <pkg>` — full record.
- `apt install -y <pkg>` — install (`-y` skips prompt).
- `apt remove`/`purge <pkg>` — uninstall (purge removes config too).

## Version staleness

`apt-cache policy <pkg>` is source of truth. If the Candidate looks old
for a fast-moving tool, check upstream with `web_search`. Workarounds:

- **Backports**: enable in `/etc/apt/sources.list.d/`, then
  `apt install -t bookworm-backports <pkg>`.
- **Upstream apt repo**: NodeSource for Node, official Go, etc. — fetch
  the `.list` + GPG key per their docs.
- **Direct download**: tarball / .deb from the release page.

Don't claim "version X unavailable" without checking the candidate and
backports.

## Common gotchas

- `apt` (friendly) vs `apt-get` (scriptable) — `apt` adds a
  "Reading package lists..." line that breaks naive parsing. Use
  `apt-get` in scripts.
- `DEBIAN_FRONTEND=noninteractive apt-get install -y` if a config prompt
  would hang.
- `apt install <pkg>=<version>` to pin a version.
