# Ubuntu skill

Base is Ubuntu ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}},
codename={{UBUNTU_CODENAME}}). Debian-derived: `apt`/`apt-get`/`dpkg`
work identically. Main differences: suite codenames (jammy, noble,
oracular, …) and PPAs as a source of fresher packages.

Run as non-root `dev` (sudo NOPASSWD). Prefix `apt update`/`install`/
`remove`/`apt-get`/`add-apt-repository`/`dpkg -i` with `sudo`. Read-only
probes (`apt-cache search`/`policy`, `dpkg -l`/`-s`) don't need it.

## Probe (only if needed)

`/etc/os-release` is in the header — do NOT re-run. Session-invariant:

- `apt --version`
- `dpkg -l` — every installed package (long; pipe to grep).
- `dpkg -l <pkg>` — installed version, or "no packages found".
- `dpkg -s <pkg>` — detailed status.

## Search and install

- `apt update` — refresh index. Required after `add-apt-repository` or
  source edits.
- `apt-cache search <pkg>` — fuzzy search.
- `apt-cache policy <pkg>` — installed + candidate + source. Always
  check before claiming a version isn't available.
- `apt install -y <pkg>` — install.
- `apt remove`/`purge <pkg>` — uninstall.

## Version staleness — PPAs are the Ubuntu shortcut

LTS releases freeze versions for years. Three escalating workarounds:

- **PPA**: `add-apt-repository -y ppa:<owner>/<name> && apt update &&
  apt install -y <pkg>`. Many language toolchains have well-maintained PPAs.
- **Upstream apt repo**: NodeSource (Node), deadsnakes (Python), official
  Go/Rust/Docker repos — fetch the `.list` + GPG key.
- **Snap**: often doesn't work inside a container; prefer apt or
  upstream repos.

Don't claim "version X unavailable" without checking `apt-cache policy`
and the relevant PPA / upstream repo.

## Common gotchas

- `add-apt-repository` requires `software-properties-common` — install
  that first if missing.
- `DEBIAN_FRONTEND=noninteractive apt-get install -y` if a config prompt
  would hang.
- `apt` adds progress lines unsuitable for scripts — use `apt-get` there.
- Snap doesn't run in most containers; don't try it as a fallback unless
  you've verified snapd is running.
