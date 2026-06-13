# Ubuntu skill
Base: Ubuntu ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}, codename={{UBUNTU_CODENAME}}). Debian-derived: apt/apt-get/dpkg work identically. Main diffs: suite codenames (jammy, noble, oracular…) + PPAs as source of fresher packages.
User=non-root `dev`, sudo NOPASSWD. Prefix apt update/install/remove, apt-get, add-apt-repository, dpkg -i with sudo. Read probes (apt-cache search/policy, dpkg -l/-s) no sudo.

## Probe (only if needed; session-invariant)
- /etc/os-release in header → no re-run.
- apt --version
- dpkg -l → all installed (long → grep).
- dpkg -l <pkg> → installed version, or "no packages found".
- dpkg -s <pkg> → detailed status.

## Search / install
- apt update → refresh index. Required after add-apt-repository or source edits.
- apt-cache search <pkg> → fuzzy.
- apt-cache policy <pkg> → installed + candidate + source. Always check before claiming a version isn't available.
- apt install -y <pkg> → install.
- apt remove/purge <pkg> → uninstall.

## Version staleness — PPAs = Ubuntu shortcut
LTS releases freeze versions for years. Three escalating workarounds:
- **PPA**: `add-apt-repository -y ppa:<owner>/<name> && apt update && apt install -y <pkg>`. Many language toolchains have well-maintained PPAs.
- **Upstream apt repo**: NodeSource (Node), deadsnakes (Python), official Go/Rust/Docker repos — fetch .list + GPG key.
- **Snap**: often doesn't work inside container → prefer apt or upstream repos.
Don't claim "version X unavailable" without checking `apt-cache policy` + relevant PPA / upstream repo.

## Gotchas
- add-apt-repository requires software-properties-common — install first if missing.
- `DEBIAN_FRONTEND=noninteractive apt-get install -y` if config prompt would hang.
- apt adds progress lines unsuitable for scripts → use apt-get there.
- Snap doesn't run in most containers; don't try as fallback unless you've verified snapd running.