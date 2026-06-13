# Debian skill
Base: Debian ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}, codename={{VERSION_CODENAME}}). Pkg mgr apt/apt-get; libc = glibc. Stable repos favor stability over freshness → versions can be months/years behind upstream.
User=non-root `dev`, sudo NOPASSWD. Prefix apt update/install/remove, apt-get, dpkg -i with sudo. Read probes (apt-cache search/policy, dpkg -l/-s) no sudo.

## Probe (only if needed; session-invariant)
- /etc/os-release in header → no re-run.
- apt --version
- dpkg -l → all installed (long → grep).
- dpkg -l <pkg> → installed version + arch, or "no packages found".
- dpkg -s <pkg> → detailed status (depends, conflicts, size).

## Search / install
- apt update → refresh index. Required after editing sources; cheap.
- apt-cache search <pkg> → fuzzy.
- apt-cache policy <pkg> → installed + candidate + source suite. Check BEFORE install to see what you'd get.
- apt-cache show <pkg> → full record.
- apt install -y <pkg> → install (-y skips prompt).
- apt remove/purge <pkg> → uninstall (purge removes config too).

## Version staleness
apt-cache policy <pkg> = source of truth. Candidate looks old for fast-moving tool → web_search upstream. Workarounds:
- **Backports**: enable in /etc/apt/sources.list.d/ → `apt install -t bookworm-backports <pkg>`.
- **Upstream apt repo**: NodeSource for Node, official Go, etc. — fetch .list + GPG key per their docs.
- **Direct download**: tarball / .deb from release page.
Don't claim "version X unavailable" without checking candidate + backports.

## Gotchas
- apt (friendly) vs apt-get (scriptable) — apt adds "Reading package lists..." line that breaks naive parsing → use apt-get in scripts.
- `DEBIAN_FRONTEND=noninteractive apt-get install -y` if config prompt would hang.
- `apt install <pkg>=<version>` to pin version.