# Debian skill

## Environment
- Base: Debian ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID, codename=$VERSION_CODENAME"}}), {{cmd:apt --version}}.
- Pkg mgr apt/apt-get; libc = glibc.
- Stable repos favor stability over freshness → versions can be months/years behind upstream.
- User=non-root `dev`, sudo NOPASSWD.
- Read probes (apt-cache search/policy, dpkg -l/-s) no sudo.

## Probe
- dpkg -l → all installed (long → grep).
- dpkg -l <pkg> → installed version + arch, or "no packages found".
- dpkg -s <pkg> → detailed status (depends, conflicts, size).

## Install order
- Order: 1) apt-get, 2) upstream site / custom repo (curl -fsSL <key-url> | sudo tee /etc/apt/keyrings/<name>.asc && echo "deb [signed-by=/etc/apt/keyrings/<name>.asc] <repo-url> <suite> main" | sudo tee /etc/apt/sources.list.d/<name>.list && apt-get update (gnupg installed)), 3) lang installer (pip, go, npm, etc.), 4) backports (DEBIAN_FRONTEND=noninteractive apt-get install -y -t {{cmd:. /etc/os-release && echo $VERSION_CODENAME}}-backports <pkg>)
- Fall to the next option only when the previous has no (or too old a) package.
- Upstream site: web_search the official install docs, then either: install script `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images); or prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin; or
- Lang installer: go install / pipx / npm i -g / cargo install, only when upstream documents it. In container these hit root-vs-dev env mismatch (GOBIN/PATH) → binary can land off PATH: verify with `which <tool>` after install and add the installer's bin dir to PATH if missing.

## Apt commands
- apt-get update → refresh index.
- Check BEFORE install to see what you'd get.
- apt-cache show <pkg> → full record.
- <pkg>=<version> pins a version.
