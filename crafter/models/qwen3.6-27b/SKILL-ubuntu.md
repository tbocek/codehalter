# Ubuntu skill

## System Context
- Base: Ubuntu ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID, codename=$UBUNTU_CODENAME"}}), {{cmd:apt --version}}.
- Debian-derived: apt/apt-get/dpkg work identically.
- Main diffs: suite codenames (jammy, noble, oracular…) + PPAs as source of fresher packages.
- User=non-root `dev`, sudo NOPASSWD.
- Prefix apt update/install/remove, apt-get, add-apt-repository, dpkg -i with sudo.
- Read probes (apt-cache search/policy, dpkg -l/-s) no sudo.

## Probe
- dpkg -l → all installed (long → grep).
- dpkg -l <pkg> → installed version, or "no packages found".
- dpkg -s <pkg> → detailed status.

## Package Management
- Order: 1) apt-get, 2) upstream site / custom repo (curl -fsSL <key-url> | sudo tee /etc/apt/keyrings/<name>.asc && echo "deb [signed-by=/etc/apt/keyrings/<name>.asc] <repo-url> <suite> main" | sudo tee /etc/apt/sources.list.d/<name>.list && apt-get update), 3) lang installer (pip, go, npm, etc.), PPA (add-apt-repository preinstalled): sudo add-apt-repository -y ppa:<owner>/<name> && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y <pkg>
- Fall to the next option only when the previous has no (or too old a) package.
- Upstream site: web_search the official install docs, then either: install script `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images); or prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin; or vendor repo when the docs offer one → add it with the custom-repo recipe above, then install via the pkg mgr (repo installs keep getting updates — prefer over a one-off binary when both exist).
- Lang installer: go install / pipx / npm i -g / cargo install, only when upstream documents it. In container these hit root-vs-dev env mismatch (GOBIN/PATH) → binary can land off PATH: verify with `which <tool>` after install and add the installer's bin dir to PATH if missing.
- apt-cache search <pkg> → fuzzy.
- DEBIAN_FRONTEND=noninteractive apt-get install -y <pkg> → install (never hangs on config prompts).
- apt-get remove/purge -y <pkg> → uninstall.
