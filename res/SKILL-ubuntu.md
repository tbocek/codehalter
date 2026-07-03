# Ubuntu skill
Base: Ubuntu ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID, codename=$UBUNTU_CODENAME"}}), {{cmd:apt --version}}. Debian-derived: apt/apt-get/dpkg work identically. Main diffs: suite codenames (jammy, noble, oracular…) + PPAs as source of fresher packages.
User=non-root `dev`, sudo NOPASSWD. Prefix apt update/install/remove, apt-get, add-apt-repository, dpkg -i with sudo. Read probes (apt-cache search/policy, dpkg -l/-s) no sudo.

## Probe
- dpkg -l → all installed (long → grep).
- dpkg -l <pkg> → installed version, or "no packages found".
- dpkg -s <pkg> → detailed status.

## Search / install
Order: 1) apt-get 2) upstream site / custom repo (below) 3) lang installer 4) PPA — details in container skill.
Use apt-get, not apt — apt's progress chatter breaks output parsing.
- apt-get update → refresh index. Required after add-apt-repository or source edits.
- apt-cache search <pkg> → fuzzy.
- apt-cache policy <pkg> → installed + candidate + source. Always check before claiming a version isn't available.
- DEBIAN_FRONTEND=noninteractive apt-get install -y <pkg> → install (never hangs on config prompts).
- apt-get remove/purge -y <pkg> → uninstall.
- Custom repo: curl -fsSL <key-url> | sudo tee /etc/apt/keyrings/<name>.asc && echo "deb [signed-by=/etc/apt/keyrings/<name>.asc] <repo-url> <suite> main" | sudo tee /etc/apt/sources.list.d/<name>.list && apt-get update.
- 4th choice PPA (add-apt-repository preinstalled): sudo add-apt-repository -y ppa:<owner>/<name> && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y <pkg>
