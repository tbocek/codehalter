# Arch

## Base System
- Arch Linux ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"}}), {{cmd:pacman -Q pacman}}. 

## Installation order
- Order: 1) pacman, 2) upstream site / custom repo, 3) lang installer (pip, go, npm, etc.), 4) AUR.
- For custom repos: append to /etc/pacman.conf: `[name]` + `Server = https://…` (key: pacman-key --recv-keys <id> && pacman-key --lsign-key <id>), then sudo pacman -Syu.
- For AUR: yay -Ss <pkg> to search, yay -S --noconfirm <pkg> to install. AUR builds from source → can be slow.
- Upstream site: web_search the official install docs, then either: install script `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images); or prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin; or vendor repo when the docs offer one → add it with the custom-repo recipe above, then install via the pkg mgr (repo installs keep getting updates — prefer over a one-off binary when both exist).
- Lang installer: go install / pipx / npm i -g / cargo install, only when upstream documents it. In container these hit root-vs-dev env mismatch (GOBIN/PATH) → binary can land off PATH: verify with `which <tool>` after install and add the installer's bin dir to PATH if missing.
- sudo pacman -S --noconfirm --needed <pkg> → install (--needed skips already-installed).
- sudo pacman -R --noconfirm <pkg> → uninstall.
