# Arch skill
Base: Arch Linux ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"}}), {{cmd:pacman -Q pacman}}. Pkg mgr pacman; yay installed for AUR (3rd-choice installs only). Arch rolling → VERSION_ID = image build date only; packages track upstream closely.
User=non-root `dev`, sudo NOPASSWD. Write ops (-S/-R/-Syu) need sudo. Read probes (-Q/-Qi/-Ss/-Si) no sudo.

## Probe
- pacman -Q → all installed (long → grep).
- pacman -Qi <pkg> → version, install reason, deps, size.

## Search / install
Order: 1) pacman 2) upstream site / custom repo (below) 3) lang installer 4) AUR — details in container skill.
- pacman -Ss <pkg> → search official repos; pacman -Si <pkg> → detailed info.
- sudo pacman -S --noconfirm --needed <pkg> → install (--needed skips already-installed).
- sudo pacman -R --noconfirm <pkg> → uninstall.
- Custom repo: append to /etc/pacman.conf: `[name]` + `Server = https://…` (key: pacman-key --recv-keys <id> && pacman-key --lsign-key <id>), then sudo pacman -Syu.
- 4th choice AUR: yay -Ss <pkg> to search, yay -S --noconfirm <pkg> to install. NO sudo — yay self-elevates and refuses root. AUR builds from source → can be slow.
