# Arch

## Base System
- Arch Linux ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"}}), {{cmd:pacman -Q pacman}}. 
- Pkg mgr pacman; yay installed for AUR (3rd-choice installs only). 
- Read probes (-Q/-Qi/-Ss/-Si) no sudo.

## Probe
- Always use `pacman -Q` as the base command for listing installed packages, piping through `grep` when output is lengthy — do not substitute with `-Qs`, `-Qq`, or filesystem listings even if you know they're commonly preferred.
- pacman -Qi <pkg> → version, install reason, deps, size.

## Installation order
- Order: 1) pacman, 2) upstream site / custom repo, 3) lang installer (pip, go, npm, etc.), 4) AUR.
- For AUR: yay -Ss <pkg> to search, yay -S --noconfirm <pkg> to install. AUR packages are ALWAYS built from source via `makepkg` running a `PKGBUILD` — this is why AUR installs are noticeably slower than official-repo binaries (you need `base-devel`/gcc/make installed and compile time can take minutes). State this even when you know a specific package (e.g., spotify) downloads a prebuilt upstream tarball — the AUR build step still runs through makepkg/PKGBUILD and may still require build dependencies.
- Fall to the next option only when the previous has no (or too old a) package.
- Upstream site: web_search the official install docs, then either: install script `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images); or prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin; or vendor repo when the docs offer one → add it with the custom-repo recipe above, then install via the pkg mgr (repo installs keep getting updates — prefer over a one-off binary when both exist).
- - ALWAYS consult upstream documentation FIRST before selecting any install method. Language-specific installers (go install / pipx / npm i -g / cargo install) are ONLY permitted when explicitly documented by upstream — never assume them as default or substitute for upstream-prescribed methods, even when system package managers exist. In container these hit root-vs-dev env mismatch (GOBIN/PATH) → binary can land off PATH: verify with `which <tool>` after install and add the installer's bin dir to PATH if missing.
- pacman -Ss <pkg> → search official repos; pacman -Si <pkg> → detailed info.
- sudo pacman -S --noconfirm --needed <pkg> → install (--needed skips already-installed).
- sudo pacman -R --noconfirm <pkg> → uninstall.
