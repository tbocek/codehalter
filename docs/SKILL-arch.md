# Arch skill

Container base is Arch Linux (`/etc/os-release` ID=arch). Package manager
is `pacman` for the official repos; `yay` (or `paru`) wraps pacman with
AUR support and is what you almost always want here.

## Probe first

- `cat /etc/os-release` — Arch is rolling, so VERSION_ID is "TEMPLATE_VERSION_ID"
  or a date; the practical version is "whatever pacman -Syu pulled last".
- `pacman --version` — pacman version.
- `pacman -Q` — every package currently installed (long; pipe to grep).
- `pacman -Qi <pkg>` — detailed info (version, install reason, depends, size)
  for one package.

These are session-invariant for the life of the container — once you see
the answer in a tool result this turn, don't re-run them.

## Search and install — ALWAYS prefer yay

`yay` searches and installs from official repos AND the AUR in one pass.
The AUR routinely carries fresher versions and tools that aren't in the
official repos at all. Starting with `pacman -S` and only falling back to
`yay` when it fails wastes a round trip — go straight to yay.

- `yay -Ss <pkg>` — search both official and AUR.
- `yay -Si <pkg>` — detailed info for one package (official or AUR).
- `yay -S --noconfirm <pkg>` — install (skip the confirmation prompt).
- `yay -Syu --noconfirm` — full system update (rarely needed here).

If yay isn't on PATH (rare), fall back to:
- `pacman -Ss <pkg>` — search official repos only (misses AUR).
- `pacman -Si <pkg>` — info, official only.
- `pacman -S --noconfirm <pkg>` — install, official only.

## Version staleness

Arch is rolling so packages are usually current within days. If something
looks stale, check that the container's mirror list is fresh
(`/etc/pacman.d/mirrorlist`) — sometimes a Dockerfile pins an old mirror.
`yay -Syu` brings everything current; `yay -S <pkg>` alone may install
against a stale cache.

## Common gotchas

- AUR packages build from source — first install may be slow.
- Some AUR packages need build deps not in `base-devel`. If `yay -S` fails
  on a missing `gcc` / `make`, `pacman -S --noconfirm base-devel` first.
- `pacman` won't run as root by default through some sudo configs; check
  `/etc/sudoers.d/` if you hit auth errors.
