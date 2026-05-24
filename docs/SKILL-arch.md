# Arch skill

Base is Arch Linux ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}). Pkg
manager is `pacman`; `yay` (or `paru`) wraps it with AUR support and is
what you almost always want. Arch is rolling, so VERSION_ID is just the
image build date.

Run as non-root `dev` (sudo NOPASSWD). Prefix `pacman -S`/`-Syu` with
`sudo`. Read-only probes (`pacman -Q`, `pacman -Qi`, `yay -Ss`) don't
need it. **`yay` REFUSES to run as root** — call `yay` directly; it
prompts for sudo when needed. Only `pacman` takes leading `sudo`.

## Probe (only if needed)

`/etc/os-release` is reflected in the header — do NOT re-run
`cat /etc/os-release`. Session-invariant:

- `pacman --version`
- `pacman -Q` — every installed package (long; pipe to grep).
- `pacman -Qi <pkg>` — version, install reason, deps, size.

## Search and install — ALWAYS prefer yay

`yay` searches and installs from official AND AUR in one pass. AUR routinely
carries fresher versions and tools not in official repos. Starting with
`pacman -S` and falling back wastes a round trip — go straight to yay.

- `yay -Ss <pkg>` — search official + AUR.
- `yay -Si <pkg>` — detailed info.
- `yay -S --noconfirm <pkg>` — install.
- `yay -Syu --noconfirm` — full update (rarely needed here).

If yay isn't on PATH (rare): `pacman -Ss`/`-Si`/`-S --noconfirm` (official
only, misses AUR).

## Version staleness

Arch is rolling; packages are usually current within days. If something
looks stale, check `/etc/pacman.d/mirrorlist` — sometimes a Dockerfile
pins an old mirror. `yay -Syu` brings everything current.

## Common gotchas

- AUR builds from source — first install may be slow.
- Some AUR packages need build deps not in `base-devel`. On `gcc`/`make`
  errors: `pacman -S --noconfirm base-devel` first.
- `pacman` may fail under some sudo configs; check `/etc/sudoers.d/`.
