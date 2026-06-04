# Arch skill

Base is Arch Linux ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}). Pkg
manager is `pacman`; `yay` (or `paru`) wraps it with AUR support and is
what you almost always want. Arch is rolling — VERSION_ID is just the
image build date.

Run as non-root `dev` (sudo NOPASSWD). To install, call **`yay` directly
with NO `sudo`** — it elevates itself and **REFUSES to run as root**, so
`sudo yay` fails. Read-only probes (`pacman -Q`, `pacman -Qi`, `yay -Ss`)
need no elevation. `sudo pacman` is the fallback-only install path (see
below); NOT how you normally install.

## Probe (only if needed)

`/etc/os-release` is in the header — do NOT re-run `cat /etc/os-release`.
Session-invariant:

- `pacman --version`
- `pacman -Q` — every installed package (long; pipe to grep).
- `pacman -Qi <pkg>` — version, install reason, deps, size.

## Search and install — ALWAYS prefer yay

`yay` searches and installs from official AND AUR in one pass. AUR
routinely carries fresher versions and tools not in official repos.
Starting with `pacman -S` and falling back wastes a round trip — go
straight to yay.

- `yay -Ss <pkg>` — search official + AUR.
- `yay -Si <pkg>` — detailed info.
- `yay -S --noconfirm <pkg>` — install. NO leading `sudo` — yay elevates
  itself. Do NOT reach for `sudo pacman -S`; that is the fallback below,
  not the default, even though installing needs root.
- `yay -Syu --noconfirm` — full update (rarely needed here).

This holds for LANGUAGE-ECOSYSTEM tools too (gopls, ruff, prettier, …): prefer
`yay -S <tool>` over `go install` / `pip install` / `npm i -g` whenever the repo
carries it (`yay -Ss <tool>` to check). In a container the language installers
hit root-vs-`dev` env mismatches (GOBIN/PATH) and land binaries off PATH; yay
just works. Only use the language installer when yay has no package.

If yay isn't on PATH (rare): `sudo pacman -Ss`/`-Si`/`-S --noconfirm`
(official only, misses AUR).

## Version staleness

Arch is rolling; packages usually current within days. If something looks
stale, check `/etc/pacman.d/mirrorlist` — sometimes a Dockerfile pins an
old mirror. `yay -Syu` brings everything current.

## Common gotchas

- AUR builds from source — first install may be slow.
- Some AUR packages need build deps not in `base-devel`. On `gcc`/`make`
  errors: `pacman -S --noconfirm base-devel` first.
- `pacman` may fail under some sudo configs; check `/etc/sudoers.d/`.
