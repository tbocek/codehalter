# Arch skill
Base: Arch Linux ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}). Pkg mgr pacman; yay (or paru) wraps pacman + AUR → almost always want yay. Arch rolling → VERSION_ID = image build date only.
User=non-root `dev`, sudo NOPASSWD. Install = call **yay direct, NO sudo** — yay self-elevates + REFUSES root → `sudo yay` fails. Read probes (pacman -Q/-Qi, yay -Ss) no elevation. `sudo pacman` = fallback-only (below), NOT normal install.

## Probe (only if needed; session-invariant)
- /etc/os-release in header → no re-cat.
- pacman --version
- pacman -Q → all installed (long → grep).
- pacman -Qi <pkg> → version, install reason, deps, size.

## Search / install — ALWAYS prefer yay
yay searches + installs official AND AUR one pass. AUR often fresher / has tools not in official. Start pacman -S then fall back = wasted round trip → go straight yay.
- yay -Ss <pkg> → search official + AUR.
- yay -Si <pkg> → detailed info.
- yay -S --noconfirm <pkg> → install. NO leading sudo (yay self-elevates). No sudo pacman -S (that = fallback, not default, even tho install needs root).
- yay -Syu --noconfirm → full update (rare here).
Same for LANG-ECOSYSTEM tools (gopls, ruff, prettier…): prefer yay -S <tool> over go install / pip install / npm i -g when repo carries it (check yay -Ss <tool>). In container, lang installers hit root-vs-dev env mismatch (GOBIN/PATH) → binaries land off PATH; yay just works. Lang installer ONLY when yay has no pkg.
yay not on PATH (rare) → sudo pacman -Ss/-Si/-S --noconfirm (official only, misses AUR).

## Version staleness
Arch rolling → pkgs usually current within days. Stale → check /etc/pacman.d/mirrorlist (Dockerfile may pin old mirror). yay -Syu brings all current.

## Gotchas
- AUR builds from source → first install may be slow.
- Some AUR pkgs need build deps not in base-devel. gcc/make error → pacman -S --noconfirm base-devel first.
- pacman may fail under some sudo configs → check /etc/sudoers.d/.