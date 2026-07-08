# Alpine

## Base System
- Base: Alpine Linux ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"}}), {{cmd:apk --version}}.
- Pkg mgr apk.
- libc=musl + gcompat installed → most prebuilt glibc binaries run; one with heavier glibc deps may still fail → then use the Alpine pkg or a static/musl build.

## Probe
- apk list -I → all installed (long → grep).

## Installation
- Order: 1) apk, 2) upstream site / custom repo (key → /etc/apk/keys/, then echo "https://host/path" >> /etc/apk/repositories && apk update), 3) lang installer (pip, go, npm, etc.)
- Fall to the next option only when the previous has no (or too old a) package.
- Upstream site: ALWAYS web_search the official install docs FIRST — before running `apk update`, `apk search`, or any package-manager probe — because upstream docs are the only authoritative source for what they actually support on this distro (Alpine repos, vendor repo, install script, or prebuilt binary). Then pick exactly one of: install script `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images); or prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin; or vendor repo when the docs offer one → add it with the custom-repo recipe above, then install via the pkg mgr (repo installs keep getting updates — prefer over a one-off binary when both exist). Do NOT skip the doc search and jump straight to `apk search` / `apk add` just because this is Alpine — the upstream may require a vendor repo or a specific binary that the distro repo does not provide.
- Lang installer: go install / pipx / npm i -g / cargo install, only when upstream documents it. In container these hit root-vs-dev env mismatch (GOBIN/PATH) → binary can land off PATH: verify with `which <tool>` after install and add the installer's bin dir to PATH if missing.
- apk search <pkg> ; -e = exact.
