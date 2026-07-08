# Alpine

## Base System
- Base: Alpine Linux ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"}}), {{cmd:apk --version}}.
- Pkg mgr apk.
- libc=musl + gcompat installed → most prebuilt glibc binaries run; one with heavier glibc deps may still fail → then use the Alpine pkg or a static/musl build.
- User=non-root `dev`, sudo NOPASSWD.
- Write ops (add/del/update) need sudo.
- Read probes (info, list -I) no sudo.

## Probe
- apk list -I → all installed (long → grep).
- apk info <pkg> → version + desc.

## Installation
- Order: 1) apk, 2) upstream site / custom repo (key → /etc/apk/keys/, then echo "https://host/path" >> /etc/apk/repositories && apk update), 3) lang installer (pip, go, npm, etc.)
- Fall to the next option only when the previous has no (or too old a) package.
- Upstream site: web_search the official install docs, then either: install script `curl -fsSL https://…/install.sh | sh` (use the shell the docs name — sh vs bash matters on minimal images); or prebuilt release binary / tarball → drop into ~/.local/bin (already on PATH) or /usr/local/bin; or vendor repo when the docs offer one → add it with the custom-repo recipe above, then install via the pkg mgr (repo installs keep getting updates — prefer over a one-off binary when both exist).
- Lang installer: go install / pipx / npm i -g / cargo install, only when upstream documents it. In container these hit root-vs-dev env mismatch (GOBIN/PATH) → binary can land off PATH: verify with `which <tool>` after install and add the installer's bin dir to PATH if missing.
- apk update → refresh index (cheap; do before search/install).
- apk search <pkg> ; -e = exact.
- apk info <pkg> → version, deps, files.
- apk add <pkg> → install (no --noconfirm; no prompts).
- apk del <pkg> → uninstall.
