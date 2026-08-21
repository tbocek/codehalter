# Alpine skill
Base: Alpine Linux ({{cmd:. /etc/os-release && echo "$PRETTY_NAME, VERSION_ID=$VERSION_ID"}}), {{cmd:apk --version}}. Pkg mgr apk. libc=musl + gcompat installed → most prebuilt glibc binaries run; one with heavier glibc deps may still fail → then use the Alpine pkg or a static/musl build.
User=non-root `dev`, sudo NOPASSWD. Write ops (add/del/update) need sudo. Read probes (info, list -I) no sudo.

## Probe
- apk list -I → all installed (long → grep).
- apk info <pkg> → version + desc.

## Search / install
Order: 1) apk 2) upstream site / custom repo (below) 3) lang installer.
- sudo apk update → refresh index (cheap; do before search/install).
- apk search -e <pkg> → exact-name check; bare apk search <pkg> = substring. Output is `<name>-<ver>-r<N>`, NO space after name → `apk search go | grep "^go "` matches 0 lines, reads as unpackaged. Never grep to prove absence.
- apk info <pkg> → version, deps, files.
- sudo apk add <pkg> → install (no --noconfirm; no prompts).
- sudo apk del <pkg> → uninstall.
- Custom repo: key → /etc/apk/keys/, then echo "https://host/path" >> /etc/apk/repositories && apk update.
- 4th choice edge (rolling — distinct from the stable community repo, which is already enabled), when stable's version is too old. Prove it: apk search -e <pkg> on stable, quote both versions. Same version in stable = no reason for edge (edge deps in a stable image = rebuild risk). One-shot pull, does NOT enable edge system-wide: apk add --repository=https://dl-cdn.alpinelinux.org/alpine/edge/community/ <pkg>
