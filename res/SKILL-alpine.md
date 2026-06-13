# Alpine skill
Base: Alpine Linux ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}). Pkg mgr apk. libc=musl NOT glibc → prebuilt glibc binaries usually no run. Prefer Alpine pkgs / static binaries.
User=non-root `dev`, sudo NOPASSWD. Write ops (add/del/update) need sudo. Read probes (info, list -I) no sudo.

## Probe (only if needed; session-invariant, reuse from history)
- /etc/os-release in header → no re-cat.
- apk --version
- apk list -I → all installed (long → grep).
- apk info <pkg> → version + desc.

## Search / install
- apk update → refresh index (cheap; do before search/install).
- apk search <pkg> ; -e = exact.
- apk info <pkg> → version, deps, files.
- apk add <pkg> → install (no --noconfirm; no prompts).
- apk del <pkg> → uninstall.
- Main + community repos split. Most dev tooling in community/ → enable in /etc/apk/repositories. Rolling: apk add --repository=https://dl-cdn.alpinelinux.org/alpine/edge/community/ <pkg>

## Version staleness
Stable freezes versions. Fast-moving tools (Go, Node, gopls, LSPs): if apk info looks old → web_search upstream before claiming unavailable. Fallback: edge repo, or apk add build-base && make from source.

## Gotchas
- musl≠glibc: prebuilt may segfault → seek Alpine/musl/static variants.
- Split pkgs: python3 vs py3-pip, go vs gopls.
- apk exits 0 on "already installed" → re-run confirms presence.