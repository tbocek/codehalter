# Alpine skill

Base is Alpine Linux ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}). Pkg
manager is `apk`. libc is **musl, not glibc** — prebuilt glibc binaries
generally won't run. Prefer Alpine packages or static binaries.

Run as non-root `dev` (sudo NOPASSWD). Prefix `apk add`/`del`/`update`
with `sudo`. Read-only probes (`apk info`, `apk list -I`) don't need it.

## Probe (only if needed)

`/etc/os-release` is in the header — do NOT re-run `cat /etc/os-release`.
Session-invariant — reuse from this turn's history:

- `apk --version`
- `apk list -I` — every installed package (long; pipe to grep).
- `apk info <pkg>` — installed version + description.

## Search and install

- `apk update` — refresh index (cheap; run before search/install).
- `apk search <pkg>` / `apk search -e <pkg>` — fuzzy / exact name.
- `apk info <pkg>` — version, deps, files.
- `apk add <pkg>` — install (no `--noconfirm`; apk doesn't prompt).
- `apk del <pkg>` — uninstall.

Main + community repos split. Most dev tooling is in `community/`; enable
in `/etc/apk/repositories` if missing. Rolling versions:
`apk add --repository=https://dl-cdn.alpinelinux.org/alpine/edge/community/ <pkg>`.

## Version staleness

Stable releases freeze versions. If `apk info` looks old for a fast-moving
tool (Go, Node, gopls, LSPs), check upstream with `web_search` before
claiming "version X isn't available." Fallback: edge, or
`apk add build-base && make` from source.

## Common gotchas

- musl ≠ glibc: prebuilt binaries often segfault. Look for "Alpine",
  "musl", or "static" download variants.
- Split packages: `python3` vs `py3-pip`, `go` vs `gopls`.
- `apk` exits 0 on "already installed" — re-running confirms presence.
