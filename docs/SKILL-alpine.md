# Alpine skill

Container base is Alpine Linux ({{PRETTY_NAME}}, VERSION_ID={{VERSION_ID}}).
Package manager is `apk`; the C library is musl, not glibc — pre-built
binaries linked against glibc generally won't run, so prefer Alpine
packages or static binaries.

## Probe (only if needed)

`/etc/os-release` is already reflected in the header above — do NOT re-run
`cat /etc/os-release`. The remaining probes ARE worth running on demand:

- `apk --version` — apk-tools version.
- `apk list -I` — every package currently installed (long; pipe to grep).
- `apk info <pkg>` — installed version + description for one package.

These are session-invariant for the life of the container — once you see
the answer in a tool result this turn, don't re-run them.

## Search and install

- `apk update` — refresh the index (run before search/install if it's been
  a while; cheap and safe).
- `apk search <pkg>` — fuzzy search across names and descriptions.
- `apk search -e <pkg>` — exact-name match only.
- `apk info <pkg>` — version, dependencies, files.
- `apk add <pkg>` — install (no `--noconfirm` flag needed; apk doesn't prompt).
- `apk del <pkg>` — uninstall.

Alpine's main + community repos are split. Most dev tooling is in
`community/`; enable it by editing `/etc/apk/repositories` if missing.
Edge (rolling) repos carry fresher versions — opt in per-install with
`apk add --repository=https://dl-cdn.alpinelinux.org/alpine/edge/community/ <pkg>`.

## Version staleness

Stable Alpine releases freeze package versions. If the candidate version
shown by `apk info` looks old for a fast-moving tool (Go, Node, gopls,
language servers), check the upstream release page with `web_search`
before claiming "version X isn't available." Edge or building from
source via `apk add build-base && wget … && make` is the fallback.

## Common gotchas

- musl ≠ glibc: prebuilt binaries from generic Linux releases often
  segfault. Look for an "Alpine" or "musl" or "static" download variant.
- Some packages are split: `python3` and `py3-pip` install separately;
  `go` includes the toolchain but `gopls` is a separate package.
- `apk` exits 0 on "already installed" — re-running an install is a
  cheap idempotent way to confirm presence.
