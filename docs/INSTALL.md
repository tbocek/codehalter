## Installing tools (reference for this fix)

Installs you run live persist only until the next rebuild — so EVERY install must
end with a matching `.devcontainer/Dockerfile` edit, or it vanishes and the next
session re-breaks.

### The loop for any missing tool

1. Install it live with the OS package manager (apk/apt/dnf — see SKILL-<os>.md),
   then verify (`<tool> --version`).
2. **Persist**: add the exact verified install as a `RUN` line in
   `.devcontainer/Dockerfile`. Self-check it landed (`grep` the Dockerfile).
3. In `respond`, tell the user to rebuild the container so the Dockerfile change
   takes effect (the live install only carries this session).

### Language runtimes come first

A formatter / LSP / MCP for a language needs that language's runtime, which the
base image may NOT have. Node is the usual gap: on a fresh JS/TS project the base
has no `node`/`npm`. Install the runtime FIRST (`apk add nodejs npm` /
`apt-get install -y nodejs npm`), persist it, THEN install the JS tool — don't
`npm install ...` before npm exists.

### Use the project's package manager — don't assume npm

Check the lockfile and match it; mixing managers corrupts `node_modules`:

- `pnpm-lock.yaml` → **pnpm**   · `yarn.lock` → **yarn**
- `bun.lockb` → **bun**         · `package-lock.json` (or none) → **npm**

Get pnpm/yarn via **corepack** (ships with node) — `corepack enable` — not
`npm install -g pnpm`. Then use it for everything: `pnpm add -D <pkg>`,
`pnpm install`. Persist `corepack enable` in the Dockerfile too. Persist the RIGHT
manager: a pnpm project's Dockerfile runs `pnpm install`, never `npm install`
(that fights the pnpm lockfile).

### pnpm inside a container

Don't point `store-dir` at a host path (`/home/<hostuser>/...` — that path
doesn't exist for the container user and the install fails). Leave pnpm's default
store, or set it under the container home (`/home/dev/.local/share/pnpm`). If
`node_modules` is half-built from a failed npm attempt, `rm -rf node_modules` and
reinstall clean with the right manager.
