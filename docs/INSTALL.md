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

### Detect the project's package manager FIRST — don't assume npm

Before ANY install, determine the project's manager and use that ONE for
everything (mixing corrupts `node_modules` — npm chokes on a pnpm `node_modules`
with `Cannot read properties of null`). Check, in this order:

1. `packageManager` field in `package.json` (e.g. `"packageManager": "pnpm@9"`) — authoritative.
2. Lockfile: `pnpm-lock.yaml`→**pnpm** · `yarn.lock`→**yarn** · `bun.lockb`→**bun** · `package-lock.json`→**npm**.
3. An already-installed `node_modules`: a `node_modules/.pnpm` dir → **pnpm** (its virtual store), `node_modules/.yarn` → yarn. The committed lockfile is often gitignored, so this is frequently the ONLY signal — check it.
4. None of the above → **npm**.

Do NOT just run `npm install` and fall back when it errors — that wastes a failed
attempt and can leave a broken `node_modules`. Get pnpm/yarn via **corepack**
(ships with node) — `corepack enable` — not `npm install -g pnpm`. Then use that
manager for everything: `<pm> add -D <pkg>`, `<pm> install`. Persist the RIGHT
manager in the Dockerfile (a pnpm project runs `pnpm install`, never `npm
install`), plus `corepack enable`.

### pnpm inside a container

pnpm hardlinks packages from a content-addressable store into `node_modules`, so
the store must be on the SAME filesystem as the project. In a devcontainer the
workspace is bind-mounted from the host while pnpm's default store
(`~/.local/share/pnpm`) is on the container's overlay — a different filesystem —
so pnpm drops a project-local store at `<repo>/.pnpm-store`. That's expected, but
it's a CACHE, not source: ensure `.pnpm-store` and `node_modules` are in
`.gitignore` (add them if missing) so they're never committed.

Don't point `store-dir` at a host path (`/home/<hostuser>/...` — doesn't exist
for the container user). If `node_modules` is half-built from a failed npm
attempt, `rm -rf node_modules` and reinstall clean with the right manager.
