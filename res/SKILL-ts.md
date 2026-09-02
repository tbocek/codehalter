# TypeScript skill
- Strict null checks ON: `T | undefined` is NOT `T`. Use `!` ONLY when you can prove non-null.
- Prefer named exports; ESM imports, not `require`, in new code.
- Always braces, even on a one-line `if`/`for`: without them a line added to the body later silently falls outside it (goto fail).

## Package manager — ONE per project, detect BEFORE installing
`package.json` `packageManager` field → lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, `bun.lockb`→bun, else npm) → an existing `node_modules/.pnpm` dir (→pnpm; the lockfile is often gitignored, so this is the real signal). Use that ONE for EVERYTHING (formatter, tsgo, scripts). Never mix npm into a pnpm/yarn project, never `npm install` as a fallback. Get pnpm/yarn via `corepack enable`.
The base image has no node → install `nodejs npm` (OS pkg mgr) + persist FIRST. pnpm in a devcontainer drops `.pnpm-store` in the repo (the store can't hardlink across the bind mount) → gitignore `.pnpm-store` + `node_modules`.

## Type-check + lint through the project's scripts
Its package-manager run (`pnpm run`/`npm run`/`yarn`, matching the lockfile) of `typecheck`/`build`/`lint`, whatever `package.json` declares. Don't reach for `npx tsc --noEmit` or eslint directly when a script exists.

## Type errors on every write — tsgo
codehalter type-checks each `.ts`/`.tsx` it writes and appends the errors to that write's result: it speaks LSP to `tsgo` (the native TypeScript compiler) itself. There is NOTHING to wire — no MCP server, no `.codehalter/mcp.toml` entry, no `npx`, no restart.
Two prerequisites, both the project's own:
1. `tsconfig.json` at the project root. Without a project there is nothing to check against, and per-file defaults would report errors the build never would, so codehalter stays silent instead of guessing.
2. `tsgo` resolvable: `<pm> add -D @typescript/native-preview` with the PROJECT'S package manager (NOT `-g`). `node_modules/.bin/tsgo` wins over a global one, so the project pins its own version. Verify with `node_modules/.bin/tsgo --version`.
That's the whole setup. The next write to a `.ts` file carries its type errors back inline.
