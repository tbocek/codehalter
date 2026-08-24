# TypeScript skill
- Strict null checks ON: `T | undefined` is NOT `T`. Use `!` ONLY when you can prove non-null.
- Prefer named exports; ESM imports, not `require`, in new code.
- Always braces, even on a one-line `if`/`for`: without them a line added to the body later silently falls outside it (goto fail).

## Package manager — ONE per project, detect BEFORE installing
`package.json` `packageManager` field → lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, `bun.lockb`→bun, else npm) → an existing `node_modules/.pnpm` dir (→pnpm; the lockfile is often gitignored, so this is the real signal). Use that ONE for EVERYTHING (formatter, lsmcp, scripts). Never mix npm into a pnpm/yarn project, never `npm install` as a fallback. Get pnpm/yarn via `corepack enable`.
The base image has no node → install `nodejs npm` (OS pkg mgr) + persist FIRST. pnpm in a devcontainer drops `.pnpm-store` in the repo (the store can't hardlink across the bind mount) → gitignore `.pnpm-store` + `node_modules`.

## Type-check + lint through the project's scripts
Its package-manager run (`pnpm run`/`npm run`/`yarn`, matching the lockfile) of `typecheck`/`build`/`lint`, whatever `package.json` declares. Don't reach for `npx tsc --noEmit` or eslint directly when a script exists.

## Code intelligence over MCP — lsmcp (gopls analog)
`@mizchi/lsmcp` = LSP→MCP server: `lsp_get_definitions`, `lsp_find_references`, `lsp_get_hover`, `lsp_get_diagnostics`, `lsp_rename_symbol`, plus `get_project_overview`/`search_symbols`. Set up ONLY when the user asks.
1. Add as project devDeps with the PROJECT'S package manager: `<pm> add -D @mizchi/lsmcp @typescript/native-preview` (NOT `-g`).
2. `npx @mizchi/lsmcp init -p tsgo` → generates `.lsmcp/config.json`.
3. Add to `.codehalter/mcp.toml`:
[[server]]
name = "lsmcp"
command = "npx"
args = ["-y", "@mizchi/lsmcp", "-p", "tsgo"]
codehalter reconciles mcp.toml at turn end — the `lsp_*`/`search_symbols` tools go live next prompt. No restart, no new session.
Stable alternative to the `@typescript/native-preview` (tsgo) preview: `<pm> add -D typescript-language-server typescript` and point lsmcp at it via `--bin` in place of `-p tsgo`.
