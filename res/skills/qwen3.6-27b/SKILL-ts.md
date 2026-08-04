# TypeScript skill
## Types
 Use `!` ONLY when you can prove non-null.

## Async
- Always handle promise rejection (`try`/`catch` or pass error up).
- No fire-and-forget promises — `await` or assign to var handled later.

## Modules
- ESM imports: `import { x } from "./y"`. Avoid CommonJS `require` in new code.

## Idioms
- Destructure objects + arrays.
 Never `==`.

## Tooling
- ONE package manager per project, detect BEFORE installing.
- Check `package.json` `packageManager` field → lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, `bun.lockb`→bun, else npm) → existing `node_modules/.pnpm` dir (→pnpm; lockfile often gitignored, so this = real signal).
- Use that ONE for EVERYTHING (formatter, lsmcp, scripts).
- Get pnpm/yarn via `corepack enable`.
- These tools need node — base image has none → install `nodejs npm` (OS pkg mgr) + persist FIRST.
- pnpm in devcontainer drops `.pnpm-store` in repo (store can't hardlink across bind-mount) → gitignore `.pnpm-store` + `node_modules`.
- Type-check + lint through project task runner — its pkg-manager run (`pnpm run`/`npm run`/`yarn`, matching lockfile) `typecheck`/`build`/`lint` or whatever `package.json` scripts declare.
- Don't call `tsc` or eslint directly.

## LSP setup
- `@mizchi/lsmcp` = LSP→MCP server: gives model real code-intelligence tools — `lsp_get_definitions`, `lsp_find_references`, `lsp_get_hover`, `lsp_get_diagnostics`, `lsp_rename_symbol`, plus `get_project_overview`/`search_symbols`.
- Setup (tsgo backend — fast native TS):
  1. Add as project devDeps with PROJECT'S package manager (matching lockfile, see Tooling): `<pm> add -D @mizchi/lsmcp @typescript/native-preview` — e.g. `pnpm add -D …` for pnpm project (persisted in package.json; NOT `-g`).
  2. `npx @mizchi/lsmcp init -p tsgo` → generates `.lsmcp/config.json`.
  3. Add MCP server to `.codehalter/mcp.toml`:
     ```toml
     [[server]]
     name = "lsmcp"
     command = "npx"
     args = ["-y", "@mizchi/lsmcp", "-p", "tsgo"]
     ```
- codehalter reconciles mcp.toml at turn end — `lsp_*`/`search_symbols` tools go live next prompt. No restart, no new session.
- Stable alternative to `@typescript/native-preview` (tsgo) preview: drive classic LSP — `<pm> add -D typescript-language-server typescript` (project's manager) + point lsmcp at it via `--bin` in place of `-p tsgo`.
