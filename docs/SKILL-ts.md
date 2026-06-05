# TypeScript skill

## Types
- Strict null checks are on. `T | undefined` is NOT the same as `T`.
- Use `?.` and `??` for safe access. Use `!` only when you can prove non-null.
- `interface` for object shapes, `type` for unions and aliases.
- Avoid `any`. Prefer `unknown` and narrow with type guards. Reach for generics before `any`.
- Don't use the non-standard `Object` or `Function` types.

## Async
- `async`/`await`, not `.then()` chains.
- Always handle promise rejection (`try`/`catch` or pass error up).
- Don't fire-and-forget promises — `await` or assign to a variable handled later.

## Modules
- ESM imports: `import { x } from "./y"`. Avoid CommonJS `require` in new code.
- Prefer named exports over default exports.
- Imports go: built-ins → third-party → local, with blank lines between groups.

## Idioms
- `const` for everything except locals you reassign; `let` for those; never `var`.
- Use destructuring for objects and arrays.
- Triple-equals `===` always. Never `==`.
- Prefer immutable updates (`{...obj, field: x}`, `[...arr, x]`) over mutation.

## Tooling
- ONE package manager per project, detected BEFORE installing — check `package.json`'s `packageManager` field, then the lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, `bun.lockb`→bun, else npm), then an existing `node_modules/.pnpm` dir (→pnpm; the lockfile is often gitignored, so this is the real signal). Use that ONE for EVERYTHING (formatter, lsmcp, scripts). Never mix npm into a pnpm/yarn project, and don't `npm install` then fall back. Get pnpm/yarn via `corepack enable`. These tools need node — if the base image has none, install `nodejs npm` (OS package manager) and persist it FIRST. pnpm in a devcontainer drops a `.pnpm-store` in the repo (its store can't hardlink across the bind-mount) — gitignore `.pnpm-store` + `node_modules`.
- Type-check and lint through the project's task runner — its package-manager run (`pnpm run`/`npm run`/`yarn`, matching the lockfile) `typecheck`/`build`/`lint` or whatever the `package.json` scripts declare. Don't call `tsc` or eslint directly.

## Code intelligence over MCP — lsmcp (the gopls analog)
`@mizchi/lsmcp` is an LSP→MCP server: it gives the model real code-intelligence
tools — `lsp_get_definitions`, `lsp_find_references`, `lsp_get_hover`,
`lsp_get_diagnostics`, `lsp_rename_symbol`, plus `get_project_overview` /
`search_symbols`. Set it up ONLY when the user asks for TS code intelligence.

Setup (tsgo backend — fast native TS):
1. Add as project devDeps with the PROJECT'S package manager (the one matching
   the lockfile, see Tooling above): `<pm> add -D @mizchi/lsmcp @typescript/native-preview`
   — e.g. `pnpm add -D …` for a pnpm project (persisted in package.json; NOT `-g`).
2. `npx @mizchi/lsmcp init -p tsgo` — generates `.lsmcp/config.json`.
3. Add the MCP server to `.codehalter/mcp.toml`:
   ```
   [[server]]
   name = "lsmcp"
   command = "npx"
   args = ["-y", "@mizchi/lsmcp", "-p", "tsgo"]
   ```
   codehalter reconciles `mcp.toml` at the end of the turn — the `lsp_*` /
   `search_symbols` tools go live next prompt. No restart, no new session.

Stable alternative to the `@typescript/native-preview` (tsgo) preview: drive the
classic LSP instead — `<pm> add -D typescript-language-server typescript` (project's manager) and
point lsmcp at it via `--bin` in place of `-p tsgo`.
