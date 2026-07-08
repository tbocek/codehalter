# JavaScript skill
## Variables
 `let` for reassign.

## Equality
 Never `==`/`!=` (silent type coercion).

## Modules
- ESM (`import`/`export`) for new code. CommonJS `require` ONLY when matching existing file's style.

## Idioms
- Optional chaining (`?.`) + nullish coalescing (`??`) for safe access — not `&&` chains.

## Style
- Match project's existing linter (eslint, biome). Don't change `.editorconfig`/lint config unless asked.

## Code intelligence over MCP — lsmcp
`@mizchi/lsmcp` gives model real code tools (`lsp_get_definitions`, `lsp_find_references`, `lsp_get_hover`, `lsp_get_diagnostics`, `search_symbols`) for JS too — LSP→MCP server.
ONE package manager per project, detect BEFORE installing: check `package.json` `packageManager` field → then lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, else npm) → then existing `node_modules/.pnpm` dir (→pnpm; lockfile often gitignored). pnpm/yarn via `corepack enable`.
Setup (tsgo backend — fast, handles JS):
1. Add as project devDeps with PROJECT'S package manager: `<pm> add -D @mizchi/lsmcp @typescript/native-preview` — e.g. `pnpm add -D …` for pnpm project (NOT `-g`).
2. `npx @mizchi/lsmcp init -p tsgo` → generates `.lsmcp/config.json`.
3. Add to `.codehalter/mcp.toml`:
[[server]]
name = "lsmcp"
command = "npx"
args = ["-y", "@mizchi/lsmcp", "-p", "tsgo"]
codehalter reconciles mcp.toml at turn end;