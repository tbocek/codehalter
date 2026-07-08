# JavaScript skill
## Variables
- `const` by default. `let` for reassign. Never `var`.
- Always declare; no implicit globals.

## Equality
- Use `===`/`!==`. Never `==`/`!=` (silent type coercion).

## Async
- `async`/`await` over `.then()` chains.
- Always handle promise rejection → `try`/`catch` around `await`, or attach `.catch`.
- No fire-and-forget promises.

## Modules
- ESM (`import`/`export`) for new code. CommonJS `require` ONLY when matching existing file's style.
- Imports: built-ins → third-party → local, blank lines between groups.

## Idioms
- Destructure objects/arrays: `const { x, y } = obj`.
- Spread for immutable updates: `[...arr, x]`, `{...obj, field: x}`. Avoid mutating shared state.
- Optional chaining (`?.`) + nullish coalescing (`??`) for safe access — not `&&` chains.
- Arrow functions for callbacks; `function` for top-level defs when hoisting matters.

## Style
- Match project's existing linter (eslint, biome). Don't change `.editorconfig`/lint config unless asked.
- No JSDoc unless rest of file uses it.

## Code intelligence over MCP — lsmcp (gopls analog)
`@mizchi/lsmcp` gives model real code tools (`lsp_get_definitions`, `lsp_find_references`, `lsp_get_hover`, `lsp_get_diagnostics`, `search_symbols`) for JS too — LSP→MCP server. Set up ONLY when user asks.
ONE package manager per project, detect BEFORE installing: check `package.json` `packageManager` field → then lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, else npm) → then existing `node_modules/.pnpm` dir (→pnpm; lockfile often gitignored). Use that ONE for everything — never mix npm into pnpm/yarn project, don't `npm install` then fall back. pnpm/yarn via `corepack enable`. These tools need node — base image has none → install `nodejs npm` + persist FIRST. pnpm drops `.pnpm-store` in repo in devcontainer → gitignore `.pnpm-store` + `node_modules`.
Setup (tsgo backend — fast, handles JS):
1. Add as project devDeps with PROJECT'S package manager: `<pm> add -D @mizchi/lsmcp @typescript/native-preview` — e.g. `pnpm add -D …` for pnpm project (NOT `-g`).
2. `npx @mizchi/lsmcp init -p tsgo` → generates `.lsmcp/config.json`.
3. Add to `.codehalter/mcp.toml`:
[[server]]
name = "lsmcp"
command = "npx"
args = ["-y", "@mizchi/lsmcp", "-p", "tsgo"]
codehalter reconciles mcp.toml at turn end; tools go live next prompt.