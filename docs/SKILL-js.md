# JavaScript skill

## Variables
- `const` by default. `let` for variables you reassign. Never `var`.
- Always declare; no implicit globals.

## Equality
- Use `===` and `!==`. Never `==` or `!=` (silent type coercion).

## Async
- `async`/`await` over `.then()` chains.
- Always handle promise rejection — `try`/`catch` around `await`, or attach `.catch`.
- No fire-and-forget promises.

## Modules
- ESM (`import`/`export`) for new code. Use CommonJS `require` only when matching an existing file's style.
- Imports: built-ins → third-party → local, with blank lines between groups.

## Idioms
- Destructure objects and arrays for clarity: `const { x, y } = obj`.
- Spread for immutable updates: `[...arr, x]`, `{...obj, field: x}`. Avoid mutating shared state.
- Optional chaining (`?.`) and nullish coalescing (`??`) for safe access — not `&&` chains.
- Arrow functions for callbacks; `function` for top-level definitions when hoisting matters.

## Style
- Match the project's existing linter (eslint, biome). Don't change `.editorconfig`/lint config without being asked.
- Don't add JSDoc unless the rest of the file uses it.

## Code intelligence over MCP — lsmcp (the gopls analog)
`@mizchi/lsmcp` gives the model real code tools (`lsp_get_definitions`,
`lsp_find_references`, `lsp_get_hover`, `lsp_get_diagnostics`, `search_symbols`)
for JS too — it's an LSP→MCP server. Set it up ONLY when the user asks.

ONE package manager per project, detected BEFORE installing: check `package.json`'s `packageManager` field, then the lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, else npm), then an existing `node_modules/.pnpm` dir (→pnpm; the lockfile is often gitignored). Use that ONE for everything — never mix npm into a pnpm/yarn project, don't `npm install` then fall back. pnpm/yarn via `corepack enable`.

Setup (tsgo backend — fast, handles JS):
1. Add as project devDeps with the PROJECT'S package manager: `<pm> add -D @mizchi/lsmcp @typescript/native-preview` — e.g. `pnpm add -D …` for a pnpm project (NOT `-g`).
2. `npx @mizchi/lsmcp init -p tsgo` — generates `.lsmcp/config.json`.
3. Add to `.codehalter/mcp.toml`:
   ```
   [[server]]
   name = "lsmcp"
   command = "npx"
   args = ["-y", "@mizchi/lsmcp", "-p", "tsgo"]
   ```
   codehalter reconciles `mcp.toml` at turn end; the tools go live next prompt.
