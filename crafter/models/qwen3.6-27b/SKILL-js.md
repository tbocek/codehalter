# JavaScript skill
## Variables
- `const` by default. Use `let` for any variable that will be reassigned — even when a functional pattern like `reduce` could technically avoid reassignment. The loop-based reassignment pattern (declare with `let`, reassign inside the loop) is the intended approach; do not sidestep `let` by passing the accumulator through `reduce` or any other functional combinator. Use `const` only for values that are truly never reassigned after initialization.

## Modules
 CommonJS `require` ONLY when matching existing file's style.

## Idioms
- Destructure objects/arrays: `const { x, y } = obj`.
- Optional chaining (`?.`) + nullish coalescing (`??`) for safe access — not `&&` chains.

## Style
- Match project's existing linter (eslint, biome). Don't change `.editorconfig`/lint config unless asked.

## Code intelligence over MCP — lsmcp (gopls analog)
`@mizchi/lsmcp` gives model real code tools (`lsp_get_definitions`, `lsp_find_references`, `lsp_get_hover`, `lsp_get_diagnostics`, `search_symbols`) for JS too — LSP→MCP server. Set up ONLY when user asks.
ONE package manager per project, detect BEFORE installing: check `package.json` `packageManager` field → then lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, else npm) → then existing `node_modules/.pnpm` dir (→pnpm; lockfile often gitignored). Use that ONE for everything — never mix npm into pnpm/yarn project, don't `npm install` then fall back. pnpm/yarn via `corepack enable`. pnpm drops `.pnpm-store` in repo in devcontainer → gitignore `.pnpm-store` + `node_modules`.
Setup (tsgo backend — fast, handles JS):
1. Add as project devDeps with PROJECT'S package manager: `<pm> add -D @mizchi/lsmcp @typescript/native-preview` — e.g. `pnpm add -D …` for pnpm project (NOT `-g`).
2. `npx @mizchi/lsmcp init -p tsgo` → generates `.lsmcp/config.json`.
3. Add to `.codehalter/mcp.toml`:
[[server]]
name = "lsmcp"
command = "npx"
args = ["-y", "@mizchi/lsmcp", "-p", "tsgo"]
codehalter reconciles mcp.toml at turn end;