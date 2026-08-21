# JavaScript skill
- ESM (`import`/`export`) for new code. CommonJS `require` ONLY when matching the existing file's style — check the file before writing either.
- Optional chaining (`?.`) + nullish coalescing (`??`) for defaults and safe access, not `&&` chains or `x === undefined` branches.
- Every promise rejection handled: `try`/`catch` around `await`, or an attached `.catch`. No fire-and-forget promises.
- Match the project's existing linter (eslint, biome). Don't change `.editorconfig`/lint config unless asked.

## Package manager — ONE per project, detect BEFORE installing
`package.json` `packageManager` field → lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, else npm) → an existing `node_modules/.pnpm` dir (→pnpm; the lockfile is often gitignored). Use that ONE for everything — never mix npm into a pnpm/yarn project. pnpm/yarn via `corepack enable`. The base image has no node → install `nodejs npm` + persist FIRST. pnpm drops `.pnpm-store` in the repo in a devcontainer → gitignore `.pnpm-store` + `node_modules`.

## Code intelligence over MCP — lsmcp (gopls analog)
`@mizchi/lsmcp` gives real code tools (`lsp_get_definitions`, `lsp_find_references`, `lsp_get_hover`, `lsp_get_diagnostics`, `search_symbols`) for JS too. Set up ONLY when the user asks.
1. `<pm> add -D @mizchi/lsmcp @typescript/native-preview` with the PROJECT'S package manager (NOT `-g`).
2. `npx @mizchi/lsmcp init -p tsgo` → generates `.lsmcp/config.json`.
3. Add to `.codehalter/mcp.toml`:
[[server]]
name = "lsmcp"
command = "npx"
args = ["-y", "@mizchi/lsmcp", "-p", "tsgo"]
codehalter reconciles mcp.toml at turn end; the tools go live next prompt.
