Missing dev tools in this %s devcontainer: %s.

PLAN ONLY — do not install anything yourself. Produce execute-phase steps, in this order, for each tool:

1. Install it with the right package manager: `apk`/`apt`/`dnf` for system tools and Go/Rust LSPs (or `go install`); for JS/TS tools (prettier) and Python tools (ruff) use the PROJECT'S package manager / environment per the install guidance below — match the lockfile, never default to npm when the project uses pnpm/yarn. Install JS tools as project devDeps so the whole project stays on its ONE package manager.
2. Verify it runs (e.g. `<tool> --version`).
3. Persist by editing `.devcontainer/Dockerfile`.
4. If the tool is MCP-capable (e.g. gopls via `gopls mcp`), add a `[[server]]` entry to `.codehalter/mcp.toml`.

Verify-phase checks: each tool runs (global tools on PATH; a JS devDep via `node_modules/.bin/<tool> --version`), and the install is persisted (a Dockerfile `RUN` line, and for JS tools the devDep is in `package.json`).
