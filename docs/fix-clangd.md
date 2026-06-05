This C/C++ project has no code-intelligence MCP configured (the gopls/lsmcp analog for C).

PLAN ONLY — do not run anything yourself. Follow the install guidance below (persist installs in the Dockerfile). Produce execute-phase steps:

1. Install `clangd` (the C/C++ LSP) via the OS package manager — e.g. `apk add clang clang-extra-tools`, `apt-get install -y clangd`, `dnf install -y clang-tools-extra`. Verify `clangd --version`.
2. Bridge clangd to MCP with lsmcp (the generic LSP→MCP server codehalter already uses for TS). That needs node, so install node + the project's package manager first if missing (see below). lsmcp drives an arbitrary LSP via `--bin`, so point it at clangd. If lsmcp can't drive clangd cleanly, use another LSP→MCP adapter — verify the `lsp_*` tools actually appear.
3. Add an ACTIVE `[[server]]` named "clangd" to `.codehalter/mcp.toml`: `command = "npx"`, `args = ["-y", "@mizchi/lsmcp", "--bin", "clangd"]`. Uncomment the WHOLE block INCLUDING the `[[server]]` header line — leaving `# [[server]]` commented makes the keys orphan top-level entries and the server never loads.
4. clangd needs `compile_commands.json` to resolve includes — generate it (CMake: `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`; Make: `bear -- make`). Note it in `respond` if it can't be produced.
5. Persist clangd + node + the package-manager install in `.devcontainer/Dockerfile`. See SKILL-c.md.

Verify: read `.codehalter/mcp.toml` back and confirm the `[[server]]` line above `name = "clangd"` is NOT commented, and `clangd` is on PATH.
