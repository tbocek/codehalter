Missing dev tools in this %s devcontainer: %s.

PLAN ONLY — do not install anything yourself. Produce execute-phase steps, in this order, for each tool:

1. Install it with the right package manager — the OS one for system tools; npm/pip/etc. for language formatters like prettier/ruff (the SKILL files cover which).
2. Verify it runs (e.g. `<tool> --version`).
3. Persist by editing `.devcontainer/Dockerfile`.
4. If the tool is MCP-capable (e.g. gopls via `gopls mcp`), add a `[[server]]` entry to `.codehalter/mcp.toml`.

Verify-phase checks: each tool on PATH, Dockerfile contains the persist line.
