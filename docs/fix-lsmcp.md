This TS/JS project has no code-intelligence MCP configured (the gopls analog).

PLAN ONLY — do not run anything yourself. Follow the install guidance below (install node first if missing, use the project's package manager, persist installs in the Dockerfile). Produce execute-phase steps:

1. Ensure node + the project's package manager are present — match the lockfile (`pnpm-lock.yaml`→pnpm, `yarn.lock`→yarn, else npm; get pnpm/yarn via `corepack enable`).
2. Add devDeps with THAT manager, e.g. `pnpm add -D @mizchi/lsmcp @typescript/native-preview`.
3. `npx @mizchi/lsmcp init -p tsgo` to write `.lsmcp/config.json`.
4. Add an ACTIVE lsmcp `[[server]]` to `.codehalter/mcp.toml` (command=`"npx"`, args=`["-y", "@mizchi/lsmcp", "-p", "tsgo"]`) — the file has a commented template; uncomment the WHOLE block INCLUDING the `[[server]]` header line. A common mistake is uncommenting name/command/args but leaving `# [[server]]` commented — then they're orphan top-level keys and the server never loads.
5. Persist node + the package-manager install in `.devcontainer/Dockerfile`. See SKILL-ts.md.

Verify: read `.codehalter/mcp.toml` back and confirm the `[[server]]` line above `name = "lsmcp"` is NOT commented.
