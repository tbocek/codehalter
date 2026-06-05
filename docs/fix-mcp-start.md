The MCP server %q in `.codehalter/mcp.toml` failed to start: %s.

Inspect the `[[server]]` entry, confirm the command is on PATH, and check any required args / env. If the server's binary is missing, install it via run_command, verify it runs by hand, then persist the install in `.devcontainer/Dockerfile` so it survives a container rebuild.
