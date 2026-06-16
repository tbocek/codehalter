Go project has no code-intelligence MCP (gopls) wired — without it the model navigates with search_text/read_file instead of go_definition/go_references.

PLAN ONLY → execute-phase steps: wire gopls as an MCP server per SKILL-go.md ("gopls as MCP server"); if gopls isn't installed, install it first (also per SKILL-go.md). Persist any install in `.devcontainer/Dockerfile`.
