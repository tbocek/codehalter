# Go prepare — Debian

A Go project was detected and `gopls` isn't on PATH. Ask the user (via the
ask_user tool) whether to install it — codehalter's `gopls mcp` exposes
go_symbols / go_references / go_definition / go_hover as MCP tools, which the
agent uses for navigation. If the user declines, append `"go"` to the
`declined` list in `.codehalter/prepare-state.toml` (create the file with
`declined = ["go"]` if it doesn't exist) and stop — do not run any install
commands. If they accept, install live, persist in the Dockerfile, wire MCP,
then verify.

## 1. Install live in this container

Run via the run_command tool:

```
sudo GOBIN=/usr/local/bin go install golang.org/x/tools/gopls@latest
```

## 2. Persist in the Dockerfile

Append before the `USER ${USERNAME}` line in .devcontainer/Dockerfile:

```
RUN go install golang.org/x/tools/gopls@latest
```

(Dockerfile already exports `GOBIN=/usr/local/bin` and that section runs as
root, so no `sudo` / `GOBIN=` prefix.)

## 3. Wire MCP

Append the gopls server entry to `.codehalter/mcp.toml` so its MCP tools
register on the next prompt. Use edit_file:

```
[[server]]
name = "gopls"
command = "gopls"
args = ["mcp"]
```

Skip this step if an entry with `name = "gopls"` is already present.

## 4. Verify

Run via the run_command tool:

```
gopls version
```

Expected: a version string like `golang.org/x/tools/gopls v0.x.y`. If not on
PATH, the install failed — investigate before reporting done.
