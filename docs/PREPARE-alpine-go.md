# Go prepare — Alpine Linux

Alpine uses musl libc, not glibc. gopls itself is pure Go and unaffected, but
heads-up if you later add cgo-based Go tooling or pull glibc-only Go
dependencies: you'll need `apk add build-base` plus possibly `gcompat` for
the glibc-shim, and some binaries may simply not work under musl. Most Go
projects assume glibc by default, so this is the most common Alpine gotcha.

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
sudo apk add --no-cache gopls
```

## 2. Persist in the Dockerfile

Append before the `USER ${USERNAME}` line in .devcontainer/Dockerfile:

```
RUN apk add --no-cache gopls
```

(That section runs as root, so no `sudo` in the Dockerfile.)

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
