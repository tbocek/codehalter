# Go bootstrap — Fedora

The devcontainer is already running. Install gopls (required: codehalter's
`gopls mcp` exposes go_symbols / go_references / go_definition / go_hover as
MCP tools), persist it in .devcontainer/Dockerfile, then verify.

## 1. Install live in this container

Run via the run_command tool:

```
sudo dnf install -y gopls
```

## 2. Persist in the Dockerfile

Append before the `USER ${USERNAME}` line in .devcontainer/Dockerfile:

```
RUN dnf install -y --setopt=install_weak_deps=False gopls && dnf clean all
```

(That section runs as root, so no `sudo` in the Dockerfile.)

## 3. Verify

Run via the run_command tool:

```
gopls version
```

Expected: a version string like `golang.org/x/tools/gopls v0.x.y`. If not on
PATH, the install failed — investigate before reporting done.
