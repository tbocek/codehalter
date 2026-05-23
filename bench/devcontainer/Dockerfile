# Default bench Dockerfile — copy of codehalter/docs/Dockerfile.devcontainer.debian.
# Edit if a test needs extra toolchains (cargo, jdk, python, etc.). Tests that
# need a different image entirely can ship their own bench/devcontainer/ overlay
# and reference it via `devcontainer = "..."` in the test's .toml.
FROM golang:1-trixie

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git sudo \
        just make \
        nodejs npm \
        firefox-esr \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${GID} ${USERNAME} \
    && useradd --uid ${UID} --gid ${GID} -m -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && install -d -o ${UID} -g ${GID} /home/${USERNAME}/.config

ENV GOBIN=/usr/local/bin
# gopls ships go_symbols / go_references etc. as MCP tools (`gopls mcp`).
# codehalter auto-seeds the entry when go.mod is present.
RUN go install golang.org/x/tools/gopls@latest

USER ${USERNAME}
