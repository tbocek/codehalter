# Monorepo prepare

This project mixes more than one stack at the repository root (e.g. a Go
backend with a TypeScript frontend, or a Rust crate beside Python tooling).
The per-stack PREPARE sections below run in order — handle each one in turn,
asking the user before starting a fresh install.

When more than one stack would touch `.devcontainer/Dockerfile`, write all
the `RUN` lines in a single Dockerfile edit (collapsed into one apt/apk/dnf
invocation when the package manager is the same) rather than re-editing the
file per stack. Same for `.codehalter/mcp.toml` — append every new `[[server]]`
entry in one edit_file call so each install only causes one container-state
delta the user needs to review.
