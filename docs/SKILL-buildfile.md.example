# Buildfile skill

When you need to build, test, lint, or format the project, USE the project's declared entry points. Don't invent ad-hoc toolchain commands.

## Use `run_task` first

Codehalter detects task runners on session start and exposes them via the `run_task` tool. The session-startup banner lists which runners and which `build`/`test`/`lint`/`format` targets are wired up. Call `run_task` with `runner:target` (e.g. `just:test`, `make:build`, `npm:lint`) — that's the canonical way to drive the project.

If a target you need is missing (e.g. there's no `lint` target), tell the user before adding one. Don't silently introduce a new build file.

## Where the targets live (in order of preference)

1. `Makefile` / `makefile` / `GNUmakefile` — `make <target>`
2. `justfile` / `Justfile` — `just <recipe>` (run `just --list` to enumerate)
3. `Taskfile.yml` — `task <name>`
4. `package.json` `scripts` block — `npm run <script>` (or `pnpm`/`yarn` matching the lockfile)
5. `Cargo.toml` — `cargo build`, `cargo test`, `cargo clippy`, `cargo fmt`
6. `build.gradle` / `pom.xml` — `./gradlew <task>` / `mvn <goal>`

## Rules

- If `make test` is defined, run that — not `go test ./...`, `pytest`, `npm test` directly, etc.
- After modifying source, run the project's `build` and `test` targets. That's the truth check, not type-checker output alone.
- Don't add new build files. If the project uses `make`, don't introduce a `justfile` alongside.
- Don't reformat the world. Run the project's existing `format` target if you need consistent style; otherwise leave formatting to the user.

## Devcontainers

Inside a devcontainer the toolchain often isn't on the host PATH — the build entry points are always wired correctly inside the container. Always go through `make`/`just`/`npm`/etc. rather than calling compilers directly. If a command fails because a tool is missing, you're probably outside the container; ask the user instead of guessing an install command.
