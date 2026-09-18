# Spec setup: `{{out_dir}}/`

A specification in `{{spec_dir}}/` ({{items}}) is about to be implemented one item per round. Before the first item, `{{out_dir}}/` needs a project that builds and runs tests. This round sets that up. It implements NO spec item yet.

## Target

{{target}}

The spec is READ-ONLY: codehalter refuses edits under `{{spec_dir}}/`. Read it for orientation only: its README or index, and the standing rules named below.

{{context}}

## Do

1. Create the project skeleton in `{{out_dir}}/` for the target: build manifest, source layout, a library crate or module for the behaviour and a thin entry point for the UI.
2. Install the toolchain the target needs inside the container (compiler, the libraries' development packages, anything tests need at runtime) and persist every install in `.devcontainer/Dockerfile` (see SKILL-base.md). Toolchain config outside the project (for example under `~/.cargo`) is fine.
3. Make the tests runnable from `{{out_dir}}/`. codehalter runs, in this order of preference: a `test` recipe in `{{out_dir}}/justfile`, else `cargo test`, `npm test` or `go test ./...` when the matching manifest is in `{{out_dir}}/`. If tests need anything special, such as a virtual display for GUI tests (`xvfb-run -a cargo test`) or an environment variable, write the `justfile` with a `test` recipe that provides it.
4. Check git: build output (for example `target/`) must be ignored, and sources, manifests and tests under `{{out_dir}}/` must NOT be. Verify with `git check-ignore -v <path>` and fix `.gitignore` where it is wrong.
5. Write one smoke test that passes.
6. Architecture, because every item will be checked by a test: keep behaviour (state, rules, data formats, flow steps) in plain testable code, and keep the UI layer thin, only rendering state and forwarding user actions. Logic inside a UI callback cannot be tested and will never count as done.

## When this round counts as done

codehalter checks it itself afterwards: a test command is found for `{{out_dir}}/`, at least one test source exists there, and the test command passes.

{{previous}}
