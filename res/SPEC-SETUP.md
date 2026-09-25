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
6. Architecture, because every item will be checked by a test: keep behaviour (state, rules, data formats, flow steps) in plain testable code, and keep the UI layer thin, only rendering state and forwarding user actions. Logic inside a UI callback cannot be tested and will never count as done. The wire itself is tested too: every interactive widget gets a stable name (`set_widget_name`, from the spec's label), a test can build a screen, find a widget by name and fire its action (GTK: `emit_by_name`, `activate_action`), and the window's construction is a function tests can call, not code that only runs in `main`. Put one such widget test next to the smoke test so every later round has the pattern to copy.
7. If the target is a desktop GUI, make its screens visible from inside this container. There is no host display here, and there never will be: everything runs in the container.
   - Install a virtual display (`xvfb-run`), fonts and an icon theme. Without fonts and icons, screenshots come out with empty boxes where text and icons belong. Persist the installs in `.devcontainer/Dockerfile`.
   - Render without a GPU: set `GSK_RENDERER=cairo` for GTK 4 (the default GL renderer needs a graphics driver the container does not have).
   - Add a snapshot entry point to the app: `--snapshot <screen> --project <fixture dir> --out <file.png>` builds the window in that screen's state from a fixture project, waits for the first frame, renders the window with the toolkit's own API (GTK 4: a `gtk::WidgetPaintable` of the window, rendered with its renderer's `render_texture`, then `save_to_png`), and exits. It takes an optional `--do <widget-name>` that fires that widget's action before rendering, so a screen can be looked at after a click, not only at rest. Name the screens after the spec's image files (`03-window`, `05-cut`, ...), so each image in the spec has a reproducible counterpart.
   - Add a `snapshot` recipe to the `justfile` that runs it under `xvfb-run -a` with `GSK_RENDERER=cairo` and writes to `{{out_dir}}/shots/<screen>.png`.
   - Prove it: render one screen, even if it is still empty, and look at it with `screenshot path={{out_dir}}/shots/<screen>.png`.

## When this round counts as done

codehalter checks it itself afterwards: a test command is found for `{{out_dir}}/`, at least one test source exists there, and the test command passes.

{{previous}}
