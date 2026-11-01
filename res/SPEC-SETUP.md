# Spec setup: `{{out_dir}}/`

A specification in `{{spec_dir}}/` ({{items}}) will be implemented one item per round. This round gives `{{out_dir}}/` a project that builds and runs tests. It implements NO spec item.

## Target

{{target}}

The spec under `{{spec_dir}}/` is READ-ONLY. Read only its README or index and the standing rules named below.

{{context}}

## Do

1. Create the project skeleton in `{{out_dir}}/` for the target: build manifest, source layout, a library crate or module for the behaviour and a thin entry point for the UI.
2. Install the toolchain the target needs inside the container (compiler, the libraries' development packages, anything tests need at runtime) and persist every install in `.devcontainer/Dockerfile` (see SKILL-base.md).
3. Make the tests runnable from `{{out_dir}}/`. codehalter runs a `test` recipe in `{{out_dir}}/justfile`, else `cargo test`, `npm test` or `go test ./...` when the matching manifest is there. If tests need something special (`xvfb-run -a cargo test`, an environment variable), write the recipe.
   - Compiled language: build tests with line tables only (Rust: `[profile.dev] debug = "line-tables-only"`; C and C++: `-gline-tables-only` with clang, `-g1` with GCC).
   - Where every test file builds its own program (Rust's `tests/*.rs`, a CMake executable per test), make ONE test program with the files as modules (Rust: `tests/all.rs` declaring the shared test module and a `suite` module, one `mod` line per file in `tests/suite/mod.rs`).
   - GUI tests bound to one thread (GTK) run on it through the toolkit's helper (gtk4-rs: `#[gtk::test]`) and build the screen they check, without starting the application.
   - The `test` recipe runs test files side by side, one process per file, at most 16 at a time (fewer with fewer cores). Each GUI worker gets its own virtual display (`xvfb-run -a` around each worker, not the whole run).
4. Build output (such as `target/`) must be ignored by git, and sources, manifests and tests under `{{out_dir}}/` must NOT be. Check with `git check-ignore -v <path>` and fix `.gitignore`.
5. Write one passing smoke test. Beside it, a shared test module with a temporary directory per test, a fake server helper that binds port 0 and returns the port, and one wait helper: it checks a condition, waits briefly, checks again, and fails after a wall-clock deadline with a message naming what never happened. Tests use it, never a fixed sleep or a fixed number of event-loop turns.
6. Keep behaviour (state, rules, data formats, flow steps) in plain testable code and the UI thin: logic in a UI callback cannot be tested. Every interactive widget gets a stable name (`set_widget_name`, from the spec's label). A test can build a screen, find a widget by name and fire its action (GTK: `emit_by_name`, `activate_action`), and the window's construction is a function tests can call, not code only in `main`. Add one such widget test beside the smoke test as the pattern to copy.
7. If the target is a desktop GUI, make its screens visible headless. Expose the host's Wayland and sound sockets as the base skill's GUI section says; tests and snapshots never touch them.
   - Run tests and snapshots with `env -u WAYLAND_DISPLAY GDK_BACKEND=x11 xvfb-run -a`, in the recipe. Install `xvfb-run`, fonts and an icon theme, or screenshots show empty boxes. Persist the installs in `.devcontainer/Dockerfile`.
   - Render without a GPU: `GSK_RENDERER=cairo` for GTK 4.
   - Add a snapshot entry point, in debug builds only: `--snapshot <screen> --project <fixture dir> --out <file.png>` builds the window in that screen's state from a fixture, takes the picture when ready, renders the window with the toolkit's API (GTK 4: a `gtk::WidgetPaintable` of the window, rendered with its renderer's `render_texture`, then `save_to_png`), and exits.
   - Ready is an event the app raises, never a timer, sleep or frame count. Every setup step that needs its widget to have a size (scrolling a list, placing a divider, the window taking its size, background data) registers as owed when scheduled and done when run. The screen is ready at the first painted frame (GTK 4: the frame clock's `after-paint`) where no shown widget owes a step. `--shot-on ready` (the default) takes the picture then; `--shot-on open:<widget-name>` also waits until the named menu or popover is shown.
   - At ready, record every shown widget's name and bounds, and take the picture one painted frame later only if none moved. If one moved, or the screen is not ready after a bound such as 600 frames, exit non-zero naming the widget or the owed step, and write no picture. Turn animations off (GTK 4: `gtk-enable-animations`), and draw anything that changes by itself (a clock, a random value, the user's name) from fixed values in snapshot mode.
   - An optional `--do <widget-name>` fires that widget's action before rendering. A menu, popover or dialog is its own surface, not in the window's paintable: render the window, then draw the open surface's own paintable onto it where it opens, at its own size. Name screens after the spec's image files (`03-window`, `05-cut`).
   - Add a `snapshot` recipe that runs it under `xvfb-run -a` with `GSK_RENDERER=cairo` and writes `{{out_dir}}/shots/<screen>.png`. Each run works in a directory at a fixed path per screen (such as `/tmp/<app>-shot/<screen>`), emptied and given a fresh fixture copy first, and held with `flock`.
   - Prove it: render one screen twice and compare the files with `cmp`: they must be identical. Then look with `screenshot path={{out_dir}}/shots/<screen>.png`. codehalter renders each screen twice before its picture check, and if they differ the next round fixes the snapshot first.
8. If the target is a web page or app, add a `snapshot` recipe that starts the app (or serves the built files), opens `<screen>`'s page in a headless browser with Playwright or a tool that can wait for a condition, writes `{{out_dir}}/shots/<screen>.png` with `page.screenshot`, and stops the app. The app sets an attribute (such as `data-ready` on `<body>`) once its data loaded and the page is laid out; the recipe waits for it and `document.fonts.ready`, never a timeout or "network idle". `firefox --headless --screenshot` cannot wait, so it fits only a page complete without scripts. Turn animations and transitions off (Playwright's `reducedMotion: 'reduce'` and a stylesheet with `animation: none; transition: none`), use fixture data, name screens after the spec's image files, and prove it twice with `cmp`.

## When this round counts as done

codehalter checks that a test command is found for `{{out_dir}}/`, a test source exists there, and the test command passes.

{{previous}}
