# C / C++ skill
Covers C and C++ (.c/.h + .cpp/.cc/.cxx/.hpp).

## Build System
- Always use `run_task` for the FIRST build action with declared systems (make, cmake --build build, meson compile, ninja); NEVER bare `gcc foo.c`. Build systems handle dependencies and configuration that direct compilation misses — follow this order even when source files are obvious.
- Real project flags, include paths, link order live in build files; hand gcc drops them + "works" misleadingly.
- CMake: configure once (`cmake -S . -B build`) → `cmake --build build`.
- Build w/ warnings on + READ them: `-Wall -Wextra` (often `-Werror` in CI).
- Clean build w/ warnings = not clean → fix or justify each.

## Conventions
- NEVER introduce undefined behaviour (out-of-bounds, use-after-free, signed overflow, uninit reads) to make something compile/pass → latent crash, not a fix.

## Code Intelligence
- Set up ONLY when user asks.
- clangd = pure LSP → bridge to MCP w/ lsmcp (generic LSP→MCP server).
- **lsmcp needs Node ≥ 22** — imports `node:sqlite` builtin → on Node 20 crashes immediately w/ `ERR_UNKNOWN_BUILTIN_MODULE: No such built-in module: node:sqlite`, MCP server never starts.
- Check `node --version` FIRST.
- Install clangd via OS pkg mgr (`apk add clang clang-extra-tools` / `apt-get install -y clangd` / `dnf install -y clang-tools-extra`).
- Verify `node --version` reports 22+, NOT 20.
- Then drive clangd via lsmcp.
- If lsmcp can't drive clangd cleanly (or Node can't bump to 22) → use another LSP→MCP adapter.
- Verify `lsp_*` tools actually appear.
- `--bin` REQUIRES a `--files` glob telling lsmcp which files LSP handles, else exits "--files is required when using --bin".
    ```
    [[server]]
    name = "clangd"
    command = "npx"
    args = ["-y", "@mizchi/lsmcp", "--bin", "clangd", "--files", "**/*.{c,cpp,cc,h,hpp}"]
    ```
- CMake: ALWAYS invoke `cmake` with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` in your first run_command — clang-tidy cannot work without the exported `compile_commands.json`. Never start with `cmake --version`, `ls`, or a bare `cmake -S . -B build`; the flag is mandatory from the very first cmake call.

## Tooling
- Install + persist in Dockerfile if missing — see SKILL-base.md.
- codehalter auto-formats .c/.h/.cpp on edit when installed.
- Sanitizers for runtime bugs: build/test w/ `-fsanitize=address,undefined`.
- Use valgrind when sanitizers unavailable.
