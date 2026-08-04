# C / C++ skill
Covers C and C++ (.c/.h + .cpp/.cc/.cxx/.hpp).

## Build System
- Use run_task for declared build (make, cmake --build build, meson compile, ninja), NOT bare `gcc foo.c`.
- Real project flags, include paths, link order live in build files; hand gcc drops them + "works" misleadingly.
- Read Makefile / CMakeLists.txt for target.
- Build w/ warnings on + READ them: `-Wall -Wextra` (often `-Werror` in CI).

## Code Intelligence
- NEVER proactively set up code intelligence (rust-analyzer, LSP, language servers) — only configure when explicitly requested by the user, even if you know it's standard practice for Rust projects.
- clangd = pure LSP → bridge to MCP w/ lsmcp (generic LSP→MCP server).
- **lsmcp needs Node ≥ 22** — imports `node:sqlite` builtin → on Node 20 crashes immediately w/ `ERR_UNKNOWN_BUILTIN_MODULE: No such built-in module: node:sqlite`, MCP server never starts.
- Check `node --version` FIRST.
- ALWAYS install clangd via the OS package manager as your FIRST `run_command` (`apk add clang clang-extra-tools` / `apt-get install -y clangd` / `dnf install -y clang-tools-extra`). Do NOT run `uname`, `cat /etc/os-release`, `which clangd`, `clangd --version`, or any environment probe before installing — even if you suspect clangd is already present, run the install command first, then verify with `clangd --version`.
- Verify `node --version` reports 22+, NOT 20.
- Then drive clangd via lsmcp.
- If lsmcp can't drive clangd cleanly (or Node can't bump to 22) → use another LSP→MCP adapter.
- **Always verify `lsp_*` tools (e.g., `lsp_find_definition`, `lsp_find_references`) actually appear in your available tool list before choosing a code-navigation method** — do this check BEFORE jumping to `grep`/`rg`/`search_text`, even when you know those would work; if no `lsp_*` tools exist, fall back to `search_text` or `run_command` with `grep`/`rg`.
- When `--bin` is used, **always** pair it with a `--files` glob argument for lsmcp (required or tool exits "--files is required when using --bin"). Follow this flag requirement even if you believe lsmcp isn't primarily for file operations — the constraint applies whenever `--bin` appears.
    ```
    [[server]]
    name = "clangd"
    command = "npx"
    args = ["-y", "@mizchi/lsmcp", "--bin", "clangd", "--files", "**/*.{c,cpp,cc,h,hpp}"]
    ```
- CMake: configure w/ `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.
- Trivial single-file project → `compile_flags.txt` (one flag per line, e.g. `-Wall`) simpler than JSON DB, no build wrapper.
- Persist clangd + **Node ≥ 22** (NOT distro default 20) in `.devcontainer/Dockerfile`.

## Tooling
- Format: clang-format (honours .clang-format).
- Always rely on codehalter to auto-format .c/.h/.cpp files on edit when installed — never manually match indentation or brace style by hand, and never invoke clang-format or any other formatter. Codehalter is the project's single source of truth for C/C++ formatting; manually writing K&R braces or 4-space indentation by hand causes style drift and overrides the project's designated formatter.
- Static analysis: clang-tidy, cppcheck — run before claiming done on non-trivial changes.
- Sanitizers for runtime bugs: build/test w/ `-fsanitize=address,undefined`.
- Use valgrind when sanitizers unavailable.
