# C / C++ skill
Covers C and C++ (.c/.h + .cpp/.cc/.cxx/.hpp). Match file you edit — no C++ idioms in C file or reverse.

## Build System
- Build through project build system — NEVER hand-invoke compiler
- Use run_task for declared build (make, cmake --build build, meson compile, ninja), NOT bare `gcc foo.c`.
- Real project flags, include paths, link order live in build files; hand gcc drops them + "works" misleadingly.
- Read Makefile / CMakeLists.txt for target.
- Make: `make` / `make <target>`.
- Check Makefile for what exists.
- CMake: configure once (`cmake -S . -B build`) → `cmake --build build`.
- Build w/ warnings on + READ them: `-Wall -Wextra` (often `-Werror` in CI).
- Clean build w/ warnings = not clean → fix or justify each.

## Conventions
- Headers: include guard (`#ifndef X_H`/`#define`/`#endif`) or `#pragma once`.
- Headers: include what you use, nothing more.
- Headers: declarations in .h, definitions in .c/.cpp.
- C: no implicit int, check every malloc/fopen return, free what you alloc, no leaks.
- C++: prefer RAII / smart pointers over raw new/delete; const-correctness; pass big objects by const&.
- NEVER introduce undefined behaviour (out-of-bounds, use-after-free, signed overflow, uninit reads) to make something compile/pass → latent crash, not a fix.

## Code Intelligence
- Code intelligence over MCP — clangd (gopls analog)
- Set up ONLY when user asks.
- clangd = pure LSP → bridge to MCP w/ lsmcp (generic LSP→MCP server).
- **lsmcp needs Node ≥ 22** — imports `node:sqlite` builtin → on Node 20 crashes immediately w/ `ERR_UNKNOWN_BUILTIN_MODULE: No such built-in module: node:sqlite`, MCP server never starts.
- Check `node --version` FIRST.
- Install clangd via OS pkg mgr (`apk add clang clang-extra-tools` / `apt-get install -y clangd` / `dnf install -y clang-tools-extra`).
- Verify `clangd --version`.
- Install **Node ≥ 22** + project pkg mgr (see SKILL-base.md).
- Verify `node --version` reports 22+, NOT 20.
- Then drive clangd via lsmcp.
- If lsmcp can't drive clangd cleanly (or Node can't bump to 22) → use another LSP→MCP adapter.
- Verify `lsp_*` tools actually appear.
- Add to `.codehalter/mcp.toml` (uncomment WHOLE block INCLUDING `[[server]]` header — commented header leaves keys orphan + server never loads).
- `--bin` REQUIRES a `--files` glob telling lsmcp which files LSP handles, else exits "--files is required when using --bin".
    ```
    [[server]]
    name = "clangd"
    command = "npx"
    args = ["-y", "@mizchi/lsmcp", "--bin", "clangd", "--files", "**/*.{c,cpp,cc,h,hpp}"]
    ```
- clangd needs `compile_commands.json` to resolve includes/flags.
- **GENERATE from build system — NEVER hand-write** (hand-authored DB duplicates build cmd + silently goes stale when Makefile changes).
- Make: `bear -- make` (install bear first if missing — `apt-get install -y bear` / `apk add bear`). bear wraps real build + records exactly what it compiled.
- CMake: configure w/ `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.
- Trivial single-file project → `compile_flags.txt` (one flag per line, e.g. `-Wall`) simpler than JSON DB, no build wrapper.
- Persist clangd + **Node ≥ 22** (NOT distro default 20) in `.devcontainer/Dockerfile`.

## Tooling
- Install + persist in Dockerfile if missing — see SKILL-base.md.
- Format: clang-format (honours .clang-format).
- codehalter auto-formats .c/.h/.cpp on edit when installed.
- Static analysis: clang-tidy, cppcheck — run before claiming done on non-trivial changes.
- Sanitizers for runtime bugs: build/test w/ `-fsanitize=address,undefined`.
- Use valgrind when sanitizers unavailable.
- Debug: gdb / lldb.
- These = OS packages (`apk add clang clang-extra-tools` / `apt-get install clang clang-tidy clang-format gdb`), NOT language-package installs.
