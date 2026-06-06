# C / C++ skill

Covers both C and C++ (`.c/.h` and `.cpp/.cc/.cxx/.hpp`). Match the file you're
editing — don't bring C++ idioms into a C file or vice versa.

## Build through the project's build system — never hand-invoke the compiler

Use `run_task` for the declared build (`make`, `cmake --build build`, `meson
compile`, `ninja`), not a bare `gcc foo.c`. A real project's flags, include
paths, and link order live in the build files; a hand `gcc` call drops them and
"works" misleadingly. Read the `Makefile` / `CMakeLists.txt` to find the target.

- Make: `make` / `make <target>`; check the `Makefile` for what exists.
- CMake: configure once (`cmake -S . -B build`), then `cmake --build build`.
- Build with warnings on and read them: `-Wall -Wextra` (often `-Werror` in CI).
  A clean build with warnings is not clean — fix or justify each.

## Conventions

- Headers: include guard (`#ifndef X_H` / `#define` / `#endif`) or `#pragma
  once`; include what you use, nothing more; declarations in `.h`, definitions in
  `.c`/`.cpp`.
- C: no implicit `int`, check every `malloc`/`fopen` return, free what you
  allocate, no leaks. C++: prefer RAII / smart pointers over raw `new`/`delete`;
  `const`-correctness; pass big objects by `const&`.
- Never introduce undefined behaviour (out-of-bounds, use-after-free, signed
  overflow, uninitialised reads) to make something compile or pass — it's a
  latent crash, not a fix.

## Code intelligence over MCP — clangd (the gopls analog)

Set up ONLY when the user asks. clangd is a pure LSP, so bridge it to MCP with
lsmcp (the generic LSP→MCP server). **lsmcp requires Node ≥ 22** — it imports the
`node:sqlite` builtin, so on Node 20 it crashes immediately with
`ERR_UNKNOWN_BUILTIN_MODULE: No such built-in module: node:sqlite` and the MCP
server never starts. Check `node --version` first.

1. Install `clangd` via the OS package manager (`apk add clang clang-extra-tools`
   / `apt-get install -y clangd` / `dnf install -y clang-tools-extra`); verify
   `clangd --version`.
2. Install **Node ≥ 22** + the project's package manager (see SKILL-container.md) —
   verify `node --version` reports 22+, NOT 20. Then drive clangd via lsmcp. If
   lsmcp can't drive clangd cleanly (or Node can't be bumped to 22), use another
   LSP→MCP adapter — verify the `lsp_*` tools actually appear.
3. Add to `.codehalter/mcp.toml` (uncomment the WHOLE block INCLUDING the
   `[[server]]` header — a commented header leaves the keys orphan and the server
   never loads):
   ```
   [[server]]
   name = "clangd"
   command = "npx"
   args = ["-y", "@mizchi/lsmcp", "--bin", "clangd"]
   ```
4. clangd needs `compile_commands.json` to resolve includes/flags. **GENERATE it
   from the build system — never hand-write it** (a hand-authored DB duplicates
   the build command and silently goes stale when the Makefile changes):
   - Make: `bear -- make` (install `bear` first if missing — `apt-get install -y
     bear` / `apk add bear`). `bear` wraps the real build and records exactly
     what it compiled.
   - CMake: configure with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`.
   For a trivial single-file project a `compile_flags.txt` (one flag per line,
   e.g. `-Wall`) is simpler than a JSON DB and needs no build wrapper.
5. Persist clangd + **Node ≥ 22** (not the distro's default 20) in
   `.devcontainer/Dockerfile`.

## Tooling (install + persist in the Dockerfile if missing — see SKILL-container.md)

- Format: `clang-format` (honours a `.clang-format`; codehalter auto-formats
  `.c/.h/.cpp` on edit when it's installed).
- Static analysis: `clang-tidy`, `cppcheck` — run before claiming done on
  non-trivial changes.
- Sanitizers for runtime bugs: build/test with `-fsanitize=address,undefined`;
  `valgrind` when sanitizers aren't available.
- Debug: `gdb` / `lldb`.

These are OS packages (`apk add clang clang-extra-tools` / `apt-get install
clang clang-tidy clang-format gdb`), not language-package installs.
