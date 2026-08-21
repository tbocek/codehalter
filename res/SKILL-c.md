# C / C++ skill
Covers C and C++ (.c/.h + .cpp/.cc/.cxx/.hpp). Match the file you edit — no C++ idioms in a C file or the reverse.

## Build through the project build system
Use `run_task` on the declared build (make, cmake --build build, meson compile, ninja), NOT bare `gcc foo.c`: real flags, include paths and link order live in the build files. Pick the ONE system the project actually uses (a Makefile that wraps cmake is not a second target to try) and read it for the target name.
Warnings on and READ them: `-Wall -Wextra` (often `-Werror` in CI). A build with warnings is not a clean build.
NEVER introduce undefined behaviour (out-of-bounds, use-after-free, signed overflow, uninit reads) to make something compile or pass → latent crash, not a fix.

## Code intelligence over MCP — clangd (gopls analog)
Set up ONLY when the user asks. clangd = pure LSP → bridge to MCP with lsmcp (generic LSP→MCP server). **lsmcp needs Node ≥ 22** — it imports the `node:sqlite` builtin, so on Node 20 it crashes immediately with `ERR_UNKNOWN_BUILTIN_MODULE: No such built-in module: node:sqlite` and the MCP server never starts. Check `node --version` FIRST.
1. Install clangd via the OS pkg mgr (`apk add clang clang-extra-tools` / `apt-get install -y clangd` / `dnf install -y clang-tools-extra`); verify `clangd --version`.
2. Install **Node ≥ 22** (NOT the distro default 20) + the project pkg manager, and persist both in `.devcontainer/Dockerfile`.
3. Add to `.codehalter/mcp.toml` (uncomment the WHOLE block INCLUDING the `[[server]]` header — a commented header leaves the keys orphaned and the server never loads). `--bin` REQUIRES a `--files` glob, else lsmcp exits "--files is required when using --bin":
[[server]]
name = "clangd"
command = "npx"
args = ["-y", "@mizchi/lsmcp", "--bin", "clangd", "--files", "**/*.{c,cpp,cc,h,hpp}"]
4. clangd needs `compile_commands.json`. GENERATE it from the build system — `bear -- make`, or cmake with `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`. NEVER hand-write it: a hand-authored DB duplicates the build command and goes stale silently. Trivial single-file project → `compile_flags.txt` (one flag per line) is simpler.

## Tooling (OS packages, not language installs — see SKILL-base.md)
clang-format (codehalter auto-formats .c/.h/.cpp on edit when installed), clang-tidy / cppcheck before claiming done on non-trivial changes, `-fsanitize=address,undefined` or valgrind for runtime bugs, gdb/lldb to debug.
