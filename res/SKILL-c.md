# C / C++ skill
Covers C and C++ (.c/.h + .cpp/.cc/.cxx/.hpp). Match the file you edit — no C++ idioms in a C file or the reverse.
Always braces, even on a one-line `if`/`for`: without them a line added to the body later silently falls outside it (goto fail).

## Build through the project build system
Use `run_task` on the declared build (make, cmake --build build, meson compile, ninja), NOT bare `gcc foo.c`: real flags, include paths and link order live in the build files. Pick the ONE system the project actually uses (a Makefile that wraps cmake is not a second target to try) and read it for the target name.
Warnings on and READ them: `-Wall -Wextra` (often `-Werror` in CI). A build with warnings is not a clean build.
NEVER introduce undefined behaviour (out-of-bounds, use-after-free, signed overflow, uninit reads) to make something compile or pass → latent crash, not a fix.

## Formatter config — pin the style that is already there
No `.clang-format` → clang-format falls back to LLVM style (2-space, 80 col), which is almost never what the code is written in, and every tool that formats disagrees with every other. Fix by MEASURING: read existing sources for indent width, tabs vs spaces, brace placement (same line vs next), pointer binding (`char *p` vs `char* p`), the column limit the code respects. Start from the closest named base (`clang-format --style=GNU --dump-config > .clang-format`, or LLVM/Google/WebKit) and correct `IndentWidth`, `UseTab`, `BreakBeforeBraces`, `PointerAlignment`, `ColumnLimit` to the measured values. Prove it with `clang-format --dry-run -Werror $(git ls-files "*.c" "*.h")`: few complaints means the config describes the code, many means the config is wrong — fix the config, do not reformat the repo to match a guess. Then format the tree as ONE formatting-only commit on a clean tree.

## Tooling (OS packages, not language installs — see SKILL-base.md)
clang-format (codehalter auto-formats .c/.h/.cpp on edit when installed), clang-tidy / cppcheck before claiming done on non-trivial changes, `-fsanitize=address,undefined` or valgrind for runtime bugs, gdb/lldb to debug.
