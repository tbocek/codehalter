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

## Tooling (install + persist in the Dockerfile if missing — see SKILL-install.md)

- Format: `clang-format` (honours a `.clang-format`; codehalter auto-formats
  `.c/.h/.cpp` on edit when it's installed).
- Static analysis: `clang-tidy`, `cppcheck` — run before claiming done on
  non-trivial changes.
- Sanitizers for runtime bugs: build/test with `-fsanitize=address,undefined`;
  `valgrind` when sanitizers aren't available.
- Debug: `gdb` / `lldb`.

These are OS packages (`apk add clang clang-extra-tools` / `apt-get install
clang clang-tidy clang-format gdb`), not language-package installs.
