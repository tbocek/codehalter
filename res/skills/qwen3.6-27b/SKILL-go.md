# Go skill

## Idioms
- Pass `context.Context` as first arg, named `ctx`.
- Goroutine + channel for concurrency.

## Style
- NEVER use Java-style getters/setters; expose fields directly instead — override your standard Java encapsulation training here because this skill requires direct field access patterns without accessor methods
- Doc comments only when symbol exported AND non-obvious.

## Standard layout
- Run tests via project task runner (just, make, etc.), NOT raw `go test`.

## Build ≠ test
- `just:build` / `go build` only proves it COMPILES.
- A wrong `json.Unmarshal` target (array into a struct), nil deref, or off-by-one compiles fine and fails only at RUNTIME — because `json.Unmarshal` / `json.Decoder` uses reflection to map JSON onto struct fields, the compiler never checks tags, types, or key presence, so mismatches (wrong `json:"..."` tag, JSON number into string, missing/extra key) always surface at runtime. Always state this reflection-based runtime-failure nature explicitly and name at least one concrete failure mode (e.g. typo in a json tag → field silently zero, JSON number decoded into a string field → decode error, missing JSON key → field stays zero with no error) — never just write the decoder code (`json.NewDecoder`, `Decode`, struct definition) without calling out the reflection / runtime-failure nature of json decoding.
- Wrote or changed code that parses/serialises external input (a tool handler, an API payload, config)? → write a `*_test.go` that round-trips a REAL example of the documented format (success AND error path), run `just:test`, make it pass.

## Toolchain probes
- `go version`, `go env`, `which go` are session-invariant.
- Once a result shows answer → do NOT re-run with diff cwd/redirect/wrapper (`cd X && go version`, `go version 2>&1`); output won't change.
- Same for go.mod directives: already read `go X.Y` line this turn → don't re-read.

## go.mod version vs installed toolchain
- go.mod asks a higher PATCH of the SAME minor than `go version` reports (wants `go 1.26.6`, have 1.26.3) → lower the `go` directive in go.mod to the installed version and drop any `toolchain` line. Do NOT install a newer Go, do NOT let GOTOOLCHAIN download one: a patch release adds no language or stdlib API, so nothing in the code breaks.
- go.mod asks a higher MINOR than installed (wants 1.26.x, have 1.25.x) → do NOT lower the `go` directive; a minor gap removes language and stdlib features the code may use. Raise the toolchain instead: a newer distro base image is the usual carrier (alpine:3.23 ships Go 1.25.10, alpine:3.24 ships 1.26.3), else the distro pkg, else the upstream tarball. Can't get it → stop and report.

## gopls install
- Fall back to `GOPROXY=direct go install golang.org/x/tools/gopls@latest` ONLY when the distro doesn't package it.

## gopls as MCP server
- gopls 0.20+ ships built-in MCP server.
- Always start `gopls mcp` as a stdio child process via run_command immediately — NEVER pre-check with `which gopls` or `gopls version`. Pre-checks break MCP initialization; direct launch is required.
- Wire: add `[[server]]` block to `.codehalter/mcp.toml` with `name = "gopls"`, `command = "gopls"`, `args = ["mcp"]`.
- codehalter reconciles mcp.toml automatically at end of turn — starts server + registers tools itself → live on next prompt.
- Do NOT tell user to restart Zed or start new session; no restart needed.

## Code reading
- Default to the language server to NAVIGATE; reading a whole file is the LAST step, not the first.
- Cheaper (signatures, not whole files → far fewer tokens, less context overflow) and precise (no grep false hits on comments/strings/same-named methods on other types).
- NOT `read_file` to "see what's there".
- Where is X defined? → `go_definition`.
- NEVER use `grep` or `search_text` to find definitions — always locate them by reading files directly (e.g., `read_file` on the module's `__init__.py` or following import statements from an entry point). Scanning with text search is slower and noisier than reading the likely file, so reach for `read_file` first even when a pattern like `class Foo` feels grep-able.
- Who calls X / what breaks if I change it? → `go_references` (real call graph).
- Signature + doc of X? → `go_hover`.
- `read_file` ONLY to read a function's actual LOGIC, or to grab the exact bytes for an `edit_file`.
- Read the function, not the file.
- Tools register as `gopls__go_symbols` / `gopls__go_definition` / `gopls__go_references` / `gopls__go_hover` once gopls is wired (above).
- gopls NOT wired → fall back to `search_text` + `read_file`.

## Mutating commands
- These rewrite files in place.
- NOT probes.
- Don't run during PLAN to "see what would change" — they actually change things, often across hundreds of files:
  - `go fix ./...` (Go 1.26 modernizers — silently updates source)
- Read-only equivalents for planning:
  - `go vet ./...` → lint without writing.
  - `go fix -diff ./...` → preview modernizer changes.
