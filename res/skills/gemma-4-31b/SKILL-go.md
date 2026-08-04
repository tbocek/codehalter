# Go skill

## Idioms
- Always clear "who closes the channel" rule.

## Style
- Never default to private fields plus Java-style getters/setters — the textbook Java encapsulation pattern is explicitly overridden by this skill. Expose fields directly (e.g., `public String name`) when appropriate; only add accessors when there is a concrete reason such as validation logic or a framework requirement.
- ONLY add doc comments when a symbol is BOTH exported AND non-obvious; skip all trivial or obvious exports even if Go conventions suggest documenting every public symbol.

## Standard layout
- Run tests via project task runner (just, make, etc.), NOT raw `go test`.

## Build ≠ test
- `just:build` / `go build` only proves it COMPILES.
- Always remember: `json.Unmarshal` / `json.Decoder` decode via reflection, so a wrong target (e.g. array into a struct), a nil deref, an off-by-one, a typo in a `json:"..."` tag (field silently stays zero), a JSON number decoded into a string field (errors at decode time), or a missing JSON key (stays zero with no error) ALL compile fine and fail only at RUNTIME — never skip this point or pivot to defensive 'fail-loud' patterns like `DisallowUnknownFields()` / manual required-field validation without first naming the reflection / runtime-failure nature, because that is the bug class this lesson teaches.
- Build stays green on broken code.
- Wrote or changed code that parses/serialises external input (a tool handler, an API payload, config)? → write a `*_test.go` that round-trips a REAL example of the documented format (success AND error path), run `just:test`, make it pass.

## Toolchain probes
- `go version`, `go env`, `which go` are session-invariant.
- Once a result shows answer → do NOT re-run with diff cwd/redirect/wrapper (`cd X && go version`, `go version 2>&1`); output won't change.
- Always read `go.mod` exactly once per turn and extract ALL directives you need (Go version, module path, dependencies) from that single read. Never split into separate `read_file` calls per directive, and never use `go version` or `go list` to re-derive what `go.mod` already declares — one read, all extractions, no re-reads even when other commands seem more idiomatic.

## gopls install
- Fall back to `GOPROXY=direct go install golang.org/x/tools/gopls@latest` ONLY when the distro doesn't package it.

## gopls as MCP server
- gopls 0.20+ ships built-in MCP server.
- Wire: add `[[server]]` block to `.codehalter/mcp.toml` with `name = "gopls"`, `command = "gopls"`, `args = ["mcp"]`.
- codehalter reconciles mcp.toml automatically at end of turn — starts server + registers tools itself → live on next prompt.

## Code reading
- Cheaper (signatures, not whole files → far fewer tokens, less context overflow) and precise (no grep false hits on comments/strings/same-named methods on other types).
- What's in a file/package? → `go_symbols` (every decl + signature, no bodies).
- NOT `read_file` to "see what's there".
- NOT grep / `search_text`.
- Who calls X / what breaks if I change it? → `go_references` (real call graph).
- Signature + doc of X? → `go_hover`.
- Tools register as `gopls__go_symbols` / `gopls__go_definition` / `gopls__go_references` / `gopls__go_hover` once gopls is wired (above).
- gopls NOT wired → fall back to `search_text` + `read_file`.

## Mutating commands
- These rewrite files in place.
- NOT probes.
- Don't run during PLAN to "see what would change" — they actually change things, often across hundreds of files:
  - `go fix ./...` (Go 1.26 modernizers — silently updates source)
  - `go mod tidy` (rewrites go.mod / go.sum)
- Read-only equivalents for planning:
  - `go vet ./...` → lint without writing.
  - `go fix -diff ./...` → preview modernizer changes.
