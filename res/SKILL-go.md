# Go skill
## Errors
- Errors = values: `if err != nil { return err }`, or wrap `fmt.Errorf("context: %w", err)`.
- NO panic for control flow → panic only for unrecoverable bugs.
- No try/catch; Go has neither.
- Never swallow an error.

## Idioms
- `:=` declare-and-assign; `=` reassign.
- Defer cleanup right after acquiring: `f, err := os.Open(...); ...; defer f.Close()`.
- `context.Context` = first arg, named `ctx`.
- Goroutine + channel for concurrency. Always a clear "who closes the channel" rule.
- Small named interfaces over big (`io.Reader`, not `BigCombinedThing`).

## Style
- gofmt: tabs indent, brace same line.
- Exports PascalCase, unexported camelCase.
- No Java-style getters/setters; expose fields directly when appropriate.
- Doc comments only when symbol exported AND non-obvious.

## Layout
- `main.go` = entry. Tests in `*_test.go` beside file under test.
- Run tests via project task runner (just, make), NOT raw `go test`.

## Build ≠ test
`go build` / `just:build` proves COMPILES only. Wrong `json.Unmarshal` target (array into struct), nil deref, off-by-one: compile fine, fail at RUNTIME → green build on broken code. Wrote/changed code parsing or serialising external input (tool handler, API payload, config)? → write `*_test.go` round-tripping a REAL example of the documented format, success AND error path, run `just:test`, make it pass. NOT just `just:build`.

## Probe toolchain once
- `go version`, `go env`, `which go` = session-invariant. Answered once → NO re-run with different cwd/redirect/wrapper (`cd X && go version`, `go version 2>&1`); output won't change.
- Same for go.mod: read the `go X.Y` line once per turn → don't re-read.

## go.mod vs installed toolchain
- go.mod wants higher PATCH, same minor (wants 1.26.6, have 1.26.3) → lower `go` directive to installed version, drop any `toolchain` line. Do NOT install newer Go, do NOT let GOTOOLCHAIN fetch one: patch adds no language/stdlib API, nothing breaks.
- go.mod wants higher MINOR (wants 1.26.x, have 1.25.x) → do NOT lower the directive; minor gap drops language + stdlib features code may use. Raise toolchain instead: newer distro base image = usual carrier (alpine:3.23 = Go 1.25.10, alpine:3.24 = 1.26.3), else distro pkg, else upstream tarball. Unobtainable → stop, report.

## Install gopls (+ other Go tools)
- OS pkg mgr first (universal rule, container skill). Fall back `GOPROXY=direct go install golang.org/x/tools/gopls@latest` ONLY when distro doesn't package it.

## gopls as MCP server
- gopls 0.20+ ships MCP server: stdio child `gopls mcp` → go_symbols / go_references / go_definition / go_hover.
- Wire: `[[server]]` in `.codehalter/mcp.toml`, `name = "gopls"`, `command = "gopls"`, `args = ["mcp"]`. codehalter reconciles mcp.toml at turn end, starts server + registers tools → live next prompt. Do NOT tell user to restart Zed or start new session; no restart needed.

## Read code: outline before bytes (gopls wired)
LSP navigates; whole-file read = LAST step, not first. Cheaper (signatures not files → fewer tokens, less overflow) + precise (no grep hits on comments/strings/same-named methods of other types).
- File/package contents? → `go_symbols` (decls+signatures, no bodies), NOT `read_file`.
- Where is X defined? → `go_definition`, NOT grep/`search_text`.
- Who calls X / what breaks if I change it? → `go_references` (real call graph); `search_text` over-matches.
- Signature + doc of X? → `go_hover`.
- `read_file` ONLY for a function's LOGIC, or exact bytes for `edit_file`. Read the function, not the file.
Tools = `gopls__go_symbols` / `gopls__go_definition` / `gopls__go_references` / `gopls__go_hover`. Not wired → `search_text` + `read_file`.

## Mutating commands — NEVER in planning
Rewrite files in place. NOT probes. Never run in PLAN to "see what would change" — they change things, often across hundreds of files:
- `go fix ./...` (Go 1.26 modernizers, silently rewrites source)
- `go mod tidy` (rewrites go.mod / go.sum)
- `gofmt -w`, `goimports -w` (rewrite formatting in place)
- `go generate ./...` (runs arbitrary //go:generate directives)
Run ONLY when the task calls for that change, ONLY in EXECUTE.
Planning equivalents: `go vet ./...` (lint, no writes), `go fix -diff ./...` (preview modernizers), `gofmt -d` / `goimports -d` (would-be diffs).
