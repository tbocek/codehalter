# Go skill
- Wrap errors with context: `fmt.Errorf("read %s: %w", path, err)`. Never swallow one.
- Tests in `*_test.go` beside the file under test. Run them via the project task runner (just, make), NOT raw `go test`.
- Read-only in planning: `go vet ./...`, `go fix -diff ./...`, `gofmt -d`. Their mutating twins — `go mod tidy`, `go fix ./...`, `gofmt -w`, `go generate ./...` — rewrite files across the repo, so run them ONLY in EXECUTE and only when the task asks for that change.

## Probe toolchain once
- `go version`, `go env`, `which go` answered once → do NOT re-run with a different cwd, redirect or wrapper (`cd X && go version`, `go version 2>&1`). Same binary, same answer.
- ONE exception worth a second look: a nested `go.mod` with its own `toolchain` line can select a different version for that directory. Read the `go`/`toolchain` lines instead of re-running the probe.

## go.mod vs installed toolchain
- go.mod wants higher PATCH, same minor (wants 1.26.6, have 1.26.3) → lower the `go` directive to the installed version, drop any `toolchain` line. Do NOT install newer Go, do NOT let GOTOOLCHAIN fetch one: a patch adds no language/stdlib API, nothing breaks. Same-minor alignment is not a downgrade of the user's intent.
- go.mod wants higher MINOR (wants 1.26.x, have 1.25.x) → do NOT lower the directive; a minor gap drops language + stdlib features the code may use. Raise the toolchain, in this order: 1) bump the base image `FROM` tag (usual carrier: alpine:3.23 = Go 1.25.10, alpine:3.24 = 1.26.3) 2) distro package 3) upstream tarball. Unobtainable → stop, report.

## Install gopls (+ other Go tools)
- OS pkg mgr first (universal rule, container skill). Fall back `GOPROXY=direct go install golang.org/x/tools/gopls@latest` ONLY when the distro doesn't package it.

## gopls as MCP server
- gopls 0.20+ ships an MCP server: stdio child `gopls mcp` → go_symbols / go_references / go_definition / go_hover.
- Wire: `[[server]]` in `.codehalter/mcp.toml`, `name = "gopls"`, `command = "gopls"`, `args = ["mcp"]`. codehalter reconciles mcp.toml at turn end, starts the server + registers the tools → live next prompt. Do NOT tell the user to restart Zed or start a new session; no restart needed.

## Read code: outline before bytes (gopls wired)
LSP navigates; a whole-file read = LAST step, not first. Cheaper (signatures not files → fewer tokens) + precise (no grep hits on comments, strings, or same-named methods of other types).
- File/package contents? → `go_symbols` (decls+signatures, no bodies), NOT `read_file`.
- Where is X defined? → `go_definition`, NOT grep/`search_text`.
- Who calls X / what breaks if I change it? → `go_references` (real call graph); `search_text` over-matches.
- Signature + doc of X? → `go_hover`.
- `read_file` ONLY for a function's LOGIC, or exact bytes for `edit_file`. Read the function, not the file.
Tools = `gopls__go_symbols` / `gopls__go_definition` / `gopls__go_references` / `gopls__go_hover`. Not wired → `search_text` + `read_file`.
