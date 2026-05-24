# Go skill

## Errors
- Errors are values. Check explicitly: `if err != nil { return err }` or wrap with `fmt.Errorf("context: %w", err)`.
- Do NOT use `panic` for control flow. Reserve panics for unrecoverable bugs.
- Do NOT use a `try`/`catch` pattern; Go has neither.

## Idioms
- Use `:=` to declare and assign in one step; `=` to reassign an existing variable.
- Defer cleanup right after acquiring a resource: `f, err := os.Open(...); ...; defer f.Close()`.
- Pass `context.Context` as the first argument, named `ctx`.
- Goroutine + channel for concurrency. Always have a clear "who closes the channel" rule.
- Prefer small, named interfaces over big ones (`io.Reader`, not `BigCombinedThing`).

## Style
- Format with `gofmt` rules: tabs for indentation, brace on the same line.
- Export with PascalCase, unexported with camelCase.
- Don't write Java-style getters/setters; expose fields directly when appropriate.
- Don't add doc comments unless the symbol is exported AND non-obvious.

## Standard layout
- `main.go` for entry point. Tests in `*_test.go` next to the file under test.
- Use `_test.go` files. Run tests through the project's task runner (`just`, `make`, etc.), not raw `go test`.

## Probe the toolchain once
- `go version`, `go env`, `which go` are session-invariant. Once a tool result in this turn's history shows the answer, do NOT re-run them with a different cwd, redirection, or wrapper (`cd X && go version`, `go version 2>&1`, etc.) — the output won't change. Scroll back and reuse the existing tool result.
- Same rule for `go.mod` directives: if you already read the `go X.Y` line in this turn, don't re-read the file just to confirm.

## gopls as an MCP server
- gopls 0.20+ ships a built-in MCP server — no third-party wrapper needed. Start it as a stdio child with `gopls mcp` (it serves go_symbols / go_references / go_definition / go_hover, etc.).
- To wire it for this project, add a `[[server]]` block to `.codehalter/mcp.toml` with `name = "gopls"`, `command = "gopls"`, `args = ["mcp"]` — codehalter spawns it on the next session.

## Mutating commands — never during planning
These rewrite files in place. They are NOT probes. Do not run them during PLAN to "see what would change" — they actually change things, often across hundreds of files:
- `go fix ./...` (Go 1.26 modernizers — silently updates source files)
- `go mod tidy` (rewrites `go.mod` / `go.sum`)
- `gofmt -w`, `goimports -w` (rewrite formatting in place)
- `go generate ./...` (runs arbitrary //go:generate directives)

Run them only when the task explicitly calls for that change, and only during EXECUTE. Bumping `go 1.25` → `go 1.26` in `go.mod` or "modernize my code" is a user-driven decision, not a side effect of researching release notes.

Read-only equivalents to use during planning instead:
- `go vet ./...` — lint without writing
- `go fix -diff ./...` — preview modernizer changes
- `gofmt -d`, `goimports -d` — show would-be diffs
