# Go skill

## Errors
- Errors are values. Check: `if err != nil { return err }` or wrap with
  `fmt.Errorf("context: %w", err)`.
- Do NOT `panic` for control flow. Reserve panics for unrecoverable bugs.
- No `try`/`catch`; Go has neither.
- Never swallow error.

## Idioms
- `:=` to declare-and-assign; `=` to reassign.
- Defer cleanup right after acquiring a resource: `f, err := os.Open(...);
  ...; defer f.Close()`.
- Pass `context.Context` as the first argument, named `ctx`.
- Goroutine + channel for concurrency. Always have a clear "who closes
  the channel" rule.
- Small named interfaces over big ones (`io.Reader`, not `BigCombinedThing`).

## Style
- `gofmt` rules: tabs for indentation, brace on same line.
- Exports PascalCase, unexported camelCase.
- No Java-style getters/setters; expose fields directly when appropriate.
- Doc comments only when symbol is exported AND non-obvious.

## Standard layout
- `main.go` for entry. Tests in `*_test.go` next to the file under test.
- Run tests through the project's task runner (`just`, `make`, etc.),
  not raw `go test`.

## Probe the toolchain once
- `go version`, `go env`, `which go` are session-invariant. Once a tool
  result shows the answer, do NOT re-run with a different cwd, redirection,
  or wrapper (`cd X && go version`, `go version 2>&1`) — output won't change.
- Same for `go.mod` directives: if you already read the `go X.Y` line
  this turn, don't re-read.

## Installing gopls (and other Go tools)
- Install via the OS package manager when it carries the tool — on Arch
  `yay -S --noconfirm gopls`. ALWAYS prefer this over `go install`: in a
  container the root vs `dev` env differs (GOBIN/GOPATH/PATH), so `go install`
  lands the binary off PATH and needs `sudo` + explicit `GOBIN=` — fragile and
  slow. The distro package is one clean command.
- Fall back to `GOPROXY=direct go install golang.org/x/tools/gopls@latest` ONLY
  when the OS does not package it (check first: `yay -Ss gopls`).

## gopls as an MCP server
- gopls 0.20+ ships a built-in MCP server. Start as stdio child with
  `gopls mcp` (serves go_symbols / go_references / go_definition / go_hover).
- To wire it: add a `[[server]]` block to `.codehalter/mcp.toml` with
  `name = "gopls"`, `command = "gopls"`, `args = ["mcp"]`. codehalter
  reconciles `mcp.toml` automatically at the end of the turn — it starts the
  server and registers its tools on its own, so they are live on the next
  prompt. Do NOT tell the user to restart Zed or start a new session to
  activate it; no restart is needed.

## Mutating commands — never during planning
These rewrite files in place. NOT probes. Do not run during PLAN to "see
what would change" — they actually change things, often across hundreds
of files:
- `go fix ./...` (Go 1.26 modernizers — silently updates source)
- `go mod tidy` (rewrites `go.mod` / `go.sum`)
- `gofmt -w`, `goimports -w` (rewrite formatting in place)
- `go generate ./...` (runs arbitrary //go:generate directives)

Run them only when the task explicitly calls for that change, and only
during EXECUTE.

Read-only equivalents for planning:
- `go vet ./...` — lint without writing.
- `go fix -diff ./...` — preview modernizer changes.
- `gofmt -d`, `goimports -d` — show would-be diffs.
