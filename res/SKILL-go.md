# Go skill
## Errors
- Errors = values. Check: `if err != nil { return err }` or wrap `fmt.Errorf("context: %w", err)`.
- NO panic for control flow → reserve panics for unrecoverable bugs.
- No try/catch; Go has neither.
- Never swallow error.

## Idioms
- `:=` declare-and-assign; `=` reassign.
- Defer cleanup right after acquiring resource: `f, err := os.Open(...); ...; defer f.Close()`.
- Pass `context.Context` as first arg, named `ctx`.
- Goroutine + channel for concurrency. Always clear "who closes the channel" rule.
- Small named interfaces over big (`io.Reader`, not `BigCombinedThing`).

## Style
- gofmt: tabs for indent, brace same line.
- Exports PascalCase, unexported camelCase.
- No Java-style getters/setters; expose fields directly when appropriate.
- Doc comments only when symbol exported AND non-obvious.

## Standard layout
- `main.go` = entry. Tests in `*_test.go` next to file under test.
- Run tests via project task runner (just, make, etc.), NOT raw `go test`.

## Probe toolchain once
- `go version`, `go env`, `which go` = session-invariant. Once a result shows answer → do NOT re-run with diff cwd/redirect/wrapper (`cd X && go version`, `go version 2>&1`); output won't change.
- Same for go.mod directives: already read `go X.Y` line this turn → don't re-read.

## Install gopls (+ other Go tools)
- Install via OS pkg mgr when it carries tool — Arch `yay -S --noconfirm gopls`. ALWAYS prefer over `go install`: in container root vs `dev` env differs (GOBIN/GOPATH/PATH) → `go install` lands binary off PATH, needs sudo + explicit `GOBIN=` — fragile + slow. Distro pkg = one clean command.
- Fall back `GOPROXY=direct go install golang.org/x/tools/gopls@latest` ONLY when OS doesn't package it (check first: `yay -Ss gopls`).

## gopls as MCP server
- gopls 0.20+ ships built-in MCP server. Start as stdio child `gopls mcp` (serves go_symbols / go_references / go_definition / go_hover).
- Wire: add `[[server]]` block to `.codehalter/mcp.toml` with `name = "gopls"`, `command = "gopls"`, `args = ["mcp"]`. codehalter reconciles mcp.toml automatically at end of turn — starts server + registers tools itself → live on next prompt. Do NOT tell user to restart Zed or start new session; no restart needed.

## Read code: outline before bytes (when gopls MCP is wired)
Default to the language server to NAVIGATE; reading a whole file is the LAST step, not the first. Cheaper (signatures, not whole files → far fewer tokens, less context overflow) and precise (no grep false hits on comments/strings/same-named methods on other types).
- What's in a file/package? → `go_symbols` (every decl + signature, no bodies). NOT `read_file` to "see what's there".
- Where is X defined? → `go_definition`. NOT grep / `search_text`.
- Who calls X / what breaks if I change it? → `go_references` (real call graph). `search_text` over-matches.
- Signature + doc of X? → `go_hover`.
- `read_file` ONLY to read a function's actual LOGIC, or to grab the exact bytes for an `edit_file`. Read the function, not the file.
Tools register as `gopls__go_symbols` / `gopls__go_definition` / `gopls__go_references` / `gopls__go_hover` once gopls is wired (above). gopls NOT wired → fall back to `search_text` + `read_file`.

## Mutating commands — NEVER during planning
These rewrite files in place. NOT probes. Don't run during PLAN to "see what would change" — they actually change things, often across hundreds of files:
- `go fix ./...` (Go 1.26 modernizers — silently updates source)
- `go mod tidy` (rewrites go.mod / go.sum)
- `gofmt -w`, `goimports -w` (rewrite formatting in place)
- `go generate ./...` (runs arbitrary //go:generate directives)
Run ONLY when task explicitly calls for that change, ONLY during EXECUTE.
Read-only equivalents for planning:
- `go vet ./...` → lint without writing.
- `go fix -diff ./...` → preview modernizer changes.
- `gofmt -d`, `goimports -d` → show would-be diffs.