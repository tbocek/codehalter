default: build improve-build

build:
    go mod tidy
    go build -o codehalter .

# Build the improve backend server.
improve-build:
    go build -o improve-server ./improve/cmd/server/

lint:
    go vet

test:
    go test ./...

improve: improve-build
    ./improve-server
# matches what the devcontainer uses (/usr/local/bin/codehalter).
install: build
    sudo install -m 0755 codehalter /usr/local/bin/codehalter

# Build the benchmark harness. Output goes to bench/bench-runner to avoid
# the collision Go hits when building a binary into a directory of the
# same name (`bench/bench`).
bench-build: build
    go build -o bench/bench-runner ./bench/

# Build then run the benchmark. Runs from bench/ so the default `tests/`
# glob and `settings.toml` paths resolve.
#
#   just bench                                       # all tests, no note
#   just bench "testing MTP"                         # all tests, tagged in results.jsonl
#   just bench "qwen30b-a3b" tests/preveltekit_go126.toml  # one test, tagged
#
# For other flags (`-work`, `-codehalter`, …) call bench-runner directly:
#   cd bench && ./bench-runner -note "x" -work /tmp/foo tests/...
bench note='' *args='': bench-build
    cd bench && ./bench-runner -note {{quote(note)}} {{args}}

# Build the skill crafter. Output goes to crafter/crafter-runner to avoid the
# directory-name collision Go hits building into crafter/.
crafter-build:
    go build -o crafter/crafter-runner ./crafter/

# Build then run the skill crafter. Runs from crafter/ so crafter.toml,
# ground-skills/ and the ../docs report path resolve.
#
#   just crafter                       # probe every SKILL in ground-skills/
#   just crafter "-ground other-dir"   # extra flags passed through
crafter *args='': crafter-build
    cd crafter && ./crafter-runner {{args}}
