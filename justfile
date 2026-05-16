default: build

build:
    go mod tidy
    go build -o codehalter .

lint:
    go vet

test:
    go test ./...

# Install the built binary to /usr/local/bin so Zed's command path
# matches what the devcontainer uses (/usr/local/bin/codehalter).
install: build
    sudo install -m 0755 codehalter /usr/local/bin/codehalter

# Build the benchmark harness. Output goes to bench/bench-runner to avoid
# the collision Go hits when building a binary into a directory of the
# same name (`bench/bench`).
bench-build: build
    go build -o bench/bench-runner ./bench/

# Build then run the benchmark. Runs from bench/ so the default `tests/`
# glob and `settings.toml` paths resolve. Pass an explicit test file as
# the first argument to run just that one (default: all under bench/tests/).
bench *args: bench-build
    cd bench && ./bench-runner {{args}}