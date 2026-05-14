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