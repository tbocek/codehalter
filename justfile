default: build

build:
    go mod tidy
    go build -o codehalter .

lint:
    go vet ./...

test:
    go test ./...

# matches what the devcontainer uses (/usr/local/bin/codehalter).
install: build
    sudo install -m 0755 codehalter /usr/local/bin/codehalter
