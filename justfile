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
