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
