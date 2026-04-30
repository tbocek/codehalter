default: build

build:
    go mod tidy
    go build -o codehalter .

lint:
    go vet

test:
    go test ./...