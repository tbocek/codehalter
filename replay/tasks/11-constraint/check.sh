go test ./... >/dev/null 2>&1 && git diff --quiet inv/inv_test.go && [ "$(git diff --name-only | wc -l)" = 1 ]
