go test ./... >/dev/null 2>&1 && grep -q 'range items\|i := 0' inv/inv.go && git diff --quiet inv/inv_test.go
