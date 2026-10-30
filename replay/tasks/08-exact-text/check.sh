grep -q '56,78 €' inv/inv.go && go test -run TestFormat ./inv >/dev/null 2>&1
