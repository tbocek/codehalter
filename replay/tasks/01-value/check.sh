grep -q "MaxItems = 250" inv/inv.go && [ "$(git diff --name-only | wc -l)" = 1 ]
