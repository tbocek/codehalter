grep -q 'PriceCents < best.PriceCents' inv/inv.go && grep -q Cheapest inv/inv_test.go && git diff --quiet -- cmd
