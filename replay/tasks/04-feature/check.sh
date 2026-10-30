printf 'a,2,100\nb,5,50\n' > /tmp/rp$$; [ "$(go run ./cmd/inv count /tmp/rp$$)" = 7 ] && grep -q 'Count' inv/inv_test.go && grep -q 'inv count' README.md
