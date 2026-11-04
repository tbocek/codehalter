[ "$(grep -c "apk add" Dockerfile)" = 1 ] && for p in git curl go just nodejs npm; do grep -qw $p Dockerfile || exit 1; done
