[ -z "$(git status --porcelain -- . ":!.codehalter")" ] && grep -q "0\.05" "$REPLAY_OUT"
