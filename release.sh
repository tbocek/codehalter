#!/usr/bin/env bash
set -euo pipefail

command -v git >/dev/null || { echo "error: git not found" >&2; exit 1; }
command -v curl >/dev/null || { echo "error: curl not found" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq not found" >&2; exit 1; }

# Check that the working tree is clean
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "error: working tree is not clean. Commit or stash changes first." >&2
  exit 1
fi

# Get the latest tag, default to v0 if none exist
latest_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0")

# Extract the version number, increment by 1
current=$(echo "$latest_tag" | sed 's/^v//')
next=$((current + 1))
new_tag="v${next}"

# Create the tag
git tag "$new_tag"
git push --tags
echo "$new_tag"

# Wait for all 5 release assets to be uploaded
EXPECTED=5
RETRIES=20
SLEEP=3
URL="https://api.github.com/repos/tbocek/codehalter/releases/tags/$new_tag"

echo "Waiting for $EXPECTED release assets at $URL ..."
for i in $(seq 1 "$RETRIES"); do
  assets=$(curl -sf "$URL" | jq '.assets | length')
  if [ "$assets" -ge "$EXPECTED" ]; then
    echo "✓ All $EXPECTED assets uploaded."
    exit 0
  fi
  echo "  $assets/$EXPECTED assets ready (attempt $i/$RETRIES)..."
  sleep "$SLEEP"
done

echo "error: release assets not ready after $((RETRIES * SLEEP))s" >&2
exit 1
