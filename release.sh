#!/usr/bin/env bash
set -euo pipefail

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

echo "$new_tag"
