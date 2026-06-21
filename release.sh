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

# Wait for the release assets. Surface a build failure explicitly instead of
# silently waiting out the timeout: poll this tag's workflow run and bail the
# moment it concludes unsuccessfully.
EXPECTED=4   # linux amd64/arm64 + darwin amd64/arm64 (no windows build)
RETRIES=20
SLEEP=3
sha="$(git rev-list -n1 "$new_tag")"
REL_URL="https://api.github.com/repos/tbocek/codehalter/releases/tags/$new_tag"
RUNS_URL="https://api.github.com/repos/tbocek/codehalter/actions/workflows/build.yml/runs?head_sha=$sha"

echo "Waiting for $EXPECTED release assets (build $sha) ..."
for i in $(seq 1 "$RETRIES"); do
  # Fail fast if CI finished without success, rather than masking it as a timeout.
  runs="$(curl -sf "$RUNS_URL" || true)"
  conclusion="$(printf '%s' "$runs" | jq -r '.workflow_runs[0].conclusion // "pending"' 2>/dev/null || echo pending)"
  run_url="$(printf '%s' "$runs" | jq -r '.workflow_runs[0].html_url // ""' 2>/dev/null || echo "")"
  case "$conclusion" in
    failure|cancelled|timed_out|startup_failure)
      echo "error: release build did not succeed (conclusion: $conclusion)${run_url:+, see $run_url}" >&2
      exit 1
      ;;
  esac

  assets="$(curl -sf "$REL_URL" | jq '.assets | length' 2>/dev/null || echo 0)"
  if [ "$assets" -ge "$EXPECTED" ]; then
    echo "✓ All $EXPECTED assets uploaded."
    exit 0
  fi
  echo "  $assets/$EXPECTED assets ready (build: $conclusion, attempt $i/$RETRIES)..."
  sleep "$SLEEP"
done

echo "error: release assets not ready after $((RETRIES * SLEEP))s (last build status: $conclusion)" >&2
exit 1
