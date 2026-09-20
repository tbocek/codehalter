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

# A release is a tag on a commit, so a commit that already carries one has
# nothing to release: tagging it again republishes the same binaries under a
# new number. That is how v61 and v62 ended up on the same commit.
if head_tags=$(git tag --points-at HEAD) && [ -n "$head_tags" ]; then
  echo "error: HEAD is already released as ${head_tags//$'\n'/, }. Commit something first." >&2
  exit 1
fi

# Tags may have been pushed from elsewhere; the numbering has to account for
# them or the push at the end fails after the tag is already made locally.
git fetch --tags --quiet || echo "warning: could not fetch tags, numbering from local ones only" >&2

# Next version = the highest vN that exists, plus one. NOT `git describe`,
# which reports the tag it picks when several point at one commit: with v61 and
# v62 both on HEAD~1 it answered v61, and the script then tried to create a v62
# that was already there.
latest_tag=$(git tag --list 'v[0-9]*' --sort=-v:refname | head -n1)
new_tag="v$(( ${latest_tag#v} + 1 ))"
if git rev-parse -q --verify "refs/tags/$new_tag" >/dev/null; then
  echo "error: $new_tag already exists even though $latest_tag is the highest. Delete it or tag by hand." >&2
  exit 1
fi

# Create the tag
git tag "$new_tag"
git push --tags
echo "$new_tag"

# Poll with as few API calls as possible: the unauthenticated GitHub API allows
# only 60 requests/hr per IP, and polling both endpoints frequently exhausts it.
# So one call every ${SLEEP}s for the run status, and only once the build
# succeeds a single call to confirm the assets. On a rate limit we abort loudly.
EXPECTED=4   # linux amd64/arm64 + darwin amd64/arm64 (no windows build)
RETRIES=20
SLEEP=10
sha="$(git rev-list -n1 "$new_tag")"
REL_URL="https://api.github.com/repos/tbocek/codehalter/releases/tags/$new_tag"
RUNS_URL="https://api.github.com/repos/tbocek/codehalter/actions/workflows/build.yml/runs?head_sha=$sha"

# gh_get URL: sets GH_BODY from the response. Aborts the whole script on a rate
# limit (403/429) or an unreachable API, rather than masking it as "not ready".
GH_BODY=""
gh_get() {
  local out code
  out="$(curl -sSL -w $'\n%{http_code}' "$1")" || { echo "error: GitHub API unreachable." >&2; exit 1; }
  code="${out##*$'\n'}"
  GH_BODY="${out%$'\n'*}"
  if [ "$code" = "403" ] || [ "$code" = "429" ]; then
    echo "error: GitHub API rate limit reached (HTTP $code): you are over the 60 requests/hr unauthenticated limit." >&2
    echo "The release may have completed anyway; check https://github.com/tbocek/codehalter/releases/tag/$new_tag" >&2
    exit 1
  fi
}

# Stage 1: wait for the build run to finish (one call per cycle).
echo "Waiting for build $sha to finish ..."
conclusion=""
for i in $(seq 1 "$RETRIES"); do
  gh_get "$RUNS_URL"
  conclusion="$(printf '%s' "$GH_BODY" | jq -r '.workflow_runs[0].conclusion // "pending"' 2>/dev/null || echo pending)"
  run_url="$(printf '%s' "$GH_BODY" | jq -r '.workflow_runs[0].html_url // ""' 2>/dev/null || echo "")"
  case "$conclusion" in
    success) break ;;
    failure|cancelled|timed_out|startup_failure)
      echo "error: release build did not succeed (conclusion: $conclusion)${run_url:+, see $run_url}" >&2
      exit 1 ;;
  esac
  echo "  not finished yet ($i/$RETRIES) ..."
  sleep "$SLEEP"
done
if [ "$conclusion" != "success" ]; then
  echo "error: build still not finished after $((RETRIES * SLEEP))s (last status: ${conclusion:-unknown})." >&2
  exit 1
fi

# Stage 2: build succeeded. One call to confirm the assets are attached.
gh_get "$REL_URL"
assets="$(printf '%s' "$GH_BODY" | jq '.assets | length' 2>/dev/null || echo 0)"
if [ "$assets" -ge "$EXPECTED" ]; then
  echo "✓ Release $new_tag is ready ($assets assets)."
  exit 0
fi
echo "error: build succeeded but the release $new_tag has only $assets of $EXPECTED assets." >&2
exit 1
