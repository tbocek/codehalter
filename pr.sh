#!/usr/bin/env bash
set -euo pipefail

# pr.sh — open (or refresh) the PR that adds/updates codehalter in the agent
# registry (https://github.com/agentclientprotocol/registry).
#
# Pins registry/codehalter/agent.json to the latest release tag, clones your fork,
# branches off UPSTREAM's current default (explicit fetch, so the PR is on top of
# the live registry, not a stale fork), copies agent.json + icon.svg, validates,
# commits, and pushes. Open the PR from the printed URL.
#
# Run AFTER release.sh has tagged and the release binaries are published, so the
# download URLs in agent.json resolve.
#
# python? Not required — build_registry.py is OPTIONAL local schema validation
# (the registry CI runs it on the PR anyway); this runs it only if `uv` is present.
# Deps: git + jq. You must have forked the registry to your account already.
#
# Config via env:
#   REGISTRY_FORK      your fork's clone URL (default https://github.com/tbocek/registry.git)
#   REGISTRY_UPSTREAM  upstream "owner/name" (default agentclientprotocol/registry)
#   PR_BRANCH          branch to push (default add-codehalter)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$REPO_ROOT/registry/codehalter"
FORK="${REGISTRY_FORK:-git@github.com:tbocek/registry.git}"
UPSTREAM="${REGISTRY_UPSTREAM:-agentclientprotocol/registry}"
UPSTREAM_URL="https://github.com/$UPSTREAM.git"
BRANCH="${PR_BRANCH:-add-codehalter}"

command -v git >/dev/null || { echo "error: git not found" >&2; exit 1; }
command -v jq  >/dev/null || { echo "error: jq not found (needed to edit agent.json)" >&2; exit 1; }
[ -f "$SRC/agent.json" ] && [ -f "$SRC/icon.svg" ] || {
  echo "error: $SRC/{agent.json,icon.svg} not found" >&2; exit 1; }

# 1. Latest release tag.
tag="$(git -C "$REPO_ROOT" describe --tags --abbrev=0 2>/dev/null || true)"
[ -n "$tag" ] || { echo "error: no git tags found — run release.sh first." >&2; exit 1; }
echo "latest tag: $tag"

# 2. Pin agent.json (version field + every release-download URL), then commit and
#    push the bump to THIS repo — but only if it actually changed.
AGENT_REL="registry/codehalter/agent.json"
tmp="$(mktemp)"
jq --arg tag "$tag" '
  .version = ($tag | ltrimstr("v")) + ".0.0"
  | .distribution.binary |= map_values(
      .archive |= sub("/releases/download/v[0-9]+/"; "/releases/download/\($tag)/")
    )
' "$SRC/agent.json" > "$tmp"
mv "$tmp" "$SRC/agent.json"
if git -C "$REPO_ROOT" diff --quiet HEAD -- "$AGENT_REL"; then
  echo "agent.json already at $tag — no bump needed"
else
  git -C "$REPO_ROOT" commit -m "bump version to $tag" -- "$AGENT_REL"
  git -C "$REPO_ROOT" push
  echo "committed + pushed: bump version to $tag"
fi

# 3. Clone your fork (origin) into a throwaway workdir.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
echo "cloning fork $FORK ..."
git clone "$FORK" "$work/registry"
cd "$work/registry"
git remote add upstream "$UPSTREAM_URL"

# 4. Branch off UPSTREAM's current default (explicit fetch), not the fork's default
#    which may lag behind. Keeps the PR on the live registry and the local
#    validation on the current schema.
default="$(git ls-remote --symref "$UPSTREAM_URL" HEAD 2>/dev/null \
  | sed -n 's#^ref: refs/heads/\([^[:space:]]*\)[[:space:]].*#\1#p')"
[ -n "$default" ] || default="main"
echo "fetching upstream/$default ..."
git fetch upstream "$default"
git checkout -b "$BRANCH" "upstream/$default"

# 5. Copy the two files.
mkdir -p codehalter
cp "$SRC/agent.json" codehalter/agent.json
cp "$SRC/icon.svg"   codehalter/icon.svg

# 6. Optional local validation (schema only; URL check skipped — CI re-validates).
if command -v uv >/dev/null && [ -f .github/workflows/build_registry.py ]; then
  echo "validating with build_registry.py ..."
  SKIP_URL_VALIDATION=1 uv run --with jsonschema .github/workflows/build_registry.py
else
  echo "skipping local validation (uv or build_registry.py absent); the registry CI will validate."
fi

# 7. Commit + push to your fork. Force so re-runs refresh the same PR branch.
git add codehalter
if git diff --cached --quiet; then
  echo "nothing changed — codehalter is already current on upstream."
  exit 0
fi
git commit -m "Add codehalter agent $tag"
git push --force -u origin "$BRANCH"

# 8. Print the PR link (open from your fork; GitHub targets upstream as the base).
forkweb="$(echo "$FORK" | sed -E 's#^git@github.com:#https://github.com/#; s#\.git$##')"
echo
echo "✓ pushed $BRANCH to your fork."
echo "Open the PR: $forkweb/pull/new/$BRANCH"
