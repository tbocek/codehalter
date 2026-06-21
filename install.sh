#!/usr/bin/env bash
set -euo pipefail

REPO="tbocek/codehalter"
BINARY="codehalter"

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)   OS="linux" ;;
    Darwin*)  OS="darwin" ;;
    *)        echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

# Detect arch
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)    ARCH="amd64" ;;
    aarch64|arm64)   ARCH="arm64" ;;
    *)               echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

# Build asset name
ASSET="${BINARY}-${OS}-${ARCH}"

# Resolve latest release tag
# Capture the API response and HTTP status (no curl -f, which would swallow the
# error body) so a non-200 such as a rate limit shows GitHub's own message to
# the user instead of failing with an empty tag.
api_url="https://api.github.com/repos/${REPO}/releases/latest"
response="$(curl -sSL -w $'\n%{http_code}' "$api_url")" || {
    echo "error: could not reach the GitHub API at ${api_url}" >&2
    exit 1
}
http_code="${response##*$'\n'}"
body="${response%$'\n'*}"
LATEST_TAG="$(printf '%s' "$body" | grep '"tag_name":' | sed -E 's/.*"tag_name": *"?([^"]+)".*/\1/')" || true
if [ "$http_code" != "200" ] || [ -z "$LATEST_TAG" ]; then
    echo "error: could not resolve the latest ${REPO} release (HTTP ${http_code}). GitHub returned:" >&2
    echo "$body" >&2
    exit 1
fi
echo "Latest release: ${LATEST_TAG}"

# Download URL
URL="https://github.com/${REPO}/releases/download/${LATEST_TAG}/${ASSET}"
echo "Downloading ${ASSET} from ${URL} ..."

TMPFILE="$(mktemp)"
trap 'rm -f "$TMPFILE"' EXIT

if ! curl -sL -o "$TMPFILE" "$URL"; then
    echo "Failed to download ${ASSET}" >&2
    exit 1
fi

chmod +x "$TMPFILE"

# Determine install directory
if [ "$(id -u)" -eq 0 ] || [ -w /usr/local/bin ]; then
    INSTALL_DIR="/usr/local/bin"
else
    INSTALL_DIR="${HOME}/.local/bin"
    mkdir -p "$INSTALL_DIR"
fi

cp "$TMPFILE" "${INSTALL_DIR}/${BINARY}"
echo "Installed ${BINARY} to ${INSTALL_DIR}/${BINARY}"

# Ensure the global config dir exists: the devcontainer bind-mounts
# ${localEnv:HOME}/.config/codehalter, and a missing bind source makes the
# container fail to start (docker --mount type=bind is strict about its source).
mkdir -p "${HOME}/.config/codehalter"
echo "Ensured ${HOME}/.config/codehalter exists"

# Record host facts codehalter reads when it scaffolds a .devcontainer. Right now
# just whether ~/.gitconfig exists, so it only bind-mounts it when present — a
# missing bind source would fail the container start.
has_gitconfig=false
[ -f "${HOME}/.gitconfig" ] && has_gitconfig=true
cat > "${HOME}/.config/codehalter/global.toml" <<EOF
# Host facts captured by install.sh, read by codehalter when scaffolding a
# devcontainer. Edit if your setup changes.
has_gitconfig_in_home = ${has_gitconfig}
EOF
echo "Wrote ${HOME}/.config/codehalter/global.toml (has_gitconfig_in_home=${has_gitconfig})"
