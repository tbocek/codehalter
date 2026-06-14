#!/usr/bin/env bash
set -euo pipefail

REPO="tbocek/codehalter"
BINARY="codehalter"

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)   OS="linux" ;;
    Darwin*)  OS="darwin" ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
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
if [ "$OS" = "windows" ]; then
    ASSET="${BINARY}-${OS}-${ARCH}.exe"
else
    ASSET="${BINARY}-${OS}-${ARCH}"
fi

# Resolve latest release tag
LATEST_TAG="$(curl -sL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name":' | sed -E 's/.*"tag_name": *"?([^"]+)".*/\1/')"
if [ -z "$LATEST_TAG" ]; then
    echo "Failed to resolve latest release tag" >&2
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
