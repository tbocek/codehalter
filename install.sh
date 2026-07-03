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

# Register codehalter as a Zed agent server. Zed keeps its settings at
# ~/.config/zed/settings.json on both Linux and macOS; an existing directory is
# the "Zed is installed" signal. The file is JSONC (Zed's default ships // header
# comments, and trailing commas are allowed), so a jq round-trip is out. Instead
# a small awk tokenizer scans the file with full string/escape/comment state, so
# "agent_servers" or "Codehalter" inside a comment or a string value never
# counts, only a real key (a string followed by ":") at the right nesting depth
# does. The entry is inserted textually, leaving every other byte of the file
# intact. The insertion may leave a trailing comma before a closing brace,
# which Zed accepts.
#
# awk exit codes: 0 = new content on stdout, 10 = Codehalter already
# configured, 11 = file shape not understood (no top-level object, or
# agent_servers maps to a non-object), anything else = awk itself failed.
ZED_DIR="${HOME}/.config/zed"
ZED_SETTINGS="${ZED_DIR}/settings.json"
ZED_AGENT_JSON='{
  "agent_servers": {
    "Codehalter": {
      "type": "custom",
      "command": "codehalter"
    }
  }
}'
if [ -d "$ZED_DIR" ]; then
    if [ ! -f "$ZED_SETTINGS" ] || ! grep -q '[^[:space:]]' "$ZED_SETTINGS"; then
        # Missing or blank file: write the whole object.
        printf '%s\n' "$ZED_AGENT_JSON" > "$ZED_SETTINGS"
        echo "Created ${ZED_SETTINGS} with the Codehalter agent server"
    else
        ZED_TMP="$(mktemp)"
        zed_status=0
        awk '
        { lines[NR] = $0 }
        END {
            # Single pass over the buffered file, character by character.
            # State: in_str/esc (inside a "..." string, respecting \"),
            # in_bc (inside /* */), depth (brace nesting), want_colon (a
            # string just closed; if the next meaningful char is ":" it was
            # a key), expect_as (the agent_servers key was seen; waiting for
            # its opening "{"). Line comments (//) skip the rest of the line.
            depth = 0; in_str = 0; esc = 0; in_bc = 0
            str = ""; want_colon = 0; expect_as = 0
            top_ln = 0; top_col = 0          # position of the top-level "{"
            as_ln = 0; as_col = 0            # position of agent_servers own "{"
            as_depth = 0; in_as = 0          # inside the agent_servers object
            found = 0; bad = 0
            for (ln = 1; ln <= NR && !bad; ln++) {
                line = lines[ln]; n = length(line)
                for (i = 1; i <= n; i++) {
                    c = substr(line, i, 1)
                    if (in_str) {
                        if (esc)            { esc = 0 }
                        else if (c == "\\") { esc = 1 }
                        else if (c == "\"") { in_str = 0; want_colon = 1 }
                        else                { str = str c }
                        continue
                    }
                    if (in_bc) {
                        if (c == "*" && substr(line, i+1, 1) == "/") { in_bc = 0; i++ }
                        continue
                    }
                    if (c == "/" && substr(line, i+1, 1) == "/") break
                    if (c == "/" && substr(line, i+1, 1) == "*") { in_bc = 1; i++; continue }
                    if (c == " " || c == "\t" || c == "\r") continue
                    if (want_colon) {
                        want_colon = 0
                        if (c == ":") {
                            if (str == "agent_servers" && depth == 1 && !as_ln) expect_as = 1
                            if (str == "Codehalter" && in_as && depth == as_depth) found = 1
                            continue
                        }
                        # Not a key, just a string value; c falls through below.
                    }
                    if (c == "\"") { in_str = 1; str = ""; continue }
                    if (expect_as) {
                        if (c == "{") {
                            depth++; as_ln = ln; as_col = i; as_depth = depth
                            in_as = 1; expect_as = 0
                            continue
                        }
                        bad = 1; break   # agent_servers maps to a non-object
                    }
                    if (c == "{") {
                        depth++
                        if (depth == 1 && !top_ln) { top_ln = ln; top_col = i }
                    } else if (c == "}") {
                        if (in_as && depth == as_depth) in_as = 0
                        depth--
                    }
                }
            }
            if (found) exit 10
            if (bad || (!as_ln && !top_ln)) exit 11
            if (as_ln) {
                # Insert the entry right after agent_servers opening "{",
                # indented one level past that line. Any same-line content
                # after the "{" moves to its own line, which JSONC allows.
                match(lines[as_ln], /^[ \t]*/)
                base = substr(lines[as_ln], 1, RLENGTH)
                ind = base "  "
                for (ln = 1; ln <= NR; ln++) {
                    if (ln != as_ln) { print lines[ln]; continue }
                    print substr(lines[ln], 1, as_col)
                    print ind "\"Codehalter\": {"
                    print ind "  \"type\": \"custom\","
                    print ind "  \"command\": \"codehalter\","
                    print ind "},"
                    rest = substr(lines[ln], as_col + 1)
                    if (rest != "") print base rest
                }
                exit 0
            }
            # No agent_servers key anywhere: insert the whole block right
            # after the top-level "{".
            for (ln = 1; ln <= NR; ln++) {
                if (ln != top_ln) { print lines[ln]; continue }
                print substr(lines[ln], 1, top_col)
                print "  \"agent_servers\": {"
                print "    \"Codehalter\": {"
                print "      \"type\": \"custom\","
                print "      \"command\": \"codehalter\","
                print "    },"
                print "  },"
                rest = substr(lines[ln], top_col + 1)
                if (rest != "") print rest
            }
            exit 0
        }' "$ZED_SETTINGS" > "$ZED_TMP" || zed_status=$?
        case "$zed_status" in
            0)
                # cat-over instead of mv: keeps the settings file inode,
                # permissions, and ownership (mktemp files are 0600).
                cat "$ZED_TMP" > "$ZED_SETTINGS"
                rm -f "$ZED_TMP"
                echo "Added the Codehalter agent server to ${ZED_SETTINGS}"
                ;;
            10)
                rm -f "$ZED_TMP"
                echo "Zed already has the Codehalter agent server configured"
                ;;
            *)
                rm -f "$ZED_TMP"
                echo "Could not update ${ZED_SETTINGS} automatically. Add this to it yourself:"
                echo "$ZED_AGENT_JSON"
                ;;
        esac
    fi
fi
