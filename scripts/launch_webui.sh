#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${AGENT_LAUNCHER_SERVICE_NAME:-personal-agent-api.service}"
WEBUI_URL="${AGENT_WEBUI_URL:-http://127.0.0.1:8765/}"
READY_PATH="${AGENT_LAUNCHER_READY_PATH:-/ready}"
WAIT_SECONDS="${AGENT_LAUNCHER_WAIT_SECONDS:-20}"
WINDOW_CHECK_SECONDS="${AGENT_LAUNCHER_WINDOW_CHECK_SECONDS:-2}"
POLL_SECONDS="${AGENT_LAUNCHER_POLL_SECONDS:-1}"
SYSTEMCTL_BIN="${AGENT_LAUNCHER_SYSTEMCTL:-systemctl}"
CURL_BIN="${AGENT_LAUNCHER_CURL:-curl}"
XDG_OPEN_BIN="${AGENT_LAUNCHER_XDG_OPEN:-xdg-open}"
WINDOW_TITLE="${AGENT_LAUNCHER_WINDOW_TITLE:-Personal Agent Web UI}"
DISPLAY_URL="${WEBUI_URL%/}"
OPEN_URL="${DISPLAY_URL}/"
BROWSER_CANDIDATES=(
    "${AGENT_LAUNCHER_BROWSER_BIN:-}"
    firefox
    google-chrome
    chromium
    chromium-browser
    brave-browser
    microsoft-edge
)

ready_url="${DISPLAY_URL}${READY_PATH}"

log() {
    printf '%s\n' "$*" >&2
}

die() {
    log "Personal Agent launcher: $*"
    exit 1
}

need_command() {
    local command_name="$1"
    local help_text="$2"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        die "$help_text"
    fi
}

need_command "$SYSTEMCTL_BIN" "systemctl is required. Try: systemctl --user status ${SERVICE_NAME}."
need_command "$CURL_BIN" "curl is required to check readiness."

_curl_body() {
    "$CURL_BIN" --fail --silent --show-error --max-time 2 "$1" 2>/dev/null
}

_webui_is_ready() {
    local body="$1"
    printf '%s' "$body" | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true'
}

_webui_frontdoor_is_live() {
    local body="$1"
    printf '%s' "$body" | grep -Eq 'personal-agent-webui'
}

_browser_window_visible() {
    local title="$1"
    if command -v wmctrl >/dev/null 2>&1; then
        if wmctrl -lx 2>/dev/null | grep -Eqi "personal-agent-webui|${title}"; then
            return 0
        fi
    fi
    if command -v xdotool >/dev/null 2>&1; then
        if xdotool search --name "$title" >/dev/null 2>&1; then
            return 0
        fi
    fi
    local browser_bin=""
    local candidate
    for candidate in "${BROWSER_CANDIDATES[@]}"; do
        if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1; then
            browser_bin="$candidate"
            break
        fi
    done
    if [ -z "$browser_bin" ]; then
        return 1
    fi
    if pgrep -af "$browser_bin" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

_launch_browser_fallback() {
    local candidate
    for candidate in "${BROWSER_CANDIDATES[@]}"; do
        if [ -z "$candidate" ]; then
            continue
        fi
        if command -v "$candidate" >/dev/null 2>&1; then
            log "Personal Agent launcher: falling back to $candidate --new-window."
            "$candidate" --new-window "$OPEN_URL" >/dev/null 2>&1 &
            return 0
        fi
    done
    return 1
}

if "$SYSTEMCTL_BIN" --user is-active --quiet "$SERVICE_NAME" >/dev/null 2>&1; then
    :
else
    if "$SYSTEMCTL_BIN" --user is-enabled --quiet "$SERVICE_NAME" >/dev/null 2>&1; then
        log "Personal Agent launcher: starting $SERVICE_NAME"
        if ! "$SYSTEMCTL_BIN" --user start "$SERVICE_NAME" >/dev/null 2>&1; then
            die "could not start $SERVICE_NAME. If this is a fresh install, run 'python -m agent setup' and try again."
        fi
    else
        log "Personal Agent launcher: registering and starting $SERVICE_NAME"
        if ! "$SYSTEMCTL_BIN" --user enable --now "$SERVICE_NAME" >/dev/null 2>&1; then
            die "could not register $SERVICE_NAME. If this is a fresh install, run 'python -m agent setup' and try again."
        fi
    fi
fi

start_epoch="$(date +%s)"
deadline_epoch="$((start_epoch + WAIT_SECONDS))"

while :; do
    now_epoch="$(date +%s)"
    if [ "$now_epoch" -ge "$deadline_epoch" ]; then
        die "the local UI did not become ready within ${WAIT_SECONDS}s. Check '$SERVICE_NAME' with 'systemctl --user status $SERVICE_NAME', then open $OPEN_URL manually."
    fi

    if ready_body="$(_curl_body "$ready_url")"; then
        if _webui_is_ready "$ready_body"; then
            break
        fi
    fi

    if frontdoor_body="$(_curl_body "$OPEN_URL")"; then
        if _webui_frontdoor_is_live "$frontdoor_body"; then
            log "Personal Agent launcher: UI frontdoor is live; opening while /ready finishes warming up."
            break
        fi
    fi

    sleep "$POLL_SECONDS"
done

log "Opening Personal Agent UI at $DISPLAY_URL"
opened_with_xdg=0
if command -v "$XDG_OPEN_BIN" >/dev/null 2>&1; then
    if "$XDG_OPEN_BIN" "$OPEN_URL" >/dev/null 2>&1; then
        opened_with_xdg=1
        sleep "$WINDOW_CHECK_SECONDS"
        if _browser_window_visible "$WINDOW_TITLE"; then
            log "Personal Agent launcher: browser window detected."
            exit 0
        fi
        log "Personal Agent launcher: xdg-open did not surface a visible window; trying browser fallback."
    else
        log "Personal Agent launcher: xdg-open could not open the UI; trying browser fallback."
    fi
else
    log "Personal Agent launcher: xdg-open is unavailable; trying browser fallback."
fi

if _launch_browser_fallback; then
    sleep "$WINDOW_CHECK_SECONDS"
    if _browser_window_visible "$WINDOW_TITLE"; then
        log "Personal Agent launcher: browser fallback launched."
        exit 0
    fi
fi

if [ "$opened_with_xdg" -eq 1 ]; then
    die "xdg-open did not result in a visible browser window. Open $DISPLAY_URL manually, or run firefox --new-window $OPEN_URL."
fi

die "Could not open $DISPLAY_URL automatically. Open it manually, or run firefox --new-window $OPEN_URL or google-chrome --new-window $OPEN_URL."
