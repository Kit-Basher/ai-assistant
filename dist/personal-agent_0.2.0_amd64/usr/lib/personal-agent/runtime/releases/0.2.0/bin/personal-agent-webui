#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${AGENT_LAUNCHER_SERVICE_NAME:-personal-agent-api.service}"
WEBUI_URL="${AGENT_WEBUI_URL:-http://127.0.0.1:8765/}"
READY_PATH="${AGENT_LAUNCHER_READY_PATH:-/ready}"
WAIT_SECONDS="${AGENT_LAUNCHER_WAIT_SECONDS:-20}"
POLL_SECONDS="${AGENT_LAUNCHER_POLL_SECONDS:-1}"
SYSTEMCTL_BIN="${AGENT_LAUNCHER_SYSTEMCTL:-systemctl}"
CURL_BIN="${AGENT_LAUNCHER_CURL:-curl}"
XDG_OPEN_BIN="${AGENT_LAUNCHER_XDG_OPEN:-xdg-open}"

ready_url="${WEBUI_URL%/}${READY_PATH}"
open_url="${WEBUI_URL%/}/"

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
        die "the local UI did not become ready within ${WAIT_SECONDS}s. Check '$SERVICE_NAME' with 'systemctl --user status $SERVICE_NAME', then open $open_url manually."
    fi

    if ready_body="$("$CURL_BIN" --fail --silent --show-error --max-time 2 "$ready_url" 2>/dev/null)"; then
        if printf '%s' "$ready_body" | grep -Eq '"ready"[[:space:]]*:[[:space:]]*true'; then
            break
        fi
    fi

    sleep "$POLL_SECONDS"
done

need_command "$XDG_OPEN_BIN" "xdg-open is unavailable. Open $open_url manually."
if ! "$XDG_OPEN_BIN" "$open_url" >/dev/null 2>&1; then
    die "the UI is ready, but the browser opener failed. Open $open_url manually."
fi

log "Personal Agent is ready. Opening $open_url"
