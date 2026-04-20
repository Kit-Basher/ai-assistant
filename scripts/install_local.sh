#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AGENT_INSTALL_PYTHON:-python3}"
systemctl_bin="${AGENT_INSTALL_SYSTEMCTL:-systemctl}"
node_bin="${AGENT_INSTALL_NODE:-node}"
npm_bin="${AGENT_INSTALL_NPM:-npm}"
install_launcher=0
check_webui_build=0

die() {
    printf '%s\n' "Personal Agent install: $*" >&2
    exit 1
}

need_command() {
    local command_name="$1"
    local help_text="$2"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        die "$help_text"
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --desktop-launcher)
            install_launcher=1
            ;;
        --check-webui-build)
            check_webui_build=1
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/install_local.sh [--desktop-launcher] [--check-webui-build]

Developer checkout install for Personal Agent.
EOF
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

need_command "$python_bin" "python3 is required. Install Python 3.11 or newer, then rerun."
if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    die "Python 3.11 or newer is required. Install Python 3.11+, then rerun."
fi
need_command "$systemctl_bin" "systemctl is required for the user service path. Install systemd user support, then rerun."
if ! "$systemctl_bin" --user show-environment >/dev/null 2>&1; then
    die "systemd --user is not available. Start a login session or run: loginctl enable-linger \"$USER\"."
fi
if [ "$check_webui_build" -eq 1 ]; then
    need_command "$node_bin" "node is required to rebuild the web UI. Install Node.js, then rerun."
    need_command "$npm_bin" "npm is required to rebuild the web UI. Install npm, then rerun."
fi

cd "$repo_root"

if [ ! -x ".venv/bin/python" ]; then
    "$python_bin" -m venv .venv
fi

".venv/bin/python" -m pip install -e .
PERSONAL_AGENT_INSTANCE=dev ".venv/bin/python" -m agent doctor --fix
AGENT_USER_SERVICE_NAME=personal-agent-api-dev.service bash "$repo_root/scripts/install_user_service.sh"

if [ "$install_launcher" -eq 1 ]; then
    AGENT_LAUNCHER_NAME=personal-agent-webui-dev \
    AGENT_LAUNCHER_DESKTOP_NAME=personal-agent-dev \
    AGENT_LAUNCHER_DISPLAY_NAME='Personal Agent (Dev)' \
    AGENT_LAUNCHER_COMMENT='Open the checkout runtime in your default browser' \
    AGENT_LAUNCHER_SERVICE_NAME=personal-agent-api-dev.service \
    AGENT_LAUNCHER_WEBUI_URL=http://127.0.0.1:18765/ \
    bash "$repo_root/scripts/install_desktop_launcher.sh"
fi

printf '%s\n' "Personal Agent install complete."
printf '%s\n' "Open: http://127.0.0.1:18765/"
printf '%s\n' "Service: systemctl --user status personal-agent-api-dev.service"
if [ "$install_launcher" -eq 1 ]; then
    printf '%s\n' "Launcher: Personal Agent (Dev) or personal-agent-webui-dev"
fi
