#!/usr/bin/env bash
set -euo pipefail

systemctl_bin="${AGENT_UNINSTALL_SYSTEMCTL:-systemctl}"
install_root="${XDG_DATA_HOME:-$HOME/.local/share}/personal-agent"
remove_state=0

die() {
    printf '%s\n' "Personal Agent Debian uninstaller: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --remove-state)
            remove_state=1
            ;;
        --install-root)
            shift
            install_root="${1-}"
            if [ -z "$install_root" ]; then
                die "--install-root requires a path"
            fi
            ;;
        -h|--help)
            cat <<'EOF'
Usage: personal-agent-uninstall [--remove-state] [--install-root PATH]

Stop and unregister the Personal Agent user service and optionally remove
user-owned state. Package-owned files are removed by apt/dpkg.
EOF
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

service_name="personal-agent-api.service"
service_path="$HOME/.config/systemd/user/$service_name"

if command -v "$systemctl_bin" >/dev/null 2>&1; then
    "$systemctl_bin" --user stop "$service_name" >/dev/null 2>&1 || true
    "$systemctl_bin" --user disable "$service_name" >/dev/null 2>&1 || true
fi
rm -f "$service_path"

if [ "$remove_state" -eq 1 ]; then
    rm -rf "$install_root"
    rm -rf "${XDG_CONFIG_HOME:-$HOME/.config}/personal-agent"
    printf '%s\n' "Removed Personal Agent user state at $install_root."
else
    mkdir -p "$install_root"
    printf '%s\n' "Removed Personal Agent user-service registration. User state preserved at $install_root."
fi
