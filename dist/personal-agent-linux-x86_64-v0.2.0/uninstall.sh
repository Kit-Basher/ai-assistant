#!/usr/bin/env bash
set -euo pipefail

python_bin="${AGENT_BUNDLE_UNINSTALL_PYTHON:-python3}"
systemctl_bin="${AGENT_BUNDLE_UNINSTALL_SYSTEMCTL:-systemctl}"
install_root="${XDG_DATA_HOME:-$HOME/.local/share}/personal-agent"
override_install_root=""
remove_state=0

die() {
    printf '%s\n' "Personal Agent bundle uninstall: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --install-root)
            shift
            override_install_root="${1-}"
            if [ -z "$override_install_root" ]; then
                die "--install-root requires a path"
            fi
            ;;
        --remove-state)
            remove_state=1
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash uninstall.sh [--install-root PATH] [--remove-state]

Remove the installed runtime, launcher, and service. User state is preserved
unless --remove-state is set.
EOF
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

if [ -n "$override_install_root" ]; then
    install_root="$override_install_root"
fi

runtime_root="$install_root/runtime"
current_root="$runtime_root/current"
stable_bin_root="$install_root/bin"
launcher_path="$stable_bin_root/personal-agent-webui"
uninstall_path="$stable_bin_root/personal-agent-uninstall"
desktop_path="${XDG_DATA_HOME:-$HOME/.local/share}/applications/personal-agent.desktop"
icon_path="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps/personal-agent.svg"
service_path="$HOME/.config/systemd/user/personal-agent-api.service"

if command -v "$systemctl_bin" >/dev/null 2>&1; then
    "$systemctl_bin" --user stop personal-agent-api.service >/dev/null 2>&1 || true
    "$systemctl_bin" --user disable personal-agent-api.service >/dev/null 2>&1 || true
    "$systemctl_bin" --user daemon-reload >/dev/null 2>&1 || true
fi

rm -f "$launcher_path" "$uninstall_path" "$desktop_path" "$icon_path" "$service_path"
rm -rf "$runtime_root" "$stable_bin_root"

if [ "$remove_state" -eq 1 ]; then
    rm -rf "$install_root"
    printf '%s\n' "Removed Personal Agent and all local state."
else
    mkdir -p "$install_root"
    printf '%s\n' "Removed Personal Agent runtime. User state was preserved at $install_root."
fi

