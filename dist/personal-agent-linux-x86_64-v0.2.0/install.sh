#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${AGENT_BUNDLE_INSTALL_PYTHON:-python3}"
systemctl_bin="${AGENT_BUNDLE_INSTALL_SYSTEMCTL:-systemctl}"
xdg_open_bin="${AGENT_BUNDLE_INSTALL_XDG_OPEN:-xdg-open}"
install_root="${XDG_DATA_HOME:-$HOME/.local/share}/personal-agent"
override_install_root=""

die() {
    printf '%s\n' "Personal Agent bundle install: $*" >&2
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
        --install-root)
            shift
            override_install_root="${1-}"
            if [ -z "$override_install_root" ]; then
                die "--install-root requires a path"
            fi
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash install.sh [--install-root PATH]

Install the release bundle into a user-local runtime root.
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

version="$(tr -d ' \n' < "$bundle_root/VERSION")"
[ -n "$version" ] || die "bundle VERSION file is missing or empty"

need_command "$python_bin" "python3 is required. Install Python 3.11 or newer, then rerun."
if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
    die "Python 3.11 or newer is required. Install Python 3.11+, then rerun."
fi
need_command "$systemctl_bin" "systemctl is required for the bundled user-service path. Install systemd user support, then rerun."
if ! "$systemctl_bin" --user show-environment >/dev/null 2>&1; then
    die "systemd --user is not available. Start a login session or run: loginctl enable-linger \"$USER\"."
fi
need_command "$xdg_open_bin" "xdg-open is required for the desktop launcher. Install it, then rerun."

state_root="$install_root"
runtime_root="$install_root/runtime"
releases_root="$runtime_root/releases"
release_root="$releases_root/$version"
current_root="$runtime_root/current"
stable_bin_root="$install_root/bin"
launcher_path="$stable_bin_root/personal-agent-webui"
uninstall_path="$stable_bin_root/personal-agent-uninstall"
desktop_root="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
desktop_path="$desktop_root/personal-agent.desktop"
icon_root="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
icon_path="$icon_root/personal-agent.svg"
service_dir="$HOME/.config/systemd/user"
service_path="$service_dir/personal-agent-api.service"

if "$systemctl_bin" --user is-active --quiet personal-agent-api.service >/dev/null 2>&1; then
    "$systemctl_bin" --user stop personal-agent-api.service >/dev/null 2>&1 || true
fi

mkdir -p "$releases_root" "$stable_bin_root" "$desktop_root" "$icon_root" "$service_dir"
rm -rf "$release_root"
cp -a "$bundle_root/payload/." "$release_root/"
install -m 755 "$bundle_root/uninstall.sh" "$release_root/bin/personal-agent-uninstall"

"$python_bin" -m venv "$release_root/.venv"
"$release_root/.venv/bin/python" -m pip install "$release_root"
"$release_root/.venv/bin/python" -m agent doctor --fix

ln -sfn "$release_root" "$current_root"
ln -sfn "$current_root/bin/personal-agent-webui" "$launcher_path"
ln -sfn "$current_root/bin/personal-agent-uninstall" "$uninstall_path"
install -m 644 "$release_root/assets/icons/personal-agent.svg" "$icon_path"

cat > "$desktop_path" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Personal Agent
Comment=Open Personal Agent in your default browser
Exec=$launcher_path
TryExec=$launcher_path
Icon=personal-agent
Terminal=false
Categories=Utility;Office;
StartupNotify=true
EOF

cat > "$service_path" <<EOF
[Unit]
Description=Personal Agent API
After=network.target

[Service]
Type=simple
WorkingDirectory=$current_root
ExecStart=$current_root/.venv/bin/python -m agent.api_server --host 127.0.0.1 --port 8765
Restart=on-failure
RestartSec=2
Environment=LLM_REGISTRY_PATH=$state_root/llm_registry.json
Environment=AGENT_SECRET_STORE_PATH=$state_root/secrets.enc.json
Environment=LLM_USAGE_STATS_PATH=$state_root/llm_usage_stats.json
Environment=AGENT_DB_PATH=$state_root/agent.db
Environment=AGENT_LOG_PATH=$state_root/agent.jsonl
Environment=AGENT_PERMISSIONS_PATH=$state_root/permissions.json
Environment=AGENT_AUDIT_LOG_PATH=$state_root/audit.jsonl
Environment=AGENT_SKILLS_PATH=$current_root/skills
Environment=AGENT_WEBUI_DIST_PATH=$current_root/agent/webui/dist
Environment=PERSONAL_AGENT_RUNTIME_ROOT=$current_root
Environment=PERSONAL_AGENT_INSTANCE=stable
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

cat > "$runtime_root/install-manifest.json" <<EOF
{
  "bundle_version": "$(printf '%s' "$version")",
  "installed_version": "$(printf '%s' "$version")",
  "install_root": "$(printf '%s' "$install_root")",
  "runtime_root": "$(printf '%s' "$runtime_root")",
  "release_root": "$(printf '%s' "$release_root")",
  "current_root": "$(printf '%s' "$current_root")"
}
EOF

"$systemctl_bin" --user daemon-reload
"$systemctl_bin" --user enable --now personal-agent-api.service

printf '%s\n' "Personal Agent bundle install complete."
printf '%s\n' "Installed version: $version"
printf '%s\n' "Open: http://127.0.0.1:8765/"
printf '%s\n' "Launcher: personal-agent-webui or the Personal Agent menu entry"
printf '%s\n' "Service: systemctl --user status personal-agent-api.service"
