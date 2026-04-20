#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
launcher_dir="$data_home/personal-agent/bin"
applications_dir="$data_home/applications"
icon_dir="$data_home/icons/hicolor/scalable/apps"
launcher_name="${AGENT_LAUNCHER_NAME:-personal-agent-webui-dev}"
desktop_name="${AGENT_LAUNCHER_DESKTOP_NAME:-personal-agent-dev}"
desktop_display_name="${AGENT_LAUNCHER_DISPLAY_NAME:-Personal Agent (Dev)}"
desktop_comment="${AGENT_LAUNCHER_COMMENT:-Open the checkout runtime in your default browser}"
service_name="${AGENT_LAUNCHER_SERVICE_NAME:-personal-agent-api-dev.service}"
webui_url="${AGENT_LAUNCHER_WEBUI_URL:-http://127.0.0.1:18765/}"
launcher_path="$launcher_dir/$launcher_name"
desktop_path="$applications_dir/$desktop_name.desktop"
icon_path="$icon_dir/personal-agent.svg"
desktop_template="$repo_root/packaging/personal-agent.desktop"
launcher_source="$repo_root/scripts/launch_webui.sh"
icon_source="$repo_root/assets/icons/personal-agent.svg"

mkdir -p "$launcher_dir" "$applications_dir" "$icon_dir" "$HOME/.local/bin"
cat > "$launcher_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export AGENT_LAUNCHER_SERVICE_NAME="$service_name"
export AGENT_WEBUI_URL="$webui_url"
exec "$launcher_source" "\$@"
EOF
chmod 755 "$launcher_path"
ln -sf "$launcher_path" "$HOME/.local/bin/$launcher_name"
install -m 644 "$icon_source" "$icon_path"

python3 - "$desktop_template" "$desktop_path" "$launcher_path" "$desktop_display_name" "$desktop_comment" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

template_path = Path(sys.argv[1])
desktop_path = Path(sys.argv[2])
launcher_path = str(sys.argv[3])
display_name = str(sys.argv[4])
comment = str(sys.argv[5])

text = template_path.read_text(encoding="utf-8")
rendered = text.replace("__PERSONAL_AGENT_LAUNCHER__", launcher_path)
rendered = rendered.replace("__PERSONAL_AGENT_NAME__", display_name)
rendered = rendered.replace("__PERSONAL_AGENT_COMMENT__", comment)
if "__PERSONAL_AGENT_LAUNCHER__" in rendered:
    raise SystemExit("desktop launcher placeholder was not replaced")
if "__PERSONAL_AGENT_NAME__" in rendered or "__PERSONAL_AGENT_COMMENT__" in rendered:
    raise SystemExit("desktop metadata placeholder was not replaced")
desktop_path.write_text(rendered, encoding="utf-8")
PY

printf '%s\n' "Installed dev desktop launcher at $desktop_path"
printf '%s\n' "Installed dev launcher script at $launcher_path"
printf '%s\n' "Installed icon at $icon_path"
