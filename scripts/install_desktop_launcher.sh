#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
launcher_dir="$data_home/personal-agent/bin"
applications_dir="$data_home/applications"
icon_dir="$data_home/icons/hicolor/scalable/apps"
launcher_path="$launcher_dir/personal-agent-webui"
desktop_path="$applications_dir/personal-agent.desktop"
icon_path="$icon_dir/personal-agent.svg"
desktop_template="$repo_root/packaging/personal-agent.desktop"
launcher_source="$repo_root/scripts/launch_webui.sh"
icon_source="$repo_root/assets/icons/personal-agent.svg"

mkdir -p "$launcher_dir" "$applications_dir" "$icon_dir" "$HOME/.local/bin"
install -m 755 "$launcher_source" "$launcher_path"
ln -sf "$launcher_path" "$HOME/.local/bin/personal-agent-webui"
install -m 644 "$icon_source" "$icon_path"

python3 - "$desktop_template" "$desktop_path" "$launcher_path" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

template_path = Path(sys.argv[1])
desktop_path = Path(sys.argv[2])
launcher_path = str(sys.argv[3])

text = template_path.read_text(encoding="utf-8")
rendered = text.replace("__PERSONAL_AGENT_LAUNCHER__", launcher_path)
if "__PERSONAL_AGENT_LAUNCHER__" in rendered:
    raise SystemExit("desktop launcher placeholder was not replaced")
desktop_path.write_text(rendered, encoding="utf-8")
PY

printf '%s\n' "Installed desktop launcher at $desktop_path"
printf '%s\n' "Installed launcher script at $launcher_path"
printf '%s\n' "Installed icon at $icon_path"
