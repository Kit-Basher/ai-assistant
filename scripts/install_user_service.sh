#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNIT_SOURCE="${REPO_ROOT}/systemd/personal-agent-api.service"
UNIT_TARGET_DIR="${HOME}/.config/systemd/user"
UNIT_TARGET="${UNIT_TARGET_DIR}/personal-agent-api.service"

mkdir -p "${UNIT_TARGET_DIR}"
cp "${UNIT_SOURCE}" "${UNIT_TARGET}"
systemctl --user daemon-reload
systemctl --user enable --now personal-agent-api.service

echo "Installed and started personal-agent-api.service"
echo "Status: systemctl --user status personal-agent-api.service"
echo "Logs:   journalctl --user -u personal-agent-api.service -f"
