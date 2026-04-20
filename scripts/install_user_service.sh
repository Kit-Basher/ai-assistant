#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UNIT_TARGET_DIR="${HOME}/.config/systemd/user"
SERVICE_NAME="${AGENT_USER_SERVICE_NAME:-personal-agent-api.service}"
UNIT_SOURCE="${REPO_ROOT}/systemd/${SERVICE_NAME}"
UNIT_TARGET="${UNIT_TARGET_DIR}/${SERVICE_NAME}"

mkdir -p "${UNIT_TARGET_DIR}"
ln -sf "${UNIT_SOURCE}" "${UNIT_TARGET}"
systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE_NAME}"

echo "Installed and started ${SERVICE_NAME}"
echo "Status: systemctl --user status ${SERVICE_NAME}"
echo "Logs:   journalctl --user -u ${SERVICE_NAME} -f"
