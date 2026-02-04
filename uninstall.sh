#!/usr/bin/env bash
set -euo pipefail

# Usage: ./uninstall.sh <linux_username>
if [ "${#}" -ne 1 ]; then
  echo "Usage: $0 <linux_username>"
  exit 2
fi

TARGET_USER="$1"
ENV_FILE="/etc/personal-agent/${TARGET_USER}.env"
SYSTEMD_UNIT="/etc/systemd/system/personal-agent@.service"

if [ "$(id -u)" -ne 0 ]; then
  echo "This uninstaller must be run as root (sudo)."
  exit 1
fi

echo "Stopping service personal-agent@${TARGET_USER}.service if running..."
systemctl stop "personal-agent@${TARGET_USER}.service" || true
systemctl disable "personal-agent@${TARGET_USER}.service" || true

echo "Removing systemd unit file: ${SYSTEMD_UNIT}"
if [ -f "${SYSTEMD_UNIT}" ]; then
  rm -f "${SYSTEMD_UNIT}"
  systemctl daemon-reload
else
  echo "Systemd unit ${SYSTEMD_UNIT} not found; nothing to remove."
fi

if [ -f "${ENV_FILE}" ]; then
  echo "Removing env file: ${ENV_FILE}"
  rm -f "${ENV_FILE}"
else
  echo "Env file ${ENV_FILE} not found; nothing to remove."
fi

echo "Uninstall complete. Repository and data under /home/${TARGET_USER}/personal-agent have been left intact."
