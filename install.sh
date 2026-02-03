#!/usr/bin/env bash
set -euo pipefail

# Usage: ./install.sh <linux_username>
if [ "${#}" -ne 1 ]; then
  echo "Usage: $0 <linux_username>"
  exit 2
fi

TARGET_USER="$1"
REPO_DIR="/home/${TARGET_USER}/personal-agent"
ENV_DIR="/etc/personal-agent"
ENV_FILE="${ENV_DIR}/${TARGET_USER}.env"
SYSTEMD_UNIT="/etc/systemd/system/personal-agent@.service"
LOCAL_UNIT_PATH="packaging/personal-agent@.service"
EXAMPLE_ENV_PATH="packaging/example.env"

if [ "$(id -u)" -ne 0 ]; then
  echo "This installer must be run as root (sudo)."
  exit 1
fi

if [ ! -d "${REPO_DIR}" ]; then
  echo "Repository not found at ${REPO_DIR}. Please clone the repo to /home/${TARGET_USER}/personal-agent and re-run."
  exit 1
fi

echo "Installing OS packages (python3-venv, pip, sqlite3) if apt-get is present..."
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y python3-venv python3-pip sqlite3
else
  echo "Warning: apt-get not found. Please ensure python3-venv, pip and sqlite3 are installed on this system."
fi

# Create venv as the target user if missing
if [ ! -x "${REPO_DIR}/.venv/bin/python" ]; then
  echo "Creating Python venv at ${REPO_DIR}/.venv as ${TARGET_USER}..."
  sudo -u "${TARGET_USER}" python3 -m venv "${REPO_DIR}/.venv"
fi

# Install requirements if present (as target user to keep venv ownership correct)
if [ -f "${REPO_DIR}/requirements.txt" ]; then
  echo "Installing Python requirements into venv..."
  sudo -u "${TARGET_USER}" "${REPO_DIR}/.venv/bin/pip" install --upgrade pip
  sudo -u "${TARGET_USER}" "${REPO_DIR}/.venv/bin/pip" install -r "${REPO_DIR}/requirements.txt"
else
  echo "No requirements.txt found in repo; skipping pip install."
fi

# Ensure memory directory exists and is writable by user
MEMORY_DIR="${REPO_DIR}/memory"
mkdir -p "${MEMORY_DIR}"
chown -R "${TARGET_USER}:" "${MEMORY_DIR}"

# Ensure logs dir (optional)
LOGS_DIR="${REPO_DIR}/logs"
mkdir -p "${LOGS_DIR}"
chown -R "${TARGET_USER}:" "${LOGS_DIR}"

# Ensure environment dir
mkdir -p "${ENV_DIR}"
chmod 755 "${ENV_DIR}"

# Copy example env if target env does not already exist
if [ -f "${ENV_FILE}" ]; then
  echo "Env file ${ENV_FILE} already exists; leaving it in place."
else
  echo "Installing example env to ${ENV_FILE} (edit to add TELEGRAM_BOT_TOKEN)..."
  cp "${REPO_DIR}/${EXAMPLE_ENV_PATH}" "${ENV_FILE}"
  chmod 640 "${ENV_FILE}"
  chown root:root "${ENV_FILE}"
fi

# Install systemd unit (copy the template unit to systemd path)
if [ -f "${REPO_DIR}/${LOCAL_UNIT_PATH}" ]; then
  echo "Installing systemd unit to ${SYSTEMD_UNIT}..."
  cp "${REPO_DIR}/${LOCAL_UNIT_PATH}" "${SYSTEMD_UNIT}"
  chmod 644 "${SYSTEMD_UNIT}"
else
  echo "Warning: local unit file ${REPO_DIR}/${LOCAL_UNIT_PATH} not found. Ensure packaging/personal-agent@.service exists in the repo."
fi

# Reload systemd and enable service for user
if command -v systemctl >/dev/null 2>&1; then
  systemctl daemon-reload
  systemctl enable --now "personal-agent@${TARGET_USER}.service"
  echo "Service enabled and started: personal-agent@${TARGET_USER}.service"
else
  echo "systemctl not available; please enable and start personal-agent@${TARGET_USER}.service manually."
fi

echo "Installation complete. Edit ${ENV_FILE} to set TELEGRAM_BOT_TOKEN, then check service with:"
echo "  systemctl status personal-agent@${TARGET_USER}.service"