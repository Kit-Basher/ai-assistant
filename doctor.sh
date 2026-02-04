#!/usr/bin/env bash
set -euo pipefail

# Usage: ./doctor.sh <linux_username>
if [ "${#}" -ne 1 ]; then
  echo "Usage: $0 <linux_username>"
  exit 2
fi

TARGET_USER="$1"
REPO_DIR="/home/${TARGET_USER}/personal-agent"
ENV_FILE="/etc/personal-agent/${TARGET_USER}.env"
SERVICE_NAME="personal-agent@${TARGET_USER}.service"
DEFAULT_DB_PATH="${REPO_DIR}/memory/agent.db"

echo "=== Service status: ${SERVICE_NAME} ==="
if command -v systemctl >/dev/null 2>&1; then
  systemctl status "${SERVICE_NAME}" --no-pager || true
else
  echo "systemctl not available on this system."
fi

echo
echo "=== Environment file check ==="
if [ -f "${ENV_FILE}" ]; then
  echo "Env file found: ${ENV_FILE}"
else
  echo "Env file missing: ${ENV_FILE}"
fi

# Determine DB_PATH from env file if present (safe, don't execute arbitrary code)
DB_PATH="${DEFAULT_DB_PATH}"
if [ -f "${ENV_FILE}" ]; then
  # Attempt to parse a DB_PATH line like: DB_PATH="/path/to/db"
  parsed=$(grep -E '^\s*DB_PATH=' "${ENV_FILE}" || true)
  if [ -n "${parsed}" ]; then
    # strip DB_PATH= and any surrounding quotes
    DB_PATH=$(echo "${parsed}" | sed -E 's/^\s*DB_PATH=//; s/^"//; s/"$//; s/^'\''//; s/'\''$//')
  fi
fi

echo
echo "=== Python import check inside venv ==="
if [ -x "${REPO_DIR}/.venv/bin/python" ]; then
  if "${REPO_DIR}/.venv/bin/python" -c "import telegram_adapter" >/dev/null 2>&1; then
    echo "OK: telegram_adapter import succeeded inside venv"
  else
    echo "FAIL: telegram_adapter import failed inside venv. Activate the venv and check dependencies."
    echo "Try: sudo -u ${TARGET_USER} ${REPO_DIR}/.venv/bin/pip install -r ${REPO_DIR}/requirements.txt"
  fi
else
  echo "Venv python not found at ${REPO_DIR}/.venv/bin/python"
fi

echo
echo "=== SQLite pending_clarifications schema ==="
if [ -f "${DB_PATH}" ]; then
  echo "Found SQLite DB at ${DB_PATH}."
  if command -v sqlite3 >/dev/null 2>&1; then
    SQL="SELECT sql FROM sqlite_master WHERE type='table' AND name='pending_clarifications';"
    result=$(sqlite3 "${DB_PATH}" "${SQL}" || true)
    if [ -n "${result}" ]; then
      echo "Schema for pending_clarifications:"
      echo "${result}"
    else
      echo "Table 'pending_clarifications' not found in DB."
    fi
  else
    echo "sqlite3 client not found; cannot query DB schema (install sqlite3)."
  fi
else
  echo "SQLite DB not found at ${DB_PATH} (default: ${DEFAULT_DB_PATH})."
fi

echo
echo "Doctor checks complete."
