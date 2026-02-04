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
UNIT_SRC="${REPO_DIR}/packaging/personal-agent@.service"
UNIT_DST="/etc/systemd/system/personal-agent@.service"

if [ "$(id -u)" -ne 0 ]; then
  echo "This installer must be run as root (sudo)."
  exit 1
fi

if [ ! -d "${REPO_DIR}" ]; then
  echo "Repo not found at ${REPO_DIR}. Clone it there before installing."
  exit 1
fi

mkdir -p "${ENV_DIR}"

if [ ! -f "${ENV_FILE}" ]; then
  if [ ! -f "${REPO_DIR}/packaging/example.env" ]; then
    echo "Missing ${REPO_DIR}/packaging/example.env. Cannot create env file."
    exit 1
  fi
  cp "${REPO_DIR}/packaging/example.env" "${ENV_FILE}"
  echo "Created env file: ${ENV_FILE}"
else
  echo "Env file already exists: ${ENV_FILE} (not overwriting)"
fi

chown root:root "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

echo "Creating venv and installing dependencies as ${TARGET_USER}..."
sudo -u "${TARGET_USER}" bash -lc "python3 -m venv '${REPO_DIR}/.venv'"
sudo -u "${TARGET_USER}" bash -lc "'${REPO_DIR}/.venv/bin/pip' install -r '${REPO_DIR}/requirements.txt'"

if [ ! -f "${UNIT_SRC}" ]; then
  echo "Missing systemd unit template at ${UNIT_SRC}."
  exit 1
fi

cp "${UNIT_SRC}" "${UNIT_DST}"
systemctl daemon-reload
systemctl enable --now "personal-agent@${TARGET_USER}.service"

echo
echo "Install complete."
echo "Next steps:"
echo "1) Edit ${ENV_FILE} and set TELEGRAM_BOT_TOKEN (and any optional vars)."
echo "2) Check status: systemctl status personal-agent@${TARGET_USER}.service --no-pager"
echo "3) Tail logs: journalctl -u personal-agent@${TARGET_USER}.service -f"
echo "4) Doctor: ./doctor.sh ${TARGET_USER}"
