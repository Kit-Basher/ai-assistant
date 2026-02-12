#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="system"
DRY_RUN="false"
ACTION="install"

if [[ "${1:-}" == "uninstall" ]]; then
  ACTION="uninstall"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --user)
      MODE="user"
      shift
      ;;
    --system)
      MODE="system"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Refusing to run as root. Please run as a normal user; sudo will be used only when needed."
  exit 1
fi

RUN_AS_USER="$(id -un)"
RUN_AS_GROUP="$(id -gn)"

if [[ "${MODE}" == "system" ]]; then
  SYSTEMD_DIR="/etc/systemd/system"
  ENV_DIR="/etc/personal-agent"
  OPS_CONFIG_TARGET="${ENV_DIR}/ops_config.json"
  UNIT_DIR_TARGET="${SYSTEMD_DIR}"
  DROPIN_AGENT_DIR="${SYSTEMD_DIR}/personal-agent.service.d"
  DROPIN_SUPERVISOR_DIR="${SYSTEMD_DIR}/personal-agent-supervisor.service.d"
  SYSTEMCTL_CMD=(systemctl)
else
  SYSTEMD_DIR="${HOME}/.config/systemd/user"
  ENV_DIR="${HOME}/.config/personal-agent"
  OPS_CONFIG_TARGET="${ENV_DIR}/ops_config.json"
  UNIT_DIR_TARGET="${SYSTEMD_DIR}"
  DROPIN_AGENT_DIR="${SYSTEMD_DIR}/personal-agent.service.d"
  DROPIN_SUPERVISOR_DIR="${SYSTEMD_DIR}/personal-agent-supervisor.service.d"
  SYSTEMCTL_CMD=(systemctl --user)
fi

AGENT_UNIT="personal-agent.service"
SUPERVISOR_UNIT="personal-agent-supervisor.service"
OBSERVE_UNIT="personal-agent-observe.service"
OBSERVE_TIMER="personal-agent-observe.timer"

AGENT_ENV="${ENV_DIR}/agent.env"
SUPERVISOR_ENV="${ENV_DIR}/supervisor.env"

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] $*"
    return 0
  fi
  "$@"
}

sudo_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] sudo $*"
    return 0
  fi
  sudo "$@"
}

confirm() {
  local prompt="$1"
  read -r -p "${prompt} [y/N]: " reply
  [[ "${reply}" == "y" || "${reply}" == "Y" ]]
}

backup_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    local ts
    ts="$(date +"%Y%m%d%H%M%S")"
    local backup="${path}.bak.${ts}"
    sudo_cmd cp "${path}" "${backup}"
    echo "Backup created: ${backup}"
  fi
}

ensure_dir() {
  local path="$1"
  if [[ "${MODE}" == "system" ]]; then
    sudo_cmd mkdir -p "${path}"
  else
    run_cmd mkdir -p "${path}"
  fi
}

write_file() {
  local content="$1"
  local target="$2"
  if [[ "${MODE}" == "system" ]]; then
    run_cmd bash -c "cat <<'EOF' | sudo tee \"${target}\" >/dev/null
${content}
EOF"
  else
    run_cmd bash -c "cat <<'EOF' > \"${target}\"
${content}
EOF"
  fi
}

install_env_file() {
  local src="$1"
  local dest="$2"
  if [[ -f "${dest}" ]]; then
    echo "Env exists: ${dest}"
    return 0
  fi
  if [[ "${MODE}" == "system" ]]; then
    sudo_cmd install -m 600 "${src}" "${dest}"
  else
    run_cmd install -m 600 "${src}" "${dest}"
  fi
  echo "Installed env: ${dest}"
}

set_env_kv() {
  local file="$1"
  local key="$2"
  local value="$3"
  if grep -q "^${key}=" "${file}" 2>/dev/null; then
    return 0
  fi
  if [[ "${MODE}" == "system" ]]; then
    run_cmd sudo bash -c "umask 077; echo '${key}=${value}' >> '${file}'"
    run_cmd sudo chmod 600 "${file}"
  else
    run_cmd bash -c "echo '${key}=${value}' >> '${file}'"
  fi
}

read_env_kv() {
  local file="$1"
  local key="$2"
  if [[ ! -f "${file}" ]]; then
    return 1
  fi
  grep -E "^${key}=" "${file}" | tail -n 1 | cut -d'=' -f2-
}

generate_hmac_key() {
  python3 - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
}

print_plan_install() {
  cat <<EOF
Install mode: ${MODE}
Repo: ${REPO_ROOT}
Unit target: ${UNIT_DIR_TARGET}
Env dir: ${ENV_DIR}
Actions:
- Copy unit files to ${UNIT_DIR_TARGET}
- Install env files to ${ENV_DIR} (600 perms)
- Install ops_config.json to ${OPS_CONFIG_TARGET}
- Create systemd drop-ins for user/group + working dir
- systemctl daemon-reload
- systemctl enable --now ${SUPERVISOR_UNIT} then ${AGENT_UNIT}
EOF
}

print_plan_uninstall() {
  cat <<EOF
Uninstall mode: ${MODE}
Actions:
- Disable services
- Remove unit files and drop-ins
- systemctl daemon-reload
- Env files kept unless you confirm removal
EOF
}

if [[ "${ACTION}" == "uninstall" ]]; then
  print_plan_uninstall
  confirm "Proceed with uninstall?" || exit 0
  if [[ "${MODE}" == "system" ]]; then
    sudo_cmd "${SYSTEMCTL_CMD[@]}" disable --now "${AGENT_UNIT}" "${SUPERVISOR_UNIT}" "${OBSERVE_TIMER}" || true
    sudo_cmd rm -f "${UNIT_DIR_TARGET}/${AGENT_UNIT}" "${UNIT_DIR_TARGET}/${SUPERVISOR_UNIT}" "${UNIT_DIR_TARGET}/${OBSERVE_UNIT}" "${UNIT_DIR_TARGET}/${OBSERVE_TIMER}"
    sudo_cmd rm -rf "${DROPIN_AGENT_DIR}" "${DROPIN_SUPERVISOR_DIR}"
    sudo_cmd "${SYSTEMCTL_CMD[@]}" daemon-reload
  else
    run_cmd "${SYSTEMCTL_CMD[@]}" disable --now "${AGENT_UNIT}" "${SUPERVISOR_UNIT}" "${OBSERVE_TIMER}" || true
    run_cmd rm -f "${UNIT_DIR_TARGET}/${AGENT_UNIT}" "${UNIT_DIR_TARGET}/${SUPERVISOR_UNIT}" "${UNIT_DIR_TARGET}/${OBSERVE_UNIT}" "${UNIT_DIR_TARGET}/${OBSERVE_TIMER}"
    run_cmd rm -rf "${DROPIN_AGENT_DIR}" "${DROPIN_SUPERVISOR_DIR}"
    run_cmd "${SYSTEMCTL_CMD[@]}" daemon-reload
  fi
  if confirm "Remove env files in ${ENV_DIR}?" ; then
    if [[ "${MODE}" == "system" ]]; then
      sudo_cmd rm -f "${AGENT_ENV}" "${SUPERVISOR_ENV}" "${OPS_CONFIG_TARGET}"
    else
      run_cmd rm -f "${AGENT_ENV}" "${SUPERVISOR_ENV}" "${OPS_CONFIG_TARGET}"
    fi
  fi
  echo "Uninstall complete."
  exit 0
fi

print_plan_install
confirm "Proceed with install?" || exit 0

ensure_dir "${ENV_DIR}"
ensure_dir "${UNIT_DIR_TARGET}"
ensure_dir "${DROPIN_AGENT_DIR}"
ensure_dir "${DROPIN_SUPERVISOR_DIR}"

if [[ ! -f "${OPS_CONFIG_TARGET}" ]]; then
  if [[ "${MODE}" == "system" ]]; then
    sudo_cmd install -m 644 "${REPO_ROOT}/ops/ops_config.json" "${OPS_CONFIG_TARGET}"
  else
    run_cmd install -m 644 "${REPO_ROOT}/ops/ops_config.json" "${OPS_CONFIG_TARGET}"
  fi
  echo "Installed ops config: ${OPS_CONFIG_TARGET}"
else
  echo "Ops config exists: ${OPS_CONFIG_TARGET}"
fi

install_env_file "${REPO_ROOT}/ops/agent.env.example" "${AGENT_ENV}"
install_env_file "${REPO_ROOT}/ops/supervisor.env.example" "${SUPERVISOR_ENV}"

HMAC_KEY="$(read_env_kv "${SUPERVISOR_ENV}" "SUPERVISOR_HMAC_KEY" || true)"
if [[ -z "${HMAC_KEY}" ]]; then
  HMAC_KEY="$(generate_hmac_key)"
  backup_file "${SUPERVISOR_ENV}"
  set_env_kv "${SUPERVISOR_ENV}" "SUPERVISOR_HMAC_KEY" "${HMAC_KEY}"
  echo "Generated SUPERVISOR_HMAC_KEY."
fi

AGENT_HMAC="$(read_env_kv "${AGENT_ENV}" "SUPERVISOR_HMAC_KEY" || true)"
if [[ -z "${AGENT_HMAC}" ]]; then
  backup_file "${AGENT_ENV}"
  set_env_kv "${AGENT_ENV}" "SUPERVISOR_HMAC_KEY" "${HMAC_KEY}"
  echo "Updated agent SUPERVISOR_HMAC_KEY."
elif [[ "${AGENT_HMAC}" != "${HMAC_KEY}" ]]; then
  echo "Agent and supervisor HMAC keys differ."
  if confirm "Update agent env to match supervisor key?"; then
    backup_file "${AGENT_ENV}"
    if [[ "${MODE}" == "system" ]]; then
      run_cmd sudo bash -c "sed -i 's/^SUPERVISOR_HMAC_KEY=.*/SUPERVISOR_HMAC_KEY=${HMAC_KEY}/' '${AGENT_ENV}'"
    else
      run_cmd sed -i "s/^SUPERVISOR_HMAC_KEY=.*/SUPERVISOR_HMAC_KEY=${HMAC_KEY}/" "${AGENT_ENV}"
    fi
  fi
fi

set_env_kv "${AGENT_ENV}" "OPS_CONFIG_PATH" "${OPS_CONFIG_TARGET}"

if confirm "Set TELEGRAM_BOT_TOKEN now?"; then
  read -r -p "TELEGRAM_BOT_TOKEN: " telegram_token
  if [[ -n "${telegram_token}" ]]; then
    backup_file "${AGENT_ENV}"
    if [[ "${MODE}" == "system" ]]; then
      run_cmd sudo bash -c "sed -i 's/^TELEGRAM_BOT_TOKEN=.*/TELEGRAM_BOT_TOKEN=${telegram_token}/' '${AGENT_ENV}'"
    else
      run_cmd sed -i "s/^TELEGRAM_BOT_TOKEN=.*/TELEGRAM_BOT_TOKEN=${telegram_token}/" "${AGENT_ENV}"
    fi
  fi
fi

if [[ "${MODE}" == "system" ]]; then
  sudo_cmd install -m 644 "${REPO_ROOT}/ops/systemd/personal-agent.service" "${UNIT_DIR_TARGET}/${AGENT_UNIT}"
  sudo_cmd install -m 644 "${REPO_ROOT}/ops/systemd/personal-agent-supervisor.service" "${UNIT_DIR_TARGET}/${SUPERVISOR_UNIT}"
  sudo_cmd install -m 644 "${REPO_ROOT}/ops/systemd/personal-agent-observe.service" "${UNIT_DIR_TARGET}/${OBSERVE_UNIT}"
  sudo_cmd install -m 644 "${REPO_ROOT}/ops/systemd/personal-agent-observe.timer" "${UNIT_DIR_TARGET}/${OBSERVE_TIMER}"
else
  run_cmd install -m 644 "${REPO_ROOT}/ops/systemd/personal-agent.service" "${UNIT_DIR_TARGET}/${AGENT_UNIT}"
  run_cmd install -m 644 "${REPO_ROOT}/ops/systemd/personal-agent-supervisor.service" "${UNIT_DIR_TARGET}/${SUPERVISOR_UNIT}"
  run_cmd install -m 644 "${REPO_ROOT}/ops/systemd/personal-agent-observe.service" "${UNIT_DIR_TARGET}/${OBSERVE_UNIT}"
  run_cmd install -m 644 "${REPO_ROOT}/ops/systemd/personal-agent-observe.timer" "${UNIT_DIR_TARGET}/${OBSERVE_TIMER}"
fi

DROPIN_AGENT_CONTENT="[Service]
User=${RUN_AS_USER}
Group=${RUN_AS_GROUP}
WorkingDirectory=${REPO_ROOT}
ExecStart=
ExecStart=${REPO_ROOT}/.venv/bin/python -m telegram_adapter
"
DROPIN_SUPERVISOR_CONTENT="[Service]
User=${RUN_AS_USER}
Group=${RUN_AS_GROUP}
WorkingDirectory=${REPO_ROOT}
ExecStart=
ExecStart=${REPO_ROOT}/.venv/bin/python ${REPO_ROOT}/ops/supervisor.py
"

write_file "${DROPIN_AGENT_CONTENT}" "${DROPIN_AGENT_DIR}/override.conf"
write_file "${DROPIN_SUPERVISOR_CONTENT}" "${DROPIN_SUPERVISOR_DIR}/override.conf"

if [[ "${MODE}" == "system" ]]; then
  sudo_cmd "${SYSTEMCTL_CMD[@]}" daemon-reload
  sudo_cmd "${SYSTEMCTL_CMD[@]}" enable --now "${SUPERVISOR_UNIT}"
  sudo_cmd "${SYSTEMCTL_CMD[@]}" enable --now "${AGENT_UNIT}"
  sudo_cmd "${SYSTEMCTL_CMD[@]}" enable --now "${OBSERVE_TIMER}"
else
  run_cmd "${SYSTEMCTL_CMD[@]}" daemon-reload
  run_cmd "${SYSTEMCTL_CMD[@]}" enable --now "${SUPERVISOR_UNIT}"
  run_cmd "${SYSTEMCTL_CMD[@]}" enable --now "${AGENT_UNIT}"
  run_cmd "${SYSTEMCTL_CMD[@]}" enable --now "${OBSERVE_TIMER}"
fi

cat <<EOF
Install complete.
Next steps:
- Check status: ${SYSTEMCTL_CMD[*]} status ${AGENT_UNIT}
- Check supervisor: ${SYSTEMCTL_CMD[*]} status ${SUPERVISOR_UNIT}
- Check observe timer: ${SYSTEMCTL_CMD[*]} status ${OBSERVE_TIMER}
- Logs: ${SYSTEMCTL_CMD[*]} status ${AGENT_UNIT}; journalctl -u ${AGENT_UNIT} -n 50 --no-pager
EOF
