#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${AGENT_WEBUI_PYTHON:-python3}"

cd "${REPO_ROOT}/desktop"
npm ci
npm run build
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/webui_build_manifest.py" write --repo-root "${REPO_ROOT}"
