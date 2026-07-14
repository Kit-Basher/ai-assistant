#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"

cd "$repo_root"

if [ ! -x ".venv/bin/python" ]; then
    "$python_bin" -m venv .venv
fi

".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -e '.[test,release]'

if [ -f "desktop/package.json" ]; then
    (cd desktop && npm ci && npm run build)
fi

printf 'repo=%s\n' "$repo_root"
".venv/bin/python" --version
".venv/bin/python" -m pip --version
if command -v node >/dev/null 2>&1; then
    node --version
fi
if command -v npm >/dev/null 2>&1; then
    npm --version
fi
