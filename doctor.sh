#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
The legacy root/system doctor wrapper is no longer supported.

Use the canonical diagnostics path instead:
  python -m agent doctor

For setup and recovery:
  python -m agent setup

Operator docs:
  docs/operator/SETUP.md
EOF

exit 1
