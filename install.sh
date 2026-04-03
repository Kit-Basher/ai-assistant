#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
The legacy root/system install path is no longer supported.

Use the canonical user-service install path instead:
1. Follow README.md Quick Start.
2. Follow docs/operator/SETUP.md for first-run, upgrade, recovery, and uninstall.

The supported runtime is the user-level systemd service:
  personal-agent-api.service
EOF

exit 1
