#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
The legacy root/system uninstall path is no longer supported.

Use the canonical user-service uninstall path instead:
1. Follow docs/operator/SETUP.md.
2. Stop and disable the user service:
     systemctl --user disable --now personal-agent-api.service
3. Remove the user service symlink and optional local state/config paths if desired.
EOF

exit 1
