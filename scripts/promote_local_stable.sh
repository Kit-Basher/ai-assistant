#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outdir="${AGENT_PROMOTE_OUTDIR:-$repo_root/dist}"
install_root="${AGENT_PROMOTE_INSTALL_ROOT:-}"
service_name="${AGENT_PROMOTE_SERVICE_NAME:-personal-agent-api.service}"
ready_url="${AGENT_PROMOTE_READY_URL:-http://127.0.0.1:8765/ready}"
ready_timeout_seconds="${AGENT_PROMOTE_READY_TIMEOUT_SECONDS:-45}"

die() {
    printf '%s\n' "Personal Agent stable promotion: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --outdir)
            shift
            outdir="${1-}"
            if [ -z "$outdir" ]; then
                die "--outdir requires a path"
            fi
            ;;
        --install-root)
            shift
            install_root="${1-}"
            if [ -z "$install_root" ]; then
                die "--install-root requires a path"
            fi
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/promote_local_stable.sh [--outdir PATH] [--install-root PATH]

Build a release bundle from the checkout and install it as the stable runtime.
EOF
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

if ! command -v bash >/dev/null 2>&1; then
    die "bash is required."
fi

if ! command -v systemctl >/dev/null 2>&1; then
    die "systemctl is required to restart $service_name after promotion."
fi

if ! command -v curl >/dev/null 2>&1; then
    die "curl is required to wait for $ready_url after promotion."
fi

wait_for_ready() {
    local deadline
    local now
    local last_error=""
    deadline=$(( $(date +%s) + ready_timeout_seconds ))
    while true; do
        if curl -fsS --max-time 2 "$ready_url" >/dev/null 2>"$outdir/promote-ready-last-error.log"; then
            return 0
        fi
        last_error="$(cat "$outdir/promote-ready-last-error.log" 2>/dev/null || true)"
        now="$(date +%s)"
        if [ "$now" -ge "$deadline" ]; then
            printf '%s\n' "Timed out waiting for $ready_url after restarting $service_name." >&2
            if [ -n "$last_error" ]; then
                printf '%s\n' "Last curl error: $last_error" >&2
            fi
            printf '%s\n' "--- systemctl --user status $service_name ---" >&2
            systemctl --user status "$service_name" --no-pager >&2 || true
            printf '%s\n' "--- journalctl --user -u $service_name -n 80 --no-pager ---" >&2
            journalctl --user -u "$service_name" -n 80 --no-pager >&2 || true
            return 1
        fi
        sleep 1
    done
}

mapfile -t build_output < <("$repo_root/scripts/build_release_bundle.sh" --clean --outdir "$outdir")
bundle_dir="${build_output[0]-}"
archive_path="${build_output[1]-}"
checksum_path="${build_output[2]-}"

if [ -z "$bundle_dir" ] || [ ! -d "$bundle_dir" ]; then
    die "bundle build did not return a bundle directory"
fi

printf '%s\n' "Built bundle: $bundle_dir"
printf '%s\n' "Archive: $archive_path"
printf '%s\n' "Checksum: $checksum_path"
printf '%s\n' "Installing stable runtime from bundle..."

if [ -n "$install_root" ]; then
    bash "$bundle_dir/install.sh" --install-root "$install_root"
else
    bash "$bundle_dir/install.sh"
fi

printf '%s\n' "Restarting $service_name..."
systemctl --user restart "$service_name"
printf '%s\n' "Waiting for $ready_url..."
wait_for_ready || die "$service_name did not become ready after promotion"

printf '%s\n' "Stable runtime promotion complete."
printf '%s\n' "Note: restarting personal-agent-api.service does not load repo checkout edits when"
printf '%s\n' "that service points at runtime/current. Run bash scripts/promote_local_stable.sh"
printf '%s\n' "after checkout changes that should affect the stable API service."
printf '%s\n' "Verify with:"
printf '%s\n' "  curl -fsS $ready_url"
printf '%s\n' "  python -m agent split_status"
printf '%s\n' "  python scripts/split_smoke.py"
