#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outdir="${AGENT_PROMOTE_OUTDIR:-$repo_root/dist}"
install_root="${AGENT_PROMOTE_INSTALL_ROOT:-}"

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

printf '%s\n' "Stable runtime promotion complete."
printf '%s\n' "Verify with:"
printf '%s\n' "  python -m agent split_status"
printf '%s\n' "  python scripts/split_smoke.py"
