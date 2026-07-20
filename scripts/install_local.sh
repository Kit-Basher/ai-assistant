#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outdir="${AGENT_LOCAL_INSTALL_OUTDIR:-$repo_root/dist}"
install_root=""

die() {
    printf '%s\n' "Personal Agent local install: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --install-root)
            shift
            install_root="${1-}"
            [ -n "$install_root" ] || die "--install-root requires a path"
            ;;
        --desktop-launcher|--check-webui-build)
            # Compatibility no-op: the stable install always builds the UI and installs its launcher.
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/install_local.sh [--install-root PATH]

Build the tracked web UI, create a release bundle, and install the stable
Personal Agent user service on http://127.0.0.1:8765/.

Developer checkout runtime: bash scripts/install_dev.sh --desktop-launcher
EOF
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

cd "$repo_root"
bash "$repo_root/scripts/build_webui.sh"
mapfile -t build_output < <(bash "$repo_root/scripts/build_release_bundle.sh" --clean --outdir "$outdir")
bundle_dir="${build_output[0]-}"
[ -n "$bundle_dir" ] && [ -d "$bundle_dir" ] || die "release bundle build did not produce an installable directory"

if [ -n "$install_root" ]; then
    bash "$bundle_dir/install.sh" --install-root "$install_root"
else
    bash "$bundle_dir/install.sh"
fi

printf '%s\n' "Stable local installation complete."
printf '%s\n' "Open: http://127.0.0.1:8765/"
printf '%s\n' "Service: systemctl --user status personal-agent-api.service"
