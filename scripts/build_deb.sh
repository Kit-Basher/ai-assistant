#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outdir="dist"
override_repo_root=""
arch="$(dpkg-architecture -qDEB_HOST_ARCH 2>/dev/null || printf '%s' amd64)"
clean=0

die() {
    printf '%s\n' "Personal Agent Debian package: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-root)
            shift
            override_repo_root="${1-}"
            if [ -z "$override_repo_root" ]; then
                die "--repo-root requires a path"
            fi
            ;;
        --outdir)
            shift
            outdir="${1-}"
            if [ -z "$outdir" ]; then
                die "--outdir requires a path"
            fi
            ;;
        --architecture)
            shift
            arch="${1-}"
            if [ -z "$arch" ]; then
                die "--architecture requires a value"
            fi
            ;;
        --clean)
            clean=1
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/build_deb.sh [--repo-root PATH] [--outdir PATH] [--architecture ARCH] [--clean]

Build a Debian package that installs the Personal Agent runtime into
/usr/lib/personal-agent and wires the desktop entry, launcher, and user service.
EOF
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
    shift
done

if [ -n "$override_repo_root" ]; then
    repo_root="$override_repo_root"
fi

repo_root="$(cd "$repo_root" && pwd)"
mkdir -p "$outdir"
outdir="$(cd "$outdir" && pwd)"

version="$(tr -d ' \n' < "$repo_root/VERSION")"
[ -n "$version" ] || die "VERSION file is missing or empty"

package_name="personal-agent"
package_version="$version"
stage_root="$outdir/${package_name}_${package_version}_${arch}"
runtime_root="$stage_root/usr/lib/personal-agent/runtime"
release_root="$runtime_root/releases/$package_version"
package_doc_root="$stage_root/usr/share/doc/$package_name"
desktop_dir="$stage_root/usr/share/applications"
icon_dir="$stage_root/usr/share/icons/hicolor/scalable/apps"
user_bin_dir="$stage_root/usr/bin"
systemd_user_dir="$stage_root/usr/lib/systemd/user"
debian_dir="$stage_root/DEBIAN"
deb_path="$outdir/${package_name}_${package_version}_${arch}.deb"
checksum_path="$deb_path.sha256"

required_inputs=(
    "$repo_root/agent"
    "$repo_root/memory"
    "$repo_root/skills"
    "$repo_root/telegram_adapter"
    "$repo_root/agent/webui/dist/index.html"
    "$repo_root/assets/icons/personal-agent.svg"
    "$repo_root/scripts/launch_webui.sh"
    "$repo_root/packaging/debian/personal-agent-uninstall.sh"
    "$repo_root/packaging/debian/personal-agent-api.service.in"
    "$repo_root/packaging/personal-agent.desktop"
    "$repo_root/llm_registry.json"
)
for path in "${required_inputs[@]}"; do
    [ -e "$path" ] || die "required packaging input missing: $path"
done

if [ "$clean" -eq 1 ]; then
    rm -rf "$stage_root" "$deb_path" "$checksum_path"
fi
rm -rf "$stage_root"
mkdir -p "$release_root" "$package_doc_root" "$desktop_dir" "$icon_dir" "$user_bin_dir" "$systemd_user_dir" "$debian_dir"

python3 - "$repo_root" "$release_root" "$package_doc_root" "$desktop_dir" "$icon_dir" "$user_bin_dir" "$systemd_user_dir" "$version" <<'PY'
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import sys

repo_root = Path(sys.argv[1]).resolve()
release_root = Path(sys.argv[2]).resolve()
package_doc_root = Path(sys.argv[3]).resolve()
desktop_dir = Path(sys.argv[4]).resolve()
icon_dir = Path(sys.argv[5]).resolve()
user_bin_dir = Path(sys.argv[6]).resolve()
systemd_user_dir = Path(sys.argv[7]).resolve()
version = sys.argv[8]
installed_runtime_root = "/usr/lib/personal-agent/runtime/current"

required = [
    repo_root / "agent",
    repo_root / "memory",
    repo_root / "skills",
    repo_root / "telegram_adapter",
    repo_root / "agent" / "webui" / "dist" / "index.html",
    repo_root / "assets" / "icons" / "personal-agent.svg",
    repo_root / "scripts" / "launch_webui.sh",
    repo_root / "packaging" / "debian" / "personal-agent-uninstall.sh",
    repo_root / "packaging" / "debian" / "personal-agent-api.service.in",
    repo_root / "packaging" / "personal-agent.desktop",
    repo_root / "llm_registry.json",
]
for path in required:
    if not path.exists():
        raise SystemExit(f"required packaging input missing: {path}")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", "node_modules", "tests"),
    )


copy_tree(repo_root / "agent", release_root / "agent")
copy_tree(repo_root / "memory", release_root / "memory")
copy_tree(repo_root / "skills", release_root / "skills")
copy_tree(repo_root / "telegram_adapter", release_root / "telegram_adapter")
(release_root / "assets" / "icons").mkdir(parents=True, exist_ok=True)
(release_root / "bin").mkdir(parents=True, exist_ok=True)
(release_root / "systemd").mkdir(parents=True, exist_ok=True)

shutil.copy2(repo_root / "assets" / "icons" / "personal-agent.svg", release_root / "assets" / "icons" / "personal-agent.svg")
shutil.copy2(repo_root / "scripts" / "launch_webui.sh", release_root / "bin" / "personal-agent-webui")
shutil.copy2(repo_root / "packaging" / "debian" / "personal-agent-uninstall.sh", release_root / "bin" / "personal-agent-uninstall")
shutil.copy2(repo_root / "llm_registry.json", release_root / "llm_registry.json")

service_template = (repo_root / "packaging" / "debian" / "personal-agent-api.service.in").read_text(encoding="utf-8")
service_rendered = service_template.replace("__PERSONAL_AGENT_RUNTIME_ROOT__", installed_runtime_root)
if "__PERSONAL_AGENT_RUNTIME_ROOT__" in service_rendered:
    raise SystemExit("service template placeholder was not replaced")
(release_root / "systemd" / "personal-agent-api.service").write_text(service_rendered, encoding="utf-8")
shutil.copy2(repo_root / "VERSION", release_root / "VERSION")

desktop_template = (repo_root / "packaging" / "personal-agent.desktop").read_text(encoding="utf-8")
desktop_rendered = desktop_template.replace("__PERSONAL_AGENT_LAUNCHER__", "/usr/bin/personal-agent-webui")
if "__PERSONAL_AGENT_LAUNCHER__" in desktop_rendered:
    raise SystemExit("desktop launcher placeholder was not replaced")
(desktop_dir / "personal-agent.desktop").write_text(desktop_rendered, encoding="utf-8")

shutil.copy2(repo_root / "assets" / "icons" / "personal-agent.svg", icon_dir / "personal-agent.svg")

(package_doc_root / "README.Debian").write_text(
    "\n".join(
        [
            "Personal Agent Debian package",
            "",
            "This package installs the browser-based UI, icon, launcher entry,",
            "and the canonical systemd user service unit.",
            "",
            "The user service is registered by the launcher on first use if it is",
            "not already enabled.",
            "",
            "Package removal deletes package-owned files. Remove user state with",
            "`personal-agent-uninstall --remove-state` before removing the package",
            "if you want a clean local reset.",
            "",
        ]
    ),
    encoding="utf-8",
)

manifest = {
    "package_name": "personal-agent",
    "package_version": version,
    "runtime_root": "/usr/lib/personal-agent/runtime",
    "release_root": f"/usr/lib/personal-agent/runtime/releases/{version}",
    "desktop_entry": "/usr/share/applications/personal-agent.desktop",
    "icon": "/usr/share/icons/hicolor/scalable/apps/personal-agent.svg",
    "launcher": "/usr/bin/personal-agent-webui",
    "uninstaller": "/usr/bin/personal-agent-uninstall",
    "service_unit": "/usr/lib/systemd/user/personal-agent-api.service",
}
(release_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

ln -sfn "/usr/lib/personal-agent/runtime/releases/$package_version" "$runtime_root/current"
ln -sfn "/usr/lib/personal-agent/runtime/current/bin/personal-agent-webui" "$user_bin_dir/personal-agent-webui"
ln -sfn "/usr/lib/personal-agent/runtime/current/bin/personal-agent-uninstall" "$user_bin_dir/personal-agent-uninstall"
ln -sfn "/usr/lib/personal-agent/runtime/current/systemd/personal-agent-api.service" "$systemd_user_dir/personal-agent-api.service"

cat > "$debian_dir/control" <<EOF
Package: $package_name
Version: $package_version
Architecture: $arch
Priority: optional
Section: utils
Maintainer: Personal Agent Contributors <noreply@example.com>
Depends: python3, python3-openai, python3-python-telegram-bot, python3-keyring, systemd, xdg-utils
Description: Local-first AI assistant with grounded runtime truth and bounded native actions
 Personal Agent uses a browser-based local UI, a canonical systemd user
 service, and preview-before-install safety for mutating actions.
EOF

build_command=(dpkg-deb --build "$stage_root" "$deb_path")
if command -v fakeroot >/dev/null 2>&1; then
    fakeroot "${build_command[@]}" >/dev/null
else
    "${build_command[@]}" >/dev/null
fi

python3 - "$deb_path" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

deb_path = Path(sys.argv[1]).resolve()
digest = hashlib.sha256(deb_path.read_bytes()).hexdigest()
checksum_path = deb_path.with_suffix(deb_path.suffix + ".sha256")
checksum_path.write_text(f"{digest}  {deb_path.name}\n", encoding="utf-8")
PY

printf '%s\n' "$stage_root"
printf '%s\n' "$deb_path"
printf '%s\n' "$checksum_path"
