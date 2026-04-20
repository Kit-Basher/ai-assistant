#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
outdir="dist"
override_repo_root=""
clean=0

die() {
    printf '%s\n' "Personal Agent release bundle: $*" >&2
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
        --clean)
            clean=1
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/build_release_bundle.sh [--repo-root PATH] [--outdir PATH] [--clean]

Build a versioned release bundle with install/uninstall entry points.
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

bundle_name="personal-agent-linux-x86_64-v${version}"
bundle_dir="$outdir/$bundle_name"
payload_dir="$bundle_dir/payload"
archive_path="$outdir/${bundle_name}.tar.gz"
checksum_path="$archive_path.sha256"

if [ "$clean" -eq 1 ]; then
    rm -rf "$bundle_dir" "$archive_path" "$checksum_path"
fi
rm -rf "$bundle_dir"
mkdir -p "$payload_dir"

python3 - "$repo_root" "$bundle_dir" "$payload_dir" "$version" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

repo_root = Path(sys.argv[1]).resolve()
bundle_dir = Path(sys.argv[2]).resolve()
payload_dir = Path(sys.argv[3]).resolve()
version = sys.argv[4]

required = [
    repo_root / "agent",
    repo_root / "memory",
    repo_root / "skills",
    repo_root / "telegram_adapter",
    repo_root / "build_backend.py",
    repo_root / "pyproject.toml",
    repo_root / "VERSION",
    repo_root / "README.md",
    repo_root / "personal_agent_bootstrap.py",
    repo_root / "personal_agent_bootstrap.pth",
    repo_root / "sitecustomize.py",
    repo_root / "agent" / "webui" / "dist" / "index.html",
    repo_root / "assets" / "icons" / "personal-agent.svg",
    repo_root / "scripts" / "launch_webui.sh",
]
for path in required:
    if not path.exists():
        raise SystemExit(f"required release-bundle input missing: {path}")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", "node_modules", "tests"),
    )


copy_tree(repo_root / "agent", payload_dir / "agent")
copy_tree(repo_root / "memory", payload_dir / "memory")
copy_tree(repo_root / "skills", payload_dir / "skills")
copy_tree(repo_root / "telegram_adapter", payload_dir / "telegram_adapter")
copy_tree(repo_root / "systemd", payload_dir / "systemd")

(payload_dir / "bin").mkdir(parents=True, exist_ok=True)
(payload_dir / "assets" / "icons").mkdir(parents=True, exist_ok=True)
(payload_dir / "agent" / "webui" / "dist").mkdir(parents=True, exist_ok=True)

shutil.copy2(repo_root / "build_backend.py", payload_dir / "build_backend.py")
shutil.copy2(repo_root / "pyproject.toml", payload_dir / "pyproject.toml")
shutil.copy2(repo_root / "requirements.txt", payload_dir / "requirements.txt")
shutil.copy2(repo_root / "VERSION", payload_dir / "VERSION")
shutil.copy2(repo_root / "README.md", payload_dir / "README.md")
shutil.copy2(repo_root / "personal_agent_bootstrap.py", payload_dir / "personal_agent_bootstrap.py")
shutil.copy2(repo_root / "personal_agent_bootstrap.pth", payload_dir / "personal_agent_bootstrap.pth")
shutil.copy2(repo_root / "sitecustomize.py", payload_dir / "sitecustomize.py")
shutil.copy2(repo_root / "assets" / "icons" / "personal-agent.svg", payload_dir / "assets" / "icons" / "personal-agent.svg")
shutil.copy2(repo_root / "scripts" / "launch_webui.sh", payload_dir / "bin" / "personal-agent-webui")
os.chmod(payload_dir / "bin" / "personal-agent-webui", 0o755)

manifest = {
    "bundle_version": version,
    "source_repo": str(repo_root),
    "payload": [
        "agent",
        "memory",
        "skills",
        "telegram_adapter",
        "systemd",
        "assets/icons/personal-agent.svg",
        "bin/personal-agent-webui",
        "build_backend.py",
        "pyproject.toml",
        "requirements.txt",
        "VERSION",
        "README.md",
        "personal_agent_bootstrap.py",
        "personal_agent_bootstrap.pth",
        "sitecustomize.py",
    ],
}
(bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cp "$repo_root/packaging/release_bundle/install.sh" "$bundle_dir/install.sh"
cp "$repo_root/packaging/release_bundle/uninstall.sh" "$bundle_dir/uninstall.sh"
cp "$repo_root/packaging/release_bundle/INSTALL.md" "$bundle_dir/INSTALL.md"
cp "$repo_root/packaging/release_bundle/UNINSTALL.md" "$bundle_dir/UNINSTALL.md"
cp "$repo_root/VERSION" "$bundle_dir/VERSION"
chmod +x "$bundle_dir/install.sh" "$bundle_dir/uninstall.sh"

tar -C "$outdir" -czf "$archive_path" "$bundle_name"
python3 - "$archive_path" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

archive_path = Path(sys.argv[1]).resolve()
digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
archive_sha = archive_path.with_suffix(archive_path.suffix + ".sha256")
archive_sha.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
PY

printf '%s\n' "$bundle_dir"
printf '%s\n' "$archive_path"
printf '%s\n' "$checksum_path"
