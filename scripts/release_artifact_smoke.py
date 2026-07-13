#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
FORBIDDEN_NAMES = {".env", "agent.db", "llm_usage_stats.json"}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".git", "node_modules"}
FORBIDDEN_SUBSTRINGS = ("/tmp/", "/home/c/", "TELEGRAM_BOT_TOKEN", "authorization_header", "Bearer ")


@dataclass
class Check:
    name: str
    status: str
    evidence: str


def _check(name: str, condition: bool, evidence: str) -> Check:
    return Check(name, "PASS" if condition else "FAIL", evidence)


def _bad_path(path: str) -> str:
    parts = set(Path(path).parts)
    name = Path(path).name
    if name in FORBIDDEN_NAMES:
        return f"forbidden_name:{name}"
    if parts & FORBIDDEN_PARTS:
        return f"forbidden_part:{sorted(parts & FORBIDDEN_PARTS)[0]}"
    if path.endswith((".pyc", ".pyo")):
        return "bytecode"
    return ""


def _text_has_forbidden(value: str) -> str:
    for needle in FORBIDDEN_SUBSTRINGS:
        if needle in value:
            return needle
    return ""


def main() -> int:
    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="pa-release-artifact-") as raw:
        outdir = Path(raw) / "dist"
        proc = subprocess.run(
            ["bash", "scripts/build_release_bundle.sh", "--outdir", str(outdir), "--clean"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        checks.append(_check("release bundle builds", proc.returncode == 0, (proc.stdout + proc.stderr).strip()[:1000]))
        if proc.returncode == 0:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            bundle_dir = Path(lines[0])
            archive_path = Path(lines[1])
            checksum_path = Path(lines[2])
            checks.append(_check("bundle version file", (bundle_dir / "VERSION").read_text(encoding="utf-8").strip() == VERSION, str(bundle_dir / "VERSION")))
            manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            checks.append(_check("bundle manifest version", manifest.get("bundle_version") == VERSION, json.dumps(manifest, sort_keys=True)[:800]))
            checks.append(_check("bundle includes release notes", (bundle_dir / "payload" / "docs" / "releases" / "v0.2.2.md").is_file(), str(bundle_dir)))
            checks.append(_check("bundle checksum exists", checksum_path.is_file(), str(checksum_path)))
            checks.append(_check("bundle archive exists", archive_path.is_file(), str(archive_path)))
            with tarfile.open(archive_path, "r:gz") as tar:
                names = tar.getnames()
                bad = [f"{name}:{_bad_path(name)}" for name in names if _bad_path(name)]
                checks.append(_check("bundle contains no forbidden paths", not bad, ", ".join(bad[:10]) or f"files={len(names)}"))
                text = (bundle_dir / "manifest.json").read_text(encoding="utf-8")
                checks.append(_check("bundle manifest has no personal source path", not _text_has_forbidden(text), _text_has_forbidden(text) or "clean"))

        import build_backend

        wheel_dir = Path(raw) / "wheel"
        sdist_dir = Path(raw) / "sdist"
        wheel_dir.mkdir()
        sdist_dir.mkdir()
        wheel_name = build_backend.build_wheel(str(wheel_dir))
        sdist_name = build_backend.build_sdist(str(sdist_dir))
        checks.append(_check("wheel builds", VERSION in wheel_name and (wheel_dir / wheel_name).is_file(), wheel_name))
        checks.append(_check("sdist builds", VERSION in sdist_name and (sdist_dir / sdist_name).is_file(), sdist_name))
        with zipfile.ZipFile(wheel_dir / wheel_name) as wheel:
            names = wheel.namelist()
            checks.append(_check("wheel metadata version", any(name.endswith("METADATA") and f"Version: {VERSION}" in wheel.read(name).decode("utf-8", "replace") for name in names), wheel_name))
            bad = [f"{name}:{_bad_path(name)}" for name in names if _bad_path(name)]
            checks.append(_check("wheel contains no forbidden paths", not bad, ", ".join(bad[:10]) or f"files={len(names)}"))
        with tarfile.open(sdist_dir / sdist_name, "r:gz") as sdist:
            names = sdist.getnames()
            checks.append(_check("sdist includes release notes", any(name.endswith("docs/releases/v0.2.2.md") for name in names), sdist_name))
            bad = [f"{name}:{_bad_path(name)}" for name in names if _bad_path(name)]
            checks.append(_check("sdist contains no forbidden paths", not bad, ", ".join(bad[:10]) or f"files={len(names)}"))

    pass_count = sum(1 for row in checks if row.status == "PASS")
    fail_count = sum(1 for row in checks if row.status == "FAIL")
    for row in checks:
        print(f"{row.status}: {row.name}: {row.evidence}")
    print(f"PASS={pass_count} WARN=0 FAIL={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
