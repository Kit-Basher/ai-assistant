#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()


@dataclass
class Check:
    name: str
    status: str
    evidence: str


def _check(name: str, condition: bool, evidence: str) -> Check:
    return Check(name, "PASS" if condition else "FAIL", evidence)


def _run(args: list[str], *, cwd: Path = ROOT, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False, timeout=timeout)


def main() -> int:
    checks: list[Check] = []

    build_script = (ROOT / "scripts" / "build_deb.sh").read_text(encoding="utf-8")
    checks.append(_check("build script does not require local llm registry", "llm_registry.json" not in build_script, "scripts/build_deb.sh"))

    with tempfile.TemporaryDirectory(prefix="pa-clean-deb-") as raw:
        outdir = Path(raw) / "dist"
        proc = _run(["bash", "scripts/build_deb.sh", "--outdir", str(outdir), "--clean"], timeout=180)
        output = (proc.stdout + proc.stderr).strip()
        checks.append(_check("Debian package builds", proc.returncode == 0, output[:1200] or f"exit={proc.returncode}"))
        if proc.returncode == 0:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip() and not line.startswith("dpkg-deb:")]
            deb_path = Path(lines[1]) if len(lines) > 1 else Path("")
            info = _run(["dpkg-deb", "-I", str(deb_path)], timeout=60)
            contents = _run(["dpkg-deb", "-c", str(deb_path)], timeout=60)
            package_text = info.stdout + contents.stdout
            checks.append(_check("package version is current", f"Version: {VERSION}" in info.stdout, info.stdout[:600]))
            checks.append(_check("package omits mutable llm registry state", "llm_registry.json" not in package_text, "no llm_registry.json in dpkg output"))
            checks.append(_check("package contains Web UI asset", "agent/webui/dist/index.html" in contents.stdout, "agent/webui/dist/index.html"))
            forbidden = [needle for needle in ("/home/c/", "/tmp/personal-agent-v022-final", "TELEGRAM_BOT_TOKEN", "Bearer ") if needle in package_text]
            checks.append(_check("package output has no personal paths or secrets", not forbidden, ", ".join(forbidden) or "clean"))

    pass_count = sum(1 for row in checks if row.status == "PASS")
    fail_count = sum(1 for row in checks if row.status == "FAIL")
    for row in checks:
        print(f"{row.status}: {row.name}: {row.evidence}")
    print(f"PASS={pass_count} WARN=0 FAIL={fail_count}")
    print(f"RELEASE_BLOCKERS={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
