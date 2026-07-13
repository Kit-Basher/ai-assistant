#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.2.2"


@dataclass
class Check:
    name: str
    status: str
    evidence: str


def _check(name: str, condition: bool, evidence: str) -> Check:
    return Check(name, "PASS" if condition else "FAIL", evidence)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=20)


def _version_from_cli_output(text: str) -> str:
    match = re.search(r"\bversion=([0-9]+\.[0-9]+\.[0-9]+)\b", text)
    return match.group(1) if match else ""


def main() -> int:
    sys.path.insert(0, str(ROOT))
    checks: list[Check] = []

    version_file = _read(ROOT / "VERSION")
    checks.append(_check("VERSION file", version_file == EXPECTED_VERSION, version_file))

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project") if isinstance(pyproject.get("project"), dict) else {}
    dynamic = project.get("dynamic") if isinstance(project.get("dynamic"), list) else []
    checks.append(_check("pyproject uses dynamic VERSION", "version" in {str(item) for item in dynamic}, json.dumps(dynamic)))

    import agent.version as version_mod
    import agent

    read_version, source = version_mod.read_version(repo_root=ROOT)
    checks.append(_check("agent.version reads product version", read_version == EXPECTED_VERSION, f"{read_version} source={source}"))
    checks.append(_check("agent.__version__ matches", str(agent.__version__) == EXPECTED_VERSION, str(agent.__version__)))

    cli = _run([sys.executable, "-m", "agent.cli", "version"])
    cli_version = _version_from_cli_output(cli.stdout)
    checks.append(_check("CLI version matches", cli.returncode == 0 and cli_version == EXPECTED_VERSION, (cli.stdout + cli.stderr).strip()[:500]))

    metadata_version = ""
    try:
        metadata_version = importlib.metadata.version("personal-agent")
    except importlib.metadata.PackageNotFoundError:
        metadata_version = ""
    checks.append(_check("installed metadata absent or matches", metadata_version in {"", EXPECTED_VERSION}, metadata_version or "not installed"))

    release_notes = ROOT / "docs" / "releases" / "v0.2.2.md"
    notes_text = release_notes.read_text(encoding="utf-8") if release_notes.exists() else ""
    checks.append(_check("release notes version present", "v0.2.2" in notes_text and "0.2.2" in notes_text, str(release_notes)))

    final_doc = ROOT / "docs" / "operator" / "V0_2_2_FINAL_RELEASE_AUDIT.md"
    final_text = final_doc.read_text(encoding="utf-8") if final_doc.exists() else ""
    checks.append(_check("version decision recorded", "Recommended final version: `v0.2.2`" in final_text, str(final_doc)))

    support_smoke = ROOT / "scripts" / "support_bundle_v2_smoke.py"
    support_text = support_smoke.read_text(encoding="utf-8") if support_smoke.exists() else ""
    checks.append(_check("support bundle version contract present", '"version.json"' in support_text, str(support_smoke)))

    build_backend = __import__("build_backend")
    wheel_name = build_backend._wheel_name()  # type: ignore[attr-defined]
    sdist_name = build_backend._sdist_name()  # type: ignore[attr-defined]
    checks.append(_check("build backend wheel version", EXPECTED_VERSION in wheel_name, wheel_name))
    checks.append(_check("build backend sdist version", EXPECTED_VERSION in sdist_name, sdist_name))

    pass_count = sum(1 for row in checks if row.status == "PASS")
    fail_count = sum(1 for row in checks if row.status == "FAIL")
    for row in checks:
        print(f"{row.status}: {row.name}: {row.evidence}")
    print(f"PASS={pass_count} WARN=0 FAIL={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
