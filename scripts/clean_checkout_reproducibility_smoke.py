#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class Check:
    name: str
    status: str
    evidence: str
    blocker: bool = True


def _check(name: str, condition: bool, evidence: str, *, warn: bool = False, blocker: bool = True) -> Check:
    if condition:
        return Check(name, "PASS", evidence, blocker)
    return Check(name, "WARN" if warn else "FAIL", evidence, blocker)


def _run(args: list[str], *, cwd: Path = ROOT, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False, timeout=timeout)


def _script_ok(script: str, *, timeout: int = 180) -> tuple[bool, str]:
    proc = _run([sys.executable, script], timeout=timeout)
    output = (proc.stdout + proc.stderr).strip()
    fail = re.search(r"\bFAIL=(\d+)\b", output)
    ok = proc.returncode == 0 and (not fail or int(fail.group(1)) == 0)
    return ok, "\n".join(output.splitlines()[-8:]) or f"exit={proc.returncode}"


def main() -> int:
    checks: list[Check] = []

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject.get("project", {}).get("optional-dependencies", {})
    checks.append(_check("test extra declared", isinstance(optional, dict) and "test" in optional, "pyproject.toml [project.optional-dependencies].test"))
    checks.append(_check("release extra declared", isinstance(optional, dict) and "release" in optional, "pyproject.toml [project.optional-dependencies].release"))

    config_test = (ROOT / "tests" / "test_config.py").read_text(encoding="utf-8")
    checks.append(_check("config test does not hard-code primary checkout", "/home/c/personal-agent/control" not in config_test, "tests/test_config.py"))

    build_deb = (ROOT / "scripts" / "build_deb.sh").read_text(encoding="utf-8")
    checks.append(_check("Debian build has no ignored llm registry dependency", "llm_registry.json" not in build_deb, "scripts/build_deb.sh"))

    root_hits: list[str] = []
    for rel in ("tests", "agent", "packaging"):
        for path in (ROOT / rel).rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "/tmp/personal-agent-v022-final" in text:
                root_hits.append(str(path.relative_to(ROOT)))
    checks.append(_check("no stale clean-worktree path embedded", not root_hits, ", ".join(root_hits[:8]) or "clean"))

    webui_index = ROOT / "agent" / "webui" / "dist" / "index.html"
    if webui_index.exists():
        checks.append(_check("desktop assets present", True, str(webui_index)))
    else:
        desktop = ROOT / "desktop" / "package.json"
        checks.append(_check("desktop assets build policy declared", desktop.exists(), "run cd desktop && npm ci && npm run build before artifact/package verification", warn=True, blocker=False))

    ok, evidence = _script_ok("scripts/clean_checkout_debian_package_smoke.py", timeout=240)
    checks.append(_check("clean Debian package smoke", ok, evidence))

    with tempfile.TemporaryDirectory(prefix="pa-clean-pkg-") as raw:
        import build_backend

        outdir = Path(raw) / "dist"
        outdir.mkdir()
        wheel = build_backend.build_wheel(str(outdir))
        sdist = build_backend.build_sdist(str(outdir))
        checks.append(_check("wheel/sdist build through backend", (outdir / wheel).is_file() and (outdir / sdist).is_file(), f"{wheel}, {sdist}"))

    npm_evidence = ROOT / "docs" / "operator" / "FRONTEND_DEPENDENCY_AUDIT_V0_2_2.md"
    text = npm_evidence.read_text(encoding="utf-8") if npm_evidence.exists() else ""
    checks.append(_check("frontend dependency audit evidence present", "npm audit --omit=dev" in text and "0 vulnerabilities" in text, str(npm_evidence)))

    pass_count = sum(1 for row in checks if row.status == "PASS")
    warn_count = sum(1 for row in checks if row.status == "WARN")
    fail_count = sum(1 for row in checks if row.status == "FAIL")
    release_blockers = sum(1 for row in checks if row.status == "FAIL" and row.blocker)
    ready = release_blockers == 0
    for row in checks:
        print(f"{row.status}: {row.name}: {row.evidence}")
    print(f"PASS={pass_count} WARN={warn_count} FAIL={fail_count}")
    print(f"RELEASE_BLOCKERS={release_blockers}")
    print(f"CLEAN_CHECKOUT_READY={'true' if ready else 'false'}")
    return 1 if release_blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
