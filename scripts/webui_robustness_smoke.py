#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"


@dataclass(frozen=True)
class Check:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    timeout_seconds: int


@dataclass(frozen=True)
class Result:
    check: Check
    ok: bool
    elapsed_seconds: float
    output: str


def _run(check: Check) -> Result:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            check.argv,
            cwd=check.cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=check.timeout_seconds,
            check=False,
        )
        output = proc.stdout or ""
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        output = f"{output}\nTIMEOUT after {check.timeout_seconds}s"
        ok = False
    return Result(check=check, ok=ok, elapsed_seconds=time.monotonic() - started, output=output)


def _summarize(output: str, *, max_lines: int = 8) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join([*lines[: max_lines // 2], "...", *lines[-(max_lines // 2) :]])


def _static_docs_checks() -> list[tuple[str, bool, str]]:
    docs = ROOT / "docs/operator/WEB_UI_ROBUSTNESS.md"
    text = docs.read_text(encoding="utf-8") if docs.is_file() else ""
    checks = [
        ("web ui robustness doc exists", docs.is_file(), str(docs.relative_to(ROOT))),
        ("send failure documented", "send failure" in text.lower(), "send failure behavior documented"),
        ("loading/busy documented", "loading" in text.lower() and "busy" in text.lower(), "loading/busy behavior documented"),
        ("large transcript/autoscroll documented", "large transcript" in text.lower() and "autoscroll" in text.lower(), "large transcript/autoscroll behavior documented"),
        ("stale cache documented", "stale frontend cache" in text.lower(), "stale frontend cache behavior documented"),
        ("refresh behavior documented", "browser refresh" in text.lower(), "browser refresh behavior documented"),
        ("transcript import/export documented", "transcript export" in text.lower() and "import is not implemented" in text.lower(), "transcript import/export status documented"),
        ("manual ui checks documented", "manual checks" in text.lower(), "remaining manual UI checks documented"),
    ]
    return checks


def main() -> int:
    node_test_files = tuple(str(path.relative_to(DESKTOP)) for path in sorted((DESKTOP / "tests").glob("*.test.js")))
    checks = [
        Check("frontend build", ("npm", "run", "build"), DESKTOP, 120),
        Check("node ui helper/static tests", ("node", "--test", *node_test_files), DESKTOP, 120),
    ]

    print("# Personal Agent Web UI Robustness Smoke")
    failed = 0
    for check in checks:
        result = _run(check)
        status = "PASS" if result.ok else "FAIL"
        print(f"## {check.name}: {status}")
        print(f"- command/API path: {' '.join(check.argv)}")
        print(f"- elapsed_s: {result.elapsed_seconds:.1f}")
        if result.output:
            print("- output:")
            print(_summarize(result.output))
        if not result.ok:
            failed += 1

    for name, ok, evidence in _static_docs_checks():
        print(f"## {name}: {'PASS' if ok else 'FAIL'}")
        print(f"- evidence: {evidence}")
        if not ok:
            failed += 1

    if failed:
        print(f"\nFAIL webui_robustness_smoke failures={failed}")
        return 1
    print("\nPASS webui_robustness_smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
