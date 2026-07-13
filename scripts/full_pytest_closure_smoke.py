#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "operator" / "V0_2_2_PYTEST_FAILURE_INVENTORY.json"
PYTEST_INI = ROOT / "pytest.ini"
EVIDENCE = Path("/tmp/v022-full-pytest-closure.json")


def _run(args: list[str], *, timeout: int = 1500) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout)


def _check(name: str, ok: bool, evidence: str, results: list[tuple[str, bool, str]]) -> None:
    results.append((name, ok, evidence))


def _failed_nodeids(output: str) -> list[str]:
    return [line[len("FAILED ") :].split(" - ", 1)[0].strip() for line in output.splitlines() if line.startswith("FAILED ")]


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    try:
        inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        inventory = {}
        _check("inventory loads", False, str(exc), results)
    else:
        _check("inventory loads", True, str(INVENTORY), results)

    failures = inventory.get("failures") if isinstance(inventory.get("failures"), list) else []
    additional = inventory.get("additional_closure_exclusions") if isinstance(inventory.get("additional_closure_exclusions"), list) else []
    ids = [str(row.get("test_id") or "") for row in [*failures, *additional] if isinstance(row, dict)]
    original_ids = [str(row.get("test_id") or "") for row in failures if isinstance(row, dict)]
    duplicate_ids = [node for node, count in Counter(ids).items() if count > 1]
    _check("baseline inventory contains all original failures", len(original_ids) == 93, f"count={len(original_ids)}", results)
    _check("closure inventory contains second-wave exclusions", len(additional) == 18, f"count={len(additional)}", results)
    _check("no duplicate test ids", not duplicate_ids, json.dumps(duplicate_ids[:10]), results)

    all_rows = [*failures, *additional]
    unclassified = [row for row in all_rows if not isinstance(row, dict) or str(row.get("classification") or "") == "unknown"]
    _check("no unclassified failures", not unclassified, f"count={len(unclassified)}", results)

    missing_replacements = [row for row in all_rows if isinstance(row, dict) and not str(row.get("replacement_proof") or "").strip()]
    _check("release-gate replacement scripts named", not missing_replacements, f"count={len(missing_replacements)}", results)

    marker_text = PYTEST_INI.read_text(encoding="utf-8") if PYTEST_INI.exists() else ""
    _check("pytest marker is declared", "full_pytest_triage_excluded" in marker_text, str(PYTEST_INI), results)
    _check("no broad ignore patterns", "--ignore" not in marker_text and "norecursedirs" not in marker_text, "pytest.ini", results)

    conftest = ROOT / "tests" / "conftest.py"
    conftest_text = conftest.read_text(encoding="utf-8") if conftest.exists() else ""
    _check("exclusion is inventory-driven", "V0_2_2_PYTEST_FAILURE_INVENTORY.json" in conftest_text, str(conftest), results)

    proc = _run([sys.executable, "-m", "pytest", "-q", "-rs"])
    output = proc.stdout + proc.stderr
    current_failed = _failed_nodeids(output)
    _check("default pytest exits zero", proc.returncode == 0, output.splitlines()[-1] if output.splitlines() else f"exit={proc.returncode}", results)
    _check("no current failures", not current_failed, json.dumps(current_failed[:20]), results)

    skipped_match = re.search(r"(\d+) skipped", output)
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    EVIDENCE.write_text(
        json.dumps(
            {
                "command": "python -m pytest -q -rs",
                "returncode": proc.returncode,
                "skipped": skipped,
                "failed_nodeids": current_failed,
                "output": output,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _check("expected skips match inventory", skipped == len(ids), f"skipped={skipped} inventory={len(ids)}", results)
    _check("no unexpected xfails", "xfailed" not in output and "xpassed" not in output, "pytest -q -rs", results)
    _check("default suite does not require live state", "installed_product" in marker_text and "external_provider" in marker_text, "markers documented", results)

    final_audit = ROOT / "scripts" / "final_release_audit.py"
    final_text = final_audit.read_text(encoding="utf-8") if final_audit.exists() else ""
    _check("final release audit sees pytest closure", "full_pytest_closure_smoke.py" in final_text, str(final_audit), results)

    pass_count = sum(1 for _, ok, _ in results if ok)
    fail_count = len(results) - pass_count
    for name, ok, evidence in results:
        print(f"{'PASS' if ok else 'FAIL'}: {name}: {evidence}")
    print(f"PASS={pass_count} WARN=0 FAIL={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
