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
NON_ENV_CLASSES = {"stale_expectation", "test_isolation_bug", "test_fixture_bug", "obsolete_test"}


def _run(args: list[str], *, timeout: int = 1500) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout)


def _check(results: list[tuple[str, bool, str]], name: str, ok: bool, evidence: str) -> None:
    results.append((name, ok, evidence))


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    parsed = json.loads(INVENTORY.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for section in ("failures", "additional_closure_exclusions"):
        values = parsed.get(section)
        if isinstance(values, list):
            rows.extend(row for row in values if isinstance(row, dict))

    counts = Counter(str(row.get("classification") or "unknown") for row in rows)
    ids = [str(row.get("test_id") or "") for row in rows]
    duplicate_ids = [node for node, count in Counter(ids).items() if count > 1]
    resolved_non_env = [
        row
        for row in rows
        if str(row.get("classification") or "") in NON_ENV_CLASSES
        and str(row.get("status") or "") in {"resolved", "removed_with_replacement"}
    ]
    env_rows = [
        row
        for row in rows
        if str(row.get("classification") or "") == "environment_dependent"
        and str(row.get("status") or "") == "environmental_exclusion"
    ]
    non_env_debt = sum(
        1
        for row in rows
        if str(row.get("classification") or "") in NON_ENV_CLASSES
        and str(row.get("status") or "") not in {"resolved", "removed_with_replacement"}
    )
    missing_replacements = [row for row in rows if not str(row.get("replacement_proof") or "").strip()]

    _check(results, "all historical entries remain recorded", len(rows) == 111, f"count={len(rows)}")
    _check(results, "no duplicate test ids", not duplicate_ids, json.dumps(duplicate_ids[:10]))
    _check(results, "all non-environmental entries are resolved", non_env_debt == 0 and len(resolved_non_env) == 89, f"resolved={len(resolved_non_env)} debt={non_env_debt}")
    _check(results, "no stale-expectation entry remains skipped", counts.get("stale_expectation", 0) == 77 and non_env_debt == 0, f"stale={counts.get('stale_expectation', 0)}")
    _check(results, "no fixture/isolation debt remains skipped", counts.get("test_fixture_bug", 0) == 5 and counts.get("test_isolation_bug", 0) == 6 and non_env_debt == 0, f"fixture={counts.get('test_fixture_bug', 0)} isolation={counts.get('test_isolation_bug', 0)}")
    _check(results, "obsolete test has replacement disposition", counts.get("obsolete_test", 0) == 1 and non_env_debt == 0, f"obsolete={counts.get('obsolete_test', 0)}")
    _check(results, "remaining exclusions are environment-dependent", len(env_rows) == 22, f"environmental={len(env_rows)}")
    _check(results, "every entry names a replacement gate", not missing_replacements, f"missing={len(missing_replacements)}")

    pytest_ini = (ROOT / "pytest.ini").read_text(encoding="utf-8")
    _check(results, "no broad ignore pattern exists", "--ignore" not in pytest_ini and "norecursedirs" not in pytest_ini, "pytest.ini")
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    _check(results, "no directory-wide exclusion exists", "item.nodeid" in conftest and "full_pytest_triage_excluded" in conftest, "inventory uses exact node ids")

    proc = _run([sys.executable, "-m", "pytest", "-q", "-rs"])
    output = proc.stdout + proc.stderr
    skipped_match = re.search(r"(\d+) skipped", output)
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    xfail_or_xpass = "xfailed" in output or "xpassed" in output
    _check(results, "default pytest exits zero", proc.returncode == 0, output.splitlines()[-1] if output.splitlines() else f"exit={proc.returncode}")
    _check(results, "skip count matches allowed environment set", skipped == len(env_rows), f"skipped={skipped} allowed={len(env_rows)}")
    _check(results, "no unexpected xfails/xpasses", not xfail_or_xpass, "pytest -q -rs")

    auth = _run([sys.executable, "scripts/capability_policy_audit.py"], timeout=120)
    _check(results, "authorization proofs remain clean", auth.returncode == 0 and "FAIL=0" in (auth.stdout + auth.stderr), (auth.stdout + auth.stderr).splitlines()[-1] if (auth.stdout + auth.stderr).splitlines() else "capability audit")

    final = (ROOT / "scripts" / "final_release_audit.py").read_text(encoding="utf-8")
    _check(results, "final release audit recognizes closure", "skipped_test_debt_closure_smoke.py" in final, "final_release_audit.py")

    pass_count = sum(1 for _, ok, _ in results if ok)
    fail_count = len(results) - pass_count
    for name, ok, evidence in results:
        print(f"{'PASS' if ok else 'FAIL'}: {name}: {evidence}")
    print(f"PASS={pass_count} WARN=0 FAIL={fail_count}")
    print(f"NON_ENVIRONMENTAL_DEBT={non_env_debt}")
    print(f"ALLOWED_ENVIRONMENTAL_SKIPS={len(env_rows)}")
    print(f"RELEASE_BLOCKERS={fail_count}")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
