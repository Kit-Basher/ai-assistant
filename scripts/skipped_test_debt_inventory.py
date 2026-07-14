#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "operator" / "V0_2_2_PYTEST_FAILURE_INVENTORY.json"
NON_ENV_CLASSES = {"stale_expectation", "test_isolation_bug", "test_fixture_bug", "obsolete_test"}


def _rows() -> list[dict[str, object]]:
    parsed = json.loads(INVENTORY.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for section in ("failures", "additional_closure_exclusions"):
        values = parsed.get(section)
        if isinstance(values, list):
            rows.extend(row for row in values if isinstance(row, dict))
    return rows


def main() -> int:
    rows = _rows()
    counts = Counter(str(row.get("classification") or "unknown") for row in rows)
    unclassified = counts.get("unknown", 0)
    environmental = counts.get("environment_dependent", 0)
    non_environmental = sum(counts.get(name, 0) for name in NON_ENV_CLASSES)
    open_non_environmental = sum(
        1
        for row in rows
        if str(row.get("classification") or "") in NON_ENV_CLASSES
        and str(row.get("status") or "") not in {"resolved", "removed_with_replacement"}
    )
    release_blockers = 0
    if len(rows) != 111:
        release_blockers += 1
    if environmental != 22 or non_environmental != 89:
        release_blockers += 1
    if counts.get("stale_expectation", 0) != 77:
        release_blockers += 1
    if counts.get("test_isolation_bug", 0) != 6:
        release_blockers += 1
    if counts.get("test_fixture_bug", 0) != 5:
        release_blockers += 1
    if counts.get("obsolete_test", 0) != 1:
        release_blockers += 1
    if open_non_environmental or unclassified:
        release_blockers += 1

    print(f"TOTAL_EXCLUDED={len(rows)}")
    print(f"ENVIRONMENTAL_EXCLUDED={environmental}")
    print(f"NON_ENVIRONMENTAL_DEBT={open_non_environmental}")
    print(f"STALE_EXPECTATION={counts.get('stale_expectation', 0)}")
    print(f"TEST_ISOLATION_BUG={counts.get('test_isolation_bug', 0)}")
    print(f"TEST_FIXTURE_BUG={counts.get('test_fixture_bug', 0)}")
    print(f"OBSOLETE_TEST={counts.get('obsolete_test', 0)}")
    print(f"UNCLASSIFIED={unclassified}")
    print(f"RELEASE_BLOCKERS={release_blockers}")
    return 1 if release_blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
