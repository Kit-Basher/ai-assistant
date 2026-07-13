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
BASELINE_OUTPUT = Path("/tmp/v022-full-pytest-baseline.txt")
CLOSURE_EVIDENCE = Path("/tmp/v022-full-pytest-closure.json")
ALLOWED_CLASSIFICATIONS = {
    "genuine_regression",
    "stale_expectation",
    "test_fixture_bug",
    "test_isolation_bug",
    "environment_dependent",
    "installed_product_only",
    "destructive_or_unsafe",
    "external_provider_dependent",
    "duplicate_but_still_valid",
    "obsolete_test",
    "unknown",
}


def _load_inventory() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def _failed_nodeids_from_text(text: str) -> list[str]:
    nodeids: list[str] = []
    for line in text.splitlines():
        if not line.startswith("FAILED "):
            continue
        nodeids.append(line[len("FAILED ") :].split(" - ", 1)[0].strip())
    return nodeids


def _current_default_pytest() -> tuple[int, str, int | None]:
    if CLOSURE_EVIDENCE.exists():
        try:
            parsed = json.loads(CLOSURE_EVIDENCE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            output = str(parsed.get("output") or "")
            return int(parsed.get("returncode") or 0), output, int(parsed.get("skipped") or 0)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=1500,
    )
    return proc.returncode, proc.stdout + proc.stderr, None


def main() -> int:
    parsed = _load_inventory()
    failures = parsed.get("failures")
    if not isinstance(failures, list):
        print("BASELINE_FAILURES=0")
        print("CURRENT_FAILURES=unknown")
        print("CLASSIFIED=0")
        print("UNCLASSIFIED=1")
        print("RESOLVED=0")
        print("EXCLUDED=0")
        print("RELEASE_BLOCKERS=1")
        return 1

    additional = parsed.get("additional_closure_exclusions")
    if not isinstance(additional, list):
        additional = []
    all_rows = [*failures, *additional]
    inventory_ids = [str(row.get("test_id") or "") for row in all_rows if isinstance(row, dict)]
    original_ids = [str(row.get("test_id") or "") for row in failures if isinstance(row, dict)]
    duplicate_ids = [node for node, count in Counter(inventory_ids).items() if count > 1]
    unclassified = [
        str(row.get("test_id") or "")
        for row in all_rows
        if not isinstance(row, dict) or str(row.get("classification") or "") not in ALLOWED_CLASSIFICATIONS
    ]
    missing_replacements = [
        str(row.get("test_id") or "")
        for row in all_rows
        if isinstance(row, dict) and not str(row.get("replacement_proof") or "").strip()
    ]

    baseline_failed: list[str] = []
    if BASELINE_OUTPUT.exists():
        baseline_failed = _failed_nodeids_from_text(BASELINE_OUTPUT.read_text(errors="replace"))
    baseline_set = set(baseline_failed)
    inventory_set = set(inventory_ids)
    baseline_uninventoried = sorted(baseline_set - inventory_set)
    inventory_not_in_baseline = sorted(set(original_ids) - baseline_set) if baseline_set else []

    current_rc, current_output, evidence_skipped = _current_default_pytest()
    current_failed = _failed_nodeids_from_text(current_output)
    skipped_match = re.search(r"(\d+) skipped", current_output)
    skipped_count = evidence_skipped if evidence_skipped is not None else (int(skipped_match.group(1)) if skipped_match else 0)
    classification_totals = Counter(str(row.get("classification") or "unknown") for row in all_rows if isinstance(row, dict))

    release_blockers = 0
    if len(original_ids) != int(parsed.get("baseline", {}).get("failed", len(original_ids))):
        release_blockers += 1
    if duplicate_ids or unclassified or missing_replacements or baseline_uninventoried:
        release_blockers += 1
    if current_rc != 0 or current_failed:
        release_blockers += 1
    if skipped_count < len(inventory_ids):
        release_blockers += 1

    print(f"BASELINE_FAILURES={parsed.get('baseline', {}).get('failed', len(original_ids))}")
    print(f"CURRENT_FAILURES={len(current_failed)}")
    print(f"CLASSIFIED={len(inventory_ids) - len(unclassified)}")
    print(f"UNCLASSIFIED={len(unclassified)}")
    print(f"RESOLVED={len(inventory_ids)}")
    print(f"EXCLUDED={skipped_count}")
    print(f"CLASSIFICATION_TOTALS={json.dumps(dict(sorted(classification_totals.items())), sort_keys=True)}")
    print(f"DUPLICATE_IDS={len(duplicate_ids)}")
    print(f"MISSING_REPLACEMENT_PROOFS={len(missing_replacements)}")
    print(f"BASELINE_UNINVENTORIED={len(baseline_uninventoried)}")
    print(f"INVENTORY_NOT_IN_BASELINE={len(inventory_not_in_baseline)}")
    print(f"RELEASE_BLOCKERS={release_blockers}")
    if release_blockers:
        if current_failed:
            print("CURRENT_FAILED_NODEIDS=" + json.dumps(current_failed[:20]))
        if baseline_uninventoried:
            print("BASELINE_UNINVENTORIED_IDS=" + json.dumps(baseline_uninventoried[:20]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
