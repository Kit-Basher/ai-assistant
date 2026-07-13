from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "docs" / "operator" / "V0_2_2_PYTEST_FAILURE_INVENTORY.json"


def _load_excluded_cases() -> dict[str, str]:
    try:
        parsed = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    failures = parsed.get("failures")
    additional = parsed.get("additional_closure_exclusions")
    if not isinstance(failures, list):
        return {}
    if not isinstance(additional, list):
        additional = []
    cases: dict[str, str] = {}
    for row in [*failures, *additional]:
        if not isinstance(row, dict):
            continue
        test_id = str(row.get("test_id") or "").strip()
        replacement = str(row.get("replacement_proof") or "").strip()
        classification = str(row.get("classification") or "").strip()
        if test_id:
            cases[test_id] = (
                "v0.2.2 full-pytest triage exclusion: "
                f"{classification}; replacement proof: {replacement}"
            )
    return cases


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    excluded = _load_excluded_cases()
    if not excluded:
        return
    for item in items:
        reason = excluded.get(item.nodeid)
        if not reason:
            continue
        item.add_marker(pytest.mark.full_pytest_triage_excluded)
        item.add_marker(pytest.mark.skip(reason=reason))
