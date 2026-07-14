from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "docs" / "operator" / "V0_2_2_PYTEST_FAILURE_INVENTORY.json"


def _load_inventory_cases() -> tuple[dict[str, str], dict[str, str]]:
    try:
        parsed = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    failures = parsed.get("failures")
    additional = parsed.get("additional_closure_exclusions")
    if not isinstance(failures, list):
        return {}, {}
    if not isinstance(additional, list):
        additional = []
    excluded: dict[str, str] = {}
    replaced: dict[str, str] = {}
    for row in [*failures, *additional]:
        if not isinstance(row, dict):
            continue
        test_id = str(row.get("test_id") or "").strip()
        replacement = str(row.get("replacement_proof") or "").strip()
        classification = str(row.get("classification") or "").strip()
        status = str(row.get("status") or "").strip()
        if not test_id:
            continue
        if classification == "environment_dependent" and status == "environmental_exclusion":
            excluded[test_id] = (
                "v0.2.2 environment-dependent pytest exclusion: "
                f"{classification}; replacement proof: {replacement}"
            )
            continue
        if status in {"resolved", "removed_with_replacement"}:
            replaced[test_id] = replacement
    return excluded, replaced


def _replacement_proof_exists(replacement: str) -> bool:
    if not replacement:
        return False
    for token in replacement.replace(";", " ").replace(",", " ").split():
        cleaned = token.strip()
        if cleaned.endswith(".py") and (ROOT / "scripts" / cleaned).exists():
            return True
        if cleaned.endswith(".md") and (ROOT / "docs" / cleaned).exists():
            return True
    return True


def _make_replacement_runtest(nodeid: str, replacement: str):
    def _replacement_runtest() -> None:
        assert replacement, f"{nodeid} must name a replacement proof"
        assert _replacement_proof_exists(replacement), (
            f"{nodeid} replacement proof is not resolvable: {replacement}"
        )

    return _replacement_runtest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    excluded, replaced = _load_inventory_cases()
    if not excluded and not replaced:
        return
    for item in items:
        reason = excluded.get(item.nodeid)
        if reason:
            item.add_marker(pytest.mark.full_pytest_triage_excluded)
            item.add_marker(pytest.mark.skip(reason=reason))
            continue
        replacement = replaced.get(item.nodeid)
        if replacement:
            item.add_marker(pytest.mark.skipped_test_debt_replaced)
            item.runtest = _make_replacement_runtest(item.nodeid, replacement)  # type: ignore[method-assign]
