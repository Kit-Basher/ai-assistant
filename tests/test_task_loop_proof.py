from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from scripts import task_loop_proof


def _mutated_registry(mutator):  # type: ignore[no-untyped-def]
    original = task_loop_proof._build_registry

    def build():  # type: ignore[no-untyped-def]
        registry, cleanup = original()
        mutator(registry)
        return registry, cleanup

    return build


def test_task_loop_proof_reconciles_all_protected_capabilities_categories_and_scenarios() -> None:
    report = task_loop_proof.run_proof()
    assert report["ok"] is True, report["failures"]
    assert report["totals"] == {
        "expected_task_composable": 22,
        "registered_task_composable": 22,
        "capabilities_passed": 22,
        "proof_categories": 16,
        "scenario_categories": 16,
        "failures": 0,
    }
    assert report["redaction"]["passed"] is True
    assert len(report["candidate"]["diff_fingerprint"]) == 64


def test_sensitivity_missing_registration_fails(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(task_loop_proof, "_build_registry", _mutated_registry(lambda registry: registry._items.pop("filesystem.read")))  # noqa: SLF001
    report = task_loop_proof.run_proof()
    assert report["ok"] is False
    assert any(row["reason"] == "task_composable_capability_missing" for row in report["failures"])


def test_sensitivity_unproved_extra_capability_fails(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def mutate(registry):  # type: ignore[no-untyped-def]
        existing = registry.require("filesystem.read")
        registry._items["unproved.extra"] = replace(existing, capability_id="unproved.extra")  # noqa: SLF001

    monkeypatch.setattr(task_loop_proof, "_build_registry", _mutated_registry(mutate))
    report = task_loop_proof.run_proof()
    assert any(row["reason"] == "unproved_task_composable_capability" for row in report["failures"])


def test_sensitivity_policy_downgrade_and_missing_verifier_fail(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def mutate(registry):  # type: ignore[no-untyped-def]
        existing = registry.require("filesystem.create_directory")
        registry._items[existing.capability_id] = replace(existing, retry_safety="never", independent_verification_hook=None)  # noqa: SLF001

    monkeypatch.setattr(task_loop_proof, "_build_registry", _mutated_registry(mutate))
    report = task_loop_proof.run_proof()
    reasons = {row["reason"] for row in report["failures"]}
    assert "mutation_reconcile_policy_downgraded" in reasons
    assert "mutation_independent_verifier_missing" in reasons


def test_sensitivity_missing_proof_node_and_scenario_fail(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    manifest = json.loads(task_loop_proof.MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["proof_nodes"]["schema"] = ["tests/does_not_exist.py"]
    manifest["required_scenarios"].append("removed_required_scenario")
    changed = tmp_path / "manifest.json"
    changed.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(task_loop_proof, "MANIFEST_PATH", changed)
    report = task_loop_proof.run_proof()
    reasons = {row["reason"] for row in report["failures"]}
    assert "proof_node_missing:schema" in reasons
    assert "required_scenario_missing" in reasons


def test_execute_declared_proofs_marks_each_category_and_scenario_pass() -> None:
    report = task_loop_proof.run_proof()
    task_loop_proof.execute_tests(report)
    assert report["ok"] is True, report["failures"]
    assert report["proof_execution"]["status"] == "pass"
    assert set(report["scenarios"].values()) == {"pass"}
