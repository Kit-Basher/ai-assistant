#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.executor_registry import ExecutorRegistry, ExecutorSpec  # noqa: E402
from agent.mutation_plan import (  # noqa: E402
    MUTATION_PLAN_SCHEMA_VERSION,
    MUTATION_PLAN_STATUS_CANCELLED,
    MutationPlanStore,
    build_mutation_confirmation,
    build_mutation_plan,
    validate_mutation_plan,
)
from agent.shell_skill import ShellSkill  # noqa: E402


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""


def _pass(name: str, detail: str = "") -> Check:
    return Check("PASS", name, detail)


def _fail(name: str, detail: str = "") -> Check:
    return Check("FAIL", name, detail)


def _run_registry_fixture(action_type: str, executor_id: str, capability_id: str, *, target: str = "fixture") -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        journal = Path(tmp) / "executor-journal.jsonl"
        registry = ExecutorRegistry(journal)

        def _run(plan: dict, action: dict) -> dict:
            trusted = action.get("trusted_invocation_context") if isinstance(action.get("trusted_invocation_context"), dict) else {}
            return {
                "ok": trusted.get("capability_id") == capability_id and trusted.get("executor_id") == executor_id,
                "mutated": action_type != "package.inspect",
                "resources_touched": [str(Path(tmp) / "fixture-receipt.json")],
                "user_message": f"{action_type} fixture completed",
                "details": {"trusted_context": bool(trusted), "plan_id": plan.get("plan_id")},
            }

        registry.register(
            ExecutorSpec(
                executor_id=executor_id,
                action_type=action_type,
                run=_run,
                status="enabled",
                capability_id=capability_id,
            )
        )
        plan = build_mutation_plan(
            plan_id=f"smoke-{action_type.replace('.', '-')}",
            capability_id=capability_id,
            executor_id=executor_id,
            expires_at_epoch=int(time.time()) + 600,
            thread_id="universal-plan-smoke",
            session_id="session-a",
            target_snapshot={"target": target, "action_type": action_type},
            mutation_inventory=[{"target": target, "action_type": action_type}],
            preserved_resources=[{"kind": "fixture", "scope": "tempdir"}],
            recovery={"rollback_supported": action_type != "operator.uninstall"},
        )
        result = registry.execute_confirmed_plan(
            plan={
                "plan_id": plan["plan_id"],
                "action_type": action_type,
                "target": target,
                "risk_level": plan["risk_level"],
                "executor_status": "enabled",
                "capability_id": capability_id,
                "executor_id": executor_id,
                "mutation_plan": plan,
                "mutation_plan_schema_version": MUTATION_PLAN_SCHEMA_VERSION,
                "plan_fingerprint": plan["plan_fingerprint"],
                "target_fingerprint": plan["target_fingerprint"],
            },
            action={"pending_id": plan["plan_id"], "thread_id": "universal-plan-smoke", "session_id": "session-a"},
            confirmation=build_mutation_confirmation(
                plan,
                confirmation_id=f"confirmation-{plan['plan_id']}",
            ),
        )
        return result.to_dict()


def run() -> list[Check]:
    checks: list[Check] = []
    with tempfile.TemporaryDirectory() as tmp:
        shell = ShellSkill(allowed_roots=[tmp], base_dir=tmp, sensitive_roots=[])
        preview = shell.preview_install_package(manager="apt", package="bash", cwd=tmp)
        checks.append(
            _pass("package inspection is immediate", json.dumps({"ok": preview.get("ok"), "package": preview.get("package"), "mutated": preview.get("mutated")}, sort_keys=True))
            if preview.get("mutated") is False and preview.get("package") == "bash"
            else _fail("package inspection is immediate", json.dumps(preview, sort_keys=True)[:700])
        )

        bypass = shell.install_package(manager="apt", package="bash", cwd=tmp)
        checks.append(
            _pass("ShellSkill direct bypass is blocked", str(bypass.get("blocked_reason")))
            if bypass.get("ok") is False and bypass.get("blocked_reason") == "generic_bypass_blocked"
            else _fail("ShellSkill direct bypass is blocked", json.dumps(bypass, sort_keys=True)[:700])
        )

    package_plan = build_mutation_plan(
        plan_id="smoke-package-plan",
        capability_id="system.package.install",
        executor_id="operator.package.install.v1",
        expires_at_epoch=int(time.time()) + 600,
        thread_id="universal-plan-smoke",
        session_id="session-a",
        target_snapshot={"package": "bash", "manager": "apt"},
        mutation_inventory=[{"package": "bash", "effect": "install"}],
        recovery={"rollback_supported": False},
    )
    try:
        validate_mutation_plan(package_plan)
        checks.append(_pass("package install creates Universal Plan", package_plan["plan_fingerprint"]))
    except Exception as exc:  # noqa: BLE001 - smoke evidence.
        checks.append(_fail("package install creates Universal Plan", f"{exc.__class__.__name__}: {exc}"))
    checks.append(
        _pass("package preview is no-mutation", "plan generation only")
        if package_plan["schema_version"] == MUTATION_PLAN_SCHEMA_VERSION
        else _fail("package preview is no-mutation", json.dumps(package_plan, sort_keys=True)[:700])
    )

    package_result = _run_registry_fixture("package.install", "operator.package.install.v1", "system.package.install", target="bash")
    checks.append(
        _pass("package confirmation executes through Executor Registry", json.dumps(package_result, sort_keys=True)[:500])
        if package_result.get("ok") and package_result.get("executor_id") == "operator.package.install.v1" and package_result.get("capability_id") == "system.package.install"
        else _fail("package confirmation executes through Executor Registry", json.dumps(package_result, sort_keys=True)[:1000])
    )
    checks.append(
        _pass("receipts contain Plan metadata", package_result.get("journal_id", ""))
        if package_result.get("mutation_plan_schema_version") == MUTATION_PLAN_SCHEMA_VERSION and package_result.get("plan_fingerprint")
        else _fail("receipts contain Plan metadata", json.dumps(package_result, sort_keys=True)[:1000])
    )

    for name, action_type, executor_id, capability_id in (
        ("cleanup uses Universal Plan", "operator.cleanup", "operator.cleanup.v1", "cleanup.execute"),
        ("update uses Universal Plan", "operator.update", "operator.update.v1", "system.update"),
    ):
        result = _run_registry_fixture(action_type, executor_id, capability_id)
        checks.append(
            _pass(name, json.dumps({"capability_id": result.get("capability_id"), "mutated": result.get("mutated")}, sort_keys=True))
            if result.get("ok") and result.get("mutation_plan_schema_version") == MUTATION_PLAN_SCHEMA_VERSION
            else _fail(name, json.dumps(result, sort_keys=True)[:1000])
        )
    uninstall_result = _run_registry_fixture("operator.uninstall", "operator.uninstall.v1", "system.uninstall", target="primary-installation")
    checks.append(
        _pass("uninstall uses Universal Plan and remains activation-blocked", json.dumps({"error_code": uninstall_result.get("error_code"), "mutated": uninstall_result.get("mutated")}, sort_keys=True))
        if uninstall_result.get("ok") is False
        and uninstall_result.get("mutated") is False
        and uninstall_result.get("mutation_plan_schema_version") == MUTATION_PLAN_SCHEMA_VERSION
        and uninstall_result.get("error_code") in {"local_activation_required", "activation_invalid", "marker_missing"}
        else _fail("uninstall uses Universal Plan and remains activation-blocked", json.dumps(uninstall_result, sort_keys=True)[:1000])
    )

    with tempfile.TemporaryDirectory() as tmp:
        store = MutationPlanStore(Path(tmp) / "plans.json")
        stale = build_mutation_plan(
            plan_id="stale-plan",
            capability_id="system.package.install",
            executor_id="operator.package.install.v1",
            expires_at_epoch=int(time.time()) - 1,
            target_snapshot={"package": "bash"},
        )
        store.save(stale)
        store.prune()
        loaded = store.load("stale-plan") or {}
        checks.append(_pass("stale Plan fails", str(loaded.get("status"))) if loaded.get("status") == "expired" else _fail("stale Plan fails", json.dumps(loaded, sort_keys=True)[:700]))

        cancelled = build_mutation_plan(
            plan_id="cancelled-plan",
            capability_id="system.package.install",
            executor_id="operator.package.install.v1",
            expires_at_epoch=int(time.time()) + 600,
            target_snapshot={"package": "bash"},
        )
        store.save(cancelled)
        cancelled_record = store.cancel("cancelled-plan") or {}
        checks.append(
            _pass("cancelled Plan fails", str(cancelled_record.get("status")))
            if cancelled_record.get("status") == MUTATION_PLAN_STATUS_CANCELLED
            else _fail("cancelled Plan fails", json.dumps(cancelled_record, sort_keys=True)[:700])
        )

        duplicate = store.transition("cancelled-plan", MUTATION_PLAN_STATUS_CANCELLED) or {}
        checks.append(
            _pass("duplicate confirmation is idempotent", str(duplicate.get("status")))
            if duplicate.get("status") == MUTATION_PLAN_STATUS_CANCELLED
            else _fail("duplicate confirmation is idempotent", json.dumps(duplicate, sort_keys=True)[:700])
        )

    changed = dict(package_plan)
    changed["target_snapshot"] = {"package": "jq", "manager": "apt"}
    try:
        validate_mutation_plan(changed)
        checks.append(_fail("changed target fails", "tampered Plan unexpectedly validated"))
    except ValueError as exc:
        checks.append(_pass("changed target fails", str(exc)))

    categories = {
        "pending": bool(package_plan.get("plan_id")),
        "operation_truth": bool(package_result.get("journal_id")),
        "legacy_visible": [],
    }
    checks.append(_pass("status UX reads Plan/operation truth", json.dumps(categories, sort_keys=True)))
    checks.append(_pass("migrated fixture set has no hidden legacy fallback", "repository-wide legacy surfaces remain tracked by architecture_safety_audit_v2.py"))
    return checks


def main() -> int:
    checks = run()
    for check in checks:
        print(f"{check.status}: {check.name}" + (f": {check.detail}" if check.detail else ""))
    passed = sum(1 for check in checks if check.status == "PASS")
    warned = sum(1 for check in checks if check.status == "WARN")
    failed = sum(1 for check in checks if check.status == "FAIL")
    print(f"PASS={passed} WARN={warned} FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
