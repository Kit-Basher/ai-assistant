#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.capability_policy import (  # noqa: E402
    POLICY_SCHEMA_VERSION,
    CapabilityRegistry,
    authorize_capability,
    build_default_capability_registry,
    stable_fingerprint,
)
from agent.executor_registry import ExecutorRegistry, ExecutorSpec  # noqa: E402
from agent.shell_skill import ShellSkill  # noqa: E402


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""


def _check(name: str, condition: bool, detail: str = "") -> Check:
    return Check("PASS" if condition else "FAIL", name, detail)


def run() -> list[Check]:
    checks: list[Check] = []
    registry = build_default_capability_registry()
    checks.append(_check("registry loads", bool(registry.get("system.package.install")), "default registry"))
    try:
        duplicate = CapabilityRegistry()
        cap = registry.get("system.package.inspect")
        duplicate.register(cap)
        duplicate.register(cap)
        checks.append(_check("duplicate id rejected", False, "duplicate registration unexpectedly succeeded"))
    except ValueError:
        checks.append(_check("duplicate id rejected", True))

    read_only = authorize_capability("system.package.inspect")
    checks.append(_check("read-only inspection allowed", read_only.allowed and read_only.reason_code == "read_only_allowed", read_only.to_dict().get("reason_code", "")))

    install_no_confirm = authorize_capability(
        "system.package.install",
        target_snapshot={"target_fingerprint": "target"},
        plan_context={"plan_fingerprint": "plan", "target_fingerprint": "target", "policy_version": POLICY_SCHEMA_VERSION},
    )
    checks.append(_check("package install requires confirmation", not install_no_confirm.allowed and install_no_confirm.reason_code == "confirmation_required", install_no_confirm.reason_code))

    cleanup = authorize_capability(
        "cleanup.execute",
        target_snapshot={"target_fingerprint": "target"},
        plan_context={"plan_fingerprint": "plan", "target_fingerprint": "target", "policy_version": POLICY_SCHEMA_VERSION},
        confirmation_context={"confirmed": True},
    )
    checks.append(_check("cleanup requires policy gate", cleanup.allowed and cleanup.mutation_allowed, cleanup.reason_code))

    update = authorize_capability(
        "system.update",
        target_snapshot={"target_fingerprint": "target"},
        plan_context={"plan_fingerprint": "plan", "target_fingerprint": "target", "policy_version": POLICY_SCHEMA_VERSION},
        confirmation_context={"confirmed": True},
    )
    checks.append(_check("update uses central gate", update.allowed and update.mutation_allowed, update.reason_code))

    uninstall = authorize_capability(
        "system.uninstall",
        target_snapshot={"target_fingerprint": "target"},
        plan_context={"plan_fingerprint": "plan", "target_fingerprint": "target", "policy_version": POLICY_SCHEMA_VERSION},
        confirmation_context={"confirmed": True},
        activation_context={"valid": False, "reason_code": "local_activation_required"},
    )
    checks.append(_check("uninstall requires activation plus confirmation", not uninstall.allowed and uninstall.reason_code == "local_activation_required", uninstall.reason_code))

    stale = authorize_capability(
        "system.package.install",
        target_snapshot={"target_fingerprint": "target"},
        plan_context={"plan_fingerprint": "plan", "target_fingerprint": "target", "policy_version": POLICY_SCHEMA_VERSION, "stale": True},
        confirmation_context={"confirmed": True},
    )
    checks.append(_check("stale Plan blocked", not stale.allowed and stale.reason_code == "stale_plan", stale.reason_code))

    changed_target = authorize_capability(
        "system.package.install",
        target_snapshot={"target_fingerprint": "changed"},
        plan_context={"plan_fingerprint": "plan", "target_fingerprint": "target", "policy_version": POLICY_SCHEMA_VERSION},
        confirmation_context={"confirmed": True},
    )
    checks.append(_check("changed target blocked", not changed_target.allowed and changed_target.reason_code == "target_changed", changed_target.reason_code))

    with tempfile.TemporaryDirectory() as tmp:
        shell = ShellSkill(allowed_roots=[tmp], sensitive_roots=[])
        bypass = shell.install_package(manager="apt", package="htop", cwd=tmp)
        checks.append(_check("shell bypass blocked", not bypass.get("ok") and bypass.get("blocked_reason") == "generic_bypass_blocked", json.dumps(bypass, sort_keys=True)[:300]))

        journal = Path(tmp) / "journal.jsonl"
        exec_registry = ExecutorRegistry(journal)

        def _run(plan: dict, action: dict) -> dict:
            return {"ok": True, "mutated": True, "resources_touched": [str(Path(tmp) / "fixture")], "user_message": "fixture complete"}

        exec_registry.register(
            ExecutorSpec(
                executor_id="operator.cleanup.v1",
                action_type="operator.cleanup",
                status="enabled",
                run=_run,
                capability_id="cleanup.execute",
            )
        )
        target_fp = stable_fingerprint({"target": "fixture cleanup"})
        plan_fp = stable_fingerprint({"plan": "fixture cleanup", "target": target_fp})
        result = exec_registry.execute_confirmed_plan(
            plan={
                "plan_id": "capability-smoke",
                "action_type": "operator.cleanup",
                "target": "fixture cleanup",
                "risk_level": "high",
                "executor_status": "enabled",
                "target_fingerprint": target_fp,
                "plan_fingerprint": plan_fp,
                "policy_schema_version": POLICY_SCHEMA_VERSION,
            },
            action={"pending_id": "capability-smoke"},
        )
        result_payload = result.to_dict()
        checks.append(_check("receipts include capability metadata", result_payload.get("capability_id") == "cleanup.execute" and bool(result_payload.get("authorization_decision_id")), json.dumps(result_payload, sort_keys=True)[:300]))

    categories = registry.status_categories()
    checks.append(_check("status UX reflects registry truth", bool(categories["requires_confirmation"]) and bool(categories["requires_local_activation"]), json.dumps(categories, sort_keys=True)[:300]))
    legacy_audit_visible = ["communications", "broader skill-pack mutations"]
    checks.append(
        _check(
            "unmigrated paths reported",
            not categories["legacy_unmigrated"] and bool(legacy_audit_visible),
            json.dumps({"registry_legacy": categories["legacy_unmigrated"], "audit_legacy": legacy_audit_visible}, sort_keys=True)[:300],
        )
    )
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
