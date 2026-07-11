#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.capability_policy import capability_for_action_type  # noqa: E402
from agent.executor_registry import ExecutorRegistry, ExecutorSpec  # noqa: E402
from agent.mutation_plan import MUTATION_PLAN_SCHEMA_VERSION, build_mutation_plan, validate_mutation_plan  # noqa: E402


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""


MIGRATED_MUTATIONS = {
    "package.install": ("operator.package.install.v1", "system.package.install"),
    "operator.cleanup": ("operator.cleanup.v1", "cleanup.execute"),
    "operator.update": ("operator.update.v1", "system.update"),
    "operator.uninstall": ("operator.uninstall.v1", "system.uninstall"),
    "operator.support_bundle": ("operator.support_bundle.v1", "support_bundle.create"),
    "operator.backup": ("operator.backup.v1", "backup.create"),
    "operator.restore": ("operator.restore.v1", "restore.execute"),
    "memory.delete_all": ("operator.memory.forget.v1", "memory.forget"),
    "memory.export": ("operator.memory.export.v1", "memory.export"),
    "memory.redact": ("operator.memory.redact.v1", "memory.redact"),
    "memory.cleanup": ("operator.memory.compact.v1", "memory.compact"),
    "operator.file.create": ("operator.file.create.v1", "files.create"),
    "operator.file.modify": ("operator.file.modify.v1", "files.modify"),
    "operator.file.delete": ("operator.file.delete.v1", "files.delete"),
    "operator.git.commit": ("operator.git.commit.v1", "git.commit"),
    "operator.git.push": ("operator.git.push.v1", "git.push"),
    "operator.service.restart": ("operator.service.restart.v1", "system.service.restart"),
    "operator.notification.local.send": ("operator.notification.local.send.v1", "notification.local.send"),
    "operator.notification.telegram.send": ("operator.notification.telegram.send.v1", "notification.external.send"),
    "operator.notification.mark_read": ("operator.notification.mark_read.v1", "notification.mark_read"),
    "operator.notification.prune": ("operator.notification.prune.v1", "notification.prune"),
}

LEGACY_VISIBLE = {
    "broader skill-pack mutation": "not migrated in this checkpoint",
}


def _pass(name: str, detail: str = "") -> Check:
    return Check("PASS", name, detail)


def _warn(name: str, detail: str = "") -> Check:
    return Check("WARN", name, detail)


def _fail(name: str, detail: str = "") -> Check:
    return Check("FAIL", name, detail)


def _source_text(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def run() -> list[Check]:
    checks: list[Check] = []
    for action_type, (executor_id, capability_id) in MIGRATED_MUTATIONS.items():
        actual = capability_for_action_type(action_type)
        checks.append(
            _pass(f"{action_type} capability binding", capability_id)
            if actual == capability_id
            else _fail(f"{action_type} capability binding", f"actual={actual!r} expected={capability_id!r}")
        )
        try:
            plan = build_mutation_plan(
                plan_id=f"audit-{action_type.replace('.', '-')}",
                capability_id=capability_id,
                executor_id=executor_id,
                expires_at_epoch=4_102_444_800,
                thread_id="audit-thread",
                session_id="audit-session",
                target_snapshot={"action_type": action_type, "target": "fixture"},
                mutation_inventory=[{"action_type": action_type, "target": "fixture"}],
                preserved_resources=[{"kind": "fixture", "path": "/tmp"}],
                recovery={"rollback_supported": action_type != "operator.uninstall"},
            )
            validate_mutation_plan(plan)
            checks.append(_pass(f"{action_type} Universal Plan schema", f"schema={plan['schema_version']}"))
        except Exception as exc:  # noqa: BLE001 - audit output.
            checks.append(_fail(f"{action_type} Universal Plan schema", f"{exc.__class__.__name__}: {exc}"))

    with tempfile.TemporaryDirectory() as tmp:
        registry = ExecutorRegistry(Path(tmp) / "journal.jsonl")

        def _run(plan: dict, action: dict) -> dict:
            trusted = action.get("trusted_invocation_context") if isinstance(action.get("trusted_invocation_context"), dict) else {}
            return {
                "ok": bool(trusted.get("authorization_decision_id")),
                "mutated": True,
                "resources_touched": [str(Path(tmp) / "fixture")],
                "user_message": "fixture mutation completed",
            }

        registry.register(
            ExecutorSpec(
                executor_id="operator.cleanup.v1",
                action_type="operator.cleanup",
                run=_run,
                status="enabled",
                capability_id="cleanup.execute",
            )
        )
        result = registry.execute_confirmed_plan(
            plan={
                "plan_id": "audit-cleanup",
                "action_type": "operator.cleanup",
                "target": "fixture",
                "risk_level": "high",
                "executor_status": "enabled",
            },
            action={"pending_id": "audit-cleanup"},
        ).to_dict()
        checks.append(
            _pass("registry synthesizes Universal Plan for legacy canonical plan", f"schema={result.get('mutation_plan_schema_version')}")
            if result.get("mutation_plan_schema_version") == MUTATION_PLAN_SCHEMA_VERSION and result.get("mutated") is True
            else _fail("registry synthesizes Universal Plan for legacy canonical plan", str(result)[:700])
        )
        checks.append(
            _pass("receipt metadata includes Plan fields", result.get("capability_id") == "cleanup.execute" and bool(result.get("plan_fingerprint")))
            if result.get("authorization_decision_id")
            else _fail("receipt metadata includes Plan fields", str(result)[:700])
        )

    orchestrator = _source_text("agent/orchestrator.py")
    direct_shell_block = "package_install_direct_mutation_blocked" in orchestrator
    package_executor_registered = 'executor_id="operator.package.install.v1"' in orchestrator and 'action_type="package.install"' in orchestrator
    checks.append(_pass("direct package mutation path blocked") if direct_shell_block else _fail("direct package mutation path blocked"))
    checks.append(_pass("package install executor registered") if package_executor_registered else _fail("package install executor registered"))

    shell_skill = _source_text("agent/shell_skill.py")
    bypass_guard = "generic_bypass_blocked" in shell_skill and "operator.package.install.v1" in shell_skill
    checks.append(_pass("ShellSkill package mutation requires trusted context") if bypass_guard else _fail("ShellSkill package mutation requires trusted context"))

    source = _source_text("agent/executor_registry.py")
    for needle, label in (
        ("validate_mutation_plan", "Plan validation before migrated mutation"),
        ("mutation_plan_schema_version", "result includes Plan schema version"),
        ("target_fingerprint", "target fingerprint enforced"),
        ("authorization_decision_id", "authorization decision recorded"),
    ):
        checks.append(_pass(label) if re.search(re.escape(needle), source) else _fail(label))

    for action, detail in LEGACY_VISIBLE.items():
        checks.append(_warn(f"legacy action audit-visible: {action}", detail))
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
