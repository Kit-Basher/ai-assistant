#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.capability_policy import build_default_capability_registry, capability_for_action_type  # noqa: E402


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""


MIGRATED_EXECUTORS = {
    "package.install": "system.package.install",
    "operator.cleanup": "cleanup.execute",
    "operator.update": "system.update",
    "operator.uninstall": "system.uninstall",
    "operator.support_bundle": "support_bundle.create",
    "operator.backup": "backup.create",
    "operator.restore": "restore.execute",
    "memory.delete_all": "memory.forget",
    "memory.export": "memory.export",
    "memory.redact": "memory.redact",
    "memory.cleanup": "memory.compact",
    "operator.file.create": "files.create",
    "operator.file.modify": "files.modify",
    "operator.file.delete": "files.delete",
    "operator.git.commit": "git.commit",
    "operator.git.push": "git.push",
    "operator.service.restart": "system.service.restart",
}

LEGACY_UNMIGRATED = {
    "communications.*": "external communication mutation adapters are not enabled as core capabilities",
    "skill_pack.*": "broader skill-pack mutation paths remain future migration work",
}


def _pass(name: str, detail: str = "") -> Check:
    return Check("PASS", name, detail)


def _warn(name: str, detail: str = "") -> Check:
    return Check("WARN", name, detail)


def _fail(name: str, detail: str = "") -> Check:
    return Check("FAIL", name, detail)


def run() -> list[Check]:
    checks: list[Check] = []
    registry = build_default_capability_registry()
    capabilities = registry.list()
    checks.append(_pass("registry loads", f"{len(capabilities)} capabilities"))
    ids = [cap.capability_id for cap in capabilities]
    checks.append(_pass("capability ids unique") if len(ids) == len(set(ids)) else _fail("capability ids unique"))
    for action_type, expected in MIGRATED_EXECUTORS.items():
        actual = capability_for_action_type(action_type)
        definition = registry.get(expected)
        if actual == expected and definition is not None and definition.implementation_status == "implemented":
            checks.append(_pass(f"{action_type} bound", expected))
        else:
            checks.append(_fail(f"{action_type} bound", f"actual={actual!r} expected={expected!r}"))
    for cap_id in (
        "system.update",
        "system.uninstall",
        "system.package.install",
        "cleanup.execute",
        "backup.create",
        "restore.execute",
        "support_bundle.create",
        "memory.forget",
        "memory.export",
        "memory.redact",
        "memory.compact",
        "files.create",
        "files.modify",
        "files.delete",
        "git.commit",
        "git.push",
        "system.service.restart",
    ):
        definition = registry.get(cap_id)
        if definition is None:
            checks.append(_fail(f"{cap_id} registered"))
            continue
        if definition.effect != "mutating":
            checks.append(_fail(f"{cap_id} mutating classification", definition.effect))
        elif definition.authorization_mode in {"plan_and_confirm", "local_activation_and_confirm"}:
            checks.append(_pass(f"{cap_id} mutation gated", definition.authorization_mode))
        else:
            checks.append(_fail(f"{cap_id} mutation gated", definition.authorization_mode))
        if definition.receipt_required:
            checks.append(_pass(f"{cap_id} receipt required"))
        else:
            checks.append(_fail(f"{cap_id} receipt required"))
        if definition.generic_bypass_forbidden:
            checks.append(_pass(f"{cap_id} generic bypass forbidden"))
        else:
            checks.append(_fail(f"{cap_id} generic bypass forbidden"))
    for action, detail in LEGACY_UNMIGRATED.items():
        checks.append(_warn(f"unmigrated action visible: {action}", detail))
    return checks


def main() -> int:
    checks = run()
    for check in checks:
        if check.detail:
            print(f"{check.status}: {check.name}: {check.detail}")
        else:
            print(f"{check.status}: {check.name}")
    passed = sum(1 for check in checks if check.status == "PASS")
    warned = sum(1 for check in checks if check.status == "WARN")
    failed = sum(1 for check in checks if check.status == "FAIL")
    print(f"PASS={passed} WARN={warned} FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
