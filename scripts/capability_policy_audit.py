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
}

LEGACY_UNMIGRATED = {
    "operator.support_bundle": "support bundle creation remains registry legacy in this checkpoint",
    "operator.backup": "Backup v1 remains behaviorally unchanged and audit-visible",
    "operator.restore": "Restore v1 remains behaviorally unchanged and audit-visible",
    "memory.lifecycle": "memory lifecycle mutation lanes remain preview-only or legacy",
    "files.*": "file mutation tools are not migrated in v1",
    "git.*": "git mutation tools are not migrated in v1",
    "communications.*": "external communication mutation adapters are not enabled as core capabilities",
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
    for cap_id in ("system.update", "system.uninstall", "system.package.install", "cleanup.execute"):
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
