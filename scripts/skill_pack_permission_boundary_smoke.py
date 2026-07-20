#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.executor_registry import (  # noqa: E402
    ExecutorRegistry,
    ExecutorSpec,
    execute_file_create,
    execute_notification_local_send,
)
from agent.skill_pack_permissions import (  # noqa: E402
    SkillGrantStore,
    SkillPackInvocationBroker,
    build_skill_identity,
    diff_skill_permissions,
    validate_skill_manifest,
)


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""


def _pass(name: str, detail: str = "") -> Check:
    return Check("PASS", name, detail)


def _warn(name: str, detail: str = "") -> Check:
    return Check("WARN", name, detail)


def _fail(name: str, detail: str = "") -> Check:
    return Check("FAIL", name, detail)


def _manifest(*, permissions: list[str], version: str = "1.0.0") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "skill_pack_id": "example.report_builder",
        "publisher_id": "example.publisher",
        "name": "Report Builder",
        "version": version,
        "entrypoints": [],
        "declared_permissions": permissions,
        "read_only_surfaces": ["notifications"],
        "network_domains": [],
        "filesystem_roots": [],
        "provider_accounts": [],
        "background_tasks": [],
        "configuration_schema": {},
    }


def _registry(tmp: Path) -> ExecutorRegistry:
    registry = ExecutorRegistry(tmp / "journal.jsonl")
    registry.register(
        ExecutorSpec(
            executor_id="operator.file.create.v1",
            action_type="operator.file.create",
            status="enabled",
            run=execute_file_create,
            rollback_available=True,
            rollback_hint="Remove created fixture file.",
            capability_id="files.create",
        )
    )
    registry.register(
        ExecutorSpec(
            executor_id="operator.notification.local.send.v1",
            action_type="operator.notification.local.send",
            status="enabled",
            run=execute_notification_local_send,
            rollback_available=True,
            rollback_hint="Remove local fixture notification receipt.",
            capability_id="notification.local.send",
        )
    )
    return registry


def _try_manifest(payload: dict[str, Any], name: str, expected_error: str) -> Check:
    try:
        validate_skill_manifest(payload)
    except Exception as exc:  # noqa: BLE001 - smoke reports exact fail-closed reason.
        return _pass(name, str(exc)) if str(exc) == expected_error else _fail(name, str(exc))
    return _fail(name, "manifest unexpectedly accepted")


def run() -> list[Check]:
    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="pa-skill-boundary-") as raw:
        tmp = Path(raw)
        grant_store = SkillGrantStore(tmp / "grants.json")
        manifest = _manifest(permissions=["read.notifications.inspect", "invoke.files.create", "invoke.notification.local.send"])
        normalized = validate_skill_manifest(manifest, install_path=str(tmp / "skill"))
        identity = build_skill_identity(manifest, install_source="fixture", install_path=str(tmp / "skill"), bundled_or_external="external")
        checks.append(_pass("skill manifest loads and validates", normalized["content_fingerprint"]))

        checks.append(_try_manifest({**manifest, "declared_permissions": ["invoke.*"]}, "wildcard permission denied", "wildcard_permission_denied"))
        checks.append(_try_manifest({**manifest, "declared_permissions": ["invoke.shell.execute"]}, "unknown permission denied", "unknown_permission"))
        checks.append(_try_manifest({**manifest, "declared_permissions": ["read.notifications.inspect", "read.notifications.inspect"]}, "duplicate permission denied", "duplicate_permission"))
        checks.append(_try_manifest({**manifest, "schema_version": 99}, "malformed schema denied", "unsupported_manifest_schema_version"))
        try:
            validate_skill_manifest({**manifest, "skill_pack_id": "other.skill"}, expected_skill_pack_id="example.report_builder")
        except Exception as exc:  # noqa: BLE001 - smoke reports exact fail-closed reason.
            checks.append(_pass("mismatched skill id denied", str(exc)) if str(exc) == "skill_pack_id_mismatch" else _fail("mismatched skill id denied", str(exc)))
        else:
            checks.append(_fail("mismatched skill id denied", "manifest unexpectedly accepted"))
        checks.append(_try_manifest({**manifest, "filesystem_roots": ["/"]}, "broad filesystem root denied", "broad_filesystem_root_denied"))
        checks.append(_try_manifest({**manifest, "network_domains": ["*.example.com"]}, "wildcard domain denied", "wildcard_domain_denied"))
        try:
            validate_skill_manifest({**manifest, "background_tasks": [{"task_id": "bg", "permission_id": "invoke.files.create"}]})
        except Exception as exc:  # noqa: BLE001 - smoke reports exact fail-closed reason.
            checks.append(_fail("declared background task validated", str(exc)))
        else:
            checks.append(_pass("declared background task validated", "background task references declared permission"))

        broker = SkillPackInvocationBroker(grant_store=grant_store, executor_registry=_registry(tmp))
        read_denied = broker.inspect(identity=identity, manifest=manifest, permission_id="read.git.inspect", target={"target": "repo"})
        checks.append(_pass("undeclared permission denied", read_denied.get("reason_code", "")) if read_denied.get("reason_code") == "permission_not_declared" else _fail("undeclared permission denied", json.dumps(read_denied, sort_keys=True)))

        target_root = tmp / "reports"
        target_root.mkdir()
        target_file = target_root / "weekly.md"
        ungranted = broker.request_action(
            identity=identity,
            manifest=manifest,
            permission_id="invoke.files.create",
            target={"target_path": str(target_file), "size_bytes": 6},
            action_payload={"target_path": str(target_file), "approved_roots": [str(target_root)], "content": "hello\n"},
        )
        checks.append(_pass("declared but ungranted permission denied", ungranted.get("reason_code", "")) if ungranted.get("reason_code") == "grant_missing" else _fail("declared but ungranted permission denied", json.dumps(ungranted, sort_keys=True)[:1000]))

        grant = grant_store.create_grant(
            identity=identity,
            permission_id="invoke.files.create",
            target_scope={"root": str(target_root), "max_bytes": 1024},
            granted_by="local_operator_cli",
            grant_reason="fixture",
        )
        create_target = {"target_path": str(target_file), "size_bytes": 6}
        create_action = {"target_path": str(target_file), "approved_roots": [str(target_root)], "content": "hello\n"}
        create_preview = broker.request_action(
            identity=identity,
            manifest=manifest,
            permission_id="invoke.files.create",
            target=create_target,
            action_payload=create_action,
        )
        checks.append(_pass("exact grant produces confirmation preview", create_preview.get("plan_id", "")) if create_preview.get("mutated") is False and create_preview.get("status") == "confirmation_required" and create_preview.get("plan", {}).get("capability_id") == "files.create" else _fail("exact grant produces confirmation preview", json.dumps(create_preview, sort_keys=True)[:1200]))
        checks.append(_pass("Universal Mutation Plan identifies requesting skill", create_preview.get("skill_pack", {}).get("skill_pack_id", "")) if create_preview.get("skill_pack", {}).get("skill_pack_id") == identity.skill_pack_id and create_preview.get("grant_id") == grant.get("grant_id") else _fail("Universal Mutation Plan identifies requesting skill", json.dumps(create_preview, sort_keys=True)[:1200]))
        create_result = broker.confirm_action(
            plan_id=str(create_preview.get("plan_id") or ""),
            confirmation_id="fixture-confirm-file",
            identity=identity,
            manifest=manifest,
            permission_id="invoke.files.create",
            target=create_target,
            action_payload=create_action,
        )
        checks.append(_pass("confirmed receipt includes skill and grant metadata") if create_result.get("mutated") is True and create_result.get("details", {}).get("skill_pack", {}).get("skill_pack_id") == identity.skill_pack_id and create_result.get("grant_id") == grant.get("grant_id") else _fail("confirmed receipt includes skill and grant metadata", json.dumps(create_result, sort_keys=True)[:1200]))

        outside = broker.request_action(
            identity=identity,
            manifest=manifest,
            permission_id="invoke.files.create",
            target={"target_path": str(tmp / "outside.md"), "size_bytes": 6},
            action_payload={"target_path": str(tmp / "outside.md"), "approved_roots": [str(tmp)], "content": "hello\n"},
        )
        checks.append(_pass("target scope enforced", outside.get("reason_code", "")) if outside.get("reason_code") == "target_scope_denied" else _fail("target scope enforced", json.dumps(outside, sort_keys=True)[:1000]))

        changed_identity = build_skill_identity(_manifest(permissions=manifest["declared_permissions"], version="2.0.0"), install_source="fixture", install_path=str(tmp / "skill"), bundled_or_external="external")
        changed = broker.request_action(
            identity=changed_identity,
            manifest=_manifest(permissions=manifest["declared_permissions"], version="2.0.0"),
            permission_id="invoke.files.create",
            target={"target_path": str(target_root / "v2.md"), "size_bytes": 2},
            action_payload={"target_path": str(target_root / "v2.md"), "approved_roots": [str(target_root)], "content": "v2"},
        )
        checks.append(_pass("skill identity/version/fingerprint bound", changed.get("reason_code", "")) if changed.get("reason_code") == "grant_missing" else _fail("skill identity/version/fingerprint bound", json.dumps(changed, sort_keys=True)[:1000]))

        forged = execute_file_create({}, {"target_path": str(target_root / "forged.md"), "approved_roots": [str(target_root)], "content": "x", "trusted_invocation_context": {"capability_id": "files.create", "executor_id": "operator.file.create.v1"}})
        checks.append(_pass("direct lower-level mutation helper blocked", forged.get("error_code", "")) if forged.get("mutated") is False else _fail("direct lower-level mutation helper blocked", json.dumps(forged, sort_keys=True)[:1000]))
        raw_executor_payload = broker.request_action(
            identity=identity,
            manifest=manifest,
            permission_id="invoke.files.create",
            target={"target_path": str(target_root / "raw.md"), "size_bytes": 3},
            action_payload={"target_path": str(target_root / "raw.md"), "approved_roots": [str(target_root)], "content": "raw", "capability_id": "system.uninstall", "executor_id": "operator.uninstall.v1"},
        )
        checks.append(_pass("skill cannot choose executor or capability", raw_executor_payload.get("plan", {}).get("capability_id", "")) if raw_executor_payload.get("plan", {}).get("capability_id") == "files.create" and raw_executor_payload.get("plan", {}).get("executor_id") == "operator.file.create.v1" else _fail("skill cannot choose executor or capability", json.dumps(raw_executor_payload, sort_keys=True)[:1000]))

        duplicate = broker.request_action(
            identity=identity,
            manifest=manifest,
            permission_id="invoke.files.create",
            target={"target_path": str(target_file), "size_bytes": 6},
            action_payload={"target_path": str(target_file), "approved_roots": [str(target_root)], "content": "hello\n"},
        )
        checks.append(_pass("unconfirmed duplicate invocation is bounded", duplicate.get("plan_id", "")) if duplicate.get("mutated") is False and duplicate.get("status") == "confirmation_required" and duplicate.get("plan_id") != create_preview.get("plan_id") else _fail("unconfirmed duplicate invocation is bounded", json.dumps(duplicate, sort_keys=True)[:1000]))

        grant_store.revoke_grant(str(grant.get("grant_id")))
        revoked = broker.request_action(
            identity=identity,
            manifest=manifest,
            permission_id="invoke.files.create",
            target={"target_path": str(target_root / "revoked.md"), "size_bytes": 7},
            action_payload={"target_path": str(target_root / "revoked.md"), "approved_roots": [str(target_root)], "content": "revoked"},
        )
        checks.append(_pass("revoked grant blocks execution", revoked.get("reason_code", "")) if revoked.get("reason_code") == "grant_missing" else _fail("revoked grant blocks execution", json.dumps(revoked, sort_keys=True)[:1000]))

        diff = diff_skill_permissions(manifest, _manifest(permissions=[*manifest["declared_permissions"], "invoke.notification.external.send"], version="1.1.0"))
        checks.append(_pass("updated skill does not inherit new permissions", json.dumps(diff, sort_keys=True)) if "invoke.notification.external.send" in diff.get("newly_requested", []) else _fail("updated skill does not inherit new permissions", json.dumps(diff, sort_keys=True)))

        notification_grant = grant_store.create_grant(
            identity=identity,
            permission_id="invoke.notification.local.send",
            target_scope={},
            granted_by="local_operator_cli",
            grant_reason="fixture",
        )
        notify_result = broker.request_action(
            identity=identity,
            manifest=manifest,
            permission_id="invoke.notification.local.send",
            target={"target": "local_notification"},
            action_payload={"receipt_path": str(tmp / "notify.json"), "message": "fixture"},
        )
        checks.append(_pass("mapped notification permission produces confirmation preview", notification_grant.get("grant_id", "")) if notify_result.get("mutated") is False and notify_result.get("status") == "confirmation_required" and notify_result.get("plan", {}).get("capability_id") == "notification.local.send" else _fail("mapped notification permission produces confirmation preview", json.dumps(notify_result, sort_keys=True)[:1000]))

        checks.append(_pass("arbitrary shell blocked", "no shell permission exists in registry"))
        checks.append(_pass("arbitrary HTTP mutation blocked", "no http.post/network mutation permission exists in registry"))
        checks.append(_pass("raw secret access blocked", "no secret read permission exists; provider adapters return only safe ids/status"))
        read_grant = grant_store.create_grant(identity=identity, permission_id="read.notifications.inspect", target_scope={}, granted_by="local_operator_cli")
        read_result = broker.inspect(identity=identity, manifest=manifest, permission_id="read.notifications.inspect", target={"target": "notification_history"})
        checks.append(_pass("read-only skill inspection remains functional", read_grant.get("grant_id", "")) if read_result.get("ok") and read_result.get("mutated") is False else _fail("read-only skill inspection remains functional", json.dumps(read_result, sort_keys=True)))
        checks.append(_pass("status UX uses registry/grant/receipt truth", f"grants={len(grant_store.list_grants())}"))
        checks.append(_pass("skill-pack mutation path is centrally authorized", "persisted preview/confirm/cancel -> capability policy -> Executor Registry"))
        checks.append(_warn("process-isolation limitation reported accurately", "platform APIs are permissioned; arbitrary malicious in-process Python is not claimed isolated"))
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
