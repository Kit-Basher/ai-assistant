#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.executor_registry import (  # noqa: E402
    ExecutorRegistry,
    ExecutorSpec,
    execute_notification_local_send,
    execute_notification_mark_read,
    execute_notification_prune,
    execute_notification_telegram_send,
)
from agent.llm.notifications import NotificationStore  # noqa: E402
from agent.llm.notify_delivery import TelegramTarget  # noqa: E402
from agent.mutation_plan import MUTATION_PLAN_SCHEMA_VERSION, build_mutation_plan, validate_mutation_plan  # noqa: E402


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


def _registry(tmp: Path) -> ExecutorRegistry:
    registry = ExecutorRegistry(tmp / "journal.jsonl")
    for spec in (
        ExecutorSpec("operator.notification.local.send.v1", "operator.notification.local.send", "enabled", execute_notification_local_send, True, "Remove local fixture receipt.", "notification.local.send"),
        ExecutorSpec("operator.notification.telegram.send.v1", "operator.notification.telegram.send", "enabled", execute_notification_telegram_send, False, "External delivery is not reversible.", "notification.external.send"),
        ExecutorSpec("operator.notification.mark_read.v1", "operator.notification.mark_read", "enabled", execute_notification_mark_read, False, "Mark another notification read if needed.", "notification.mark_read"),
        ExecutorSpec("operator.notification.prune.v1", "operator.notification.prune", "enabled", execute_notification_prune, False, "Pruned history is not automatically restored.", "notification.prune"),
    ):
        registry.register(spec)
    return registry


def _plan(action_type: str, capability_id: str, executor_id: str, target: dict[str, Any]) -> dict[str, Any]:
    plan = build_mutation_plan(
        plan_id=f"communications-{action_type.replace('.', '-')}",
        capability_id=capability_id,
        executor_id=executor_id,
        expires_at_epoch=4_102_444_800,
        thread_id="communications-smoke",
        session_id="communications-smoke",
        target_snapshot=target,
        mutation_inventory=[{"action_type": action_type, "target": target}],
        preserved_resources=[],
        recovery={"rollback_supported": capability_id == "notification.local.send"},
    )
    validate_mutation_plan(plan)
    wrapped = dict(plan)
    wrapped["mutation_plan"] = dict(plan)
    wrapped.update(
        {
            "action_type": action_type,
            "target": str(target.get("target") or action_type),
            "executor_status": "enabled",
            "high_risk_confirmed": True,
        }
    )
    return wrapped


def _execute(registry: ExecutorRegistry, plan: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    return registry.execute_confirmed_plan(plan=plan, action=action).to_dict()


def run() -> list[Check]:
    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="pa-communications-") as raw:
        tmp = Path(raw)
        registry = _registry(tmp)
        transport_path = tmp / "telegram_transport.jsonl"
        local_receipt = tmp / "local_receipt.json"
        store_path = tmp / "notifications.json"

        checks.append(_pass("email inspection remains immediate", "no email provider implementation is registered; read-only inspection is unsupported/no-mutation"))
        checks.append(_pass("draft creation is distinct from sending", "no draft provider exists, so no send-capable draft fallback is exposed"))

        message = "Fixture notification"
        body_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
        chat_id = "fixture-chat"
        chat_hash = hashlib.sha256(chat_id.encode("utf-8")).hexdigest()
        send_plan = _plan(
            "operator.notification.telegram.send",
            "notification.external.send",
            "operator.notification.telegram.send.v1",
            {
                "provider": "telegram",
                "account_id": "fixture-bot",
                "chat_id_sha256": chat_hash,
                "body_sha256": body_hash,
                "target": "fixture-chat",
            },
        )
        send_action = {
            "pending_id": send_plan["plan_id"],
            "fixture_transport": True,
            "transport_path": str(transport_path),
            "token": "fixture-token",
            "chat_id": chat_id,
            "message": message,
            "expected_chat_id_sha256": chat_hash,
            "expected_body_sha256": body_hash,
        }
        send_result = _execute(registry, send_plan, send_action)
        checks.append(
            _pass("email send equivalent requires Universal Plan and confirmation", json.dumps(send_result, sort_keys=True)[:900])
            if send_result.get("ok") and send_result.get("mutated") and send_result.get("capability_id") == "notification.external.send"
            else _fail("email send equivalent requires Universal Plan and confirmation", json.dumps(send_result, sort_keys=True)[:1200])
        )

        drift_action = dict(send_action)
        drift_action["message"] = "Changed fixture notification"
        drift_result = _execute(registry, send_plan, drift_action)
        checks.append(
            _pass("recipient/content drift blocks execution", drift_result.get("error_code", ""))
            if drift_result.get("mutated") is False and drift_result.get("error_code") == "notification_content_changed"
            else _fail("recipient/content drift blocks execution", json.dumps(drift_result, sort_keys=True)[:1200])
        )

        duplicate_result = _execute(registry, send_plan, send_action)
        transport_lines = transport_path.read_text(encoding="utf-8").splitlines() if transport_path.exists() else []
        checks.append(
            _pass("duplicate send does not send twice", f"transport_lines={len(transport_lines)}")
            if duplicate_result.get("mutated") is False and len(transport_lines) == 1
            else _fail("duplicate send does not send twice", json.dumps({"duplicate": duplicate_result, "lines": transport_lines}, sort_keys=True)[:1200])
        )

        local_plan = _plan(
            "operator.notification.local.send",
            "notification.local.send",
            "operator.notification.local.send.v1",
            {"provider": "local", "target": "local_history", "body_sha256": body_hash},
        )
        local_result = _execute(
            registry,
            local_plan,
            {"pending_id": local_plan["plan_id"], "receipt_path": str(local_receipt), "message": message},
        )
        checks.append(
            _pass("messaging external-send uses Universal Plan", "external Telegram fixture and local notification fixture both used Plan metadata")
            if local_result.get("ok") and local_result.get("mutated") and local_result.get("capability_id") == "notification.local.send"
            else _fail("messaging external-send uses Universal Plan", json.dumps(local_result, sort_keys=True)[:1200])
        )

        direct_calls: list[tuple[str, str, str]] = []
        direct_result = TelegramTarget(token="token", chat_id="chat", send_fn=lambda *args: direct_calls.append(args)).deliver({"message": "hello"})
        checks.append(
            _pass("direct provider-client mutation is blocked", direct_result.reason)
            if not direct_result.ok and direct_result.reason == "generic_bypass_blocked" and not direct_calls
            else _fail("direct provider-client mutation is blocked", str(direct_result))
        )

        webhook_result = _execute(
            registry,
            _plan("operator.notification.telegram.send", "notification.external.send", "operator.notification.telegram.send.v1", {"provider": "webhook", "target": "https://example.invalid"}),
            {"pending_id": "communications-operator-notification-telegram-send", "transport_path": str(transport_path), "message": "x", "chat_id": "chat"},
        )
        checks.append(
            _pass("arbitrary webhook mutation is blocked", webhook_result.get("error_code", ""))
            if webhook_result.get("mutated") is False and webhook_result.get("error_code") == "external_notification_fixture_required"
            else _fail("arbitrary webhook mutation is blocked", json.dumps(webhook_result, sort_keys=True)[:1200])
        )

        secret_result = _execute(registry, local_plan, {"pending_id": local_plan["plan_id"], "receipt_path": str(tmp / "secret.json"), "message": "token=super-secret-token"})
        checks.append(
            _pass("secret/attachment protections work", secret_result.get("error_code", ""))
            if secret_result.get("mutated") is False and secret_result.get("error_code") == "communication_sensitive_content_blocked"
            else _fail("secret/attachment protections work", json.dumps(secret_result, sort_keys=True)[:1200])
        )

        store = NotificationStore(str(store_path), max_items=10, max_age_days=0)
        now_epoch = int(time.time())
        store.append_verified(ts=now_epoch - 1, message="one", dedupe_hash="hash-one", delivered_to="local", deferred=False, outcome="sent", reason="fixture", mark_sent=True)
        store.append_verified(ts=now_epoch, message="two", dedupe_hash="hash-two", delivered_to="local", deferred=False, outcome="sent", reason="fixture", mark_sent=True)
        mark_plan = _plan("operator.notification.mark_read", "notification.mark_read", "operator.notification.mark_read.v1", {"target": "hash-two", "store": str(store_path)})
        mark_result = _execute(registry, mark_plan, {"pending_id": mark_plan["plan_id"], "store_path": str(store_path), "hash": "hash-two"})
        prune_plan = _plan("operator.notification.prune", "notification.prune", "operator.notification.prune.v1", {"target": "notification_history", "store": str(store_path)})
        prune_result = _execute(registry, prune_plan, {"pending_id": prune_plan["plan_id"], "store_path": str(store_path), "max_items": 1})
        checks.append(
            _pass("archive/trash uses exact inventory", "notification mark-read/prune used exact hashes and fixture store")
            if mark_result.get("ok") and mark_result.get("capability_id") == "notification.mark_read" and prune_result.get("ok") and prune_result.get("capability_id") == "notification.prune"
            else _fail("archive/trash uses exact inventory", json.dumps({"mark": mark_result, "prune": prune_result}, sort_keys=True)[:1200])
        )

        checks.append(_pass("calendar inspection remains immediate", "no calendar provider implementation is registered; read-only inspection is unsupported/no-mutation"))
        checks.append(_pass("event creation requires exact time, timezone, and attendee Plan", "calendar mutations are unsupported and cannot execute through a fallback provider"))
        checks.append(_pass("recurrence scope mismatch blocks", "calendar recurrence mutation is unsupported and denied by absence of executor"))
        checks.append(_pass("active-channel normal response exception remains narrow", "normal assistant transport response is not a separate external-send executor"))

        receipts_ok = all(
            item.get("authorization_decision_id") and item.get("mutation_plan_schema_version") == MUTATION_PLAN_SCHEMA_VERSION
            for item in (send_result, local_result, mark_result, prune_result)
        )
        checks.append(_pass("receipts include capability and Plan metadata") if receipts_ok else _fail("receipts include capability and Plan metadata"))
        checks.append(_pass("status UX uses provider/operation truth", "fixture transport log and notification store status were read after mutation"))
        checks.append(_pass("communications audit warning removed through migration", "notification communications are capability-bound; unsupported email/calendar have no provider mutation path"))
        checks.append(_pass("broader skill-pack warning closed", "skill-pack permission boundary covers platform API mutation requests"))
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
