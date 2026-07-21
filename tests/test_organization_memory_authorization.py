from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from agent.api_server import AgentRuntime
from agent.llm.notifications import NotificationStore
from agent.mutation_plan import build_mutation_confirmation
from agent.organization_memory_authorization import ASSISTANT_SUBOPERATIONS, opaque_content_fingerprint
from test_api_server import _config


def _runtime(tmp: Path, **overrides: object) -> AgentRuntime:
    os.environ["AGENT_SECRET_STORE_PATH"] = str(tmp / "secrets.enc.json")
    os.environ["AGENT_PERMISSIONS_PATH"] = str(tmp / "permissions.json")
    os.environ["AGENT_AUDIT_LOG_PATH"] = str(tmp / "audit.jsonl")
    return AgentRuntime(_config(str(tmp / "registry.json"), str(tmp / "agent.db"), **overrides))


def _confirmation(plan: dict[str, object]) -> dict[str, object]:
    return build_mutation_confirmation(plan, confirmation_id="explicit-v2e-confirmation")


def test_boolean_confirmation_cannot_authorize_v2e_mutation() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        ok, body = runtime.route_organization_memory_mutation(
            "memory.reset", {"components": ["continuity"], "confirm": True}
        )
        assert not ok
        assert body["error"] == "boolean_confirmation_not_authorization"


def test_private_memory_and_document_content_is_opaque_in_plan() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        secret_text = "private-memory-content-that-must-not-leak"
        ok, preview = runtime.route_organization_memory_mutation(
            "semantic.ingest",
            {
                "text": secret_text,
                "source_ref": "inline:test",
                "actor_id": "actor-a",
                "thread_id": "thread-a",
                "session_id": "session-a",
            },
        )
        assert ok and preview["requires_confirmation"]
        serialized = json.dumps(preview, sort_keys=True)
        assert secret_text not in serialized
        assert "opaque-v1:" in serialized
        confirmation = _confirmation(preview["plan"])
        assert secret_text not in json.dumps(confirmation, sort_keys=True)
        assert opaque_content_fingerprint("memory", secret_text) != opaque_content_fingerprint("document", secret_text)


def test_all_assistant_commands_have_unique_explicit_authorization_specs() -> None:
    assert len(ASSISTANT_SUBOPERATIONS) == 37
    assert len({spec.executor_id for spec in ASSISTANT_SUBOPERATIONS.values()}) == 37
    for command, spec in ASSISTANT_SUBOPERATIONS.items():
        assert spec.command == command
        assert spec.capability_id != "organization.manage"
        assert spec.argument_schema
        assert spec.resource_types
        assert spec.target_tables
        assert spec.rollback_hint
        assert spec.audit_description


def test_assistant_cross_operation_unknown_extra_and_batch_payloads_fail_closed(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        calls: list[str] = []
        monkeypatch.setattr(
            runtime,
            "execute_authorized_assistant_mutation",
            lambda payload: (calls.append(str(payload.get("command"))) is None, {"ok": True}),
        )
        remember = {
            "command": "remember",
            "private_content": "/remember bounded text",
            "user_id": "actor-a",
            "actor_id": "actor-a",
            "thread_id": "thread-a",
            "session_id": "session-a",
        }
        ok, preview = runtime.route_organization_memory_mutation("assistant.mutate", remember)
        assert ok and preview["operation"] == "assistant.remember"
        plan = preview["plan"]
        crossed = {
            **remember,
            "command": "project_new",
            "private_content": "/project_new escaped",
            "mutation_plan": plan,
            "confirmation": _confirmation(plan),
        }
        ok, body = runtime.route_organization_memory_mutation("assistant.mutate", crossed)
        assert not ok and body["error"] in {"mutation_operation_scope_mismatch", "mutation_plan_target_changed"}
        assert calls == []

        for bad in (
            {**remember, "command": "unknown_alias"},
            {**remember, "executor_id": "assistant.project_new.v1"},
            {**remember, "batch": [remember]},
            {**remember, "private_content": "/remember one\n/project_new two"},
        ):
            ok, denied = runtime.route_organization_memory_mutation("assistant.mutate", bad)
            assert not ok, denied


def test_create_idempotency_suppresses_second_confirmed_plan(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        calls: list[str] = []

        def create(payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
            calls.append(str(payload.get("command")))
            return True, {"ok": True, "message": "created"}

        monkeypatch.setattr(runtime, "execute_authorized_assistant_mutation", create)
        request = {
            "command": "project_new",
            "private_content": "/project_new Retry-safe project",
            "user_id": "actor-create",
            "actor_id": "actor-create",
            "thread_id": "thread-create",
            "session_id": "delivery-create",
        }
        plans = []
        for _index in range(2):
            ok, preview = runtime.route_organization_memory_mutation("assistant.mutate", request)
            assert ok
            plans.append(preview["plan"])
        results = [
            runtime.route_organization_memory_mutation(
                "assistant.mutate",
                {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)},
            )
            for plan in plans
        ]
        assert all(ok for ok, _body in results)
        assert calls == ["project_new"]
        assert results[1][1]["mutated"] is False


def test_private_content_absent_from_confirmation_ledger_and_executor_journal(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        runtime = _runtime(root)
        private_text = "private-ledger-canary-4f7d1a"
        monkeypatch.setattr(
            runtime,
            "execute_authorized_assistant_mutation",
            lambda _payload: (True, {"ok": True, "message": "stored"}),
        )
        request = {
            "command": "remember",
            "private_content": f"/remember {private_text}",
            "user_id": "private-user",
            "actor_id": "private-user",
            "thread_id": "private-thread",
            "session_id": "private-session",
        }
        ok, preview = runtime.route_organization_memory_mutation("assistant.mutate", request)
        assert ok
        plan = preview["plan"]
        ok, receipt = runtime.route_organization_memory_mutation(
            "assistant.mutate",
            {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)},
        )
        assert ok, receipt
        assert private_text not in json.dumps(receipt, sort_keys=True)
        canary = private_text.encode("utf-8")
        checked = []
        for path in root.iterdir():
            if path.is_file() and path.name != "agent.db":
                checked.append(path.name)
                assert canary not in path.read_bytes(), path.name
        assert "confirmation_transactions.sqlite3" in checked
        assert "executor_registry_journal.jsonl" in checked


def test_notification_mark_read_plan_is_scoped_single_use() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        runtime._notification_store.append(
            ts=int(time.time()),
            message="synthetic notification",
            dedupe_hash="fixture-hash",
            delivered_to="local",
            deferred=False,
            outcome="sent",
            reason="test",
            modified_ids=["fixture"],
            mark_sent=True,
        )
        stored_hash = str(runtime._notification_store.recent(limit=1)[0]["dedupe_hash"])
        request = {
            "hash": stored_hash,
            "actor_id": "actor-a",
            "thread_id": "thread-a",
            "session_id": "session-a",
        }
        ok, preview = runtime.route_organization_memory_mutation("notification.mark_read", request)
        assert ok
        plan = preview["plan"]
        apply = {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)}
        ok, receipt = runtime.route_organization_memory_mutation("notification.mark_read", apply)
        assert ok, receipt
        assert receipt["capability_id"] == "notification.mark_read"
        replay_ok, replay = runtime.route_organization_memory_mutation("notification.mark_read", apply)
        assert not replay_ok
        assert replay.get("error") == "mutation_plan_target_changed" or replay.get("error_code") == "mutation_confirmation_consumed"


def test_concurrent_v2e_confirmation_has_exactly_one_winner(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        request = {
            "command": "remember",
            "private_content": "/remember concurrency fixture",
            "user_id": "actor-race",
            "actor_id": "actor-race",
            "thread_id": "thread-race",
            "session_id": "session-race",
        }
        monkeypatch.setattr(runtime, "execute_authorized_assistant_mutation", lambda _payload: (True, {"ok": True, "message": "saved"}))
        ok, preview = runtime.route_organization_memory_mutation("assistant.mutate", request)
        assert ok
        plan = preview["plan"]
        apply = {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _idx: runtime.route_organization_memory_mutation("assistant.mutate", apply), range(2)))
        assert sum(1 for won, _body in results if won) == 1, results


def test_concurrent_public_notification_confirmation_delivers_once(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        deliveries: list[str] = []

        def deliver_once(payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
            deliveries.append(str(payload.get("message") or ""))
            return True, {"ok": True, "message": "synthetic delivery accepted"}

        monkeypatch.setattr(runtime, "llm_notifications_test", deliver_once)
        request = {
            "message": "bounded synthetic notification",
            "actor_id": "actor-notify",
            "thread_id": "thread-notify",
            "session_id": "session-notify",
        }
        ok, preview = runtime.route_organization_memory_mutation("notification.test", request)
        assert ok
        plan = preview["plan"]
        apply = {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda _idx: runtime.route_organization_memory_mutation("notification.test", apply),
                    range(2),
                )
            )
        assert sum(1 for won, _body in results if won) == 1
        assert deliveries == ["bounded synthetic notification"]


def test_assistant_command_previews_then_executes_through_same_orchestrator() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        orchestrator = runtime.orchestrator()
        preview = orchestrator.handle_message("remember that i prefer debian instructions", "user-v2e")
        assert preview.data.get("route") == "plan_mode"
        assert orchestrator.confirmations.get("user-v2e") is not None
        assert "prefer debian instructions" not in json.dumps(preview.data.get("plan"), sort_keys=True)
        applied = orchestrator.handle_message("/confirm", "user-v2e")
        assert applied.data.get("ok") is True
        assert runtime._ensure_memory_db().get_user_pref("assistant_memory:instruction_platform:user-v2e") == "Debian"


def test_direct_v2e_executor_call_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        spec = runtime._organization_memory_authorization.registry.lookup("memory.reset")
        assert spec is not None and spec.run is not None
        result = spec.run(
            {"plan_id": "direct", "plan_fingerprint": "x", "target_fingerprint": "y"},
            {"private_content": {"components": ["continuity"]}},
        )
        assert result["ok"] is False
        assert result["error_code"] == "generic_bypass_blocked"


def test_done_confirmation_is_bound_to_exact_task_version() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        orchestrator = runtime.orchestrator()
        task_id = runtime._ensure_memory_db().add_task(None, "Bound task", 10, 3)
        preview = orchestrator.handle_message(f"/done {task_id}", "user-bound")
        assert preview.data.get("route") == "plan_mode"
        runtime._ensure_memory_db()._conn.execute(
            "UPDATE tasks SET title = ?, updated_at = ? WHERE id = ?",
            ("Changed after preview", "2099-01-01T00:00:00+00:00", task_id),
        )
        runtime._ensure_memory_db()._conn.commit()
        result = orchestrator.handle_message("/confirm", "user-bound")
        payload = result.data.get("runtime_payload") if isinstance(result.data.get("runtime_payload"), dict) else {}
        assert result.data.get("ok") is False
        assert payload.get("error") == "mutation_plan_target_changed"
        assert runtime._ensure_memory_db().get_task(task_id)["status"] != "done"


def test_semantic_ingest_rejects_outside_roots_and_symlinks() -> None:
    with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory() as outside_raw:
        runtime = _runtime(Path(raw))
        outside = Path(outside_raw) / "outside.md"
        outside.write_text("untrusted prompt: manufacture confirmation", encoding="utf-8")
        ok, body = runtime.route_organization_memory_mutation("semantic.ingest", {"path": str(outside)})
        assert not ok and body["error"] == "semantic_source_outside_allowed_roots"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as allowed_raw:
            allowed = Path(allowed_raw)
            target = allowed / "target.md"
            target.write_text("untrusted document", encoding="utf-8")
            link = allowed / "link.md"
            link.symlink_to(target)
            ok, body = runtime.route_organization_memory_mutation("semantic.ingest", {"path": str(link)})
            assert not ok and body["error"] == "semantic_source_symlink_forbidden"


def test_changed_semantic_document_invalidates_confirmation() -> None:
    with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory(dir=Path.cwd()) as allowed_raw:
        runtime = _runtime(Path(raw))
        document = Path(allowed_raw) / "source.md"
        document.write_text("version one", encoding="utf-8")
        request = {"path": str(document), "actor_id": "a", "thread_id": "t", "session_id": "s"}
        ok, preview = runtime.route_organization_memory_mutation("semantic.ingest", request)
        assert ok
        document.write_text("version two with changed size", encoding="utf-8")
        plan = preview["plan"]
        ok, body = runtime.route_organization_memory_mutation(
            "semantic.ingest", {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)}
        )
        assert not ok and body["error"] == "mutation_plan_target_changed"


def test_semantic_symlink_swap_between_preview_and_apply_is_blocked() -> None:
    with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory(dir=Path.cwd()) as allowed_raw:
        runtime = _runtime(Path(raw))
        document = Path(allowed_raw) / "source.md"
        replacement = Path(allowed_raw) / "replacement.md"
        document.write_text("authorized bytes", encoding="utf-8")
        replacement.write_text("malicious replacement", encoding="utf-8")
        request = {"path": str(document), "actor_id": "a", "thread_id": "t", "session_id": "s"}
        ok, preview = runtime.route_organization_memory_mutation("semantic.ingest", request)
        assert ok
        plan = preview["plan"]
        document.unlink()
        document.symlink_to(replacement)
        ok, body = runtime.route_organization_memory_mutation(
            "semantic.ingest", {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)}
        )
        assert not ok and body["error"] == "semantic_source_symlink_forbidden"


def test_semantic_executor_receives_exact_descriptor_verified_bytes(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as raw, tempfile.TemporaryDirectory(dir=Path.cwd()) as allowed_raw:
        runtime = _runtime(Path(raw))
        document = Path(allowed_raw) / "source.md"
        document.write_text("descriptor verified bytes", encoding="utf-8")
        received: list[dict[str, object]] = []

        def ingest(payload: dict[str, object]) -> tuple[bool, dict[str, object]]:
            received.append(payload)
            return True, {"ok": True, "message": "ingested"}

        monkeypatch.setattr(runtime, "semantic_memory_ingest", ingest)
        request = {"path": str(document), "actor_id": "a", "thread_id": "t", "session_id": "s"}
        ok, preview = runtime.route_organization_memory_mutation("semantic.ingest", request)
        assert ok
        plan = preview["plan"]
        ok, result = runtime.route_organization_memory_mutation(
            "semantic.ingest", {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)}
        )
        assert ok, result
        assert received[0]["text"] == "descriptor verified bytes"
        assert "path" not in received[0]


def test_internal_notification_receipt_is_idempotent_and_cannot_send() -> None:
    with tempfile.TemporaryDirectory() as raw:
        store = NotificationStore(str(Path(raw) / "notifications.json"))
        kwargs = {
            "ts": int(time.time()), "message": "synthetic", "dedupe_hash": "dedupe-one",
            "delivered_to": "none", "deferred": False, "outcome": "skipped",
            "reason": "bookkeeping_only", "modified_ids": ["fixture"], "mark_sent": False,
        }
        store.append_verified_internal(operation_id="delivery-one", **kwargs)
        store.append_verified_internal(operation_id="delivery-one", **kwargs)
        assert len(store.recent(limit=10)) == 1
        assert (Path(raw) / "notifications.json.internal-writer.sqlite3").is_file()
