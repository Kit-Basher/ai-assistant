from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import tempfile

from agent.api_server import AgentRuntime
from agent.llm.notifications import NotificationStore
from agent.mutation_plan import build_mutation_confirmation
from agent.pack_search_authorization import SPECS
from test_api_server import _config


def _runtime(tmp: Path, **overrides: object) -> AgentRuntime:
    os.environ["AGENT_SECRET_STORE_PATH"] = str(tmp / "secrets.enc.json")
    os.environ["AGENT_PERMISSIONS_PATH"] = str(tmp / "permissions.json")
    os.environ["AGENT_AUDIT_LOG_PATH"] = str(tmp / "audit.jsonl")
    return AgentRuntime(_config(str(tmp / "registry.json"), str(tmp / "agent.db"), **overrides))


def _confirmation(plan: dict[str, object]) -> dict[str, object]:
    return build_mutation_confirmation(plan, confirmation_id="explicit-v2f-confirmation")


def test_all_v2f_operations_have_distinct_capability_executor_bindings() -> None:
    assert len(SPECS) == 14
    assert len({row.executor_id for row in SPECS.values()}) == len(SPECS)
    assert all(row.capability_id and row.rollback for row in SPECS.values())


def test_legacy_tokens_booleans_unknown_operations_and_permission_escalation_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw))
        for payload, error in (
            ({"actions": {}, "confirm": True}, "boolean_or_legacy_confirmation_not_authorization"),
            ({"actions": {}, "confirmation_token": "old"}, "legacy_confirmation_token_rejected"),
            ({"actions": {"*": True}}, "permission_wildcard_forbidden"),
            ({"actions": {"unknown.capability": True}}, "permission_capability_unknown"),
        ):
            ok, body = runtime.route_pack_search_mutation("permission.policy.update", payload)
            assert not ok
            assert body["error"] == error
        ok, body = runtime.route_pack_search_mutation("unknown.operation", {})
        assert not ok and body["error"] == "mutation_operation_unknown"


def test_remote_combined_fetch_install_is_explicitly_denied_but_local_install_is_planned() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        runtime = _runtime(tmp)
        ok, body = runtime.route_pack_search_mutation("external_pack.install", {"source": "https://example.invalid/a.zip"})
        assert not ok and body["error"] == "remote_pack_fetch_stage_unimplemented_denied"
        for remote_payload in (
            {"source": {"kind": "generic_archive_url", "url": "https://example.invalid/a.zip"}},
            {"download_url": "https://example.invalid/a.zip"},
            {"archive_url": "https://example.invalid/a.zip"},
            {"source": {"base_url": "https://example.invalid/a.zip"}},
        ):
            ok, body = runtime.packs_install(remote_payload)
            assert not ok and body["error"] == "remote_pack_fetch_stage_unimplemented_denied"
        pack = tmp / "pack"
        pack.mkdir()
        (pack / "SKILL.md").write_text("---\nid: fixture\nname: fixture\nversion: 1.0.0\n---\nSafe text.\n", encoding="utf-8")
        ok, body = runtime.route_pack_search_mutation("external_pack.install", {"path": str(pack)})
        assert ok and body["requires_confirmation"]
        assert body["plan"]["capability_id"] == "pack.lifecycle.install"


def test_cross_operation_stale_target_and_single_use_confirmation_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw), llm_autopilot_safe_mode=False)
        request = {"actions": {"llm.notifications.test": False}}
        ok, preview = runtime.route_pack_search_mutation("permission.policy.update", request)
        assert ok
        plan = preview["plan"]
        wrong = dict(plan)
        wrong["executor_id"] = "external_pack.enable.v1"
        ok, body = runtime.route_pack_search_mutation("permission.policy.update", {**request, "mutation_plan": wrong, "confirmation": _confirmation(plan)})
        assert not ok and body["error"] in {"mutation_operation_scope_mismatch", "mutation_plan_fingerprint_mismatch"}

        runtime.permission_store.update({"actions": {"llm.notifications.send": True}})
        ok, body = runtime.route_pack_search_mutation("permission.policy.update", {**request, "mutation_plan": plan, "confirmation": _confirmation(plan)})
        assert not ok and body["error"] == "mutation_plan_target_changed"

        ok, fresh = runtime.route_pack_search_mutation("permission.policy.update", request)
        assert ok
        fresh_plan = fresh["plan"]
        apply_payload = {**request, "mutation_plan": fresh_plan, "confirmation": _confirmation(fresh_plan)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _i: runtime.route_pack_search_mutation("permission.policy.update", apply_payload), range(2)))
        assert sum(1 for ok, _body in results if ok) == 1


def test_search_image_tag_or_runtime_plan_drift_invalidates_confirmation() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw), llm_autopilot_safe_mode=False)
        configured = {
            "ok": True,
            "setup_mode": "managed_container",
            "service_id": "searxng",
            "image": "docker.io/searxng/searxng:latest",
            "container_name": "personal-agent-searxng",
            "loopback_bind": "127.0.0.1:8888:8080",
            "_execution_plan": {
                "setup_mode": "managed_container",
                "service_id": "searxng",
                "image": "docker.io/searxng/searxng:latest",
                "container_name": "personal-agent-searxng",
                "loopback_bind": "127.0.0.1:8888:8080",
                "executor_pending": {"approved_image": "docker.io/searxng/searxng:latest"},
            },
        }
        runtime._build_search_setup_execution_plan = lambda _payload: copy.deepcopy(configured)  # type: ignore[method-assign]
        ok, preview = runtime.route_pack_search_mutation("search.setup", {})
        assert ok and preview["requires_confirmation"]
        plan = preview["plan"]

        def drifted_builder(payload: dict[str, object]) -> dict[str, object]:
            _ = payload
            built = copy.deepcopy(configured)
            execution = built.get("_execution_plan") if isinstance(built.get("_execution_plan"), dict) else built
            execution["image"] = "docker.io/searxng/searxng:unexpected-tag"
            pending = execution.get("executor_pending") if isinstance(execution.get("executor_pending"), dict) else {}
            pending["approved_image"] = "docker.io/searxng/searxng:unexpected-tag"
            execution["executor_pending"] = pending
            return built

        runtime._build_search_setup_execution_plan = drifted_builder  # type: ignore[method-assign]
        ok, body = runtime.route_pack_search_mutation(
            "search.setup",
            {"mutation_plan": plan, "confirmation": _confirmation(plan)},
        )
        assert not ok
        assert body["error"] == "mutation_plan_target_changed"


def test_safe_mode_keeps_legacy_search_and_generic_llm_mutations_blocked() -> None:
    with tempfile.TemporaryDirectory() as raw:
        runtime = _runtime(Path(raw), safe_mode_enabled=True)

        ok_search, search_body = runtime.route_pack_search_mutation("search.setup", {})
        ok_fix, fix_body = runtime.route_provider_model_mutation("llm.fix", {})

        assert not ok_search
        assert search_body["error"] == "safe_mode_mutation_blocked"
        assert search_body["failure_stage"] == "policy"
        assert search_body["requires_confirmation"] is False
        assert "plan" not in search_body
        assert not ok_fix
        assert fix_body["error"] == "safe_mode_mutation_blocked"


def test_notification_delivery_ledger_blocks_duplicates_and_reconciles_crashes() -> None:
    with tempfile.TemporaryDirectory() as raw:
        store = NotificationStore(str(Path(raw) / "notifications.json"))
        args = {"operation_id": "delivery-1", "transport": "local", "target_fingerprint": "target", "content_fingerprint": "content"}
        with ThreadPoolExecutor(max_workers=4) as pool:
            won = list(pool.map(lambda _i: store.reserve_delivery(**args), range(4)))
        assert won.count(True) == 1
        assert store.mark_delivery_executing("delivery-1")
        restarted = NotificationStore(str(Path(raw) / "notifications.json"))
        rows = restarted.delivery_reconciliation_status()["deliveries"]
        assert rows[0]["state"] == "indeterminate"
        assert restarted.reserve_delivery(**args) is False
        assert restarted.delivery_reconciliation_status()["exactly_once_claimed"] is False
