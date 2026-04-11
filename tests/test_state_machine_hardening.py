from __future__ import annotations

import json
import os
import tempfile
import unittest
import time
from unittest.mock import patch

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.packs.manifest import PackManifest
from agent.packs.store import PackStore
from agent.runtime_lifecycle import RuntimeLifecyclePhase
from agent.state_transitions import (
    confirmation_transition_state,
    install_pack_write_is_noop,
    pack_approval_hash_write_is_noop,
    pack_enabled_write_is_noop,
    runtime_phase_transition_allowed,
    startup_phase_transition_allowed,
)
from agent.ux.llm_fixit_wizard import confirm_token_for_plan_rows


def _config(registry_path: str, db_path: str) -> Config:
    base = Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path="/tmp/skills",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="llama3",
        ollama_model_sentinel=None,
        ollama_model_worker=None,
        allow_cloud=True,
        prefer_local=True,
        llm_timeout_seconds=15,
        llm_provider="none",
        enable_llm_presentation=False,
        openai_base_url=None,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        llm_selector="single",
        llm_broker_policy_path=None,
        llm_allow_remote=True,
        openrouter_api_key=None,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_model="openai/gpt-4o-mini",
        openrouter_site_url=None,
        openrouter_app_name=None,
        llm_registry_path=registry_path,
        llm_routing_mode="auto",
        llm_retry_attempts=1,
        llm_retry_base_delay_ms=0,
        llm_circuit_breaker_failures=2,
        llm_circuit_breaker_window_seconds=60,
        llm_circuit_breaker_cooldown_seconds=30,
        llm_usage_stats_path=os.path.join(os.path.dirname(db_path), "usage_stats.json"),
        llm_health_state_path=os.path.join(os.path.dirname(db_path), "llm_health_state.json"),
        llm_automation_enabled=False,
        model_scout_state_path=os.path.join(os.path.dirname(db_path), "model_scout_state.json"),
        autopilot_notify_store_path=os.path.join(os.path.dirname(db_path), "llm_notifications.json"),
    )
    return base.__class__(**base.__dict__)


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.client_address = ("127.0.0.1", 12345)
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload or {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:  # type: ignore[override]
        self.status_code = status
        self.content_type = "application/json"
        self.body = json.dumps(payload, ensure_ascii=True).encode("utf-8")

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        *,
        content_type: str,
        cache_control: str | None = None,
    ) -> None:
        _ = cache_control
        self.status_code = status
        self.content_type = content_type
        self.body = body

    def _read_json(self) -> dict[str, object]:
        return self._payload


class TestStateMachineHardening(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.env_backup)
        self.tmpdir.cleanup()

    def test_pack_lifecycle_idempotency_and_removal_reuse_are_safe(self) -> None:
        store = PackStore(self.db_path)
        manifest = PackManifest(
            pack_id="pack.demo",
            version="1.0.0",
            title="Demo Pack",
            description="demo skill",
            entrypoints=("skills.demo:handler",),
            trust="trusted",
            permissions={"ifaces": ["demo.run"]},
        )
        manifest_path = os.path.join(self.tmpdir.name, "pack.json")

        first = store.install_pack(manifest, manifest_path=manifest_path, enable=False)
        second = store.install_pack(manifest, manifest_path=manifest_path, enable=False)
        self.assertEqual(first, second)
        self.assertTrue(
            install_pack_write_is_noop(
                second,
                version=manifest.version,
                trust=manifest.trust,
                manifest_path=manifest_path,
                permissions_json=json.dumps(second.get("permissions") or {}, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                permissions_hash=str(second.get("permissions_hash") or ""),
                approved_permissions_hash=str(second.get("approved_permissions_hash") or "").strip() or None,
                enabled_value=bool(second.get("enabled", False)),
            )
        )

        enabled_once = store.set_enabled("pack.demo", True)
        enabled_twice = store.set_enabled("pack.demo", True)
        self.assertEqual(enabled_once, enabled_twice)
        self.assertTrue(bool((enabled_twice or {}).get("enabled", False)))
        self.assertTrue(pack_enabled_write_is_noop(enabled_twice, enabled=True))

        approval_hash = str((enabled_twice or {}).get("permissions_hash") or "")
        approved_once = store.set_approval_hash("pack.demo", approval_hash)
        approved_twice = store.set_approval_hash("pack.demo", approval_hash)
        self.assertEqual(approved_once, approved_twice)
        self.assertTrue(pack_approval_hash_write_is_noop(approved_twice, approval_hash=approval_hash))

        canonical_pack = {
            "id": "pack.external.demo",
            "name": "External Demo",
            "version": "1.0.0",
            "type": "skill",
            "pack_identity": {
                "canonical_id": "pack.external.demo",
                "content_hash": "hash",
                "source_fingerprint": "source-a",
            },
            "source_history": [{"source_fingerprint": "source-a", "fetched_at": 1}],
            "versions": [{"canonical_id": "pack.external.demo", "content_hash": "hash", "seen_at": 1}],
            "trust_anchor": {
                "first_seen_at": "1",
                "first_seen_source": "source-a",
                "local_review_status": "unreviewed",
                "user_approved_hashes": [],
            },
        }
        first_external = store.record_external_pack(
            canonical_pack=dict(canonical_pack),
            classification="portable_text_skill",
            status="normalized",
            risk_report={"level": "low", "score": 0.1, "flags": []},
            review_envelope={"review_required": True, "why_risk": []},
            quarantine_path=os.path.join(self.tmpdir.name, "quarantine"),
            normalized_path=os.path.join(self.tmpdir.name, "normalized"),
        )
        second_external = store.record_external_pack(
            canonical_pack=dict(canonical_pack),
            classification="portable_text_skill",
            status="normalized",
            risk_report={"level": "low", "score": 0.1, "flags": []},
            review_envelope={"review_required": True, "why_risk": []},
            quarantine_path=os.path.join(self.tmpdir.name, "quarantine"),
            normalized_path=os.path.join(self.tmpdir.name, "normalized"),
        )
        self.assertEqual(first_external, second_external)

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.pack_store = store
        removed_first = runtime.delete_external_pack("pack.external.demo")
        removed_second = runtime.delete_external_pack("pack.external.demo")
        self.assertTrue(removed_first[0])
        self.assertEqual(True, removed_first[1].get("ok"))
        self.assertTrue(removed_second[0])
        self.assertEqual(True, removed_second[1].get("ok"))
        self.assertTrue(bool(removed_second[1].get("already_removed")))
        self.assertIn("already completed", str(removed_second[1].get("message") or "").lower())

    def test_external_pack_record_reuse_is_noop_for_identical_safe_content(self) -> None:
        store = PackStore(self.db_path)
        canonical_pack = {
            "id": "pack.example",
            "name": "Example Pack",
            "version": "1.0.0",
            "type": "skill",
            "pack_identity": {"canonical_id": "pack.example", "content_hash": "hash"},
            "source_history": [{"source_fingerprint": "source-a", "fetched_at": 1}],
            "versions": [{"canonical_id": "pack.example", "content_hash": "hash", "seen_at": 1}],
            "trust_anchor": {"first_seen_at": "1", "first_seen_source": "source-a", "local_review_status": "unreviewed", "user_approved_hashes": []},
        }
        risk_report = {"level": "low", "score": 0.1, "flags": []}
        review_envelope = {"review_required": True, "why_risk": []}

        first = store.record_external_pack(
            canonical_pack=dict(canonical_pack),
            classification="portable_text_skill",
            status="normalized",
            risk_report=dict(risk_report),
            review_envelope=dict(review_envelope),
            quarantine_path=str(self.tmpdir.name),
            normalized_path=str(self.tmpdir.name),
        )
        second = store.record_external_pack(
            canonical_pack=dict(canonical_pack),
            classification="portable_text_skill",
            status="normalized",
            risk_report=dict(risk_report),
            review_envelope=dict(review_envelope),
            quarantine_path=str(self.tmpdir.name),
            normalized_path=str(self.tmpdir.name),
        )

        self.assertEqual(first, second)

    def test_runtime_transition_rules_and_startup_guard_are_explicit(self) -> None:
        self.assertTrue(startup_phase_transition_allowed("starting", "listening"))
        self.assertTrue(startup_phase_transition_allowed("warming", "ready"))
        self.assertTrue(runtime_phase_transition_allowed(RuntimeLifecyclePhase.BOOT, RuntimeLifecyclePhase.WARMUP))
        self.assertTrue(runtime_phase_transition_allowed(RuntimeLifecyclePhase.DEGRADED, RuntimeLifecyclePhase.READY))
        self.assertFalse(startup_phase_transition_allowed("ready", "starting"))
        self.assertFalse(runtime_phase_transition_allowed(RuntimeLifecyclePhase.READY, RuntimeLifecyclePhase.BOOT))

        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.startup_phase = "ready"
        runtime._set_startup_phase("ready")
        self.assertEqual("ready", runtime.startup_phase)
        runtime._set_startup_phase("starting")
        self.assertEqual("ready", runtime.startup_phase)

    def test_fixit_confirmation_reuse_is_consumed_and_truthful(self) -> None:
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime.set_listening("127.0.0.1", 8765)
        now_epoch = int(time.time())
        plan = [
            {
                "id": "01_provider.test",
                "kind": "safe_action",
                "action": "provider.test",
                "reason": "Check provider health.",
                "params": {"provider": "openrouter"},
                "safe_to_execute": True,
            }
        ]
        confirm_token = confirm_token_for_plan_rows(plan)
        runtime._llm_fixit_store.save(  # type: ignore[attr-defined]
            {
                "active": True,
                "issue_hash": "hash",
                "issue_code": "openrouter_down",
                "step": "awaiting_confirm",
                "question": "Apply this fix-it plan now?",
                "choices": [],
                "pending_plan": plan,
                "pending_confirm_token": confirm_token,
                "pending_created_ts": now_epoch - 5,
                "pending_expires_ts": now_epoch + 300,
                "pending_issue_code": "openrouter_down",
                "last_prompt_ts": now_epoch - 5,
                "last_confirm_token": None,
                "last_confirmed_ts": None,
                "last_confirmed_issue_code": None,
            }
        )

        with patch.object(
            runtime,
            "_execute_llm_fixit_plan",
            return_value=(True, {"ok": True, "executed_steps": [{"id": "01_provider.test"}], "blocked_steps": [], "failed_steps": []}),
        ):
            first = _HandlerForTest(runtime, "/llm/fixit", {"confirm": True})
            first.do_POST()

        self.assertEqual(200, first.status_code)
        first_payload = json.loads(first.body.decode("utf-8"))
        self.assertTrue(first_payload["ok"])
        self.assertTrue(first_payload["did_work"])
        self.assertFalse(bool(runtime._llm_fixit_store.state.get("active")))  # type: ignore[attr-defined]
        self.assertEqual(confirm_token, runtime._llm_fixit_store.state.get("last_confirm_token"))  # type: ignore[attr-defined]

        second = _HandlerForTest(runtime, "/llm/fixit", {"confirm": True})
        second.do_POST()
        self.assertEqual(200, second.status_code)
        second_payload = json.loads(second.body.decode("utf-8"))
        self.assertEqual("already_consumed", second_payload.get("status"))
        self.assertIn("already used", str(second_payload.get("message") or "").lower())

    def test_invalid_transition_helpers_fail_cleanly_for_missing_and_stale_states(self) -> None:
        store = PackStore(self.db_path)
        self.assertIsNone(store.set_enabled("pack.missing", True))
        self.assertIsNone(store.set_approval_hash("pack.missing", "approval"))
        self.assertIsNone(store.remove_external_pack("pack.missing"))

        self.assertEqual(
            "missing_pending_plan",
            confirmation_transition_state(
                {},
                provided_token="",
                recomputed_token="",
                now_epoch=int(time.time()),
            ).get("state"),
        )
        stale = confirmation_transition_state(
            {
                "step": "awaiting_confirm",
                "pending_plan": [{"id": "01_provider.test", "action": "provider.test", "kind": "safe_action", "reason": "x", "params": {}, "safe_to_execute": True}],
                "pending_confirm_token": "token-a",
                "pending_expires_ts": int(time.time()) - 1,
            },
            provided_token="token-a",
            recomputed_token="token-a",
            now_epoch=int(time.time()),
        )
        self.assertEqual("expired", stale.get("state"))

    def test_confirmation_state_helper_reports_reuse_and_mismatch_cleanly(self) -> None:
        plan = [
            {
                "id": "01_provider.test",
                "kind": "safe_action",
                "action": "provider.test",
                "reason": "Check provider health.",
                "params": {"provider": "openrouter"},
                "safe_to_execute": True,
            }
        ]
        token = confirm_token_for_plan_rows(plan)
        allowed = confirmation_transition_state(
            {
                "step": "awaiting_confirm",
                "pending_plan": plan,
                "pending_confirm_token": token,
                "pending_expires_ts": 400,
            },
            provided_token="",
            recomputed_token=token,
            now_epoch=200,
        )
        self.assertTrue(bool(allowed.get("allowed")))
        mismatch = confirmation_transition_state(
            {
                "step": "awaiting_confirm",
                "pending_plan": plan,
                "pending_confirm_token": token,
                "pending_expires_ts": 400,
            },
            provided_token="wrong",
            recomputed_token=token,
            now_epoch=200,
        )
        self.assertEqual("token_mismatch", mismatch.get("state"))
        consumed = confirmation_transition_state(
            {
                "step": "idle",
                "last_confirm_token": token,
            },
            provided_token="",
            recomputed_token=token,
            now_epoch=200,
        )
        self.assertEqual("already_consumed", consumed.get("state"))


if __name__ == "__main__":
    unittest.main()
