from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import sqlite3
import tempfile
import unittest
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.packs.capability_recommendation import recommend_packs_for_capability
from agent.packs.state_truth import build_pack_state_snapshot
from agent.runtime_contract import normalize_user_facing_status
from agent.runtime_truth_service import RuntimeTruthService
from agent.ux.llm_fixit_wizard import confirm_token_for_plan_rows


def _config(registry_path: str, db_path: str, skills_path: str) -> Config:
    return Config(
        telegram_bot_token="token",
        openai_api_key=None,
        openai_model="gpt-4o-mini",
        openai_model_worker=None,
        agent_timezone="UTC",
        db_path=db_path,
        log_path=os.path.join(os.path.dirname(db_path), "agent.log"),
        skills_path=skills_path,
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


class _PackSourceDiscovery:
    def __init__(self, listings: list[dict[str, object]]) -> None:
        self._listings = list(listings)

    def list_sources(self) -> list[dict[str, object]]:
        return [
            {
                "id": "local",
                "name": "Local Catalog",
                "kind": "local_catalog",
                "enabled": True,
                "allowed_by_policy": True,
            }
        ]

    def search(self, source_id: str, term: str) -> dict[str, object]:
        _ = source_id
        needle = str(term or "").strip().lower()
        results = [
            row
            for row in self._listings
            if needle in str(row.get("name") or "").lower()
            or needle in str(row.get("summary") or "").lower()
            or needle in " ".join(str(item) for item in (row.get("tags") if isinstance(row.get("tags"), list) else [])).lower()
        ]
        return {"search": {"results": results}, "from_cache": False, "stale": False}


class _BrokenDiscovery:
    def list_sources(self) -> list[dict[str, object]]:
        raise sqlite3.OperationalError("database is locked")

    def search(self, source_id: str, term: str) -> dict[str, object]:  # pragma: no cover - never reached
        _ = source_id
        _ = term
        return {"search": {"results": []}, "from_cache": False, "stale": False}


class _FakeRuntimeTruth:
    def __init__(self, provider: str = "ollama", model: str = "qwen2.5:3b-instruct") -> None:
        self._provider = provider
        self._model = model

    def current_chat_target_status(self) -> dict[str, object]:
        return {
            "effective_provider": self._provider,
            "effective_model": f"{self._provider}:{self._model}",
            "health_reason": None,
            "qualification_reason": None,
            "degraded_reason": None,
            "active_provider_health_status": "ok",
            "active_model_health_status": "ok",
        }

    def providers_status(self) -> dict[str, object]:
        return {"providers": [{"provider": self._provider, "local": True}]}


class _LoopRuntime:
    def __init__(self, *, phase_context: dict[str, object], started_at: datetime) -> None:
        self._phase_context = dict(phase_context)
        self.started_at = started_at
        self.version = "0.6.0"
        self.git_commit = "abc123"
        self.pid = 1234
        self.started_at_iso = started_at.isoformat()
        self._startup_last_error = None

    def _runtime_observability_context(self) -> dict[str, object]:
        return dict(self._phase_context)

    def safe_mode_target_status(self) -> dict[str, object]:
        return {"enabled": False}

    def _model_watch_hf_status_snapshot(self) -> dict[str, object]:
        return {}

    def _ready_recent_telegram_messages(self, limit: int = 5) -> list[dict[str, object]]:
        _ = limit
        return []

    def _runtime_surface_message(
        self,
        *,
        startup_phase: str,
        normalized_status: dict[str, object],
        onboarding_summary: str | None = None,
        onboarding_next_action: str | None = None,
        safe_mode_target: dict[str, object] | None = None,
        failure_recovery: dict[str, object] | None = None,
    ) -> str:
        _ = safe_mode_target
        recovery_message = str((failure_recovery or {}).get("message") or "").strip()
        runtime_mode = str(normalized_status.get("runtime_mode") or "").strip().lower()
        if recovery_message and runtime_mode != "ready":
            return recovery_message
        if startup_phase in {"starting", "listening", "warming"}:
            return "Starting up... retrying. Try /ready again in a moment."
        if str(onboarding_summary or "").strip():
            next_action = str(onboarding_next_action or "").strip()
            return (
                f"{str(onboarding_summary).strip()} Next: {next_action}"
                if next_action
                else str(onboarding_summary).strip()
            )
        return str(normalized_status.get("summary") or "Ready.")

    def runtime_truth_service(self) -> _FakeRuntimeTruth:
        return _FakeRuntimeTruth()

    def get_defaults(self) -> dict[str, object]:
        return {
            "routing_mode": "auto",
            "default_provider": "ollama",
            "default_model": "qwen2.5:3b-instruct",
            "resolved_default_model": "ollama:qwen2.5:3b-instruct",
            "allow_remote_fallback": True,
        }

    def llm_control_mode_status(self) -> dict[str, object]:
        return {"mode": "safe", "mode_label": "SAFE MODE"}

    def _ensure_memory_db(self) -> None:
        raise RuntimeError("memory db unavailable")


class TestRegressionSoak(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self.skills_path = os.path.join(self.tmpdir.name, "skills")
        os.makedirs(self.skills_path, exist_ok=True)
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")
        os.environ["AGENT_LLM_FIXIT_WIZARD_STATE_PATH"] = os.path.join(self.tmpdir.name, "fixit.json")
        self.runtime = AgentRuntime(_config(self.registry_path, self.db_path, self.skills_path), defer_bootstrap_warmup=True)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _make_skill_source(self, name: str) -> Path:
        source_dir = Path(self.tmpdir.name) / name
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "SKILL.md").write_text(
            "---\n"
            f"id: {name}\n"
            f"name: {name.replace('_', ' ').title()}\n"
            "version: 1.0.0\n"
            "description: safe text skill\n"
            "---\n"
            f"# {name.replace('_', ' ').title()}\n\n"
            "Use the reference notes only.\n",
            encoding="utf-8",
        )
        return source_dir

    @staticmethod
    def _packs_snapshot_is_consistent(payload: dict[str, object]) -> None:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        packs = payload.get("packs") if isinstance(payload.get("packs"), list) else []
        available = payload.get("available_packs") if isinstance(payload.get("available_packs"), list) else []
        rows = [row for row in packs + available if isinstance(row, dict)]
        assert int(summary.get("total") or 0) == len(rows)
        assert int(summary.get("installed") or 0) == len([row for row in packs if isinstance(row, dict)])
        assert int(summary.get("blocked") or 0) == sum(
            1 for row in rows if str(row.get("state") or "").strip().lower() in {"blocked", "installed_blocked"}
        )
        assert int(summary.get("available") or 0) == sum(
            1 for row in available if str(row.get("state") or "").strip().lower() == "available"
        )
        assert int(summary.get("usable") or 0) == sum(1 for row in packs if bool(row.get("usable", False)))

    @staticmethod
    def _assert_runtime_views_consistent(runtime: AgentRuntime) -> None:
        ready_payload = runtime.ready_status()
        ui_payload = runtime.ui_state()
        packs_payload = runtime.packs_state()
        for _ in range(2):
            again_ready = runtime.ready_status()
            again_ui = runtime.ui_state()
            again_packs = runtime.packs_state()
            assert again_ready["state_label"] == ready_payload["state_label"]
            assert again_ready["reason"] == ready_payload["reason"]
            assert again_ui["runtime"]["summary"] == ui_payload["runtime"]["summary"]
            assert again_packs["summary"] == packs_payload["summary"]

        assert str(ready_payload["message"] or "").strip()
        assert str(ui_payload["runtime"]["summary"] or "").strip()
        assert str(ready_payload["next_step"] or "").strip() or ready_payload["ready"]
        assert bool(ready_payload["ok"])
        assert bool(packs_payload["ok"])

    def test_repeated_pack_install_remove_cycles_converge_cleanly(self) -> None:
        source_dir = self._make_skill_source("soak_install_remove")
        canonical_ids: set[str] = set()
        for cycle in range(3):
            ok, install_body = self.runtime.packs_install({"path": str(source_dir)})
            self.assertTrue(ok, install_body)
            pack = install_body["pack"]
            canonical_id = str(pack["canonical_id"])
            canonical_ids.add(canonical_id)
            self.assertEqual(1, int(self.runtime.packs_state()["summary"]["installed"]))
            self.assertEqual(1, int(self.runtime.packs_state()["summary"]["total"]))
            self._packs_snapshot_is_consistent(self.runtime.packs_state())
            self.assertIsNotNone(self.runtime.pack_store.get_external_pack(canonical_id))
            self._assert_runtime_views_consistent(self.runtime)

            removed = self.runtime.pack_store.remove_external_pack(canonical_id, removed_by="test", reason=f"cycle-{cycle}")
            self.assertIsNotNone(removed)
            self.assertTrue(bool(removed.get("removed", False)))
            self.assertIsNone(self.runtime.pack_store.get_external_pack(canonical_id))
            self.assertIsNotNone(self.runtime.pack_store.get_external_pack_removal(canonical_id))
            self.assertEqual(1, len(self.runtime.pack_store.list_external_pack_removals()))
            self.assertEqual(0, int(self.runtime.packs_state()["summary"]["installed"]))
            self.assertEqual(0, int(self.runtime.packs_state()["summary"]["total"]))
            self._packs_snapshot_is_consistent(self.runtime.packs_state())
            self._assert_runtime_views_consistent(self.runtime)

        self.assertEqual(1, len(canonical_ids))
        removals = self.runtime.pack_store.list_external_pack_removals()
        self.assertEqual(1, len(removals))

    def test_repeated_recommendation_cycles_before_and_after_install_remove_stay_deterministic(self) -> None:
        available_listings = [
            {
                "remote_id": "voice-local-fast",
                "name": "Local Voice",
                "summary": "Lightweight voice output for this machine.",
                "artifact_type_hint": "portable_text_skill",
                "installable_by_current_policy": True,
                "preview_available": True,
                "tags": ["voice_output", "lightweight"],
            },
            {
                "remote_id": "voice-studio",
                "name": "Studio Voice",
                "summary": "Full voice output with broader features.",
                "artifact_type_hint": "portable_text_skill",
                "installable_by_current_policy": True,
                "preview_available": True,
                "tags": ["voice_output", "broader"],
            },
        ]
        discovery = _PackSourceDiscovery(available_listings)
        store = SimpleNamespace(list_external_packs=lambda: [])
        prompt = "Talk to me out loud."

        first = recommend_packs_for_capability(prompt, pack_store=store, pack_registry_discovery=discovery)
        second = recommend_packs_for_capability(prompt, pack_store=store, pack_registry_discovery=discovery)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first["recommended_pack"]["name"], second["recommended_pack"]["name"])
        self.assertEqual(first["comparison_mode"], second["comparison_mode"])
        self.assertEqual("single_recommendation", first["comparison_mode"])
        self.assertEqual("Local Voice", first["recommended_pack"]["name"])
        self.assertIsNone(first.get("alternate_pack"))

        installed_rows = [
            {
                "pack_id": "voice-local-fast",
                "name": "Local Voice",
                "status": "normalized",
                "enabled": True,
                "normalized_path": str(self.tmpdir.name),
                "canonical_pack": {
                    "pack_identity": {"canonical_id": "voice-local-fast", "content_hash": "abc", "source_key": "local"},
                    "display_name": "Local Voice",
                    "source": {"name": "Local"},
                    "capabilities": {"declared": ["voice_output"], "summary": "Lightweight voice output for this machine."},
                },
                "review_envelope": {"pack_name": "Local Voice"},
            }
        ]
        installed_store = SimpleNamespace(list_external_packs=lambda: list(installed_rows))
        installed_result = recommend_packs_for_capability(
            prompt,
            pack_store=installed_store,
            pack_registry_discovery=discovery,
        )
        self.assertIsNotNone(installed_result)
        self.assertEqual("installed_healthy", installed_result["status"])
        self.assertIsNotNone(installed_result["installed_pack"])
        self.assertEqual("Local Voice", installed_result["installed_pack"]["name"])

        removed_result = recommend_packs_for_capability(
            prompt,
            pack_store=SimpleNamespace(list_external_packs=lambda: []),
            pack_registry_discovery=discovery,
        )
        self.assertIsNotNone(removed_result)
        self.assertEqual(first["recommended_pack"]["name"], removed_result["recommended_pack"]["name"])
        self.assertEqual(first["comparison_mode"], removed_result["comparison_mode"])

        self.assertTrue(str(first["next_step"] or "").strip())
        self.assertIn("preview", str(first["next_step"] or "").lower())

    def test_repeated_recommendation_fallback_when_discovery_is_unavailable_is_stable(self) -> None:
        class _BrokenDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                raise sqlite3.OperationalError("database is locked")

        store = SimpleNamespace(list_external_packs=lambda: [])
        prompt = "Use the avatar."
        for _ in range(3):
            result = recommend_packs_for_capability(prompt, pack_store=store, pack_registry_discovery=_BrokenDiscovery())
            self.assertIsNotNone(result)
            self.assertEqual("text_only", result["fallback"])
            self.assertEqual("missing", result["status"])
            self.assertTrue(result["source_errors"])

    def test_repeated_runtime_boot_warmup_ready_degraded_cycles_stay_explainable(self) -> None:
        for _ in range(3):
            runtime = _LoopRuntime(
                phase_context={
                    "ready": False,
                    "phase": "warmup",
                    "startup_phase": "starting",
                    "warmup_remaining": ["native_packs"],
                    "normalized_status": normalize_user_facing_status(
                        ready=False,
                        bootstrap_required=True,
                        failure_code="no_chat_model",
                        phase="starting",
                        provider="ollama",
                        model="qwen2.5:3b-instruct",
                        local_providers={"ollama"},
                    ),
                    "telegram": {"state": "running", "effective_state": "running", "enabled": True},
                    "telegram_state": "running",
                    "telegram_enabled": True,
                    "llm_status": {
                        "active_provider_health": {"status": "ok"},
                        "active_model_health": {"status": "ok"},
                    },
                },
                started_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            )
            service = RuntimeTruthService(runtime)

        for phase_name, startup_phase, ready_flag, failure_code, expected_kind in (
                ("boot", "starting", False, "no_chat_model", "runtime_initializing"),
                ("warm", "warming", False, "no_chat_model", "runtime_initializing"),
                ("ready", "ready", True, None, None),
                ("degraded", "degraded", False, "llm_unavailable", "runtime_degraded"),
                ("recovered", "ready", True, None, None),
            ):
                runtime._phase_context = {
                    **runtime._phase_context,
                    "phase": phase_name,
                    "startup_phase": startup_phase,
                    "ready": ready_flag,
                    "warmup_remaining": ["native_packs"] if startup_phase in {"starting", "warming"} else [],
                    "normalized_status": normalize_user_facing_status(
                        ready=ready_flag,
                        bootstrap_required=not ready_flag and startup_phase in {"starting", "warming"},
                        failure_code=failure_code,
                        phase=startup_phase,
                        provider="ollama",
                        model="qwen2.5:3b-instruct",
                        local_providers={"ollama"},
                    ),
                }
                for _ in range(3):
                    ready_payload = service.ready_status()
                    ui_payload = service.ui_state(ready_payload=ready_payload)
                    self.assertTrue(bool(ready_payload["ok"]))
                    if expected_kind is None:
                        self.assertTrue(bool(ready_payload["ready"]))
                        self.assertEqual("Ready", ready_payload["state_label"])
                        self.assertIsNone(ready_payload["failure_recovery"])
                    else:
                        self.assertFalse(bool(ready_payload["ready"]))
                        self.assertEqual(expected_kind, ready_payload["failure_recovery"]["kind"])
                        self.assertTrue(str(ready_payload["failure_recovery"]["next_step"] or "").strip())
                    self.assertTrue(str(ui_payload["runtime"]["summary"] or "").strip())
                    self.assertTrue(str(ui_payload["runtime"]["next_action"] or "").strip() or ui_payload["runtime"]["status"] == "unknown")

    def test_repeated_fixit_confirmation_cycles_execute_once_and_replay_safely(self) -> None:
        now_epoch = int(time.time())
        plan = [
            {
                "id": "defaults.set",
                "kind": "safe_action",
                "action": "defaults.set",
                "reason": "update defaults",
                "params": {"default_provider": "ollama"},
                "safe_to_execute": True,
            }
        ]
        token = confirm_token_for_plan_rows(plan)
        execute_calls: list[int] = []

        def _fake_execute(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            execute_calls.append(1)
            return (
                True,
                {
                    "ok": True,
                    "executed_steps": [{"id": "defaults.set"}],
                    "blocked_steps": [],
                    "failed_steps": [],
                    "provider_tests": {},
                },
            )

        for cycle in range(3):
            self.runtime._llm_fixit_store.save(  # type: ignore[attr-defined]
                {
                    "active": True,
                    "issue_hash": f"cycle-{cycle}",
                    "issue_code": "needs_clarification",
                    "step": "awaiting_confirm",
                    "question": "Confirm the pending step?",
                    "choices": [],
                    "pending_plan": plan,
                    "pending_confirm_token": token,
                    "pending_created_ts": now_epoch - 5,
                    "pending_expires_ts": now_epoch + 300,
                    "pending_issue_code": "needs_clarification",
                    "confirm_token": token,
                    "last_confirm_token": None,
                    "last_confirmed_ts": None,
                    "last_confirmed_issue_code": None,
                }
            )
            with patch.object(
                self.runtime,
                "_llm_fixit_status_payload",
                return_value={
                    "ok": True,
                    "runtime_mode": "READY",
                    "ready": True,
                    "resolved_default_model": "ollama:qwen2.5:3b-instruct",
                    "providers": [],
                    "models": [],
                    "safe_mode": {"safe_mode": False},
                },
            ), patch("agent.api_server.evaluate_wizard_decision", return_value=SimpleNamespace(status="ok", issue_code="ok", question=None)):
                with patch.object(
                    self.runtime,
                    "_execute_llm_fixit_plan",
                    side_effect=_fake_execute,
                ):
                    first_ok, first_body = self.runtime.llm_fixit({"confirm": True})
                    second_ok, second_body = self.runtime.llm_fixit({"confirm": True})
            self.assertTrue(first_ok)
            self.assertTrue(second_ok)
            self.assertEqual("already_consumed", str(second_body.get("status") or ""))
            self.assertFalse(bool(self.runtime._llm_fixit_store.state.get("active")))  # type: ignore[attr-defined]
            self.assertEqual([], self.runtime._llm_fixit_store.state.get("pending_plan"))  # type: ignore[attr-defined]
            self.assertTrue(str(self.runtime._llm_fixit_store.state.get("last_confirm_token") or "").strip())  # type: ignore[attr-defined]
        self.assertEqual(3, len(execute_calls))


if __name__ == "__main__":
    unittest.main()
