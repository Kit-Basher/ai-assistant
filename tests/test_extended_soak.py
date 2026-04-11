from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.config import Config
from agent.packs.capability_recommendation import recommend_packs_for_capability
from agent.packs.state_truth import build_pack_state_snapshot
from agent.packs.store import PackStore
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

    def list_packs(self, source_id: str) -> dict[str, object]:
        _ = source_id
        return {"packs": list(self._listings), "from_cache": False, "stale": False}


class _BrokenDiscovery:
    def list_sources(self) -> list[dict[str, object]]:
        raise sqlite3.OperationalError("database is locked")

    def search(self, source_id: str, term: str) -> dict[str, object]:  # pragma: no cover - defensive
        _ = source_id
        _ = term
        return {"search": {"results": []}, "from_cache": False, "stale": False}


class _LoopRuntime:
    def __init__(self, *, phase_context: dict[str, object], started_at: datetime) -> None:
        self._phase_context = dict(phase_context)
        self.started_at = started_at
        self.version = "0.8.0"
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

    def runtime_truth_service(self) -> SimpleNamespace:
        return SimpleNamespace(
            current_chat_target_status=lambda: {
                "effective_provider": "ollama",
                "effective_model": "ollama:qwen2.5:3b-instruct",
                "health_reason": None,
                "qualification_reason": None,
                "degraded_reason": None,
                "active_provider_health_status": "ok",
                "active_model_health_status": "ok",
            }
        )

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


class TestExtendedSoak(unittest.TestCase):
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
        (source_dir / "references").mkdir(parents=True, exist_ok=True)
        (source_dir / "references" / "guide.md").write_text("# Guide\n\nSafe reference text.\n", encoding="utf-8")
        return source_dir

    @staticmethod
    def _pack_snapshot(payload: dict[str, object]) -> str:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        return (
            f"total={int(summary.get('total') or 0)} "
            f"installed={int(summary.get('installed') or 0)} "
            f"enabled={int(summary.get('enabled') or 0)} "
            f"healthy={int(summary.get('healthy') or 0)} "
            f"blocked={int(summary.get('blocked') or 0)} "
            f"available={int(summary.get('available') or 0)}"
        )

    @staticmethod
    def _ready_snapshot(payload: dict[str, object]) -> str:
        return (
            f"label={payload.get('state_label')} "
            f"reason={payload.get('reason')} "
            f"next={payload.get('next_step')} "
            f"message={payload.get('message')}"
        )

    @staticmethod
    def _ui_snapshot(payload: dict[str, object]) -> str:
        runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
        return (
            f"summary={runtime.get('summary')} "
            f"next_action={runtime.get('next_action')} "
            f"status={runtime.get('status')}"
        )

    def _assert_ready_is_stable(self, *, label: str, iterations: int = 4) -> None:
        ready = self.runtime.ready_status()
        ui = self.runtime.ui_state()
        packs = self.runtime.packs_state()
        self.assertTrue(str(ready.get("message") or "").strip(), msg=f"{label}: empty ready message")
        self.assertTrue(str(self._ui_snapshot(ui)).strip(), msg=f"{label}: empty ui snapshot")
        self.assertTrue(str(self._pack_snapshot(packs)).strip(), msg=f"{label}: empty packs snapshot")
        for index in range(iterations):
            again_ready = self.runtime.ready_status()
            again_ui = self.runtime.ui_state()
            again_packs = self.runtime.packs_state()
            self.assertEqual(
                self._ready_snapshot(ready),
                self._ready_snapshot(again_ready),
                msg=f"{label}: ready drift at poll {index}",
            )
            self.assertEqual(
                self._ui_snapshot(ui),
                self._ui_snapshot(again_ui),
                msg=f"{label}: ui drift at poll {index}",
            )
            self.assertEqual(
                self._pack_snapshot(packs),
                self._pack_snapshot(again_packs),
                msg=f"{label}: packs drift at poll {index}",
            )

    def _pack_snapshot(self, payload: dict[str, object]) -> str:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        return (
            f"installed={int(summary.get('installed') or 0)} "
            f"total={int(summary.get('total') or 0)} "
            f"enabled={int(summary.get('enabled') or 0)} "
            f"usable={int(summary.get('usable') or 0)} "
            f"blocked={int(summary.get('blocked') or 0)} "
            f"available={int(summary.get('available') or 0)}"
        )

    def test_pack_lifecycle_extended_soak_keeps_tombstones_and_artifacts_clean(self) -> None:
        source_dir = self._make_skill_source("extended_soak_pack")
        seen_canonical_ids: set[str] = set()

        for cycle in range(8):
            install_ok, install_body = self.runtime.packs_install({"path": str(source_dir)})
            self.assertTrue(install_ok, msg=f"install failed at cycle {cycle}: {install_body}")
            pack = install_body["pack"]
            canonical_id = str(pack["canonical_id"] or "").strip()
            self.assertTrue(canonical_id, msg=f"empty canonical id at cycle {cycle}: {install_body}")
            seen_canonical_ids.add(canonical_id)

            install_state = self.runtime.packs_state()
            self.assertEqual(1, int(install_state["summary"]["installed"]), msg=f"installed drift at cycle {cycle}: {install_state}")
            self.assertEqual(1, int(install_state["summary"]["total"]), msg=f"total drift at cycle {cycle}: {install_state}")
            self.assertEqual(1, len(self.runtime.pack_store.list_external_packs()), msg=f"duplicate installed rows at cycle {cycle}")
            self.assertIsNotNone(self.runtime.pack_store.get_external_pack(canonical_id), msg=f"missing installed pack at cycle {cycle}")
            self._assert_ready_is_stable(label=f"install-cycle-{cycle}")

            removed_ok, removed_body = self.runtime.delete_external_pack(canonical_id)
            self.assertTrue(removed_ok, msg=f"remove transport failed at cycle {cycle}: {removed_body}")
            self.assertTrue(bool(removed_body.get("ok", False)), msg=f"remove body not ok at cycle {cycle}: {removed_body}")
            self.assertFalse(bool(removed_body.get("already_removed", False)), msg=f"remove unexpectedly reported already_removed at cycle {cycle}: {removed_body}")
            self.assertIn("I removed external pack", str(removed_body.get("message") or ""), msg=f"remove message drift at cycle {cycle}: {removed_body}")

            removal = removed_body.get("removal") if isinstance(removed_body.get("removal"), dict) else {}
            normalized_path = Path(str(removal.get("normalized_path") or "").strip())
            quarantine_path = Path(str(removal.get("quarantine_path") or "").strip())
            if str(normalized_path) != ".":
                self.assertFalse(normalized_path.exists(), msg=f"normalized path leaked at cycle {cycle}: {normalized_path}")
            if str(quarantine_path) != ".":
                self.assertFalse(quarantine_path.exists(), msg=f"quarantine path leaked at cycle {cycle}: {quarantine_path}")

            removed_again_ok, removed_again_body = self.runtime.delete_external_pack(canonical_id)
            self.assertTrue(removed_again_ok, msg=f"repeat remove transport failed at cycle {cycle}: {removed_again_body}")
            self.assertTrue(bool(removed_again_body.get("already_removed", False)), msg=f"repeat remove did not report already_removed at cycle {cycle}: {removed_again_body}")
            self.assertIn("already completed", str(removed_again_body.get("message") or "").lower(), msg=f"repeat remove message drift at cycle {cycle}: {removed_again_body}")

            after_remove_state = self.runtime.packs_state()
            self.assertEqual(0, int(after_remove_state["summary"]["installed"]), msg=f"installed not cleared at cycle {cycle}: {after_remove_state}")
            self.assertEqual(0, int(after_remove_state["summary"]["total"]), msg=f"total not cleared at cycle {cycle}: {after_remove_state}")
            self.assertIsNone(self.runtime.pack_store.get_external_pack(canonical_id), msg=f"removed pack still visible at cycle {cycle}")
            self.assertIsNotNone(self.runtime.pack_store.get_external_pack_removal(canonical_id), msg=f"removal tombstone missing at cycle {cycle}")
            self.assertEqual(1, len(self.runtime.pack_store.list_external_pack_removals()), msg=f"tombstone duplicated at cycle {cycle}")
            self._assert_ready_is_stable(label=f"remove-cycle-{cycle}")

        self.assertEqual(1, len(seen_canonical_ids), msg=f"canonical id drifted across cycles: {seen_canonical_ids}")
        self.assertEqual(1, len(self.runtime.pack_store.list_external_pack_removals()))

    def test_runtime_pack_toggle_extended_soak_keeps_enablement_idempotent(self) -> None:
        pack_id = "soak.runtime.native"
        permissions = {"ifaces": [], "fs": {"read": [], "write": []}, "net": {"allow_domains": [], "deny": []}, "proc": {"spawn": []}}

        for cycle in range(10):
            enabled_row = self.runtime.pack_store.ensure_native_pack(
                pack_id=pack_id,
                version="1.0.0",
                permissions=permissions,
            )
            self.assertTrue(bool(enabled_row.get("enabled", False)), msg=f"native ensure did not enable at cycle {cycle}: {enabled_row}")
            self.assertEqual(pack_id, enabled_row["pack_id"])
            self.assertEqual(1, len(self.runtime.pack_store.list_runtime_packs()), msg=f"runtime pack duplicated at cycle {cycle}")

            repeat_install = self.runtime.pack_store.ensure_native_pack(
                pack_id=pack_id,
                version="1.0.0",
                permissions=permissions,
            )
            self.assertEqual(int(enabled_row.get("updated_at") or 0), int(repeat_install.get("updated_at") or 0), msg=f"noop install churned timestamp at cycle {cycle}")
            self.assertEqual(enabled_row.get("approved_permissions_hash"), repeat_install.get("approved_permissions_hash"), msg=f"noop install approval drift at cycle {cycle}")

            disabled_row = self.runtime.pack_store.set_enabled(pack_id, False)
            self.assertIsNotNone(disabled_row, msg=f"disable failed at cycle {cycle}")
            assert disabled_row is not None
            self.assertFalse(bool(disabled_row.get("enabled", True)), msg=f"disable did not stick at cycle {cycle}")

            disabled_again = self.runtime.pack_store.set_enabled(pack_id, False)
            self.assertIsNotNone(disabled_again, msg=f"repeat disable failed at cycle {cycle}")
            assert disabled_again is not None
            self.assertFalse(bool(disabled_again.get("enabled", True)), msg=f"repeat disable drift at cycle {cycle}")

            reenabled_row = self.runtime.pack_store.set_enabled(pack_id, True)
            self.assertIsNotNone(reenabled_row, msg=f"reenable failed at cycle {cycle}")
            assert reenabled_row is not None
            self.assertTrue(bool(reenabled_row.get("enabled", False)), msg=f"reenable did not stick at cycle {cycle}")

            reenabled_again = self.runtime.pack_store.set_enabled(pack_id, True)
            self.assertIsNotNone(reenabled_again, msg=f"repeat reenable failed at cycle {cycle}")
            assert reenabled_again is not None
            self.assertTrue(bool(reenabled_again.get("enabled", False)), msg=f"repeat reenable drift at cycle {cycle}")

            runtime_pack = self.runtime.pack_store.get_pack(pack_id)
            self.assertIsNotNone(runtime_pack, msg=f"runtime pack missing at cycle {cycle}")
            assert runtime_pack is not None
            self.assertTrue(bool(runtime_pack.get("enabled", False)), msg=f"runtime pack not enabled at cycle {cycle}")

    def test_pack_state_snapshot_extended_soak_stays_aligned_with_runtime_surface(self) -> None:
        source_dir = self._make_skill_source("snapshot_alignment")

        for cycle in range(10):
            install_ok, install_body = self.runtime.packs_install({"path": str(source_dir)})
            self.assertTrue(install_ok, msg=f"install failed at cycle {cycle}: {install_body}")
            canonical_id = str(install_body["pack"]["canonical_id"] or "").strip()
            snapshot = build_pack_state_snapshot(pack_store=self.runtime.pack_store, discovery=self.runtime._pack_registry_discovery())
            runtime_snapshot = self.runtime.packs_state()
            self.assertEqual(snapshot["summary"], runtime_snapshot["summary"], msg=f"summary drift at cycle {cycle}")
            self.assertEqual(snapshot["state_label"], runtime_snapshot["state_label"], msg=f"state label drift at cycle {cycle}")
            self.assertEqual(snapshot["recovery"], runtime_snapshot["recovery"], msg=f"recovery drift at cycle {cycle}")
            self.assertTrue(str(snapshot["packs"][0]["status_note"] or "").strip(), msg=f"empty installed status note at cycle {cycle}")

            removed = self.runtime.pack_store.remove_external_pack(canonical_id, removed_by="test", reason=f"snapshot-cycle-{cycle}")
            self.assertIsNotNone(removed, msg=f"remove failed at cycle {cycle}: {canonical_id}")
            snapshot_after_remove = build_pack_state_snapshot(pack_store=self.runtime.pack_store, discovery=self.runtime._pack_registry_discovery())
            runtime_after_remove = self.runtime.packs_state()
            self.assertEqual(snapshot_after_remove["summary"], runtime_after_remove["summary"], msg=f"remove summary drift at cycle {cycle}")
            self.assertEqual(snapshot_after_remove["state_label"], runtime_after_remove["state_label"], msg=f"remove state drift at cycle {cycle}")
            self.assertEqual(snapshot_after_remove["recovery"], runtime_after_remove["recovery"], msg=f"remove recovery drift at cycle {cycle}")
            self.assertEqual(0, int(runtime_after_remove["summary"]["installed"]), msg=f"installed rows leaked at cycle {cycle}: {runtime_after_remove}")

    def test_restart_churn_extended_soak_rebuilds_clean_state_each_cycle(self) -> None:
        for cycle in range(8):
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
            final_ready: dict[str, object] | None = None
            for phase_name, startup_phase, ready_flag, failure_code, expected_kind in (
                ("boot", "starting", False, "no_chat_model", "runtime_initializing"),
                ("warming", "warming", False, "no_chat_model", "runtime_initializing"),
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
                ready_payload = service.ready_status()
                ui_payload = service.ui_state(ready_payload=ready_payload)
                self.assertTrue(str(ready_payload["message"] or "").strip(), msg=f"cycle={cycle} phase={phase_name}: empty ready message")
                self.assertTrue(str(ui_payload["runtime"]["summary"] or "").strip(), msg=f"cycle={cycle} phase={phase_name}: empty ui summary")
                if expected_kind is None:
                    self.assertTrue(bool(ready_payload["ready"]), msg=f"cycle={cycle} phase={phase_name}: unexpectedly not ready")
                    self.assertEqual("Ready", ready_payload["state_label"], msg=f"cycle={cycle} phase={phase_name}: bad state label")
                    self.assertIsNone(ready_payload["failure_recovery"], msg=f"cycle={cycle} phase={phase_name}: stale recovery")
                else:
                    self.assertFalse(bool(ready_payload["ready"]), msg=f"cycle={cycle} phase={phase_name}: unexpectedly ready")
                    self.assertEqual(expected_kind, ready_payload["failure_recovery"]["kind"], msg=f"cycle={cycle} phase={phase_name}: recovery drift")
                    self.assertTrue(str(ready_payload["failure_recovery"]["next_step"] or "").strip(), msg=f"cycle={cycle} phase={phase_name}: missing next step")
                final_ready = ready_payload

            assert final_ready is not None
            self.assertTrue(bool(final_ready["ready"]), msg=f"cycle={cycle}: ready state did not recover")
            self.assertEqual("Ready", final_ready["state_label"], msg=f"cycle={cycle}: sticky startup label")
            self.assertNotIn("initializing", str(final_ready["message"] or "").lower(), msg=f"cycle={cycle}: sticky startup wording")

    def test_record_external_pack_noop_loop_does_not_churn_timestamp(self) -> None:
        store = PackStore(self.db_path)
        source_dir = self._make_skill_source("noop_external_pack")
        canonical_pack = {
            "id": "noop_external_pack",
            "name": "Noop External Pack",
            "display_name": "Noop External Pack",
            "version": "1.0.0",
            "type": "skill",
            "pack_identity": {
                "canonical_id": "noop_external_pack",
                "content_hash": "hash",
                "source_key": "local:noop",
            },
            "source": {"name": "Local", "kind": "local_catalog"},
            "source_history": [{"source_fingerprint": "source-a", "fetched_at": 1}],
            "versions": [{"canonical_id": "noop_external_pack", "content_hash": "hash", "seen_at": 1}],
            "trust_anchor": {
                "first_seen_at": "1",
                "first_seen_source": "source-a",
                "local_review_status": "unreviewed",
                "user_approved_hashes": [],
            },
            "capabilities": {"declared": ["voice_output"], "summary": "Safe text skill."},
        }
        risk_report = {"level": "low", "score": 0.1, "flags": []}
        review_envelope = {"review_required": True, "why_risk": []}

        first = store.record_external_pack(
            canonical_pack=dict(canonical_pack),
            classification="portable_text_skill",
            status="normalized",
            risk_report=dict(risk_report),
            review_envelope=dict(review_envelope),
            quarantine_path=str(source_dir),
            normalized_path=str(source_dir),
        )
        first_updated_at = int(first.get("updated_at") or 0)
        for index in range(12):
            again = store.record_external_pack(
                canonical_pack=dict(canonical_pack),
                classification="portable_text_skill",
                status="normalized",
                risk_report=dict(risk_report),
                review_envelope=dict(review_envelope),
                quarantine_path=str(source_dir),
                normalized_path=str(source_dir),
            )
            self.assertEqual(first_updated_at, int(again.get("updated_at") or 0), msg=f"noop timestamp churn at iteration {index}: {again}")
            self.assertEqual(first.get("canonical_json"), again.get("canonical_json"), msg=f"noop canonical drift at iteration {index}")
            self.assertEqual(first.get("risk_json"), again.get("risk_json"), msg=f"noop risk drift at iteration {index}")
            self.assertEqual(first.get("review_json"), again.get("review_json"), msg=f"noop review drift at iteration {index}")
        self.assertEqual(0, len(store.list_external_pack_removals()))

    def test_recommendation_extended_soak_tracks_install_remove_and_degraded_discovery(self) -> None:
        discovery = _PackSourceDiscovery(
            [
                {
                    "remote_id": "voice-local-fast",
                    "name": "Local Voice",
                    "summary": "Lightweight voice output for this machine.",
                    "artifact_type_hint": "portable_text_skill",
                    "installable_by_current_policy": True,
                    "preview_available": True,
                    "tags": ["voice_output", "lightweight"],
                }
            ]
        )
        installed_rows: list[dict[str, object]] = []
        store = SimpleNamespace(list_external_packs=lambda: list(installed_rows))
        prompt = "Talk to me out loud."
        missing_reference: dict[str, object] | None = None
        installed_reference: dict[str, object] | None = None
        installed_path = str(Path(__file__).resolve())

        for cycle in range(12):
            missing = recommend_packs_for_capability(prompt, pack_store=store, pack_registry_discovery=discovery)
            self.assertIsNotNone(missing, msg=f"missing recommendation vanished at cycle {cycle}")
            assert missing is not None
            self.assertEqual("missing", missing["status"], msg=f"missing status drift at cycle {cycle}: {missing}")
            self.assertEqual("install_preview", missing["fallback"], msg=f"missing fallback drift at cycle {cycle}: {missing}")
            self.assertEqual("Local Voice", missing["recommended_pack"]["name"], msg=f"missing recommendation drift at cycle {cycle}: {missing}")
            if missing_reference is None:
                missing_reference = missing
            else:
                self.assertEqual(missing_reference["comparison_mode"], missing["comparison_mode"], msg=f"missing comparison drift at cycle {cycle}")
                self.assertEqual(missing_reference["recommended_pack"]["name"], missing["recommended_pack"]["name"], msg=f"missing name drift at cycle {cycle}")
                self.assertEqual(missing_reference["next_step"], missing["next_step"], msg=f"missing next step drift at cycle {cycle}")

            installed_rows[:] = [
                {
                    "pack_id": "pack.voice.local_fast",
                    "name": "Local Voice",
                    "status": "normalized",
                    "enabled": True,
                    "normalized_path": installed_path,
                    "canonical_pack": {
                        "display_name": "Local Voice",
                        "pack_identity": {"canonical_id": "pack.voice.local_fast"},
                        "source": {"name": "Local", "kind": "local_catalog"},
                        "capabilities": {
                            "summary": "Local speech output for this machine.",
                            "declared": ["voice_output"],
                        },
                    },
                    "review_envelope": {"pack_name": "Local Voice"},
                }
            ]
            installed = recommend_packs_for_capability(prompt, pack_store=store, pack_registry_discovery=discovery)
            self.assertIsNotNone(installed, msg=f"installed recommendation vanished at cycle {cycle}")
            assert installed is not None
            self.assertEqual("installed_healthy", installed["status"], msg=f"installed status drift at cycle {cycle}: {installed}")
            self.assertIsNotNone(installed["installed_pack"], msg=f"installed pack missing at cycle {cycle}: {installed}")
            if installed_reference is None:
                installed_reference = installed
            else:
                self.assertEqual(installed_reference["comparison_mode"], installed["comparison_mode"], msg=f"installed comparison drift at cycle {cycle}")
                self.assertEqual(installed_reference["installed_pack"]["name"], installed["installed_pack"]["name"], msg=f"installed name drift at cycle {cycle}")
                self.assertEqual(installed_reference["next_step"], installed["next_step"], msg=f"installed next step drift at cycle {cycle}")

            installed_rows.clear()
            removed = recommend_packs_for_capability(prompt, pack_store=store, pack_registry_discovery=discovery)
            self.assertIsNotNone(removed, msg=f"removed recommendation vanished at cycle {cycle}")
            assert removed is not None
            self.assertEqual("missing", removed["status"], msg=f"removed status drift at cycle {cycle}: {removed}")
            self.assertEqual("Local Voice", removed["recommended_pack"]["name"], msg=f"removed name drift at cycle {cycle}: {removed}")
            self.assertEqual(missing_reference["comparison_mode"], removed["comparison_mode"], msg=f"removed comparison drift at cycle {cycle}")

        class _BrokenDiscovery:
            def list_sources(self) -> list[dict[str, object]]:
                raise sqlite3.OperationalError("database is locked")

        for cycle in range(6):
            degraded = recommend_packs_for_capability(prompt, pack_store=store, pack_registry_discovery=_BrokenDiscovery())
            self.assertIsNotNone(degraded, msg=f"degraded recommendation missing at cycle {cycle}")
            assert degraded is not None
            self.assertEqual("text_only", degraded["fallback"], msg=f"degraded fallback drift at cycle {cycle}: {degraded}")
            self.assertTrue(degraded["source_errors"], msg=f"degraded source errors missing at cycle {cycle}: {degraded}")

    def test_fixit_extended_soak_consumes_tokens_and_clears_pending_state(self) -> None:
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
        execute_calls = 0

        def _fake_execute(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            nonlocal execute_calls
            execute_calls += 1
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

        for cycle in range(8):
            now_epoch = int(time.time())
            token = confirm_token_for_plan_rows(plan)
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
                with patch.object(self.runtime, "_execute_llm_fixit_plan", side_effect=_fake_execute):
                    first_ok, first_body = self.runtime.llm_fixit({"confirm": True})
                    second_ok, second_body = self.runtime.llm_fixit({"confirm": True})
            self.assertTrue(first_ok, msg=f"confirm failed at cycle {cycle}: {first_body}")
            self.assertTrue(bool(first_body.get("did_work", False)), msg=f"confirm did not work at cycle {cycle}: {first_body}")
            self.assertTrue(second_ok, msg=f"replay failed at cycle {cycle}: {second_body}")
            self.assertEqual("already_consumed", str(second_body.get("status") or "").strip(), msg=f"replay drift at cycle {cycle}: {second_body}")
            self.assertFalse(bool(self.runtime._llm_fixit_store.state.get("active")))  # type: ignore[attr-defined]
            self.assertEqual([], self.runtime._llm_fixit_store.state.get("pending_plan"))  # type: ignore[attr-defined]
            self.assertTrue(str(self.runtime._llm_fixit_store.state.get("last_confirm_token") or "").strip())  # type: ignore[attr-defined]

        self.assertEqual(8, execute_calls, msg=f"confirm execute count drifted: {execute_calls}")

    def test_runtime_phase_extended_soak_stays_canonical(self) -> None:
        for cycle in range(10):
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
            phase_expectations = (
                ("boot", "starting", False, "no_chat_model", "runtime_initializing"),
                ("warming", "warming", False, "no_chat_model", "runtime_initializing"),
                ("ready", "ready", True, None, None),
                ("degraded", "degraded", False, "llm_unavailable", "runtime_degraded"),
                ("recovered", "ready", True, None, None),
            )
            for phase_name, startup_phase, ready_flag, failure_code, expected_kind in phase_expectations:
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
                baseline = service.ready_status()
                self.assertTrue(str(baseline.get("message") or "").strip(), msg=f"cycle={cycle} phase={phase_name}: empty ready message")
                for poll in range(5):
                    again = service.ready_status()
                    self.assertEqual(
                        baseline.get("state_label"),
                        again.get("state_label"),
                        msg=f"cycle={cycle} phase={phase_name} poll={poll}: state label drift",
                    )
                    self.assertEqual(
                        baseline.get("reason"),
                        again.get("reason"),
                        msg=f"cycle={cycle} phase={phase_name} poll={poll}: reason drift",
                    )
                    self.assertEqual(
                        baseline.get("next_step"),
                        again.get("next_step"),
                        msg=f"cycle={cycle} phase={phase_name} poll={poll}: next step drift",
                    )
                    self.assertEqual(
                        baseline.get("failure_recovery", {}).get("kind") if isinstance(baseline.get("failure_recovery"), dict) else None,
                        again.get("failure_recovery", {}).get("kind") if isinstance(again.get("failure_recovery"), dict) else None,
                        msg=f"cycle={cycle} phase={phase_name} poll={poll}: recovery drift",
                    )
                if expected_kind is None:
                    self.assertTrue(bool(baseline.get("ready", False)), msg=f"cycle={cycle} phase={phase_name}: not ready unexpectedly")
                    self.assertEqual("Ready", baseline.get("state_label"))
                else:
                    self.assertFalse(bool(baseline.get("ready", True)), msg=f"cycle={cycle} phase={phase_name}: ready unexpectedly")
                    self.assertEqual(expected_kind, baseline.get("failure_recovery", {}).get("kind"))


if __name__ == "__main__":
    unittest.main()
