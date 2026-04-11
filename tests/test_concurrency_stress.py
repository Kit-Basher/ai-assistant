from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent.api_server import APIServerHandler, AgentRuntime
from agent.config import Config
from agent.state_transitions import startup_phase_transition_allowed
from agent.ux.llm_fixit_wizard import confirm_token_for_plan_rows


def _config(registry_path: str, db_path: str, skills_path: str) -> Config:
    base = Config(
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
    return base


class _HandlerForTest(APIServerHandler):
    def __init__(self, runtime_obj: AgentRuntime, path: str, payload: dict[str, object] | None = None) -> None:
        self.runtime = runtime_obj
        self.path = path
        self.headers = {}
        self.status_code = 0
        self.content_type = ""
        self.body = b""
        self._payload = payload or {}

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
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


class TestConcurrencyStress(unittest.TestCase):
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
    def _assert_pack_state_snapshot_consistent(payload: dict[str, object]) -> None:
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        packs = payload.get("packs") if isinstance(payload.get("packs"), list) else []
        available_packs = payload.get("available_packs") if isinstance(payload.get("available_packs"), list) else []
        rows = [row for row in packs + available_packs if isinstance(row, dict)]
        installed_rows = [row for row in packs if isinstance(row, dict)]
        available_rows = [row for row in available_packs if isinstance(row, dict)]
        if summary:
            assert int(summary.get("total") or 0) == len(rows)
            assert int(summary.get("installed") or 0) == len(installed_rows)
            assert int(summary.get("available") or 0) == sum(
                1
                for row in available_rows
                if str(row.get("state") or "").strip().lower() == "available"
            )
            assert int(summary.get("blocked") or 0) == sum(
                1
                for row in rows
                if str(row.get("state") or "").strip().lower() in {"blocked", "installed_blocked"}
            )
            assert int(summary.get("usable") or 0) == sum(1 for row in rows if bool(row.get("usable", False)))
        for row in installed_rows:
            assert bool(row.get("installed", False))
            assert str(row.get("state") or "").strip().lower() != "available"
        for row in available_rows:
            assert not bool(row.get("installed", False))

    def test_pack_polling_while_install_is_blocked_stays_truthful(self) -> None:
        source_dir = self._make_skill_source("polling_install")
        entered = threading.Event()
        release = threading.Event()
        install_result: list[dict[str, object]] = []
        errors: list[BaseException] = []
        original_record = self.runtime.pack_store.record_external_pack

        def blocked_record(*args, **kwargs):  # type: ignore[no-untyped-def]
            entered.set()
            if not release.wait(3):
                raise RuntimeError("install release timed out")
            return original_record(*args, **kwargs)

        def install_worker() -> None:
            try:
                ok, body = self.runtime.packs_install({"source": str(source_dir)})
                install_result.append({"ok": ok, "body": body})
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        with mock.patch.object(self.runtime.pack_store, "record_external_pack", side_effect=blocked_record):
            thread = threading.Thread(target=install_worker, daemon=True)
            thread.start()
            self.assertTrue(entered.wait(3))

            for _ in range(20):
                ready = self.runtime.ready_status()
                packs = self.runtime.packs_state()
                self.assertTrue(bool(ready.get("ok", False)))
                self.assertTrue(bool(packs.get("ok", False)))
                self._assert_pack_state_snapshot_consistent(packs)

            release.set()
            thread.join(timeout=5)

        self.assertFalse(errors, errors)
        self.assertTrue(install_result)
        self.assertTrue(bool(install_result[0]["ok"]))
        final_packs = self.runtime.packs_state()
        self.assertEqual(1, int(final_packs["summary"]["installed"]))
        self.assertEqual(1, int(final_packs["summary"]["total"]))

    def test_pack_polling_while_remove_is_blocked_stays_truthful(self) -> None:
        source_dir = self._make_skill_source("polling_remove")
        ok, install_body = self.runtime.packs_install({"source": str(source_dir)})
        self.assertTrue(ok)
        canonical_id = str(install_body["pack"]["canonical_id"])
        self.assertTrue(canonical_id)

        entered = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []
        remove_result: list[dict[str, object]] = []
        original_remove = self.runtime.pack_store.remove_external_pack

        def blocked_remove(*args, **kwargs):  # type: ignore[no-untyped-def]
            entered.set()
            if not release.wait(3):
                raise RuntimeError("remove release timed out")
            return original_remove(*args, **kwargs)

        def remove_worker() -> None:
            try:
                ok, body = self.runtime.delete_external_pack(canonical_id)
                remove_result.append({"ok": ok, "body": body})
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        with mock.patch.object(self.runtime.pack_store, "remove_external_pack", side_effect=blocked_remove):
            thread = threading.Thread(target=remove_worker, daemon=True)
            thread.start()
            self.assertTrue(entered.wait(3))

            for _ in range(20):
                packs = self.runtime.packs_state()
                self.assertTrue(bool(packs.get("ok", False)))
                self._assert_pack_state_snapshot_consistent(packs)

            release.set()
            thread.join(timeout=5)

        self.assertFalse(errors, errors)
        self.assertTrue(remove_result)
        self.assertTrue(bool(remove_result[0]["ok"]))
        final_packs = self.runtime.packs_state()
        self.assertEqual(0, int(final_packs["summary"]["installed"]))
        self.assertFalse(any(row.get("pack_id") == canonical_id for row in self.runtime.pack_store.list_external_packs()))

    def test_fixit_confirm_overlap_executes_once_and_replays_cleanly(self) -> None:
        self.runtime.set_listening("127.0.0.1", 8765)
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
        self.runtime._llm_fixit_store.save(  # type: ignore[attr-defined]
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

        execute_started = threading.Event()
        allow_finish = threading.Event()
        execute_calls: list[int] = []
        errors: list[BaseException] = []
        original_execute = self.runtime._execute_llm_fixit_plan

        def blocked_execute(*args, **kwargs):  # type: ignore[no-untyped-def]
            execute_calls.append(1)
            execute_started.set()
            if not allow_finish.wait(3):
                raise RuntimeError("fixit release timed out")
            return original_execute(*args, **kwargs)

        def confirm_worker(results: list[dict[str, object]]) -> None:
            try:
                handler = _HandlerForTest(self.runtime, "/llm/fixit", {"confirm": True})
                handler.do_POST()
                results.append(json.loads(handler.body.decode("utf-8")))
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        results: list[dict[str, object]] = []
        with mock.patch.object(
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
        ), mock.patch("agent.api_server.evaluate_wizard_decision", return_value=SimpleNamespace(status="ok", issue_code="ok", question=None)):
            with mock.patch.object(self.runtime, "_execute_llm_fixit_plan", side_effect=blocked_execute):
                first = threading.Thread(target=confirm_worker, args=(results,), daemon=True)
                second = threading.Thread(target=confirm_worker, args=(results,), daemon=True)
                first.start()
                self.assertTrue(execute_started.wait(3))
                second.start()
                time.sleep(0.1)
                self.assertEqual(1, len(execute_calls))
                allow_finish.set()
                first.join(timeout=5)
                second.join(timeout=5)

        self.assertFalse(errors, errors)
        self.assertEqual(2, len(results))
        self.assertEqual(1, len(execute_calls))
        statuses = {row.get("status") for row in results}
        self.assertIn("already_consumed", statuses)
        self.assertTrue(any(str(row.get("status") or "") != "already_consumed" for row in results))
        self.assertFalse(bool(self.runtime._llm_fixit_store.state.get("active")))  # type: ignore[attr-defined]

    def test_registry_writes_remain_parseable_under_overlap(self) -> None:
        service = self.runtime._pack_registry_discovery()
        service.create_catalog_source(
            {
                "source_id": "local-a",
                "name": "Local A",
                "kind": "local_catalog",
                "base_url": "/tmp/a.json",
                "enabled": True,
                "discovery_only": True,
                "supports_search": True,
                "supports_preview": True,
                "supports_compare_hint": True,
                "notes": "initial",
            }
        )
        start = threading.Barrier(2)
        errors: list[BaseException] = []

        def update_worker(note: str) -> None:
            try:
                start.wait(timeout=3)
                service.update_catalog_source(
                    "local-a",
                    {
                        "name": "Local A",
                        "kind": "local_catalog",
                        "base_url": "/tmp/a.json",
                        "enabled": True,
                        "discovery_only": True,
                        "supports_search": True,
                        "supports_preview": True,
                        "supports_compare_hint": True,
                        "notes": note,
                    },
                )
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        first = threading.Thread(target=update_worker, args=("note-one",), daemon=True)
        second = threading.Thread(target=update_worker, args=("note-two",), daemon=True)
        first.start()
        second.start()
        first.join(timeout=5)
        second.join(timeout=5)

        self.assertFalse(errors, errors)
        catalog_path = Path(service.sources_path)
        parsed = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.assertIn("sources", parsed)
        self.assertTrue(any(row.get("id") == "local-a" for row in parsed["sources"]))
        source = service.get_catalog_source("local-a")
        self.assertEqual("local-a", source["source"]["id"])
        self.assertIn(str(source["source"]["notes"]), {"note-one", "note-two"})

    def test_retry_helpers_handle_busy_then_exhaust_cleanly(self) -> None:
        store = self.runtime.pack_store
        attempts = {"write": 0, "read": 0}

        def flaky_write() -> str:
            attempts["write"] += 1
            if attempts["write"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        def flaky_read() -> str:
            attempts["read"] += 1
            if attempts["read"] < 3:
                raise sqlite3.OperationalError("database is busy")
            return "ok"

        self.assertEqual("ok", store._write_with_retry(flaky_write))
        self.assertEqual("ok", store._read_with_retry(flaky_read))
        self.assertEqual(3, attempts["write"])
        self.assertEqual(3, attempts["read"])

        write_attempts = {"count": 0}

        def always_busy() -> None:
            write_attempts["count"] += 1
            raise sqlite3.OperationalError("database is locked")

        with self.assertRaises(sqlite3.OperationalError):
            store._write_with_retry(always_busy)
        self.assertEqual(3, write_attempts["count"])

    def test_startup_phase_polling_remains_truthful_under_transition_pressure(self) -> None:
        self.assertTrue(startup_phase_transition_allowed("starting", "listening"))
        self.assertTrue(startup_phase_transition_allowed("listening", "warming"))
        self.assertTrue(startup_phase_transition_allowed("warming", "ready"))
        self.assertTrue(startup_phase_transition_allowed("ready", "degraded"))

        errors: list[BaseException] = []
        seen_phases: list[str] = []
        stop = threading.Event()

        def writer() -> None:
            for phase in ("listening", "warming", "ready", "degraded"):
                self.runtime._set_startup_phase(phase)
                time.sleep(0.02)
            stop.set()

        def reader() -> None:
            try:
                while not stop.is_set():
                    ready = self.runtime.ready_status()
                    seen_phases.append(str(ready.get("startup_phase") or ""))
                    self.assertTrue(bool(ready.get("ok", False)))
                    self.assertIn(str(ready.get("startup_phase") or ""), {"starting", "listening", "warming", "ready", "degraded"})
                    self.assertIn(str(ready.get("phase") or ""), {"boot", "warmup", "ready", "degraded", "recovering"})
            except BaseException as exc:  # pragma: no cover - defensive
                errors.append(exc)

        writer_thread = threading.Thread(target=writer, daemon=True)
        readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
        for thread in readers:
            thread.start()
        writer_thread.start()
        writer_thread.join(timeout=5)
        for thread in readers:
            thread.join(timeout=5)

        self.assertFalse(errors, errors)
        self.assertTrue(seen_phases)
        self.assertIn("degraded", seen_phases)


if __name__ == "__main__":
    unittest.main()
