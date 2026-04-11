from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from agent.api_server import AgentRuntime
from agent.memory_runtime import MemoryRuntime
from agent.working_memory import append_turn

from tests.test_agent_workflows import _config
from tests.test_behavioral_eval_battery import _BehavioralRuntimeTruth, _first_line


class TestFailureRecoveryPaths(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.tmpdir.name, "registry.json")
        self.db_path = os.path.join(self.tmpdir.name, "agent.db")
        self._env_backup = dict(os.environ)
        os.environ["AGENT_SECRET_STORE_PATH"] = os.path.join(self.tmpdir.name, "secrets.enc.json")
        os.environ["AGENT_PERMISSIONS_PATH"] = os.path.join(self.tmpdir.name, "permissions.json")
        os.environ["AGENT_AUDIT_LOG_PATH"] = os.path.join(self.tmpdir.name, "audit.jsonl")

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)
        self.tmpdir.cleanup()

    def _runtime(self) -> tuple[AgentRuntime, _BehavioralRuntimeTruth]:
        truth = _BehavioralRuntimeTruth()
        runtime = AgentRuntime(_config(self.registry_path, self.db_path))
        runtime._runtime_truth_service = truth  # type: ignore[assignment]
        return runtime, truth

    def _chat(
        self,
        runtime: AgentRuntime,
        text: str,
        *,
        user_id: str = "user1",
        expect_transport_ok: bool | None = True,
        expect_body_ok: bool | None = True,
    ) -> dict[str, object]:
        with patch("agent.orchestrator.route_inference", side_effect=AssertionError("LLM should not run")), patch.object(
            runtime,
            "_auto_bootstrap_local_chat_model",
            return_value=None,
        ):
            ok, body = runtime.chat(
                {
                    "messages": [{"role": "user", "content": text}],
                    "source_surface": "api",
                    "user_id": user_id,
                }
            )
        if expect_transport_ok is True:
            self.assertTrue(ok, msg=str(body))
        elif expect_transport_ok is False:
            self.assertFalse(ok, msg=str(body))
        if expect_body_ok is True:
            self.assertTrue(bool(body.get("ok")), msg=str(body))
        elif expect_body_ok is False:
            self.assertFalse(bool(body.get("ok")), msg=str(body))
        return body

    def test_provider_unavailable_returns_actionable_setup_guidance(self) -> None:
        runtime, truth = self._runtime()
        truth.openrouter_health_status = "down"
        truth.openrouter_health_reason = "provider unavailable"
        truth.openrouter_configured = False
        truth.openrouter_secret_present = False

        response = self._chat(runtime, "is openrouter configured?")
        text = str((response.get("assistant") or {}).get("content") or "")
        self.assertIn("OpenRouter", text)
        self.assertIn("api key", text.lower())
        self.assertNotIn("need more context", text.lower())

    def test_discovery_source_failure_continues_and_preserves_debug_metadata(self) -> None:
        runtime, truth = self._runtime()
        truth.discovery_payload = {
            "ok": True,
            "query": None,
            "message": "Found 1 model(s) across 3 source(s), but some sources failed. Source errors: huggingface: hf timeout.",
            "models": [
                {
                    "id": "openrouter:vendor/tiny-gemma",
                    "provider": "openrouter",
                    "source": "openrouter",
                    "capabilities": ["chat"],
                    "local": False,
                    "installable": False,
                    "confidence": 0.74,
                }
            ],
            "sources": [
                {"source": "huggingface", "enabled": True, "queried": True, "ok": False, "count": 0, "error_kind": "fetch_failed", "error": "hf timeout"},
                {"source": "openrouter", "enabled": True, "queried": True, "ok": True, "count": 1, "error_kind": None, "error": None},
                {"source": "ollama", "enabled": False, "queried": False, "ok": True, "count": 0, "error_kind": "disabled", "error": None},
            ],
            "debug": {
                "source_registry": ["huggingface", "openrouter", "ollama", "external_snapshots"],
                "source_errors": {"huggingface": {"error_kind": "fetch_failed", "error": "hf timeout"}},
                "source_counts": {"openrouter": 1, "ollama": 0},
                "matched_count": 1,
            },
        }

        response = self._chat(runtime, "look for information on a new model from hugging face")
        text = str((response.get("assistant") or {}).get("content") or "")
        self.assertTrue(text.lower().startswith("the closest matches look like"))
        self.assertIn("hf timeout", text.lower())
        self.assertIn("model_discovery_manager", response.get("meta", {}).get("used_tools", []))
        self.assertNotIn("does not exist", text.lower())

    def test_missing_config_is_actionable_and_not_vague(self) -> None:
        runtime, truth = self._runtime()
        truth.openrouter_secret_present = False
        response = self._chat(runtime, "configure openrouter")
        text = str((response.get("assistant") or {}).get("content") or "")
        self.assertIn("Paste your OpenRouter API key", text)
        self.assertNotIn("need more context", text.lower())

    def test_runtime_degraded_status_explains_state_and_next_step(self) -> None:
        runtime, truth = self._runtime()
        truth.runtime_status = lambda kind="runtime_status": {  # type: ignore[assignment]
            "scope": "ready",
            "ready": False,
            "runtime_mode": "DEGRADED",
            "failure_code": "provider_unhealthy",
            "provider": truth.effective_provider,
            "model": truth.effective_model,
            "configured_provider": truth.default_provider,
            "configured_model": truth.default_model,
            "qualification_reason": "The runtime is degraded because the active provider is unavailable.",
            "summary": "The runtime is degraded because the active provider is unavailable.",
        }

        response = self._chat(runtime, "what is the runtime status?")
        text = str((response.get("assistant") or {}).get("content") or "")
        self.assertIn("degraded", text.lower())
        self.assertIn("provider", text.lower())
        self.assertNotIn("{", _first_line(text))

    def test_install_failure_produces_next_step_not_success(self) -> None:
        runtime, truth = self._runtime()
        with patch.object(
            truth,
            "acquire_chat_model_target",
            return_value=(False, {"ok": False, "error_kind": "provider_unavailable", "message": "Ollama is down. Restart it and try again.", "next_action": "Restart Ollama and retry."}),
        ):
            preview = self._chat(runtime, "install ollama:qwen2.5:7b-instruct")
            preview_text = str((preview.get("assistant") or {}).get("content") or "")
            self.assertIn("Say yes to continue, or no to cancel.", preview_text)
            failure = self._chat(runtime, "yes", expect_transport_ok=None, expect_body_ok=False)
        failure_text = str((failure.get("assistant") or {}).get("content") or "")
        self.assertIn("restart it and try again", failure_text.lower())
        self.assertNotIn("Started acquiring", failure_text)
        self.assertNotIn("success", failure_text.lower())

    def test_pack_enablement_returns_actionable_failure(self) -> None:
        runtime, _truth = self._runtime()
        ok, body = runtime.packs_enable({"pack_id": "missing-pack", "enabled": True})
        self.assertFalse(ok)
        self.assertEqual("pack_not_found", body.get("error"))
        self.assertIn("Install the pack first", str(body.get("next_question") or ""))

    def test_brief_surface_remains_user_facing_and_non_silent(self) -> None:
        runtime, _truth = self._runtime()
        orch = runtime.orchestrator()

        def _insert_system_facts(snapshot_id: str, taken_at: str, load_1m: float, mem_used: int, disk_used: int) -> None:
            facts = {
                "schema": {"name": "system_facts", "version": 1},
                "snapshot": {
                    "snapshot_id": snapshot_id,
                    "taken_at": taken_at,
                    "timezone": "UTC",
                    "collector": {
                        "agent_version": "0.6.0",
                        "hostname": "host",
                        "boot_id": "boot",
                        "uptime_s": 1,
                        "collection_duration_ms": 1,
                        "partial": False,
                        "errors": [],
                    },
                    "provenance": {"sources": []},
                },
                "os": {"kernel": {"release": "6.0.0", "arch": "x86_64"}},
                "cpu": {"load": {"load_1m": load_1m, "load_5m": load_1m, "load_15m": load_1m}},
                "memory": {
                    "ram_bytes": {
                        "total": 16 * 1024**3,
                        "used": mem_used,
                        "free": 0,
                        "available": (16 * 1024**3) - mem_used,
                        "buffers": 0,
                        "cached": 0,
                    },
                    "swap_bytes": {"total": 0, "free": 0, "used": 0},
                    "pressure": {"psi_supported": False, "memory_some_avg10": None, "io_some_avg10": None, "cpu_some_avg10": None},
                },
                "filesystems": {
                    "mounts": [
                        {
                            "mountpoint": "/",
                            "device": "/dev/sda1",
                            "fstype": "ext4",
                            "total_bytes": 100 * 1024**3,
                            "used_bytes": disk_used,
                            "avail_bytes": 100 * 1024**3 - disk_used,
                            "used_pct": (float(disk_used) / float(100 * 1024**3)) * 100.0,
                            "inodes": {"total": None, "used": None, "avail": None, "used_pct": None},
                        }
                    ]
                },
                "process_summary": {"top_cpu": [], "top_mem": [{"pid": 1, "name": "proc", "cpu_pct": None, "rss_bytes": mem_used // 4}]},
                "integrity": {"content_hash_sha256": "0" * 64, "signed": False, "signature": None},
            }
            facts_json = json.dumps(facts, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            orch.db.insert_system_facts_snapshot(
                id=snapshot_id,
                user_id="user1",
                taken_at=taken_at,
                boot_id="boot",
                schema_version=1,
                facts_json=facts_json,
                content_hash_sha256="0" * 64,
                partial=False,
                errors_json="[]",
            )

        def _observe_now(_ctx: object, user_id: str | None = None) -> dict[str, object]:
            _insert_system_facts("snap-1", "2026-02-06T00:00:00+00:00", load_1m=0.1, mem_used=2 * 1024**3, disk_used=60 * 1024**3)
            return {"text": "Snapshot taken: 2026-02-06T00:00:00+00:00 (UTC)", "payload": {}}

        orch._call_skill = lambda *args, **kwargs: _observe_now({}, user_id="user1")  # type: ignore[assignment]

        response = orch.handle_message("/brief", "user1")
        self.assertIn("i have a baseline now", response.text.lower())
        self.assertNotIn("/resource_report", response.text)
        self.assertNotIn("debug", response.text.lower())


if __name__ == "__main__":
    unittest.main()
